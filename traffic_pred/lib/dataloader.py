"""
Data loader for NYC traffic demand prediction.

Supports three data layouts:

  Layout A — ST-SSL style:
    data_dir/train.npz, val.npz, test.npz, adj_mx.npz
    Non-consecutive 35-step windows.

  Layout B — Continuous (recommended):
    Single demand.npz with (T, N, C) raw time series.
    Consecutive sliding window, real timestamps.
    Triggered by: data_mode: continuous in config.

  Fallback — synthetic data for pipeline testing.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lib.utils import StandardScaler, get_project_path


# ============================================================
# Helpers
# ============================================================

def load_adj(data_dir: str):
    adj_path = os.path.join(data_dir, 'adj_mx.npz')
    if os.path.exists(adj_path):
        npz = np.load(adj_path, allow_pickle=True)
        key = 'adj_mx' if 'adj_mx' in npz else list(npz.keys())[0]
        return npz[key].astype(np.float32)
    return None


def _load_xy_from_npz(path: str):
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())
    x_key = 'X' if 'X' in keys else ('x' if 'x' in keys else None)
    y_key = 'Y' if 'Y' in keys else ('y' if 'y' in keys else None)
    if x_key and y_key:
        return npz[x_key].astype(np.float32), npz[y_key].astype(np.float32)
    return None


def _make_loader(x, y, hour, dow, batch_size, shuffle):
    ds = TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y),
                       torch.LongTensor(hour), torch.LongTensor(dow))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)


# ============================================================
# Layout B: Continuous sliding window
# ============================================================

def _build_continuous(config):
    """
    Load (T, N, C) demand tensor, split temporally, create sliding windows.
    Returns: train_loader, val_loader, test_loader, scaler, adj
    """
    dcfg = config['data']
    tcfg = config['train']
    batch_size = tcfg['batch_size']
    input_len = dcfg['input_len']
    output_len = dcfg['output_len']
    start_dow = dcfg.get('start_weekday', 3)  # 2015-01-01 = Thursday

    # --- Load demand ---
    demand_path = dcfg['demand_path']
    if not os.path.isabs(demand_path):
        demand_path = get_project_path(demand_path)

    raw = np.load(demand_path)
    key = 'data' if 'data' in raw else list(raw.keys())[0]
    data = raw[key].astype(np.float32)  # (T, N, C)
    T, N, C = data.shape
    print(f"[DATA] Continuous mode: {demand_path}")
    print(f"  Shape: ({T}, {N}, {C}) = {T/48:.1f} days")

    # --- Load adjacency (optional) ---
    adj = None
    # Try ST-SSL adj first
    stssl_dir = get_project_path('data', 'processed', 'NYCTaxi')
    if os.path.isdir(stssl_dir):
        adj = load_adj(stssl_dir)
        if adj is not None:
            print(f"  Adj: loaded from {stssl_dir}, shape={adj.shape}")

    # --- Temporal split (no shuffling across splits!) ---
    n_train = int(T * dcfg['train_ratio'])
    n_val = int(T * dcfg['val_ratio'])
    n_test = T - n_train - n_val

    data_train = data[:n_train]
    data_val = data[n_train:n_train + n_val]
    data_test = data[n_train + n_val:]

    print(f"  Split: train={n_train} val={n_val} test={n_test} steps")

    # --- Normalize (fit on train only) ---
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_val = scaler.transform(data_val)
    data_test = scaler.transform(data_test)

    # --- Sliding window + time indices ---
    window = input_len + output_len

    def make_windows(d, offset):
        """
        d: (T_split, N, C) normalized data
        offset: absolute time step index of d[0] in the full series
        Returns: x, y, hour_slots, day_of_weeks
        """
        n = len(d) - window + 1
        if n <= 0:
            raise ValueError(f"Split too short: {len(d)} < {window}")

        xs = np.zeros((n, input_len, N, C), dtype=np.float32)
        ys = np.zeros((n, output_len, N, C), dtype=np.float32)
        hours = np.zeros(n, dtype=np.int64)
        dows = np.zeros(n, dtype=np.int64)

        for i in range(n):
            xs[i] = d[i:i + input_len]
            ys[i] = d[i + input_len:i + window]

            # Target time step = offset + i + input_len
            target_step = offset + i + input_len
            hours[i] = target_step % 48                         # 0-47
            dows[i] = ((target_step // 48) + start_dow) % 7     # 0=Mon

        return xs, ys, hours, dows

    x_train, y_train, h_train, d_train = make_windows(data_train, 0)
    x_val, y_val, h_val, d_val = make_windows(data_val, n_train)
    x_test, y_test, h_test, d_test = make_windows(data_test, n_train + n_val)

    print(f"  Samples: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    print(f"  x: ({x_train.shape[0]}, {input_len}, {N}, {C}) → "
          f"y: ({y_train.shape[0]}, {output_len}, {N}, {C})")

    # --- DataLoaders ---
    train_loader = _make_loader(x_train, y_train, h_train, d_train, batch_size, True)
    val_loader = _make_loader(x_val, y_val, h_val, d_val, batch_size, False)
    test_loader = _make_loader(x_test, y_test, h_test, d_test, batch_size, False)

    return train_loader, val_loader, test_loader, scaler, adj


# ============================================================
# Layout A: ST-SSL pre-windowed
# ============================================================

def _build_stssl(config):
    """Load ST-SSL split files (train/val/test.npz)."""
    dcfg = config['data']
    tcfg = config['train']
    batch_size = tcfg['batch_size']

    data_dir = dcfg.get('data_dir', '')
    if data_dir and not os.path.isabs(data_dir):
        data_dir = get_project_path(data_dir)

    print(f"[DATA] ST-SSL layout in {data_dir}")
    adj = load_adj(data_dir)

    x_train, y_train = _load_xy_from_npz(os.path.join(data_dir, 'train.npz'))
    x_val, y_val = _load_xy_from_npz(os.path.join(data_dir, 'val.npz'))
    x_test, y_test = _load_xy_from_npz(os.path.join(data_dir, 'test.npz'))

    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")

    # Normalize
    scaler = StandardScaler()
    all_train = x_train.reshape(-1, x_train.shape[-2], x_train.shape[-1])
    scaler.fit(all_train)

    def normalize(x, y):
        sx, sy = x.shape, y.shape
        xn = scaler.transform(x.reshape(-1, sx[-2], sx[-1])).reshape(sx)
        yn = scaler.transform(y.reshape(-1, sy[-2], sy[-1])).reshape(sy)
        return xn, yn

    x_train, y_train = normalize(x_train, y_train)
    x_val, y_val = normalize(x_val, y_val)
    x_test, y_test = normalize(x_test, y_test)

    # Time indices (approximate, from sample index)
    def compute_time_indices(n_samples, offset):
        idx = np.arange(n_samples) + offset
        return (idx % 48).astype(np.int64), (((idx // 48) + 3) % 7).astype(np.int64)

    offset0 = 144  # 3-day lookback
    h_tr, d_tr = compute_time_indices(len(x_train), offset0)
    h_va, d_va = compute_time_indices(len(x_val), offset0 + len(x_train))
    h_te, d_te = compute_time_indices(len(x_test), offset0 + len(x_train) + len(x_val))

    print(f"  Samples: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    train_loader = _make_loader(x_train, y_train, h_tr, d_tr, batch_size, True)
    val_loader = _make_loader(x_val, y_val, h_va, d_va, batch_size, False)
    test_loader = _make_loader(x_test, y_test, h_te, d_te, batch_size, False)

    return train_loader, val_loader, test_loader, scaler, adj


# ============================================================
# Main entry
# ============================================================

def build_dataloaders(config):
    """
    Build train/val/test dataloaders based on config.

    Returns: (train_loader, val_loader, test_loader, scaler, adj)
        Each loader yields: (x, y, hour_slot, day_of_week)
        - x: (B, T_in, N, C)   input sequence
        - y: (B, T_out, N, C)  prediction target
        - hour_slot: (B,)      0-47, target time's half-hour of day
        - day_of_week: (B,)    0-6, Mon=0
    """
    dcfg = config['data']

    # Route to appropriate layout
    mode = dcfg.get('data_mode', 'auto')

    if mode == 'continuous':
        return _build_continuous(config)

    # Auto-detect: check for ST-SSL files
    data_dir = dcfg.get('data_dir', '')
    if data_dir and not os.path.isabs(data_dir):
        data_dir = get_project_path(data_dir)
    train_path = os.path.join(data_dir, 'train.npz') if data_dir else ''

    if data_dir and os.path.exists(train_path):
        return _build_stssl(config)

    # Fallback: check for demand file
    demand_path = dcfg.get('demand_path', '')
    if demand_path:
        config['data']['data_mode'] = 'continuous'
        return _build_continuous(config)

    raise FileNotFoundError(
        "No data found. Set data_mode: continuous with demand_path, "
        "or provide ST-SSL files in data_dir."
    )
