"""
Data loader for NYC traffic demand prediction.

Supports two data layouts:

  Layout A — ST-SSL style (recommended):
    data_dir/train.npz, val.npz, test.npz, adj_mx.npz
    Each .npz contains:
      X: (samples, lookback_window, nodes, features)
      Y: (samples, predict_horizon, nodes, features)

  Layout B — single file:
    A single .npz with key 'data' containing (T, N, C) raw time series

  Fallback — synthetic data for pipeline testing.

Download ST-SSL data (NYCTaxi, NYCBike1, NYCBike2, BJTaxi):
  Google Drive: https://drive.google.com/file/d/1n0y6X8pWNVwHxtFUuY8WsTYZHwBe9GeS
  Then unzip and place e.g. NYCTaxi/ under data/processed/
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.utils import StandardScaler, get_project_path


# ============================================================
# Dataset (for single-file raw time series)
# ============================================================

class TrafficDataset(Dataset):
    """Sliding-window dataset: raw (T, N, C) → (x, y) pairs."""

    def __init__(self, data: np.ndarray, input_len: int, output_len: int):
        super().__init__()
        self.data = data.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.num_samples = len(data) - self.total_len + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Data length {len(data)} too short for "
                f"input_len={input_len} + output_len={output_len}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_len]
        y = self.data[idx + self.input_len: idx + self.total_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ============================================================
# NPZ loading helpers
# ============================================================

def _load_xy_from_npz(path: str):
    """
    Load pre-windowed (X, Y) from ST-SSL format .npz.
    Handles both uppercase (X, Y) and lowercase (x, y) keys.
    Returns (X, Y) as float32 arrays, or None if not found.
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    # try uppercase first (ST-SSL convention), then lowercase
    x_key = 'X' if 'X' in keys else ('x' if 'x' in keys else None)
    y_key = 'Y' if 'Y' in keys else ('y' if 'y' in keys else None)

    if x_key and y_key:
        X = npz[x_key].astype(np.float32)
        Y = npz[y_key].astype(np.float32)
        return X, Y
    return None


def _load_raw_array(path: str) -> np.ndarray:
    """Load a raw (T, N, C) array from .npz, trying common key names."""
    npz = np.load(path, allow_pickle=True)
    for key in ['data', 'arr_0', 'flow_data', 'demand']:
        if key in npz:
            return npz[key].astype(np.float32)
    first_key = list(npz.keys())[0]
    return npz[first_key].astype(np.float32)


def load_adj(data_dir: str) -> np.ndarray:
    """Load adjacency matrix from adj_mx.npz in data_dir."""
    adj_path = os.path.join(data_dir, 'adj_mx.npz')
    if not os.path.exists(adj_path):
        return None
    npz = np.load(adj_path, allow_pickle=True)
    keys = list(npz.keys())
    for key in ['adj_mx', 'adj', 'arr_0']:
        if key in keys:
            adj = npz[key]
            # handle 0-d object arrays (scipy sparse stored as object)
            if adj.ndim == 0:
                adj = adj.item()
                if hasattr(adj, 'toarray'):
                    adj = adj.toarray()
            return np.array(adj, dtype=np.float32)
    first_key = keys[0]
    return np.array(npz[first_key], dtype=np.float32)


# ============================================================
# Synthetic data for testing
# ============================================================

def generate_synthetic_data(
    num_steps: int = 4032,
    num_nodes: int = 200,
    num_features: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic traffic data with daily periodicity + noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(num_steps)
    daily = np.sin(2 * np.pi * t / 48)
    weekly = 0.3 * np.sin(2 * np.pi * t / 336)
    base = rng.exponential(scale=30, size=(num_nodes, num_features))
    data = np.zeros((num_steps, num_nodes, num_features), dtype=np.float32)
    for n in range(num_nodes):
        for c in range(num_features):
            trend = base[n, c] * (1 + 0.5 * daily + weekly)
            noise = rng.normal(0, base[n, c] * 0.1, size=num_steps)
            data[:, n, c] = np.maximum(trend + noise, 0)
    print(f"Generated synthetic data: {data.shape}")
    return data


# ============================================================
# Main entry point
# ============================================================

def build_dataloaders(config: dict):
    """
    Build train/val/test DataLoaders.

    Returns: train_loader, val_loader, test_loader, scaler, adj
      - adj may be None if adj_mx.npz is not found
    """
    dcfg = config['data']
    tcfg = config['train']
    batch_size = tcfg['batch_size']

    # ---- Try Layout A: ST-SSL split files ----
    data_dir = dcfg.get('data_dir', '')
    if data_dir and not os.path.isabs(data_dir):
        data_dir = get_project_path(data_dir)
    train_path = os.path.join(data_dir, 'train.npz') if data_dir else ''

    if data_dir and os.path.exists(train_path):
        print(f"[DATA] Found ST-SSL layout in {data_dir}")
        adj = load_adj(data_dir)

        # Load pre-windowed X, Y
        result = _load_xy_from_npz(train_path)
        if result is not None:
            x_train, y_train = result
            x_val, y_val = _load_xy_from_npz(os.path.join(data_dir, 'val.npz'))
            x_test, y_test = _load_xy_from_npz(os.path.join(data_dir, 'test.npz'))

            print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
            print(f"  x_val:   {x_val.shape},   y_val:   {y_val.shape}")
            print(f"  x_test:  {x_test.shape},  y_test:  {y_test.shape}")
            if adj is not None:
                print(f"  adj:     {adj.shape}")

            # Fit scaler on training data
            scaler = StandardScaler()
            # reshape to (all_timesteps, N, C) for fitting
            all_train = x_train.reshape(-1, x_train.shape[-2], x_train.shape[-1])
            scaler.fit(all_train)

            # Normalize all splits
            def normalize(x, y):
                sx = x.shape
                x_norm = scaler.transform(x.reshape(-1, sx[-2], sx[-1])).reshape(sx)
                sy = y.shape
                y_norm = scaler.transform(y.reshape(-1, sy[-2], sy[-1])).reshape(sy)
                return x_norm, y_norm

            x_train, y_train = normalize(x_train, y_train)
            x_val, y_val = normalize(x_val, y_val)
            x_test, y_test = normalize(x_test, y_test)

            def make_loader(x, y, shuffle):
                ds = TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
                return DataLoader(ds, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=shuffle)

            print(f"  Samples: train={len(x_train)}, "
                  f"val={len(x_val)}, test={len(x_test)}")

            return (make_loader(x_train, y_train, True),
                    make_loader(x_val, y_val, False),
                    make_loader(x_test, y_test, False),
                    scaler, adj)
        else:
            # npz has raw time series, not pre-windowed
            print("  No X/Y keys found, trying raw time series format...")
            # (falls through to Layout B logic below)

    # ---- Fallback: synthetic ----
    print("[WARNING] No real data found, using synthetic data for testing.")
    data = generate_synthetic_data(
        num_nodes=dcfg['num_nodes'], num_features=dcfg['input_dim'])

    total = len(data)
    t1 = int(total * dcfg['train_ratio'])
    t2 = int(total * (dcfg['train_ratio'] + dcfg['val_ratio']))

    scaler = StandardScaler().fit(data[:t1])
    splits = [scaler.transform(d) for d in [data[:t1], data[t1:t2], data[t2:]]]

    input_len = dcfg['input_len']
    output_len = dcfg['output_len']
    loaders = []
    for d, shuf in zip(splits, [True, False, False]):
        ds = TrafficDataset(d, input_len, output_len)
        loaders.append(DataLoader(ds, batch_size=batch_size,
                                  shuffle=shuf, drop_last=shuf))

    print(f"  Samples: train={len(loaders[0].dataset)}, "
          f"val={len(loaders[1].dataset)}, test={len(loaders[2].dataset)}")

    return (*loaders, scaler, None)
