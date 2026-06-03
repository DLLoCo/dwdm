"""
Data loader for NYC traffic demand prediction.

Supports two data layouts:
  Layout A — ST-SSL style (recommended):
    data_dir/train.npz, val.npz, test.npz, adj_mx.npz
  Layout B — single file with (T, N, C)
  Fallback — synthetic data for pipeline testing

v6 addition: when config has use_cluster=True, computes dual clusters
and returns cluster_info as 6th return value.
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
    """Load pre-windowed (X, Y) from ST-SSL format .npz."""
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())
    x_key = 'X' if 'X' in keys else ('x' if 'x' in keys else None)
    y_key = 'Y' if 'Y' in keys else ('y' if 'y' in keys else None)
    if x_key and y_key:
        return npz[x_key].astype(np.float32), npz[y_key].astype(np.float32)
    return None


def _load_raw_array(path: str) -> np.ndarray:
    """Load a raw (T, N, C) array from .npz."""
    npz = np.load(path, allow_pickle=True)
    for key in ['data', 'arr_0', 'flow_data', 'demand']:
        if key in npz:
            return npz[key].astype(np.float32)
    keys = list(npz.keys())
    if keys:
        return npz[keys[0]].astype(np.float32)
    raise KeyError(f"No data found in {path}")


def load_adj(data_dir: str):
    """Load adjacency matrix from adj_mx.npz if it exists."""
    adj_path = os.path.join(data_dir, 'adj_mx.npz')
    if os.path.exists(adj_path):
        npz = np.load(adj_path, allow_pickle=True)
        for key in ['adj_mx', 'adj', 'arr_0']:
            if key in npz:
                return npz[key].astype(np.float32)
        keys = list(npz.keys())
        if keys:
            return npz[keys[0]].astype(np.float32)
    return None


def generate_synthetic_data(num_nodes=200, num_features=2,
                            num_days=84, steps_per_day=48):
    """Generate synthetic data for pipeline testing."""
    T = num_days * steps_per_day
    t = np.arange(T)
    daily = np.sin(2 * np.pi * (t % steps_per_day) / steps_per_day)
    weekly = 0.3 * np.sin(2 * np.pi * (t % (7 * steps_per_day)) /
                          (7 * steps_per_day))
    base = daily + weekly
    data = np.zeros((T, num_nodes, num_features), dtype=np.float32)
    rng = np.random.RandomState(42)
    for n in range(num_nodes):
        scale = rng.exponential(50)
        for c in range(num_features):
            noise = rng.randn(T) * 0.1
            data[:, n, c] = np.maximum(0, scale * (base + noise + 1))
    return data


# ============================================================
# Main builder
# ============================================================

def build_dataloaders(config):
    """
    Build train/val/test DataLoaders from config.

    Returns: (train_loader, val_loader, test_loader, scaler, adj, cluster_info)
      - adj may be None
      - cluster_info is None when use_cluster=False, otherwise a dict with:
          'spatial_labels':   (N,) int64 tensor
          'n_spatial':        int
          'n_temporal':       int
    """
    dcfg = config['data']
    tcfg = config['train']
    mcfg = config['model']
    batch_size = tcfg['batch_size']

    use_cluster = mcfg.get('use_cluster', False)

    # ---- Try Layout A: ST-SSL split files ----
    data_dir = dcfg.get('data_dir', '')
    if data_dir and not os.path.isabs(data_dir):
        data_dir = get_project_path(data_dir)
    train_path = os.path.join(data_dir, 'train.npz') if data_dir else ''

    if data_dir and os.path.exists(train_path):
        print(f"[DATA] Found ST-SSL layout in {data_dir}")
        adj = load_adj(data_dir)

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

            # ---- Clustering (v6): compute BEFORE normalization ----
            cluster_info = None
            tc_train = tc_val = tc_test = None

            if use_cluster:
                from lib.cluster_builder import build_clusters
                n_sp = mcfg.get('n_spatial_clusters', 5)
                n_tp = mcfg.get('n_temporal_clusters', 5)
                cl = build_clusters(x_train, x_val, x_test,
                                    n_spatial=n_sp, n_temporal=n_tp)
                cluster_info = {
                    'spatial_labels': torch.LongTensor(cl['spatial_labels']),
                    'n_spatial': cl['n_spatial'],
                    'n_temporal': cl['n_temporal'],
                    'detail': cl,  # for visualization
                }
                tc_train = cl['temporal_train']
                tc_val = cl['temporal_val']
                tc_test = cl['temporal_test']

            # ---- Normalize ----
            scaler = StandardScaler()
            all_train = x_train.reshape(-1, x_train.shape[-2],
                                        x_train.shape[-1])
            scaler.fit(all_train)

            def normalize(x, y):
                sx = x.shape
                x_n = scaler.transform(
                    x.reshape(-1, sx[-2], sx[-1])).reshape(sx)
                sy = y.shape
                y_n = scaler.transform(
                    y.reshape(-1, sy[-2], sy[-1])).reshape(sy)
                return x_n, y_n

            x_train, y_train = normalize(x_train, y_train)
            x_val, y_val = normalize(x_val, y_val)
            x_test, y_test = normalize(x_test, y_test)

            # ---- Time indices (approximate) ----
            def compute_time_indices(n_samples, offset):
                indices = np.arange(n_samples) + offset
                hour_slot = indices % 48
                day_of_week = ((indices // 48) + 3) % 7  # Jan 1 2015 = Thu
                return hour_slot.astype(np.int64), day_of_week.astype(np.int64)

            offset0 = 144
            h_train, d_train = compute_time_indices(len(x_train), offset0)
            h_val, d_val = compute_time_indices(
                len(x_val), offset0 + len(x_train))
            h_test, d_test = compute_time_indices(
                len(x_test), offset0 + len(x_train) + len(x_val))
            print(f"  Time indices: hour_slot [0-47], day_of_week [0-6]")

            # ---- Build DataLoaders ----
            def make_loader(x, y, hour, dow, tc, shuffle):
                tensors = [torch.FloatTensor(x), torch.FloatTensor(y),
                           torch.LongTensor(hour), torch.LongTensor(dow)]
                if tc is not None:
                    tensors.append(torch.LongTensor(tc))
                ds = TensorDataset(*tensors)
                return DataLoader(ds, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=shuffle)

            print(f"  Samples: train={len(x_train)}, "
                  f"val={len(x_val)}, test={len(x_test)}")

            return (make_loader(x_train, y_train, h_train, d_train,
                                tc_train, True),
                    make_loader(x_val, y_val, h_val, d_val,
                                tc_val, False),
                    make_loader(x_test, y_test, h_test, d_test,
                                tc_test, False),
                    scaler, adj, cluster_info)

    # ---- Fallback: synthetic ----
    print("[WARNING] No real data found, using synthetic data for testing.")
    data = generate_synthetic_data(
        num_nodes=dcfg['num_nodes'], num_features=dcfg['input_dim'])

    total = len(data)
    t1 = int(total * dcfg['train_ratio'])
    t2 = int(total * (dcfg['train_ratio'] + dcfg['val_ratio']))

    scaler = StandardScaler().fit(data[:t1])
    input_len = dcfg['input_len']
    output_len = dcfg['output_len']
    win = input_len + output_len

    loaders = []
    offset = 0
    for d_raw, shuf in zip([data[:t1], data[t1:t2], data[t2:]],
                           [True, False, False]):
        d = scaler.transform(d_raw)
        n = len(d) - win + 1
        xs, ys = [], []
        for i in range(n):
            xs.append(d[i:i + input_len])
            ys.append(d[i + input_len:i + win])
        x_t = torch.FloatTensor(np.array(xs))
        y_t = torch.FloatTensor(np.array(ys))
        indices = np.arange(n) + offset
        h_t = torch.LongTensor(indices % 48)
        d_t = torch.LongTensor((indices // 48) % 7)
        ds = TensorDataset(x_t, y_t, h_t, d_t)
        loaders.append(DataLoader(ds, batch_size=batch_size,
                                  shuffle=shuf, drop_last=shuf))
        offset += n

    print(f"  Samples: train={len(loaders[0].dataset)}, "
          f"val={len(loaders[1].dataset)}, test={len(loaders[2].dataset)}")

    return (*loaders, scaler, None, None)
