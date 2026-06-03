"""
Dual Clustering for Spatio-Temporal Heterogeneity
==================================================
Challenge 2: 出行热点区域在时间维度上的动态迁移规律

Two clustering dimensions:
  1. Spatial clustering:  group 200 nodes by demand pattern similarity
     → "商业核心", "通勤起点", "低需求冷区" etc.
  2. Temporal clustering: group samples by recent demand level
     → "深夜低谷", "早高峰上升", "白天平台", "晚高峰" etc.

Both are learned from training data only (no data leakage).
Temporal clusters are inferred for val/test via .predict().

Usage:
    from lib.cluster_builder import build_clusters
    cluster_info = build_clusters(x_train, x_val, x_test,
                                  n_spatial=5, n_temporal=5)
    # cluster_info['spatial_labels']   : (N_nodes,)
    # cluster_info['temporal_train']   : (N_train,)
    # cluster_info['temporal_val']     : (N_val,)
    # cluster_info['temporal_test']    : (N_test,)
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SKScaler


def _node_features(x_all: np.ndarray) -> np.ndarray:
    """
    Extract per-node feature vector from training data.

    x_all: (N_samples, T, N_nodes, C)
    Returns: (N_nodes, 4) — [mean_pickup, std_pickup, mean_dropoff, std_dropoff]

    Nodes with similar demand magnitude and variability cluster together.
    """
    # Flatten samples & timesteps → per-node statistics
    # x_all shape: (S, T, N, C)
    N = x_all.shape[2]
    C = x_all.shape[3]
    feats = []
    for c in range(C):
        vals = x_all[:, :, :, c]           # (S, T, N)
        node_mean = vals.mean(axis=(0, 1))  # (N,)
        node_std = vals.std(axis=(0, 1))    # (N,)
        feats.append(node_mean)
        feats.append(node_std)
    return np.stack(feats, axis=1)          # (N, C*2)


def _sample_features(x: np.ndarray, last_k: int = 8) -> np.ndarray:
    """
    Extract per-sample feature vector from recent demand.

    x: (N_samples, T, N_nodes, C)
    Returns: (N_samples, last_k * C) — recent global demand profile

    Uses the last `last_k` steps (= recent 4 hours for 30-min intervals).
    Global mean over nodes captures "what demand level is the city at".
    """
    recent = x[:, -last_k:, :, :]          # (S, last_k, N, C)
    profile = recent.mean(axis=2)           # (S, last_k, C)
    return profile.reshape(len(x), -1)      # (S, last_k * C)


def build_spatial_clusters(x_train: np.ndarray,
                           n_clusters: int = 5,
                           random_state: int = 42) -> dict:
    """
    Cluster nodes by demand pattern similarity.

    Returns dict with:
      'labels':     (N_nodes,) int array, cluster assignment
      'centers':    (K, feat_dim) cluster centroids
      'features':   (N_nodes, feat_dim) raw features (for visualization)
    """
    features = _node_features(x_train)
    scaler = SKScaler()
    features_norm = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    labels = km.fit_predict(features_norm)

    # Sort clusters by mean demand (cluster 0 = coldest, K-1 = hottest)
    cluster_means = []
    for k in range(n_clusters):
        mask = labels == k
        cluster_means.append(features[mask, 0].mean())  # mean of mean_pickup
    order = np.argsort(cluster_means)
    remap = np.zeros(n_clusters, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    labels = remap[labels]

    print(f"  Spatial clustering: {n_clusters} clusters")
    for k in range(n_clusters):
        count = (labels == k).sum()
        mean_demand = features[labels == k, 0].mean()
        print(f"    Cluster {k}: {count:3d} nodes, "
              f"mean_pickup={mean_demand:.1f}")

    return {
        'labels': labels.astype(np.int64),
        'centers': km.cluster_centers_,
        'features': features,
        'n_clusters': n_clusters,
    }


def build_temporal_clusters(x_train: np.ndarray,
                            x_val: np.ndarray,
                            x_test: np.ndarray,
                            n_clusters: int = 5,
                            last_k: int = 8,
                            random_state: int = 42) -> dict:
    """
    Cluster samples by recent demand level.

    Fit K-Means on training data, predict for val/test.
    Returns dict with:
      'train_labels':  (N_train,)
      'val_labels':    (N_val,)
      'test_labels':   (N_test,)
      'n_clusters':    int
    """
    feat_train = _sample_features(x_train, last_k)
    feat_val = _sample_features(x_val, last_k)
    feat_test = _sample_features(x_test, last_k)

    scaler = SKScaler()
    feat_train_norm = scaler.fit_transform(feat_train)
    feat_val_norm = scaler.transform(feat_val)
    feat_test_norm = scaler.transform(feat_test)

    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    train_labels = km.fit_predict(feat_train_norm)
    val_labels = km.predict(feat_val_norm)
    test_labels = km.predict(feat_test_norm)

    # Sort by mean demand level
    cluster_demand = []
    for k in range(n_clusters):
        mask = train_labels == k
        cluster_demand.append(feat_train[mask].mean())
    order = np.argsort(cluster_demand)
    remap = np.zeros(n_clusters, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    train_labels = remap[train_labels]
    val_labels = remap[val_labels]
    test_labels = remap[test_labels]

    print(f"  Temporal clustering: {n_clusters} clusters")
    for k in range(n_clusters):
        count = (train_labels == k).sum()
        mean_level = feat_train[train_labels == k].mean()
        print(f"    Cluster {k}: {count:4d} samples, "
              f"mean_demand_level={mean_level:.3f}")

    return {
        'train_labels': train_labels.astype(np.int64),
        'val_labels': val_labels.astype(np.int64),
        'test_labels': test_labels.astype(np.int64),
        'n_clusters': n_clusters,
    }


def build_clusters(x_train: np.ndarray,
                   x_val: np.ndarray,
                   x_test: np.ndarray,
                   n_spatial: int = 5,
                   n_temporal: int = 5,
                   random_state: int = 42) -> dict:
    """
    Build both spatial and temporal clusters.

    Returns dict with all cluster info needed by model and dataloader.
    """
    print("[CLUSTER] Building dual clusters...")

    spatial = build_spatial_clusters(
        x_train, n_clusters=n_spatial, random_state=random_state)

    temporal = build_temporal_clusters(
        x_train, x_val, x_test,
        n_clusters=n_temporal, random_state=random_state)

    return {
        'spatial_labels': spatial['labels'],        # (N_nodes,)
        'n_spatial': spatial['n_clusters'],
        'temporal_train': temporal['train_labels'],  # (N_train,)
        'temporal_val': temporal['val_labels'],      # (N_val,)
        'temporal_test': temporal['test_labels'],    # (N_test,)
        'n_temporal': temporal['n_clusters'],
        'spatial_detail': spatial,
        'temporal_detail': temporal,
    }


# ============================================================
# Visualization helper (for explore_data.py / reports)
# ============================================================

def plot_clusters(cluster_info: dict, grid_rows: int = 10,
                  grid_cols: int = 20, save_path: str = None):
    """Plot spatial clusters on the grid map."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = cluster_info['spatial_labels']
    K = cluster_info['n_spatial']
    grid = labels.reshape(grid_rows, grid_cols)

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.get_cmap('Set2', K)
    im = ax.imshow(grid, cmap=cmap, aspect='auto',
                   vmin=-0.5, vmax=K - 0.5)
    cbar = plt.colorbar(im, ax=ax, ticks=range(K))
    cbar.set_label('Cluster ID')
    ax.set_title('Spatial Node Clusters (by demand pattern)')
    ax.set_xlabel('Grid Column (West → East)')
    ax.set_ylabel('Grid Row (South → North)')

    # Annotate counts
    for k in range(K):
        count = (labels == k).sum()
        cbar.ax.text(1.5, k, f' n={count}', va='center',
                     fontsize=9, transform=cbar.ax.get_yaxis_transform())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
