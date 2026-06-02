"""
Adjacency matrix construction for grid-based traffic data.
Supports geographic adjacency and data-driven similarity.
"""
import numpy as np
from scipy.sparse import coo_matrix


def build_grid_adj(rows: int, cols: int, include_diag: bool = False) -> np.ndarray:
    """
    Build adjacency matrix for a rows x cols grid.
    Each cell connects to its 4 neighbors (up/down/left/right).
    If include_diag=True, also connects to 4 diagonal neighbors.

    Returns: (N, N) adjacency matrix where N = rows * cols.
    """
    n = rows * cols
    adj = np.zeros((n, n), dtype=np.float32)

    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            # 4-directional neighbors
            neighbors = []
            if r > 0:          neighbors.append(idx(r - 1, c))
            if r < rows - 1:   neighbors.append(idx(r + 1, c))
            if c > 0:          neighbors.append(idx(r, c - 1))
            if c < cols - 1:   neighbors.append(idx(r, c + 1))
            # diagonal neighbors
            if include_diag:
                if r > 0 and c > 0:             neighbors.append(idx(r - 1, c - 1))
                if r > 0 and c < cols - 1:      neighbors.append(idx(r - 1, c + 1))
                if r < rows - 1 and c > 0:      neighbors.append(idx(r + 1, c - 1))
                if r < rows - 1 and c < cols - 1: neighbors.append(idx(r + 1, c + 1))

            for j in neighbors:
                adj[i, j] = 1.0

    return adj


def build_similarity_adj(data: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Build adjacency matrix from demand time-series similarity.
    Uses Gaussian kernel on PCC (Pearson Correlation Coefficient).

    Args:
        data: (T, N, C) demand tensor
        k: number of nearest neighbors to keep per node

    Returns: (N, N) adjacency matrix
    """
    # flatten features: (T, N, C) -> (N, T*C)
    n = data.shape[1]
    features = data.reshape(-1, n, data.shape[2])  # (T, N, C)
    features = features.transpose(1, 0, 2).reshape(n, -1)  # (N, T*C)

    # compute pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(features)  # (N, N)

    # keep only top-k neighbors per node
    adj = np.zeros_like(sim)
    for i in range(n):
        top_k = np.argsort(sim[i])[-k - 1:-1]  # exclude self
        adj[i, top_k] = sim[i, top_k]
        adj[top_k, i] = sim[top_k, i]  # symmetrize

    # clip negative values
    adj = np.maximum(adj, 0)
    return adj.astype(np.float32)


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency matrix: D^{-1} A."""
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0  # avoid div-by-zero
    return (adj / row_sum).astype(np.float32)


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    """Add identity to adjacency matrix."""
    return adj + np.eye(adj.shape[0], dtype=np.float32)
