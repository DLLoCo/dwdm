"""
General utilities: config loading, reproducibility, data scaling, etc.
"""
import os
import yaml
import random
import numpy as np
import torch

# Project root: traffic_pred/ directory (parent of lib/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_path(*parts) -> str:
    """Get absolute path relative to project root.
    Example: get_project_path('configs', 'nyctaxi.yaml')
    """
    return os.path.join(PROJECT_ROOT, *parts)


def load_config(path: str) -> dict:
    """Load YAML config file. If path is relative, resolve from project root."""
    if not os.path.isabs(path):
        path = get_project_path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu: int = 0) -> torch.device:
    """Get torch device."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


class StandardScaler:
    """
    Z-score normalization: (x - mean) / std.
    Fitted on training data, applied to all splits.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Compute mean and std from data. data shape: (T, N, C)"""
        self.mean = data.mean(axis=0, keepdims=True)  # (1, N, C)
        self.std = data.std(axis=0, keepdims=True)     # (1, N, C)
        self.std[self.std < 1e-6] = 1.0  # avoid div-by-zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Supports both numpy and torch tensors."""
        if isinstance(data, torch.Tensor):
            mean = torch.FloatTensor(self.mean).to(data.device)
            std = torch.FloatTensor(self.std).to(data.device)
            return data * std + mean
        return data * self.std + self.mean


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def print_model_params(model: torch.nn.Module):
    """Print total number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
