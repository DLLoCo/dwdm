"""
Evaluation metrics for traffic demand prediction.
All metrics support masking to ignore near-zero regions.
"""
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def masked_mae(pred, true, mask_val=0.0):
    """Mean Absolute Error, ignoring entries where true <= mask_val."""
    pred, true = _to_numpy(pred), _to_numpy(true)
    mask = true > mask_val
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def masked_rmse(pred, true, mask_val=0.0):
    """Root Mean Squared Error, ignoring entries where true <= mask_val."""
    pred, true = _to_numpy(pred), _to_numpy(true)
    mask = true > mask_val
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def masked_mape(pred, true, mask_val=5.0):
    """Mean Absolute Percentage Error, ignoring small values (default >5)."""
    pred, true = _to_numpy(pred), _to_numpy(true)
    mask = true > mask_val
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100)


def compute_all_metrics(pred, true, mask_val=5.0):
    """Compute MAE, RMSE, MAPE and return as dict."""
    return {
        'MAE':  masked_mae(pred, true, mask_val=0.0),
        'RMSE': masked_rmse(pred, true, mask_val=0.0),
        'MAPE': masked_mape(pred, true, mask_val=mask_val),
    }


class MaskedMAELoss(torch.nn.Module):
    """Differentiable MAE loss with masking, for use in training."""

    def __init__(self, mask_val=0.0):
        super().__init__()
        self.mask_val = mask_val

    def forward(self, pred, true):
        mask = (true > self.mask_val).float()
        loss = torch.abs(pred - true) * mask
        # avoid div-by-zero when all entries are masked
        return loss.sum() / (mask.sum() + 1e-8)
