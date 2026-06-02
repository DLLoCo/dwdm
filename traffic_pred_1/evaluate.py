"""
Evaluation script — detailed test analysis.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from lib.utils import load_config, set_seed, get_device, get_project_path
from lib.dataloader import build_dataloaders
from lib.metrics import compute_all_metrics
from model.net import TrafficPredNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nyctaxi.yaml')
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def run_prediction(model, loader, scaler, device, base_adj):
    """Run model on entire loader, return predictions and ground truth."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x, base_adj)
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def analyze_by_flow_type(preds, trues):
    """Separate metrics for pickup (inflow) and dropoff (outflow)."""
    print("\n--- Per-flow-type metrics ---")
    names = ['Pickup (inflow)', 'Dropoff (outflow)']
    for c, name in enumerate(names):
        m = compute_all_metrics(preds[..., c], trues[..., c])
        print(f"  {name:20s}: MAE={m['MAE']:.4f}  "
              f"RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%")


def analyze_by_volume(preds, trues, thresholds=(10, 50, 100)):
    """Metrics grouped by ground-truth demand volume level."""
    print("\n--- Per-volume-level metrics ---")
    flat_true = trues.flatten()
    flat_pred = preds.flatten()

    bounds = [0] + list(thresholds) + [float('inf')]
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (flat_true >= lo) & (flat_true < hi)
        if mask.sum() == 0:
            continue
        p, t = flat_pred[mask], flat_true[mask]
        mae = np.mean(np.abs(p - t))
        label = f"[{lo:.0f}, {hi:.0f})" if hi < float('inf') else f"[{lo:.0f}, +inf)"
        print(f"  Volume {label:12s} ({mask.sum():7d} pts): MAE={mae:.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])
    device = get_device(args.gpu)

    print("=" * 60)
    print("Evaluation: Traffic Demand Prediction")
    print("=" * 60)

    # Data
    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)
    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
    else:
        from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
        adj_np = normalize_adj(add_self_loops(
            build_grid_adj(config['data']['grid_rows'], config['data']['grid_cols'])))
        base_adj = torch.FloatTensor(adj_np).to(device)

    # Model
    model = TrafficPredNet.from_config(config).to(device)
    ckpt_path = get_project_path(args.checkpoint)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARNING] No checkpoint at {ckpt_path}, using random weights")

    # Run predictions
    print("\nRunning predictions on test set...")
    preds, trues = run_prediction(model, test_loader, scaler, device, base_adj)
    print(f"  Predictions: {preds.shape}, Ground truth: {trues.shape}")

    # Overall metrics
    print("\n--- Overall test metrics ---")
    metrics = compute_all_metrics(preds, trues)
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # Per-flow-type
    if preds.shape[-1] >= 2:
        analyze_by_flow_type(preds, trues)

    # Per-volume-level
    analyze_by_volume(preds, trues)

    # Save predictions for visualization
    save_dir = get_project_path('checkpoints')
    save_path = os.path.join(save_dir, 'test_predictions.npz')
    np.savez(save_path, preds=preds, trues=trues)
    print(f"\nPredictions saved to {save_path}")
    print("Run visualize.py next for plots.")


if __name__ == '__main__':
    main()
