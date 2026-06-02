"""
Evaluation script — detailed test analysis.
Usage: python evaluate.py
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


def load_od_patterns(config, device):
    if not config['model'].get('use_flow_gate', False):
        return None
    od_path = get_project_path('data', 'processed', 'NYCTaxi_OD', 'od_hourly.npz')
    if not os.path.exists(od_path):
        od_path = get_project_path('data', 'processed', 'NYCTaxi_OD', 'od_avg.npz')
    if os.path.exists(od_path):
        from model.flow_gate import FlowGateManager
        mgr = FlowGateManager(od_path, device)
        if mgr.available:
            return mgr.get_patterns()
    return None


def run_prediction(model, loader, scaler, device, base_adj,
                   od_patterns=None, use_time=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            hour = batch[2].to(device) if use_time and len(batch) > 2 else None
            dow = batch[3].to(device) if use_time and len(batch) > 3 else None

            pred = model(x, base_adj, od_patterns=od_patterns,
                         hour_idx=hour, dow_idx=dow)
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])
    device = get_device(args.gpu)
    mcfg = config['model']
    use_te = mcfg.get('use_time_embed', False)

    print("=" * 60)
    print("Evaluation: Traffic Demand Prediction")
    print("=" * 60)

    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)
    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
    else:
        from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
        base_adj = torch.FloatTensor(normalize_adj(add_self_loops(
            build_grid_adj(config['data']['grid_rows'],
                           config['data']['grid_cols'])))).to(device)

    od_patterns = load_od_patterns(config, device)

    model = TrafficPredNet.from_config(config).to(device)
    ckpt_path = get_project_path(args.checkpoint)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARNING] No checkpoint found at {ckpt_path}")

    print("\nRunning predictions on test set...")
    preds, trues = run_prediction(model, test_loader, scaler, device,
                                   base_adj, od_patterns, use_time=use_te)
    print(f"  Predictions: {preds.shape}, Ground truth: {trues.shape}")

    print("\n--- Overall test metrics ---")
    metrics = compute_all_metrics(preds, trues)
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    if preds.shape[-1] >= 2:
        print("\n--- Per-flow-type metrics ---")
        for c, name in enumerate(['Pickup (inflow)', 'Dropoff (outflow)']):
            m = compute_all_metrics(preds[..., c], trues[..., c])
            print(f"  {name:20s}: MAE={m['MAE']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%")

    print("\n--- Per-volume-level metrics ---")
    flat_t, flat_p = trues.flatten(), preds.flatten()
    for lo, hi in [(0,10),(10,50),(50,100),(100,float('inf'))]:
        mask = (flat_t >= lo) & (flat_t < hi)
        if mask.sum() == 0: continue
        mae = np.mean(np.abs(flat_p[mask] - flat_t[mask]))
        label = f"[{lo:.0f}, {hi:.0f})" if hi < float('inf') else f"[{lo:.0f}, +inf)"
        print(f"  Volume {label:12s} ({mask.sum():7d} pts): MAE={mae:.4f}")

    save_dir = get_project_path('checkpoints')
    np.savez(os.path.join(save_dir, 'test_predictions.npz'),
             preds=preds, trues=trues)
    print(f"\nPredictions saved. Run visualize.py for plots.")


if __name__ == '__main__':
    main()
