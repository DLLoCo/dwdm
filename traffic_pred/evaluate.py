"""
Evaluation script — detailed test analysis with per-period breakdown.

Generates:
  - Overall metrics (MAE / RMSE / MAPE)
  - Per-period metrics (morning peak / evening peak / late night / midday)
  - Per-node-type metrics (hot nodes vs cold nodes)

Usage:
  python evaluate.py --config configs/nyctaxi_v5.yaml
  python evaluate.py --config configs/nyctaxi.yaml     # baseline without TE/FG
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from lib.utils import load_config, set_seed, get_device, get_project_path, ensure_dir
from lib.dataloader import build_dataloaders
from lib.metrics import compute_all_metrics
from model.net import TrafficPredNet


# ============================================================
# Time periods (half-hour slot indices 0-47)
# ============================================================
PERIODS = {
    'Morning Peak (7-10)':  (14, 20),
    'Midday (11-14)':       (22, 28),
    'Evening Peak (17-20)': (34, 40),
    'Late Night (0-4)':     (0, 8),
}


def slot_to_time(slot):
    h, m = divmod(slot * 30, 60)
    return f"{h:02d}:{m:02d}"


# ============================================================
# Load helpers
# ============================================================

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


# ============================================================
# Run prediction — collect predictions + hour/dow labels
# ============================================================

def run_prediction(model, loader, scaler, device, base_adj,
                   od_patterns=None, use_time=False):
    """Returns: preds, trues, hours, dows (all numpy)."""
    model.eval()
    preds, trues, hours, dows = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            hour = batch[2] if len(batch) > 2 else None
            dow = batch[3] if len(batch) > 3 else None

            hour_gpu = hour.to(device) if use_time and hour is not None else None
            dow_gpu = dow.to(device) if use_time and dow is not None else None

            pred = model(x, base_adj, od_patterns=od_patterns,
                         hour_idx=hour_gpu, dow_idx=dow_gpu)

            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())
            if hour is not None:
                hours.append(hour.numpy())
            if dow is not None:
                dows.append(dow.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    hours = np.concatenate(hours) if hours else None
    dows = np.concatenate(dows) if dows else None
    return preds, trues, hours, dows


# ============================================================
# Per-period analysis (报告结果4)
# ============================================================

def period_analysis(preds, trues, hours):
    """Compute metrics for each time period."""
    if hours is None:
        print("  [SKIP] No hour labels available")
        return {}

    print(f"\n{'Period':<25s} {'Samples':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAPE(%)':>10s}")
    print("-" * 68)

    results = {}
    for name, (start, end) in PERIODS.items():
        mask = (hours >= start) & (hours < end)
        n = mask.sum()
        if n == 0:
            continue
        m = compute_all_metrics(preds[mask], trues[mask])
        results[name] = m
        print(f"  {name:<23s} {n:>8d} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>10.2f}")

    # Other (not in any defined period)
    all_mask = np.zeros(len(hours), dtype=bool)
    for start, end in PERIODS.values():
        all_mask |= (hours >= start) & (hours < end)
    other_mask = ~all_mask
    if other_mask.sum() > 0:
        m = compute_all_metrics(preds[other_mask], trues[other_mask])
        results['Other'] = m
        print(f"  {'Other':<23s} {other_mask.sum():>8d} {m['MAE']:>10.4f} "
              f"{m['RMSE']:>10.4f} {m['MAPE']:>10.2f}")

    print("-" * 68)
    return results


# ============================================================
# Weekday vs Weekend analysis
# ============================================================

def weekday_weekend_analysis(preds, trues, dows):
    """Compare weekday vs weekend performance."""
    if dows is None:
        print("  [SKIP] No day-of-week labels available")
        return

    is_weekend = (dows == 0) | (dows == 6)  # Mon=0..Sun=6, Sat=5,Sun=6
    # Actually in our encoding: 0=Mon, so Sat=5, Sun=6
    is_weekend = (dows == 5) | (dows == 6)

    print(f"\n{'Day Type':<20s} {'Samples':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAPE(%)':>10s}")
    print("-" * 58)

    for label, mask in [('Weekday', ~is_weekend), ('Weekend', is_weekend)]:
        n = mask.sum()
        if n == 0:
            continue
        m = compute_all_metrics(preds[mask], trues[mask])
        print(f"  {label:<18s} {n:>8d} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>10.2f}")
    print("-" * 58)


# ============================================================
# Hot vs Cold node analysis
# ============================================================

def node_analysis(preds, trues):
    """Compare prediction quality on hot nodes vs cold nodes."""
    # trues shape: (N_samples, 1, 200, 2)
    node_avg = np.abs(trues).mean(axis=(0, 1, 3))  # (200,)
    threshold = np.percentile(node_avg, 80)  # top 20%

    hot_mask = node_avg >= threshold
    cold_mask = ~hot_mask

    print(f"\n{'Node Type':<20s} {'Nodes':>6s} {'MAE':>10s} {'RMSE':>10s} {'MAPE(%)':>10s}")
    print("-" * 58)

    for label, mask in [('Hot (top 20%)', hot_mask), ('Cold (bottom 80%)', cold_mask)]:
        p = preds[:, :, mask, :]
        t = trues[:, :, mask, :]
        m = compute_all_metrics(p, t)
        print(f"  {label:<18s} {mask.sum():>6d} {m['MAE']:>10.4f} "
              f"{m['RMSE']:>10.4f} {m['MAPE']:>10.2f}")
    print("-" * 58)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])
    device = get_device(args.gpu)
    mcfg = config['model']
    use_te = mcfg.get('use_time_embed', False)

    print("=" * 68)
    print("  Evaluation: Traffic Demand Prediction")
    print("=" * 68)
    print(f"  Config: {args.config}")
    print(f"  FlowGate: {mcfg.get('use_flow_gate', False)}")
    print(f"  TimeEmbed: {use_te}")

    # Data
    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)
    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
    else:
        from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
        base_adj = torch.FloatTensor(normalize_adj(add_self_loops(
            build_grid_adj(config['data']['grid_rows'],
                           config['data']['grid_cols'])))).to(device)

    od_patterns = load_od_patterns(config, device)

    # Model
    model = TrafficPredNet.from_config(config).to(device)
    ckpt_path = get_project_path(args.checkpoint)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        print(f"  Checkpoint: {ckpt_path}")
    else:
        print(f"  [WARNING] No checkpoint at {ckpt_path}, using random weights!")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")

    # Predict
    preds, trues, hours, dows = run_prediction(
        model, test_loader, scaler, device, base_adj,
        od_patterns=od_patterns, use_time=use_te)

    # 1. Overall metrics
    overall = compute_all_metrics(preds, trues)
    print(f"\n{'=' * 68}")
    print(f"  Overall Results")
    print(f"{'=' * 68}")
    print(f"  MAE:  {overall['MAE']:.4f}")
    print(f"  RMSE: {overall['RMSE']:.4f}")
    print(f"  MAPE: {overall['MAPE']:.2f}%")

    # 2. Per-period analysis (报告结果4)
    print(f"\n{'=' * 68}")
    print(f"  Per-Period Analysis (Morning Peak / Evening Peak / Late Night)")
    print(f"{'=' * 68}")
    period_results = period_analysis(preds, trues, hours)

    # 3. Weekday vs Weekend
    print(f"\n{'=' * 68}")
    print(f"  Weekday vs Weekend")
    print(f"{'=' * 68}")
    weekday_weekend_analysis(preds, trues, dows)

    # 4. Hot vs Cold nodes
    print(f"\n{'=' * 68}")
    print(f"  Hot Nodes vs Cold Nodes")
    print(f"{'=' * 68}")
    node_analysis(preds, trues)

    # 5. Save predictions for visualization
    save_dir = get_project_path('figures')
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, 'test_predictions.npz')
    save_dict = {'preds': preds, 'trues': trues}
    if hours is not None:
        save_dict['hours'] = hours
    if dows is not None:
        save_dict['dows'] = dows
    np.savez(save_path, **save_dict)
    print(f"\n  Predictions saved to: {save_path}")

    print(f"\n{'=' * 68}")
    print(f"  Done!")
    print(f"{'=' * 68}")


if __name__ == '__main__':
    main()
