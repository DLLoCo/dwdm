"""
Evaluation script for traffic demand prediction.

Usage:
    python evaluate.py
    python evaluate.py --config configs/nyctaxi_cluster.yaml
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from lib.utils import load_config, set_seed, get_device, get_project_path, \
    ensure_dir, print_model_params
from lib.dataloader import build_dataloaders
from lib.metrics import masked_mae, masked_rmse, masked_mape
from model.net import TrafficPredNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nyctaxi_memory.yaml')
    p.add_argument('--checkpoint', default='checkpoints/v7_memory/best_model.pt')
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def load_od_patterns(config, device):
    mcfg = config['model']
    if not mcfg.get('use_flow_gate', False):
        return None
    od_dir = config['data'].get('data_dir', '').replace('NYCTaxi', 'NYCTaxi_OD')
    if not os.path.isabs(od_dir):
        od_dir = get_project_path(od_dir)
    od_path = os.path.join(od_dir, 'od_hourly.npz')
    if os.path.exists(od_path):
        npz = np.load(od_path, allow_pickle=True)
        for key in ['od_hourly', 'od', 'arr_0', 'data']:
            if key in npz:
                od = npz[key].astype(np.float32)
                t = torch.FloatTensor(od).to(device)
                print(f"[FlowGate] Loaded {t.shape[0]} OD patterns: {t.shape}")
                return t
        first_key = list(npz.keys())[0]
        od = npz[first_key].astype(np.float32)
        t = torch.FloatTensor(od).to(device)
        print(f"[FlowGate] Loaded {t.shape[0]} OD patterns (key='{first_key}'): {t.shape}")
        return t
    return None


def predict_all(model, loader, scaler, device, base_adj,
                od_patterns=None, use_time=False, use_cluster=False,
                spatial_cluster=None):
    """Run predictions on all batches, return (preds, trues) in real scale."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            hour = batch[2].to(device) if use_time and len(batch) > 2 else None
            dow = batch[3].to(device) if use_time and len(batch) > 3 else None
            tc = batch[4].to(device) if use_cluster and len(batch) > 4 else None

            pred = model(x, base_adj, od_patterns=od_patterns,
                         hour_idx=hour, dow_idx=dow,
                         spatial_cluster=spatial_cluster,
                         temporal_cluster=tc)
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])
    device = get_device(args.gpu)
    mcfg = config['model']

    use_fg = mcfg.get('use_flow_gate', False)
    use_te = mcfg.get('use_time_embed', False)
    use_cl = (mcfg.get('use_cluster', False)
          or mcfg.get('use_spatial_cluster', False)
          or mcfg.get('use_temporal_cluster', False))
    use_time = use_te or use_fg

    print("=" * 68)
    print("  Evaluation: Traffic Demand Prediction")
    print("=" * 68)
    print(f"  Config: {args.config}")
    print(f"  FlowGate: {use_fg}, TimeEmbed: {use_te}, Cluster: {use_cl}")

    # ---- Data ----
    result = build_dataloaders(config)
    train_loader, val_loader, test_loader, scaler, adj = result[:5]
    cluster_info = result[5] if len(result) > 5 else None

    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
    else:
        from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
        base_adj = torch.FloatTensor(normalize_adj(add_self_loops(
            build_grid_adj(config['data']['grid_rows'],
                           config['data']['grid_cols'])))).to(device)

    od_patterns = load_od_patterns(config, device)

    spatial_cluster = None
    if use_cl and cluster_info is not None:
        spatial_cluster = cluster_info['spatial_labels'].to(device)

    # ---- Model ----
    model = TrafficPredNet.from_config(config).to(device)
    ckpt_path = get_project_path(args.checkpoint)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        print(f"  Checkpoint: {ckpt_path}")
    else:
        print(f"  [WARNING] No checkpoint found at {ckpt_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Predict ----
    preds, trues = predict_all(
        model, test_loader, scaler, device, base_adj,
        od_patterns=od_patterns, use_time=use_time, use_cluster=use_cl,
        spatial_cluster=spatial_cluster)

    print(f"  Predictions: {preds.shape}, Ground truth: {trues.shape}")

    # ---- Overall ----
    print("=" * 68)
    print("  Overall Results")
    print("=" * 68)
    mae = masked_mae(preds, trues)
    rmse = masked_rmse(preds, trues)
    mape = masked_mape(preds, trues)
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # ---- Per-flow-type ----
    print("=" * 68)
    print("  Per-Flow-Type Metrics")
    print("=" * 68)
    flow_names = ['Pickup (inflow)', 'Dropoff (outflow)']
    for c, name in enumerate(flow_names):
        p_c = preds[:, :, :, c]
        t_c = trues[:, :, :, c]
        print(f"  {name:25s}: MAE={masked_mae(p_c, t_c):.4f}  "
              f"RMSE={masked_rmse(p_c, t_c):.4f}  "
              f"MAPE={masked_mape(p_c, t_c):.2f}%")

    # ---- Per-period analysis ----
    print("=" * 68)
    print("  Per-Period Analysis (Morning Peak / Evening Peak / Late Night)")
    print("=" * 68)

    # Use time indices from dataloader
    n_test = len(test_loader.dataset)
    offset = 144 + len(train_loader.dataset) + len(val_loader.dataset)
    test_hours = (np.arange(n_test) + offset) % 48  # 0-47

    periods = {
        'Morning Peak (7-10)': (14, 20),     # slots 14-19
        'Midday (11-14)': (22, 28),           # slots 22-27
        'Evening Peak (17-20)': (34, 40),     # slots 34-39
        'Late Night (0-4)': (0, 8),           # slots 0-7
        'Other': None,
    }

    print(f"{'Period':30s} {'Samples':>8s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE(%)':>10s}")
    print("-" * 68)

    assigned = np.zeros(n_test, dtype=bool)
    for name, slot_range in periods.items():
        if slot_range is not None:
            lo, hi = slot_range
            if lo < hi:
                mask = (test_hours >= lo) & (test_hours < hi)
            else:
                mask = (test_hours >= lo) | (test_hours < hi)
        else:
            mask = ~assigned

        assigned |= mask
        count = mask.sum()
        if count == 0:
            continue

        p_sub = preds[mask]
        t_sub = trues[mask]
        m = masked_mae(p_sub, t_sub)
        r = masked_rmse(p_sub, t_sub)
        mp = masked_mape(p_sub, t_sub)
        print(f"  {name:28s} {count:8d} {m:10.4f} {r:10.4f} {mp:10.2f}")

    print("-" * 68)

    # ---- Weekday vs Weekend ----
    print("=" * 68)
    print("  Weekday vs Weekend")
    print("=" * 68)
    test_dow = ((np.arange(n_test) + offset) // 48 + 3) % 7
    print(f"{'Day Type':22s} {'Samples':>8s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE(%)':>10s}")
    print("-" * 58)
    for label, mask in [('Weekday', (test_dow >= 0) & (test_dow < 5)),
                        ('Weekend', (test_dow >= 5))]:
        count = mask.sum()
        if count == 0:
            continue
        m = masked_mae(preds[mask], trues[mask])
        r = masked_rmse(preds[mask], trues[mask])
        mp = masked_mape(preds[mask], trues[mask])
        print(f"  {label:20s} {count:8d} {m:10.4f} {r:10.4f} {mp:10.2f}")
    print("-" * 58)

    # ---- Hot vs Cold nodes ----
    print("=" * 68)
    print("  Hot Nodes vs Cold Nodes")
    print("=" * 68)
    node_mean = trues.mean(axis=(0, 1, 3))  # (N,)
    top_k = int(0.2 * len(node_mean))
    hot_nodes = np.argsort(node_mean)[-top_k:]
    cold_nodes = np.argsort(node_mean)[:-top_k]

    print(f"{'Node Type':22s} {'Nodes':>6s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE(%)':>10s}")
    print("-" * 58)
    for label, nds in [('Hot (top 20%)', hot_nodes),
                       ('Cold (bottom 80%)', cold_nodes)]:
        p_sub = preds[:, :, nds, :]
        t_sub = trues[:, :, nds, :]
        m = masked_mae(p_sub, t_sub)
        r = masked_rmse(p_sub, t_sub)
        mp = masked_mape(p_sub, t_sub)
        print(f"  {label:20s} {len(nds):6d} {m:10.4f} {r:10.4f} {mp:10.2f}")
    print("-" * 58)

    # ---- Per-cluster analysis (if clustering enabled) ----
    if use_cl and cluster_info is not None:
        print("=" * 68)
        print("  Per-Spatial-Cluster Analysis")
        print("=" * 68)
        sp_labels = cluster_info['spatial_labels'].numpy()
        n_sp = cluster_info['n_spatial']
        print(f"{'Cluster':22s} {'Nodes':>6s} {'MAE':>10s} {'RMSE':>10s} "
              f"{'MAPE(%)':>10s}")
        print("-" * 58)
        for k in range(n_sp):
            nds = np.where(sp_labels == k)[0]
            if len(nds) == 0:
                continue
            p_sub = preds[:, :, nds, :]
            t_sub = trues[:, :, nds, :]
            m = masked_mae(p_sub, t_sub)
            r = masked_rmse(p_sub, t_sub)
            mp = masked_mape(p_sub, t_sub)
            avg_demand = trues[:, :, nds, :].mean()
            print(f"  Cluster {k} (avg={avg_demand:.1f})"
                  f" {len(nds):6d} {m:10.4f} {r:10.4f} {mp:10.2f}")
        print("-" * 58)

        print("=" * 68)
        print("  Per-Temporal-Cluster Analysis")
        print("=" * 68)
        # Get temporal cluster labels for test set
        tc_test = result[5]['detail']['temporal_test'] if result[5] else None
        if tc_test is not None:
            n_tp = cluster_info['n_temporal']
            print(f"{'Cluster':22s} {'Samples':>8s} {'MAE':>10s} {'RMSE':>10s} "
                  f"{'MAPE(%)':>10s}")
            print("-" * 58)
            for k in range(n_tp):
                mask = tc_test == k
                count = mask.sum()
                if count == 0:
                    continue
                m = masked_mae(preds[mask], trues[mask])
                r = masked_rmse(preds[mask], trues[mask])
                mp = masked_mape(preds[mask], trues[mask])
                avg_demand = trues[mask].mean()
                print(f"  Cluster {k} (avg={avg_demand:.1f})"
                      f" {count:8d} {m:10.4f} {r:10.4f} {mp:10.2f}")
            print("-" * 58)

    # ---- Save predictions ----
    fig_dir = get_project_path('figures')
    ensure_dir(fig_dir)
    save_path = os.path.join(fig_dir, 'test_predictions.npz')
    np.savez(save_path, predictions=preds, ground_truth=trues)
    print(f"  Predictions saved to: {save_path}")

    print("=" * 68)
    print("  Done!")
    print("=" * 68)


if __name__ == '__main__':
    main()
