"""
消融实验一键脚本
==================
训练 5 个模型变体，输出对比表格。

Usage:
    python run_ablation.py              # 训练 + 评估全部
    python run_ablation.py --eval_only  # 只评估（已训练过）
    python run_ablation.py --epochs 50  # 快速测试

预计耗时：每组 ~10-15 min（100 epochs），总计 ~1h
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from lib.utils import load_config, set_seed, get_device, ensure_dir, get_project_path
from lib.dataloader import build_dataloaders
from lib.metrics import MaskedMAELoss, masked_mae, masked_rmse, masked_mape
from model.net import TrafficPredNet


EXPERIMENTS = [
    ('Full model',           'configs/ablation/full.yaml'),
    ('w/o Residual',         'configs/ablation/no_residual.yaml'),
    ('w/o Spatial Cluster',  'configs/ablation/no_spatial.yaml'),
    ('w/o Temporal Cluster', 'configs/ablation/no_temporal.yaml'),
    ('w/o Both Clusters',    'configs/ablation/no_cluster.yaml'),
]


def train_one(config, device):
    """Train one model, return best val MAE."""
    set_seed(config['train']['seed'])
    tcfg = config['train']
    mcfg = config['model']

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

    use_cl = (mcfg.get('use_cluster', False)
              or mcfg.get('use_spatial_cluster', False)
              or mcfg.get('use_temporal_cluster', False))

    spatial_cluster = None
    if use_cl and cluster_info is not None:
        spatial_cluster = cluster_info['spatial_labels'].to(device)

    model = TrafficPredNet.from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg['lr'],
                                 weight_decay=tcfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = MaskedMAELoss()

    save_dir = get_project_path(tcfg['save_dir'])
    ensure_dir(save_dir)

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(1, tcfg['epochs'] + 1):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            tc = batch[4].to(device) if use_cl and len(batch) > 4 else None

            optimizer.zero_grad()
            pred = model(x, base_adj, spatial_cluster=spatial_cluster,
                         temporal_cluster=tc)
            loss = criterion(pred, y)
            loss.backward()
            if tcfg['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tcfg['max_grad_norm'])
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                tc = batch[4].to(device) if use_cl and len(batch) > 4 else None
                pred = model(x, base_adj, spatial_cluster=spatial_cluster,
                             temporal_cluster=tc)
                val_losses.append(criterion(pred, y).item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pt'))
        else:
            patience += 1
            if patience >= tcfg['early_stop_patience']:
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}, val_loss={val_loss:.4f}, "
                  f"best={best_val_loss:.4f}")

    return save_dir


def evaluate_one(config, device):
    """Evaluate one model, return metrics dict."""
    set_seed(config['train']['seed'])
    mcfg = config['model']

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

    use_cl = (mcfg.get('use_cluster', False)
              or mcfg.get('use_spatial_cluster', False)
              or mcfg.get('use_temporal_cluster', False))

    spatial_cluster = None
    if use_cl and cluster_info is not None:
        spatial_cluster = cluster_info['spatial_labels'].to(device)

    model = TrafficPredNet.from_config(config).to(device)
    ckpt = os.path.join(get_project_path(config['train']['save_dir']),
                        'best_model.pt')
    if not os.path.exists(ckpt):
        return None
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1]
            tc = batch[4].to(device) if use_cl and len(batch) > 4 else None
            pred = model(x, base_adj, spatial_cluster=spatial_cluster,
                         temporal_cluster=tc)
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # Per-period analysis
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    offset = 144 + n_train + n_val
    test_hours = (np.arange(len(preds)) + offset) % 48

    night_mask = test_hours < 8
    morning_mask = (test_hours >= 14) & (test_hours < 20)
    evening_mask = (test_hours >= 34) & (test_hours < 40)

    # Hot/cold nodes
    node_mean = trues.mean(axis=(0, 1, 3))
    top_k = int(0.2 * len(node_mean))
    hot = np.argsort(node_mean)[-top_k:]
    cold = np.argsort(node_mean)[:-top_k]

    return {
        'MAE': masked_mae(preds, trues),
        'RMSE': masked_rmse(preds, trues),
        'MAPE': masked_mape(preds, trues),
        'Morning MAE': masked_mae(preds[morning_mask], trues[morning_mask]),
        'Evening MAE': masked_mae(preds[evening_mask], trues[evening_mask]),
        'Night MAE': masked_mae(preds[night_mask], trues[night_mask]),
        'Hot MAE': masked_mae(preds[:,:,hot,:], trues[:,:,hot,:]),
        'Cold MAE': masked_mae(preds[:,:,cold,:], trues[:,:,cold,:]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = get_device(args.gpu)

    print("=" * 80)
    print("  Ablation Study")
    print("=" * 80)

    all_results = {}

    for name, cfg_path in EXPERIMENTS:
        print(f"\n{'─' * 80}")
        print(f"  {name} ({cfg_path})")
        print(f"{'─' * 80}")

        config = load_config(cfg_path)
        if args.epochs:
            config['train']['epochs'] = args.epochs

        # Train
        if not args.eval_only:
            t0 = time.time()
            save_dir = train_one(config, device)
            elapsed = time.time() - t0
            print(f"  Trained in {elapsed:.0f}s → {save_dir}")
        else:
            print("  [eval_only] Skipping training")

        # Evaluate
        metrics = evaluate_one(config, device)
        if metrics is None:
            print("  [SKIP] No checkpoint found")
            continue

        all_results[name] = metrics
        print(f"  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
              f"MAPE={metrics['MAPE']:.1f}%  "
              f"Night={metrics['Night MAE']:.2f}")

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n\n{'=' * 80}")
    print("  Ablation Results Summary")
    print(f"{'=' * 80}\n")

    header = (f"{'Variant':<25s} {'MAE':>7s} {'RMSE':>7s} {'MAPE%':>7s} "
              f"{'Morn':>7s} {'Eve':>7s} {'Night':>7s} "
              f"{'Hot':>7s} {'Cold':>7s}")
    print(header)
    print("─" * len(header))

    for name, m in all_results.items():
        print(f"{name:<25s} {m['MAE']:7.2f} {m['RMSE']:7.2f} "
              f"{m['MAPE']:6.1f}% "
              f"{m['Morning MAE']:7.2f} {m['Evening MAE']:7.2f} "
              f"{m['Night MAE']:7.2f} "
              f"{m['Hot MAE']:7.2f} {m['Cold MAE']:7.2f}")

    print(f"\n{'─' * len(header)}")

    # Save results JSON
    out_path = get_project_path('figures', 'ablation_results.json')
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
