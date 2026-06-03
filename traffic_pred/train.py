"""
Training script for traffic demand prediction.

Usage:
    python train.py                                    # default config
    python train.py --config configs/nyctaxi.yaml      # old pipeline (no cluster)
    python train.py --config configs/nyctaxi_cluster.yaml  # v6 with clustering
    python train.py --epochs 50 --lr 0.003
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from lib.utils import load_config, set_seed, get_device, ensure_dir, \
    print_model_params, get_project_path
from lib.dataloader import build_dataloaders
from lib.metrics import MaskedMAELoss, compute_all_metrics
from model.net import TrafficPredNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/ablation/with_flowgate.yaml')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def load_od_patterns(config, device):
    """Load OD flow patterns for Flow Gate if available."""
    mcfg = config['model']
    if not mcfg.get('use_flow_gate', False):
        return None
    od_dir = config['data'].get('data_dir', '').replace('NYCTaxi', 'NYCTaxi_OD')
    if not os.path.isabs(od_dir):
        od_dir = get_project_path(od_dir)
    od_path = os.path.join(od_dir, 'od_hourly.npz')
    if os.path.exists(od_path):
        npz = np.load(od_path, allow_pickle=True)
        # Try common key names
        for key in ['od_hourly', 'od', 'arr_0', 'data']:
            if key in npz:
                od = npz[key].astype(np.float32)
                t = torch.FloatTensor(od).to(device)
                print(f"[FlowGate] Loaded {t.shape[0]} OD patterns: {t.shape}")
                return t
        # Fallback: use first key
        first_key = list(npz.keys())[0]
        od = npz[first_key].astype(np.float32)
        t = torch.FloatTensor(od).to(device)
        print(f"[FlowGate] Loaded {t.shape[0]} OD patterns (key='{first_key}'): {t.shape}")
        return t
    print("[FlowGate] WARNING: od_hourly.npz not found, Flow Gate disabled")
    return None


def evaluate(model, loader, criterion, scaler, device, base_adj,
             od_flow=None, use_time=False, use_cluster=False,
             spatial_cluster=None):
    """Run evaluation on a data loader, return loss and metrics."""
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            hour = batch[2].to(device) if use_time and len(batch) > 2 else None
            dow = batch[3].to(device) if use_time and len(batch) > 3 else None
            tc = batch[4].to(device) if use_cluster and len(batch) > 4 else None

            pred = model(x, base_adj, od_patterns=od_flow,
                         hour_idx=hour, dow_idx=dow,
                         spatial_cluster=spatial_cluster,
                         temporal_cluster=tc)

            loss = criterion(pred, y)
            losses.append(loss.item())

            pred_real = scaler.inverse_transform(pred.cpu())
            true_real = scaler.inverse_transform(y.cpu())
            preds.append(pred_real.numpy())
            trues.append(true_real.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    metrics = compute_all_metrics(preds, trues)
    return np.mean(losses), metrics


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.lr: config['train']['lr'] = args.lr
    if args.batch_size: config['train']['batch_size'] = args.batch_size

    tcfg = config['train']
    mcfg = config['model']
    set_seed(tcfg['seed'])
    device = get_device(args.gpu)
    save_dir = get_project_path(tcfg['save_dir'])
    ensure_dir(save_dir)

    use_fg = mcfg.get('use_flow_gate', False)
    use_te = mcfg.get('use_time_embed', False)
    use_cl = (mcfg.get('use_cluster', False)
          or mcfg.get('use_spatial_cluster', False)
          or mcfg.get('use_temporal_cluster', False))
    use_time = use_te or use_fg  # need time indices for either

    print("=" * 60)
    print("Training: Traffic Demand Prediction")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"FlowGate: {use_fg}, TimeEmbed: {use_te}, Cluster: {use_cl}")

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

    # Spatial cluster labels → to device (stays there, reused every batch)
    spatial_cluster = None
    if use_cl and cluster_info is not None:
        spatial_cluster = cluster_info['spatial_labels'].to(device)
        print(f"[CLUSTER] Spatial: {cluster_info['n_spatial']} clusters, "
              f"Temporal: {cluster_info['n_temporal']} clusters")

    # ---- Model ----
    model = TrafficPredNet.from_config(config).to(device)
    print_model_params(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = MaskedMAELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [],
               'val_MAE': [], 'val_RMSE': [], 'val_MAPE': []}

    n_epochs = tcfg['epochs']
    log_every = tcfg.get('log_every', 10)
    n_batches = len(train_loader)

    print(f"\nStart training: {n_epochs} epochs, lr={tcfg['lr']}, "
          f"batch_size={tcfg['batch_size']}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses = []

        for i, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            hour = batch[2].to(device) if use_time and len(batch) > 2 else None
            dow = batch[3].to(device) if use_time and len(batch) > 3 else None
            tc = batch[4].to(device) if use_cl and len(batch) > 4 else None

            optimizer.zero_grad()
            pred = model(x, base_adj, od_patterns=od_patterns,
                         hour_idx=hour, dow_idx=dow,
                         spatial_cluster=spatial_cluster,
                         temporal_cluster=tc)

            loss = criterion(pred, y)
            loss.backward()

            if tcfg['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tcfg['max_grad_norm'])
            optimizer.step()
            epoch_losses.append(loss.item())

            if log_every > 0 and (i + 1) % log_every == 0:
                print(f"    batch {i+1}/{n_batches}, "
                      f"loss={loss.item():.4f}")

        train_loss = np.mean(epoch_losses)

        # ---- Validation ----
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, scaler, device, base_adj,
            od_flow=od_patterns, use_time=use_time, use_cluster=use_cl,
            spatial_cluster=spatial_cluster)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_MAE'].append(val_metrics['MAE'])
        history['val_RMSE'].append(val_metrics['RMSE'])
        history['val_MAPE'].append(val_metrics['MAPE'])

        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"MAE={val_metrics['MAE']:.3f} "
              f"RMSE={val_metrics['RMSE']:.3f} "
              f"MAPE={val_metrics['MAPE']:.2f}% | "
              f"lr={lr:.6f} | {elapsed:.1f}s")

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            ckpt = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= tcfg['early_stop_patience']:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best: epoch {best_epoch})")
                break

    # ---- Save history ----
    hist_path = os.path.join(save_dir, 'train_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {hist_path}")

    # ---- Test with best model ----
    print("\n" + "=" * 60)
    print("Evaluating on test set (best model)...")
    print("=" * 60)

    best_ckpt = os.path.join(save_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                     weights_only=True))

    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, scaler, device, base_adj,
        od_flow=od_patterns, use_time=use_time, use_cluster=use_cl,
        spatial_cluster=spatial_cluster)

    print(f"\n  Test MAE:  {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"\n  Best epoch: {best_epoch}")


if __name__ == '__main__':
    main()
