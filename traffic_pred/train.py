"""
Training script for traffic demand prediction.
Usage: python train.py
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
    p.add_argument('--config', default='configs/nyctaxi.yaml')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
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
    print("[WARNING] OD data not found, flow gate disabled")
    return None


def evaluate(model, loader, criterion, scaler, device, base_adj,
             od_patterns=None, use_time=False):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            hour = batch[2].to(device) if use_time and len(batch) > 2 else None
            dow = batch[3].to(device) if use_time and len(batch) > 3 else None

            pred = model(x, base_adj, od_patterns=od_patterns,
                         hour_idx=hour, dow_idx=dow)
            losses.append(criterion(pred, y).item())
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y.cpu()).numpy())

    return np.mean(losses), compute_all_metrics(
        np.concatenate(preds), np.concatenate(trues))


def train_one_epoch(model, loader, criterion, optimizer, device, base_adj,
                    max_grad_norm, log_every, od_patterns=None, use_time=False):
    model.train()
    losses = []
    for i, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        hour = batch[2].to(device) if use_time and len(batch) > 2 else None
        dow = batch[3].to(device) if use_time and len(batch) > 3 else None

        optimizer.zero_grad()
        pred = model(x, base_adj, od_patterns=od_patterns,
                     hour_idx=hour, dow_idx=dow)
        loss = criterion(pred, y)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(loss.item())

        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"    batch {i+1}/{len(loader)}, loss={loss.item():.4f}")

    return np.mean(losses)


def main():
    args = parse_args()
    config = load_config(args.config)
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

    print("=" * 60)
    print("Training: Traffic Demand Prediction")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Flow Gate: {use_fg}, Time Embed: {use_te}")

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

    print(f"\nStart training: {tcfg['epochs']} epochs, "
          f"lr={tcfg['lr']}, batch_size={tcfg['batch_size']}")
    print("-" * 60)

    total_start = time.time()
    for epoch in range(1, tcfg['epochs'] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, base_adj,
            tcfg['max_grad_norm'], tcfg['log_every'],
            od_patterns=od_patterns, use_time=use_te)

        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, scaler, device, base_adj,
            od_patterns=od_patterns, use_time=use_te)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_MAE'].append(val_metrics['MAE'])
        history['val_RMSE'].append(val_metrics['RMSE'])
        history['val_MAPE'].append(val_metrics['MAPE'])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{tcfg['epochs']} | "
              f"train={train_loss:.4f} val={val_loss:.4f} | "
              f"MAE={val_metrics['MAE']:.3f} RMSE={val_metrics['RMSE']:.3f} "
              f"MAPE={val_metrics['MAPE']:.2f}% | "
              f"lr={lr_now:.6f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= tcfg['early_stop_patience']:
                print(f"\nEarly stopping at epoch {epoch}. Best: {best_epoch}")
                break

    total_time = time.time() - total_start
    print(f"\nTraining finished in {total_time/60:.1f} min")

    # Test
    print("\n" + "=" * 60)
    print("Evaluating on test set (best model)...")
    print("=" * 60)
    model.load_state_dict(torch.load(
        os.path.join(save_dir, 'best_model.pt'), weights_only=True))
    _, test_metrics = evaluate(
        model, test_loader, criterion, scaler, device, base_adj,
        od_patterns=od_patterns, use_time=use_te)

    print(f"\n  Test MAE:  {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

    results = {'best_epoch': best_epoch, 'test_metrics': test_metrics,
               'history': history, 'config': config,
               'total_time_min': total_time / 60}
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
