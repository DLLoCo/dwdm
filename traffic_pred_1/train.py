"""
Training script for traffic demand prediction.

Usage:
    python train.py
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


def evaluate(model, loader, criterion, scaler, device, base_adj):
    """Run evaluation on a data loader, return loss and metrics."""
    model.eval()
    losses = []
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, base_adj)
            loss = criterion(pred, y)
            losses.append(loss.item())

            # inverse transform for real-scale metrics
            pred_real = scaler.inverse_transform(pred.cpu())
            y_real = scaler.inverse_transform(y.cpu())
            preds.append(pred_real.numpy())
            trues.append(y_real.numpy())

    avg_loss = np.mean(losses)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = compute_all_metrics(preds, trues)

    return avg_loss, metrics


def train_one_epoch(model, loader, criterion, optimizer, device, base_adj,
                    max_grad_norm, log_every):
    """Train for one epoch, return average loss."""
    model.train()
    losses = []

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x, base_adj)
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

    # --- Config ---
    config = load_config(args.config)
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.lr: config['train']['lr'] = args.lr
    if args.batch_size: config['train']['batch_size'] = args.batch_size

    tcfg = config['train']
    set_seed(tcfg['seed'])
    device = get_device(args.gpu)
    save_dir = get_project_path(tcfg['save_dir'])
    ensure_dir(save_dir)

    print("=" * 60)
    print("Training: Traffic Demand Prediction")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {config['data']['dataset']}")

    # --- Data ---
    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)

    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
    else:
        from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
        adj_np = normalize_adj(add_self_loops(
            build_grid_adj(config['data']['grid_rows'], config['data']['grid_cols'])))
        base_adj = torch.FloatTensor(adj_np).to(device)

    # --- Model ---
    model = TrafficPredNet.from_config(config).to(device)
    print_model_params(model)

    # --- Optimizer & Loss ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = MaskedMAELoss()

    # --- Training loop ---
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

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, base_adj,
            tcfg['max_grad_norm'], tcfg['log_every'])

        # Validate
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, scaler, device, base_adj)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_MAE'].append(val_metrics['MAE'])
        history['val_RMSE'].append(val_metrics['RMSE'])
        history['val_MAPE'].append(val_metrics['MAPE'])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{tcfg['epochs']} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"MAE={val_metrics['MAE']:.3f} RMSE={val_metrics['RMSE']:.3f} "
              f"MAPE={val_metrics['MAPE']:.2f}% | "
              f"lr={lr_now:.6f} | {elapsed:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= tcfg['early_stop_patience']:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best epoch: {best_epoch}")
                break

    total_time = time.time() - total_start
    print(f"\nTraining finished in {total_time/60:.1f} min")

    # --- Test ---
    print("\n" + "=" * 60)
    print("Evaluating on test set (best model)...")
    print("=" * 60)

    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_model.pt'),
                   weights_only=True))
    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, scaler, device, base_adj)

    print(f"\n  Test MAE:  {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")

    # Save results
    results = {
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'history': history,
        'config': config,
        'total_time_min': total_time / 60,
    }
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == '__main__':
    main()
