"""
Baseline comparison script with trainable models.

Baselines:
  1. HA          - Historical Average (no training)
  2. Last-value  - Copy last input step (no training)
  3. MLP         - Flatten 35 steps as features, per-node MLP
  4. LSTM        - Standard LSTM on 35-step sequence
  5. TCN         - 1D temporal convolution (no sequential assumption)
  6. Transformer - Self-attention over 35 steps

Usage:
    python baselines.py
    python baselines.py --epochs 30
    python baselines.py --only MLP TCN
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
from lib.utils import load_config, set_seed, get_device, get_project_path
from lib.dataloader import build_dataloaders
from lib.metrics import compute_all_metrics, MaskedMAELoss


# ============================================================
# Baseline Models
# ============================================================

class MLPBaseline(nn.Module):
    """Flatten (T, C) per node -> MLP -> (output_dim)."""
    def __init__(self, input_len, input_dim, output_dim, num_nodes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, T * C)
        out = self.net(x)
        return out.unsqueeze(1)


class LSTMBaseline(nn.Module):
    """Per-node LSTM over T steps."""
    def __init__(self, input_dim, output_dim, num_nodes, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden, output_dim)

    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.proj(out)
        return out.reshape(B, N, -1).unsqueeze(1)


class TCNBaseline(nn.Module):
    """1D temporal convolution - no sequential assumption."""
    def __init__(self, input_len, input_dim, output_dim, num_nodes, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, output_dim)

    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        out = self.conv(x).squeeze(-1)
        out = self.proj(out)
        return out.reshape(B, N, -1).unsqueeze(1)


class TransformerBaseline(nn.Module):
    """Transformer encoder over T steps per node."""
    def __init__(self, input_len, input_dim, output_dim, num_nodes,
                 d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        x = self.input_proj(x) + self.pos_embed
        out = self.encoder(x)
        out = out.mean(dim=1)
        out = self.proj(out)
        return out.reshape(B, N, -1).unsqueeze(1)


# ============================================================
# Training helper
# ============================================================

def train_model(model, train_loader, val_loader, device, epochs=30,
                lr=0.001, patience=8, name="Model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MaskedMAELoss()
    best_val = float('inf')
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(criterion(model(x), y).item())

        tl, vl = np.mean(losses), np.mean(val_losses)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{name:12s}] Epoch {epoch:3d}: train={tl:.4f} val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{name:12s}] Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def eval_model(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())
    return compute_all_metrics(np.concatenate(preds), np.concatenate(trues))


# ============================================================
# Non-trainable baselines
# ============================================================

def baseline_ha(train_loader, test_loader, scaler):
    all_y = [y.numpy() for _, y in train_loader]
    mean_pred = np.concatenate(all_y).mean(axis=0, keepdims=True)
    preds, trues = [], []
    for _, y in test_loader:
        preds.append(scaler.inverse_transform(
            torch.FloatTensor(np.tile(mean_pred, (y.shape[0],1,1,1)))).numpy())
        trues.append(scaler.inverse_transform(y).numpy())
    return compute_all_metrics(np.concatenate(preds), np.concatenate(trues))


def baseline_last_value(test_loader, scaler):
    preds, trues = [], []
    for x, y in test_loader:
        preds.append(scaler.inverse_transform(x[:, -1:]).numpy())
        trues.append(scaler.inverse_transform(y).numpy())
    return compute_all_metrics(np.concatenate(preds), np.concatenate(trues))


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nyctaxi.yaml')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--only', nargs='*', default=None,
                   help='e.g. --only MLP TCN Transformer')
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])
    device = get_device(args.gpu)

    print("=" * 70)
    print("Baseline Comparison")
    print("=" * 70)
    print(f"Device: {device}, Training epochs: {args.epochs}")

    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)

    x_sample, y_sample = next(iter(train_loader))
    B, T, N, C = x_sample.shape
    out_dim = y_sample.shape[-1]
    print(f"Input: ({B}, {T}, {N}, {C}), Output dim: {out_dim}\n")

    results = {}
    run_all = args.only is None

    # --- Non-trainable ---
    if run_all or 'HA' in (args.only or []):
        print("[1] HA (Historical Average)...")
        results['HA'] = baseline_ha(train_loader, test_loader, scaler)
        print(f"    MAE={results['HA']['MAE']:.4f}\n")

    if run_all or 'Last' in (args.only or []):
        print("[2] Last-value...")
        results['Last-value'] = baseline_last_value(test_loader, scaler)
        print(f"    MAE={results['Last-value']['MAE']:.4f}\n")

    # --- Trainable ---
    trainable = {
        'MLP': lambda: MLPBaseline(T, C, out_dim, N, hidden=128),
        'LSTM': lambda: LSTMBaseline(C, out_dim, N, hidden=64, num_layers=2),
        'TCN': lambda: TCNBaseline(T, C, out_dim, N, hidden=64),
        'Transformer': lambda: TransformerBaseline(T, C, out_dim, N, d_model=64, nhead=4, num_layers=2),
    }

    idx = 3
    for name, builder in trainable.items():
        if not run_all and name not in (args.only or []):
            continue
        print(f"[{idx}] {name}...")
        model = builder()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Params: {params:,}")
        t0 = time.time()
        model = train_model(model, train_loader, val_loader, device,
                            epochs=args.epochs, name=name)
        results[name] = eval_model(model, test_loader, scaler, device)
        print(f"    MAE={results[name]['MAE']:.4f}  Time: {time.time()-t0:.0f}s\n")
        idx += 1

    # --- Our CCRNN ---
    ckpt = get_project_path('checkpoints', 'best_model.pt')
    if os.path.exists(ckpt) and (run_all or 'CCRNN' in (args.only or [])):
        from model.net import TrafficPredNet
        print(f"[{idx}] CCRNN (ours, from checkpoint)...")
        model = TrafficPredNet.from_config(config).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        if adj is not None:
            base_adj = torch.FloatTensor(adj).to(device)
        else:
            base_adj = None
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device), base_adj)
                preds.append(scaler.inverse_transform(pred.cpu()).numpy())
                trues.append(scaler.inverse_transform(y).numpy())
        results['CCRNN (ours)'] = compute_all_metrics(
            np.concatenate(preds), np.concatenate(trues))
        print(f"    MAE={results['CCRNN (ours)']['MAE']:.4f}\n")

    # --- Final table ---
    print("=" * 70)
    print(f"{'Model':<20s} {'MAE':>10s} {'RMSE':>10s} {'MAPE(%)':>10s} {'Params':>12s}")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda x: x[1]['MAE']):
        print(f"{name:<20s} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>10.2f}")
    print("=" * 70)

    best = min(results, key=lambda k: results[k]['MAE'])
    print(f"\nBest: {best} (MAE={results[best]['MAE']:.4f})")


if __name__ == '__main__':
    main()
