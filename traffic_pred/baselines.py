"""
Baseline comparison — all 8 models, compatible with 4-tuple dataloaders.

Baselines:
  1. HA           - Historical Average
  2. Last-value   - Copy last input step
  3. MLP          - Per-node feedforward
  4. LSTM         - Per-node LSTM
  5. TCN          - 1D temporal convolution
  6. Transformer  - Self-attention over time
  7. STGCN        - ChebNet (fixed adj) + temporal conv
  8. CCRNN        - CGC (learned adj) + GRU

Usage:
    python baselines.py --config configs/nyctaxi_cont.yaml
    python baselines.py --config configs/nyctaxi_cont.yaml --epochs 30
    python baselines.py --config configs/nyctaxi_cont.yaml --only STGCN CCRNN
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
# Batch unpacking helper (handles both 2-tuple and 4-tuple)
# ============================================================

def unpack_batch(batch):
    """Unpack (x, y) or (x, y, hour, dow) batch."""
    return batch[0], batch[1]


# ============================================================
# Simple Baselines
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, input_len, input_dim, output_dim, num_nodes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, T * C)
        return self.net(x).unsqueeze(1)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden, output_dim)
    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        out, _ = self.lstm(x)
        out = self.proj(out[:, -1, :])
        return out.reshape(B, N, -1).unsqueeze(1)


class TCNBaseline(nn.Module):
    def __init__(self, input_len, input_dim, output_dim, num_nodes, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, output_dim)
    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        out = self.conv(x).squeeze(-1)
        return self.proj(out).reshape(B, N, -1).unsqueeze(1)


class TransformerBaseline(nn.Module):
    def __init__(self, input_len, input_dim, output_dim, num_nodes,
                 d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.proj = nn.Linear(d_model, output_dim)
    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        x = self.input_proj(x) + self.pos_embed
        out = self.encoder(x).mean(dim=1)
        return self.proj(out).reshape(B, N, -1).unsqueeze(1)


# ============================================================
# STGCN Baseline — ChebNet (fixed adj) + Gated Temporal Conv
# ============================================================

class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_dim, out_dim))
            for _ in range(K)
        ])
        self._reset()

    def _reset(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x, adj):
        out = x @ self.weights[0]
        if self.K > 1:
            T0 = x
            T1 = torch.matmul(adj, x)
            out = out + T1 @ self.weights[1]
            for k in range(2, self.K):
                T2 = 2 * torch.matmul(adj, T1) - T0
                out = out + T2 @ self.weights[k]
                T0, T1 = T1, T2
        return out


class STGCNBlock(nn.Module):
    def __init__(self, dim, num_nodes, K=3, kernel_size=3):
        super().__init__()
        self.tconv = nn.Conv1d(dim, dim * 2, kernel_size,
                               padding=(kernel_size - 1) // 2)
        self.gconv = ChebConv(dim, dim, K)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        residual = x
        B, T, N, D = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        h = self.tconv(h)
        h_val, h_gate = h.chunk(2, dim=1)
        h = h_val * torch.sigmoid(h_gate)
        h = h.reshape(B, N, D, T).permute(0, 3, 1, 2)
        h = h.reshape(B * T, N, D)
        h = self.gconv(h, adj)
        h = h.reshape(B, T, N, D)
        return self.norm(h + residual)


class STGCNBaseline(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=3, K=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            STGCNBlock(hidden_dim, num_nodes, K) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, base_adj)
        h = h.mean(dim=1)
        return self.output_proj(h).unsqueeze(1)


# ============================================================
# CCRNN Baseline — CGC (learned adj) + GRU
# ============================================================

class GraphConvGRUCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, cheb_k=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        gate_in = in_dim + hidden_dim
        self.Wz = nn.ParameterList([nn.Parameter(torch.randn(gate_in, hidden_dim) * 0.05) for _ in range(cheb_k)])
        self.Wr = nn.ParameterList([nn.Parameter(torch.randn(gate_in, hidden_dim) * 0.05) for _ in range(cheb_k)])
        self.Wh = nn.ParameterList([nn.Parameter(torch.randn(gate_in, hidden_dim) * 0.05) for _ in range(cheb_k)])

    def _diff_conv(self, x, adj, weights):
        out = x @ weights[0]
        power = x
        for k in range(1, len(weights)):
            power = torch.matmul(adj, power)
            out = out + power @ weights[k]
        return out

    def forward(self, x_t, h_prev, adj):
        combined = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self._diff_conv(combined, adj, self.Wz))
        r = torch.sigmoid(self._diff_conv(combined, adj, self.Wr))
        combined_r = torch.cat([x_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self._diff_conv(combined_r, adj, self.Wh))
        return z * h_prev + (1 - z) * h_tilde


class CCRNNBaseline(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=2, cheb_k=3, embed_dim=30):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.cells = nn.ModuleList()
        for m in range(num_layers):
            dim_in = input_dim if m == 0 else hidden_dim
            self.cells.append(GraphConvGRUCell(dim_in, hidden_dim, cheb_k))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, base_adj=None):
        B, T, N, C = x.shape
        device = x.device
        adj = torch.softmax(torch.relu(self.E1 @ self.E2.t()), dim=-1)
        if base_adj is not None:
            adj = 0.5 * base_adj + 0.5 * adj
        h = [torch.zeros(B, N, self.hidden_dim, device=device)
             for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t, :, :]
            for m in range(self.num_layers):
                h[m] = self.cells[m](inp, h[m], adj)
                inp = h[m]
        return self.output_proj(h[-1]).unsqueeze(1)


# ============================================================
# Training + Eval (4-tuple safe)
# ============================================================

def train_model(model, train_loader, val_loader, device, base_adj,
                epochs=30, lr=0.001, patience=8, name="Model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MaskedMAELoss()
    best_val = float('inf')
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, base_adj), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                val_losses.append(criterion(model(x, base_adj), y).item())

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


def eval_model(model, loader, scaler, device, base_adj):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            pred = model(x, base_adj)
            preds.append(scaler.inverse_transform(pred.cpu()).numpy())
            trues.append(scaler.inverse_transform(y).numpy())
    return compute_all_metrics(np.concatenate(preds), np.concatenate(trues))


def baseline_ha(train_loader, test_loader, scaler):
    all_y = [batch[1].numpy() for batch in train_loader]
    mean_pred = np.concatenate(all_y).mean(axis=0, keepdims=True)
    preds, trues = [], []
    for batch in test_loader:
        y = batch[1]
        preds.append(scaler.inverse_transform(
            torch.FloatTensor(np.tile(mean_pred, (y.shape[0], 1, 1, 1)))).numpy())
        trues.append(scaler.inverse_transform(y).numpy())
    return compute_all_metrics(np.concatenate(preds), np.concatenate(trues))


def baseline_last_value(test_loader, scaler):
    preds, trues = [], []
    for batch in test_loader:
        x, y = batch[0], batch[1]
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
                   help='Run specific baselines, e.g. --only STGCN CCRNN MLP')
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['train']['seed'])
    device = get_device(args.gpu)

    print("=" * 70)
    print("Baseline Comparison (Continuous Data)")
    print("=" * 70)
    print(f"Device: {device}, Epochs: {args.epochs}")
    print(f"Config: {args.config}")

    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)

    # Adjacency
    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
    else:
        from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
        adj_np = normalize_adj(add_self_loops(
            build_grid_adj(config['data']['grid_rows'], config['data']['grid_cols'])))
        base_adj = torch.FloatTensor(adj_np).to(device)

    # Get dims from a batch
    batch = next(iter(train_loader))
    x_sample, y_sample = batch[0], batch[1]
    B, T, N, C = x_sample.shape
    out_dim = y_sample.shape[-1]
    print(f"Input: ({B}, {T}, {N}, {C}), Output dim: {out_dim}\n")

    results = {}
    run_all = args.only is None

    # --- Non-trainable ---
    if run_all or 'HA' in (args.only or []):
        print("[HA] Historical Average...")
        results['HA'] = baseline_ha(train_loader, test_loader, scaler)
        print(f"    MAE={results['HA']['MAE']:.4f}\n")

    if run_all or 'Last' in (args.only or []):
        print("[Last-value]...")
        results['Last-value'] = baseline_last_value(test_loader, scaler)
        print(f"    MAE={results['Last-value']['MAE']:.4f}\n")

    # --- All trainable baselines ---
    trainable = {
        'MLP': lambda: MLPBaseline(T, C, out_dim, N, hidden=128),
        'LSTM': lambda: LSTMBaseline(C, out_dim, N, hidden=64, num_layers=2),
        'TCN': lambda: TCNBaseline(T, C, out_dim, N, hidden=64),
        'Transformer': lambda: TransformerBaseline(T, C, out_dim, N),
        'STGCN': lambda: STGCNBaseline(N, C, out_dim, hidden_dim=64,
                                        num_layers=3, K=3),
        'CCRNN': lambda: CCRNNBaseline(N, C, out_dim, hidden_dim=64,
                                        num_layers=2, cheb_k=3, embed_dim=30),
    }

    for name, builder in trainable.items():
        if not run_all and name not in (args.only or []):
            continue
        print(f"[{name}] Training...")
        model = builder()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Params: {params:,}")
        t0 = time.time()
        model = train_model(model, train_loader, val_loader, device, base_adj,
                            epochs=args.epochs, name=name)
        results[name] = eval_model(model, test_loader, scaler, device, base_adj)
        print(f"    MAE={results[name]['MAE']:.4f}  Time: {time.time()-t0:.0f}s\n")

    # --- Our model (from checkpoint, if exists) ---
    ckpt = get_project_path('checkpoints', 'best_model.pt')
    if os.path.exists(ckpt) and (run_all or 'Ours' in (args.only or [])):
        from model.net import TrafficPredNet
        print("[Ours] Loading from checkpoint...")
        try:
            model = TrafficPredNet.from_config(config).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device,
                                             weights_only=True))
            od_patterns = None
            od_path = config['data'].get('od_path', '')
            if od_path:
                if not os.path.isabs(od_path):
                    od_path = get_project_path(od_path)
                if os.path.exists(od_path):
                    od_npz = np.load(od_path)
                    od_key = 'od' if 'od' in od_npz else list(od_npz.keys())[0]
                    od_patterns = torch.FloatTensor(od_npz[od_key]).to(device)

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device)
                    y = batch[1]
                    hour = batch[2].to(device) if len(batch) > 2 else None
                    dow = batch[3].to(device) if len(batch) > 3 else None
                    pred = model(x, base_adj, od_patterns=od_patterns,
                                 hour_idx=hour, dow_idx=dow)
                    preds.append(scaler.inverse_transform(pred.cpu()).numpy())
                    trues.append(scaler.inverse_transform(y).numpy())
            results['Ours (TC+CGC)'] = compute_all_metrics(
                np.concatenate(preds), np.concatenate(trues))
            print(f"    MAE={results['Ours (TC+CGC)']['MAE']:.4f}\n")
        except Exception as e:
            print(f"    [SKIP] Checkpoint incompatible: {e}\n")

    # --- Results table ---
    print("=" * 70)
    print(f"{'Model':<20s} {'MAE':>10s} {'RMSE':>10s} {'MAPE(%)':>10s}")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda x: x[1]['MAE']):
        print(f"{name:<20s} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>10.2f}")
    print("=" * 70)

    best = min(results, key=lambda k: results[k]['MAE'])
    print(f"\nBest: {best} (MAE={results[best]['MAE']:.4f})")


if __name__ == '__main__':
    main()
