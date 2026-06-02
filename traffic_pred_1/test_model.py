"""
Step 2 验证脚本：确认模型前向传播跑通。
运行: python test_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from lib.utils import load_config, set_seed, print_model_params, get_device
from lib.dataloader import build_dataloaders
from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
from lib.metrics import MaskedMAELoss
from model.net import TrafficPredNet


def main():
    print("=" * 60)
    print("Step 2: Model Forward Pass Test")
    print("=" * 60)

    # 1. Setup
    config = load_config('configs/nyctaxi.yaml')
    set_seed(config['train']['seed'])
    device = get_device()
    print(f"\n[1/5] Device: {device}")

    # 2. Load data
    print("\n[2/5] Loading data...")
    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)
    x, y = next(iter(train_loader))
    print(f"  x: {x.shape}, y: {y.shape}")

    # 3. Build adjacency matrix
    print("\n[3/5] Building adjacency matrix...")
    if adj is not None:
        base_adj = torch.FloatTensor(adj).to(device)
        print(f"  Using loaded adj: {base_adj.shape}")
    else:
        rows = config['data']['grid_rows']
        cols = config['data']['grid_cols']
        adj_np = normalize_adj(add_self_loops(
            build_grid_adj(rows, cols)))
        base_adj = torch.FloatTensor(adj_np).to(device)
        print(f"  Using grid adj: {base_adj.shape}")

    # 4. Build model
    print("\n[4/5] Building model...")
    model = TrafficPredNet.from_config(config).to(device)
    print_model_params(model)

    # 5. Test forward pass
    print("\n[5/5] Testing forward pass...")
    x = x.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x, base_adj)

    print(f"  Input:      {x.shape}")
    print(f"  Prediction: {pred.shape}")
    print(f"  Target:     {y.shape}")
    assert pred.shape == y.shape, \
        f"Shape mismatch! pred={pred.shape}, target={y.shape}"
    print("  Shape check PASSED.")

    # Test backward pass
    print("\n  Testing backward pass...")
    model.train()
    pred = model(x, base_adj)
    criterion = MaskedMAELoss()
    loss = criterion(pred, y)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print("  Backward pass PASSED.")

    # Summary
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — Model is ready!")
    print("=" * 60)
    print(f"\n  Next step: write train.py (Step 3-4)")


if __name__ == '__main__':
    main()
