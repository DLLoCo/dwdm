"""
Step 1 验证脚本：确认数据管线完整跑通。

用法:
  1. 无数据时（测试管线）: python test_data_pipeline.py
  2. 有数据时: 
     先把 NYCTaxi/ 文件夹放到 data/processed/NYCTaxi/
     然后: python test_data_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from lib.utils import load_config, set_seed
from lib.dataloader import build_dataloaders
from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops
from lib.metrics import masked_mae, masked_rmse, masked_mape


def main():
    print("=" * 60)
    print("Step 1: Data Pipeline Test")
    print("=" * 60)

    # 1. Load config
    config = load_config('configs/nyctaxi.yaml')
    set_seed(config['train']['seed'])
    print("\n[1/5] Config loaded.")
    print(f"  Dataset: {config['data']['dataset']}")
    print(f"  Data dir: {config['data']['data_dir']}")

    # 2. Build dataloaders
    print("\n[2/5] Building dataloaders...")
    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)

    # 3. Check one batch
    print("\n[3/5] Checking batch shapes...")
    x, y = next(iter(train_loader))
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  x dtype: {x.dtype}")
    print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")

    # Validate dimensions
    B, T_in, N, C = x.shape
    _, T_out, N2, C2 = y.shape
    print(f"  Batch={B}, T_in={T_in}, T_out={T_out}, N={N}, C={C}")
    assert N == N2, f"Node mismatch: x has {N}, y has {N2}"
    assert C == C2, f"Feature mismatch: x has {C}, y has {C2}"
    print("  Shape checks PASSED.")

    # 4. Check adjacency matrix
    print("\n[4/5] Checking adjacency matrix...")
    if adj is not None:
        print(f"  Loaded from data: shape={adj.shape}")
        print(f"  Value range: [{adj.min():.4f}, {adj.max():.4f}]")
        print(f"  Density: {(adj > 0).sum() / adj.size:.4f}")
        adj_tensor = torch.FloatTensor(adj)
    else:
        print("  No adj_mx.npz found, building grid adjacency...")
        rows = config['data']['grid_rows']
        cols = config['data']['grid_cols']
        adj_np = build_grid_adj(rows, cols, include_diag=False)
        adj_np = add_self_loops(adj_np)
        adj_np = normalize_adj(adj_np)
        print(f"  Built grid adj: shape={adj_np.shape}")
        print(f"  Density: {(adj_np > 0).sum() / adj_np.size:.4f}")
        adj_tensor = torch.FloatTensor(adj_np)

    print(f"  Adj tensor: {adj_tensor.shape}")

    # 5. Test metrics with fake predictions
    print("\n[5/5] Testing metrics...")
    fake_pred = y + torch.randn_like(y) * 0.1
    y_real = scaler.inverse_transform(y)
    pred_real = scaler.inverse_transform(fake_pred)

    mae = masked_mae(pred_real, y_real)
    rmse = masked_rmse(pred_real, y_real)
    mape = masked_mape(pred_real, y_real)
    print(f"  Fake prediction: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")

    # Summary
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — Data pipeline is ready!")
    print("=" * 60)
    print(f"\n  Model input:  ({B}, {T_in}, {N}, {C})")
    print(f"  Model output: ({B}, {T_out}, {N}, {C})")
    print(f"  Adj matrix:   {adj_tensor.shape}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")


if __name__ == '__main__':
    main()
