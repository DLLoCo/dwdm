"""
验证连续数据管线: 加载 demand.npz → 滑窗 → DataLoader → 检查 shapes + 时间索引。
用法: python test_cont_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from lib.utils import load_config, set_seed
from lib.dataloader import build_dataloaders
from lib.adj_builder import build_grid_adj, normalize_adj, add_self_loops


def main():
    print("=" * 60)
    print("  Test: Continuous Data Pipeline")
    print("=" * 60)

    config = load_config('configs/nyctaxi_cont.yaml')
    set_seed(config['train']['seed'])

    # Build dataloaders
    train_loader, val_loader, test_loader, scaler, adj = build_dataloaders(config)

    # Check one batch
    batch = next(iter(train_loader))
    x, y, hour, dow = batch
    print(f"\n--- Batch shapes ---")
    print(f"  x:    {x.shape}  (B, T_in, N, C)")
    print(f"  y:    {y.shape}  (B, T_out, N, C)")
    print(f"  hour: {hour.shape} range=[{hour.min()}, {hour.max()}]")
    print(f"  dow:  {dow.shape}  range=[{dow.min()}, {dow.max()}]")

    # Verify time indices make sense
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\n--- First 5 samples' target time ---")
    for i in range(min(5, len(hour))):
        h, m = divmod(int(hour[i]) * 30, 60)
        print(f"  sample {i}: {h:02d}:{m:02d} {day_names[int(dow[i])]}")

    # Check scaler
    print(f"\n--- Scaler ---")
    print(f"  mean shape: {scaler.mean.shape}")
    print(f"  std  shape: {scaler.std.shape}")

    # Check x range (should be roughly normalized)
    print(f"\n--- Normalized x stats ---")
    print(f"  x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  x mean:  {x.mean():.4f}")
    print(f"  x std:   {x.std():.4f}")

    # Inverse transform check
    y_real = scaler.inverse_transform(y.numpy().reshape(-1, 200, 2))
    print(f"\n--- Inverse-transformed y (real demand) ---")
    print(f"  range: [{y_real.min():.0f}, {y_real.max():.0f}]")
    print(f"  mean:  {y_real.mean():.2f}")

    # Adjacency matrix
    if adj is not None:
        print(f"\n--- Adjacency ---")
        print(f"  shape: {adj.shape}, density: {(adj > 0).mean():.4f}")
    else:
        print(f"\n--- No adj loaded, will use grid adj ---")
        rows = config['data']['grid_rows']
        cols = config['data']['grid_cols']
        adj_np = normalize_adj(add_self_loops(build_grid_adj(rows, cols)))
        print(f"  Built grid adj: {adj_np.shape}, density: {(adj_np > 0).mean():.4f}")

    # Dataset sizes
    print(f"\n--- Dataset sizes ---")
    print(f"  train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"  val:   {len(val_loader.dataset)} samples")
    print(f"  test:  {len(test_loader.dataset)} samples")

    print(f"\n{'=' * 60}")
    print("  Pipeline OK! Ready to run baselines.")
    print(f"  Next: python baselines.py --config configs/nyctaxi_cont.yaml")
    print("=" * 60)


if __name__ == '__main__':
    main()
