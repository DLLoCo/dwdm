"""
检查原始预处理数据的格式，确认后再写 dataloader。
用法: python inspect_demand.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from lib.utils import get_project_path

def main():
    print("=" * 70)
    print("  Inspect preprocessed data")
    print("=" * 70)

    # ============================================================
    # 1. demand.npz
    # ============================================================
    demand_path = get_project_path('data', 'processed', 'NYCTaxi_OD', 'demand.npz')
    if not os.path.exists(demand_path):
        print(f"\n[ERROR] Not found: {demand_path}")
        print("  Run preprocess_raw.py first")
        return

    npz = np.load(demand_path)
    print(f"\n--- demand.npz ---")
    print(f"  Keys: {list(npz.keys())}")
    for k in npz.keys():
        arr = npz[k]
        print(f"  '{k}': shape={arr.shape}, dtype={arr.dtype}")

    # 假设 key 是 'data'，如果不是就用第一个 key
    key = 'data' if 'data' in npz else list(npz.keys())[0]
    demand = npz[key]

    T, N, C = demand.shape
    print(f"\n--- Demand tensor ---")
    print(f"  T={T} time steps, N={N} nodes, C={C} features")
    print(f"  时间跨度: {T} steps × 30min = {T/48:.1f} 天")
    print(f"  起始: 2015-01-01 00:00 (推测)")
    print(f"  结束: 约 2015-{1 + T//(48*30):02d}-{(T%(48*30))//48 + 1:02d}")

    print(f"\n--- 基本统计 ---")
    print(f"  全局: min={demand.min():.0f}, max={demand.max():.0f}, "
          f"mean={demand.mean():.2f}, std={demand.std():.2f}")
    print(f"  Pickup  (feat 0): mean={demand[:,:,0].mean():.2f}, "
          f"max={demand[:,:,0].max():.0f}, zeros={100*(demand[:,:,0]==0).mean():.1f}%")
    print(f"  Dropoff (feat 1): mean={demand[:,:,1].mean():.2f}, "
          f"max={demand[:,:,1].max():.0f}, zeros={100*(demand[:,:,1]==0).mean():.1f}%")

    # 按天检查是否完整
    full_days = T // 48
    remainder = T % 48
    print(f"\n--- 时间完整性 ---")
    print(f"  完整天数: {full_days}, 剩余 slots: {remainder}")

    # 每天总需求（看是否有异常天）
    if full_days > 0:
        daily_total = np.zeros(full_days)
        for d in range(full_days):
            daily_total[d] = demand[d*48:(d+1)*48].sum()
        print(f"\n--- 每日总需求 ---")
        print(f"  mean={daily_total.mean():.0f}, std={daily_total.std():.0f}")
        print(f"  min day: day {daily_total.argmin()} = {daily_total.min():.0f}")
        print(f"  max day: day {daily_total.argmax()} = {daily_total.max():.0f}")
        # 打印前几天和后几天
        print(f"  前5天: {[f'{v:.0f}' for v in daily_total[:5]]}")
        print(f"  后5天: {[f'{v:.0f}' for v in daily_total[-5:]]}")

    # 样本量估算（滑窗）
    for input_len in [8, 12, 16, 24]:
        n_samples = T - input_len - 1 + 1  # input + 1 output
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.1)
        n_test = n_samples - n_train - n_val
        print(f"\n  input_len={input_len} ({input_len*0.5:.0f}h): "
              f"total={n_samples}, train={n_train}, val={n_val}, test={n_test}")

    # ============================================================
    # 2. od_hourly.npz
    # ============================================================
    od_path = get_project_path('data', 'processed', 'NYCTaxi_OD', 'od_hourly.npz')
    if os.path.exists(od_path):
        od_npz = np.load(od_path)
        print(f"\n--- od_hourly.npz ---")
        print(f"  Keys: {list(od_npz.keys())}")
        for k in od_npz.keys():
            arr = od_npz[k]
            print(f"  '{k}': shape={arr.shape}, dtype={arr.dtype}")
        
        od_key = 'od' if 'od' in od_npz else list(od_npz.keys())[0]
        od = od_npz[od_key]
        print(f"  mean={od.mean():.4f}, max={od.max():.2f}, "
              f"sparsity={100*(od==0).mean():.1f}%")
    else:
        print(f"\n  [WARN] od_hourly.npz not found at {od_path}")

    # ============================================================
    # 3. adj_mx.npz (if exists)
    # ============================================================
    adj_path = get_project_path('data', 'processed', 'NYCTaxi', 'adj_mx.npz')
    if os.path.exists(adj_path):
        adj_npz = np.load(adj_path, allow_pickle=True)
        print(f"\n--- adj_mx.npz ---")
        print(f"  Keys: {list(adj_npz.keys())}")
        for k in adj_npz.keys():
            arr = adj_npz[k]
            if isinstance(arr, np.ndarray):
                print(f"  '{k}': shape={arr.shape}, dtype={arr.dtype}, "
                      f"density={(arr>0).mean():.4f}")

    # ============================================================
    # 4. 确认 demand 的一个样本窗口
    # ============================================================
    print(f"\n--- 样本窗口示例 (input_len=12) ---")
    idx = 48 * 3  # 第4天开始
    x = demand[idx:idx+12]       # (12, 200, 2)
    y = demand[idx+12:idx+13]    # (1, 200, 2)
    
    # 时间信息
    hour_slot = (idx + 12) % 48
    dow = ((idx + 12) // 48) % 7  # 周四=0 if starting from Thursday
    # 2015-01-01 is Thursday (weekday=3 in Python, but we can use 0-6 starting from day 0)
    actual_dow = ((idx + 12) // 48 + 3) % 7  # 3=Thursday for Jan 1

    h, m = divmod(hour_slot * 30, 60)
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    
    print(f"  x: shape={x.shape}, 从 step {idx} 到 {idx+11}")
    print(f"  y: shape={y.shape}, step {idx+12}")
    print(f"  目标时间: slot {hour_slot} = {h:02d}:{m:02d}, "
          f"day_of_week={actual_dow} ({day_names[actual_dow]})")
    print(f"  x 总需求范围: [{x.sum(axis=(1,2)).min():.0f}, {x.sum(axis=(1,2)).max():.0f}]")
    print(f"  y 总需求: {y.sum():.0f}")

    print(f"\n{'=' * 70}")
    print("  Done. Use these results to configure the new dataloader.")
    print("=" * 70)


if __name__ == '__main__':
    main()