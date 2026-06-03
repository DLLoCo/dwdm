"""
NYC Taxi 出行需求数据探索 — 生成报告所需的全部数据可视化

生成图表:
  1. 数据概览统计（打印）
  2. 空间热力图：平均 pickup / dropoff 需求 (10×20 网格)
  3. 时间模式：48 个半小时的日内需求曲线
  4. 早高峰 / 晚高峰 / 深夜低谷 热力图对比
  5. 工作日 vs 周末 对比
  6. OD 流量分析：Top OD 对 + OD 矩阵热力图
  7. 需求分布直方图 + 长尾分析
  8. 数据集拆分统计

用法:
    python explore_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from lib.utils import get_project_path, ensure_dir

# ============================================================
# 配置
# ============================================================
GRID_ROWS, GRID_COLS = 10, 20
NUM_NODES = GRID_ROWS * GRID_COLS
SLOTS_PER_DAY = 48           # 30-min intervals
START_DOW = 3                 # 2015-01-01 is Thursday

LAT_MIN, LAT_MAX = 40.60, 40.90
LON_MIN, LON_MAX = -74.06, -73.77

SAVE_DIR = get_project_path('figures', 'exploration')

# 美观配色
CMAP_HEAT = 'YlOrRd'
CMAP_BLUE = 'YlGnBu'
COLORS = {
    'pickup':  '#E74C3C',
    'dropoff': '#3498DB',
    'morning': '#F39C12',
    'evening': '#E74C3C',
    'night':   '#2C3E50',
    'weekday': '#E74C3C',
    'weekend': '#3498DB',
}

# 时段定义（半小时 slot 编号 0-47）
PERIODS = {
    'morning_peak':  (14, 20),   # 7:00 - 10:00
    'evening_peak':  (34, 40),   # 17:00 - 20:00
    'late_night':    (0, 8),     # 0:00 - 4:00
    'midday':        (22, 28),   # 11:00 - 14:00
}


def slot_to_time(slot):
    """Convert slot index (0-47) to time string."""
    h, m = divmod(slot * 30, 60)
    return f"{h:02d}:{m:02d}"


# ============================================================
# 数据加载
# ============================================================
def load_all_data():
    """Load ST-SSL split data + OD patterns."""
    data_dir = get_project_path('data', 'processed', 'NYCTaxi')

    splits = {}
    for name in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f'{name}.npz')
        npz = np.load(path, allow_pickle=True)
        keys = list(npz.keys())
        x_key = 'X' if 'X' in keys else 'x'
        y_key = 'Y' if 'Y' in keys else 'y'
        splits[name] = {
            'X': npz[x_key].astype(np.float32),
            'Y': npz[y_key].astype(np.float32),
        }
        print(f"  {name}: X={splits[name]['X'].shape}, Y={splits[name]['Y'].shape}")

    # Adjacency
    adj_path = os.path.join(data_dir, 'adj_mx.npz')
    adj = None
    if os.path.exists(adj_path):
        adj = np.load(adj_path, allow_pickle=True)['adj_mx']
        print(f"  adj: {adj.shape}")

    # OD hourly patterns
    od_path = get_project_path('data', 'processed', 'NYCTaxi_OD', 'od_hourly.npz')
    od_hourly = None
    if os.path.exists(od_path):
        od_hourly = np.load(od_path)['od']
        print(f"  od_hourly: {od_hourly.shape}")
    else:
        print("  [WARN] od_hourly.npz not found, OD analysis will be skipped")

    return splits, adj, od_hourly


def reconstruct_demand_series(splits):
    """
    从 ST-SSL 的 Y 标签（每个样本的预测目标）重建连续的需求时间序列。
    Y shape: (samples, 1, 200, 2) — 每个样本对应 1 个时间步的 ground truth。
    
    注意: 样本之间是滑窗生成的，相邻样本的 Y 对应相邻时间步。
    """
    # 合并 train/val/test 的 Y 来获得完整时间序列
    all_y = np.concatenate([
        splits['train']['Y'],
        splits['val']['Y'],
        splits['test']['Y'],
    ], axis=0)  # (total_samples, 1, 200, 2)

    demand = all_y[:, 0, :, :]  # (T, 200, 2)
    return demand


def compute_time_indices(n_samples, offset=144):
    """为每个样本计算对应的 hour_slot (0-47) 和 day_of_week (0-6)."""
    indices = np.arange(n_samples) + offset
    hour_slot = indices % SLOTS_PER_DAY
    day_of_week = ((indices // SLOTS_PER_DAY) + START_DOW) % 7
    return hour_slot, day_of_week


# ============================================================
# 图1: 数据概览统计（打印）
# ============================================================
def print_data_overview(splits, demand, od_hourly):
    """打印完整的数据集统计信息."""
    print("\n" + "=" * 70)
    print("                     数据概览 / Data Overview")
    print("=" * 70)

    n_train = len(splits['train']['X'])
    n_val = len(splits['val']['X'])
    n_test = len(splits['test']['X'])
    n_total = n_train + n_val + n_test

    print(f"\n--- 数据集拆分 ---")
    print(f"  训练集:  {n_train:>5} samples ({n_train/n_total*100:.1f}%)")
    print(f"  验证集:  {n_val:>5} samples ({n_val/n_total*100:.1f}%)")
    print(f"  测试集:  {n_test:>5} samples ({n_test/n_total*100:.1f}%)")
    print(f"  总计:    {n_total:>5} samples")

    X0 = splits['train']['X']
    print(f"\n--- 输入特征 ---")
    print(f"  X shape: {X0.shape}  →  (samples, {X0.shape[1]} time steps, "
          f"{X0.shape[2]} nodes, {X0.shape[3]} features)")
    print(f"  输入窗口: {X0.shape[1]} steps (含近期 + 3天前同时段上下文)")
    print(f"  空间网格: {GRID_ROWS}×{GRID_COLS} = {NUM_NODES} 区域")
    print(f"  特征: [pickup_demand, dropoff_demand]")
    print(f"  时间间隔: 30 分钟")

    print(f"\n--- 需求统计 (原始值, 合并全部样本的 Y) ---")
    print(f"  demand shape: {demand.shape}  →  (time_steps, nodes, features)")
    print(f"  Pickup  — mean: {demand[:,:,0].mean():.2f}, "
          f"std: {demand[:,:,0].std():.2f}, "
          f"max: {demand[:,:,0].max():.0f}, "
          f"非零比例: {(demand[:,:,0]>0).mean()*100:.1f}%")
    print(f"  Dropoff — mean: {demand[:,:,1].mean():.2f}, "
          f"std: {demand[:,:,1].std():.2f}, "
          f"max: {demand[:,:,1].max():.0f}, "
          f"非零比例: {(demand[:,:,1]>0).mean()*100:.1f}%")

    # 稀疏性
    zero_ratio = (demand == 0).mean() * 100
    print(f"\n  数据稀疏度: {zero_ratio:.1f}% 的 (time, node, feature) 值为 0")

    # 时间跨度
    total_days = n_total / SLOTS_PER_DAY
    print(f"\n--- 时间跨度 ---")
    print(f"  约 {total_days:.1f} 天 ({n_total} 个 30 分钟时间片)")
    print(f"  起始日: 2015-01-01 (周四)")

    if od_hourly is not None:
        print(f"\n--- OD 流量 (od_hourly) ---")
        print(f"  shape: {od_hourly.shape}  →  (48 half-hour slots, "
              f"{od_hourly.shape[1]} origins, {od_hourly.shape[2]} destinations)")
        print(f"  mean: {od_hourly.mean():.4f}, max: {od_hourly.max():.2f}")
        total_flow = od_hourly.sum(axis=(1, 2))
        peak_slot = total_flow.argmax()
        print(f"  最繁忙半小时: slot {peak_slot} ({slot_to_time(peak_slot)}) "
              f"— 总流量 {total_flow[peak_slot]:.0f}")


# ============================================================
# 图2: 空间需求热力图
# ============================================================
def plot_spatial_heatmap(demand, save_dir):
    """平均 pickup 和 dropoff 需求的空间分布热力图."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for idx, (feat, label, cmap) in enumerate([
        (0, 'Pickup (上车)', CMAP_HEAT),
        (1, 'Dropoff (下车)', CMAP_BLUE),
    ]):
        avg = demand[:, :, feat].mean(axis=0)  # (200,)
        grid = avg.reshape(GRID_ROWS, GRID_COLS)

        ax = axes[idx]
        im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='lower',
                       extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(f'Average {label} Demand per 30min', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Avg trips / 30min', shrink=0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_spatial_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 01_spatial_heatmap.png")


# ============================================================
# 图3: 日内时间模式（48 个半小时 slot）
# ============================================================
def plot_temporal_pattern(demand, save_dir):
    """一天 48 个半小时的平均需求曲线 — 揭示早晚高峰."""
    hour_slots, _ = compute_time_indices(len(demand))

    # 按 slot 聚合
    pickup_by_slot = np.zeros(SLOTS_PER_DAY)
    dropoff_by_slot = np.zeros(SLOTS_PER_DAY)
    count_by_slot = np.zeros(SLOTS_PER_DAY)

    for t in range(len(demand)):
        s = hour_slots[t]
        pickup_by_slot[s] += demand[t, :, 0].sum()
        dropoff_by_slot[s] += demand[t, :, 1].sum()
        count_by_slot[s] += 1

    count_by_slot[count_by_slot == 0] = 1
    pickup_by_slot /= count_by_slot
    dropoff_by_slot /= count_by_slot

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(SLOTS_PER_DAY)

    ax.fill_between(x, pickup_by_slot, alpha=0.2, color=COLORS['pickup'])
    ax.fill_between(x, dropoff_by_slot, alpha=0.2, color=COLORS['dropoff'])
    ax.plot(x, pickup_by_slot, '-o', color=COLORS['pickup'],
            markersize=4, linewidth=2, label='Pickup (上车)')
    ax.plot(x, dropoff_by_slot, '-s', color=COLORS['dropoff'],
            markersize=4, linewidth=2, label='Dropoff (下车)')

    # 标注高峰时段
    for name, (s, e) in PERIODS.items():
        color = {'morning_peak': COLORS['morning'],
                 'evening_peak': COLORS['evening'],
                 'late_night': COLORS['night'],
                 'midday': '#95A5A6'}[name]
        label_cn = {'morning_peak': '早高峰',
                    'evening_peak': '晚高峰',
                    'late_night': '深夜低谷',
                    'midday': '午间'}[name]
        ax.axvspan(s, e, alpha=0.08, color=color, label=label_cn)

    ax.set_xticks(np.arange(0, 48, 4))
    ax.set_xticklabels([slot_to_time(s) for s in range(0, 48, 4)], fontsize=9)
    ax.set_xlabel('Time of Day', fontsize=11)
    ax.set_ylabel('Total Demand (all regions)', fontsize=11)
    ax.set_title('Daily Temporal Pattern — Peak Hours vs Late Night',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.5, 47.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_temporal_pattern.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 02_temporal_pattern.png")

    return pickup_by_slot, dropoff_by_slot


# ============================================================
# 图4: 早高峰 / 晚高峰 / 深夜 热力图对比（报告必需）
# ============================================================
def plot_peak_comparison_heatmap(demand, save_dir):
    """三个时段的空间需求热力图并排对比."""
    hour_slots, _ = compute_time_indices(len(demand))

    period_names = ['morning_peak', 'evening_peak', 'late_night']
    period_labels = ['Morning Peak\n(7:00-10:00)',
                     'Evening Peak\n(17:00-20:00)',
                     'Late Night\n(0:00-4:00)']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 全局 vmax 统一色阶
    all_means = []
    for pname in period_names:
        s, e = PERIODS[pname]
        mask = (hour_slots >= s) & (hour_slots < e)
        if mask.sum() > 0:
            all_means.append(demand[mask].mean(axis=0))

    vmax_pickup = max(m[:, 0].max() for m in all_means)
    vmax_dropoff = max(m[:, 1].max() for m in all_means)

    for col, (pname, plabel) in enumerate(zip(period_names, period_labels)):
        s, e = PERIODS[pname]
        mask = (hour_slots >= s) & (hour_slots < e)
        avg = demand[mask].mean(axis=0) if mask.sum() > 0 else np.zeros((NUM_NODES, 2))

        for row, (feat, feat_label, cmap, vmax) in enumerate([
            (0, 'Pickup', CMAP_HEAT, vmax_pickup),
            (1, 'Dropoff', CMAP_BLUE, vmax_dropoff),
        ]):
            grid = avg[:, feat].reshape(GRID_ROWS, GRID_COLS)
            ax = axes[row, col]
            im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='lower',
                           vmin=0, vmax=vmax,
                           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
            if row == 0:
                ax.set_title(plabel, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{feat_label}\nLatitude', fontsize=10)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Longitude' if row == 1 else '', fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Spatial Demand: Peak Hours vs Late Night Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_peak_comparison_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 03_peak_comparison_heatmap.png")


# ============================================================
# 图5: 工作日 vs 周末
# ============================================================
def plot_weekday_vs_weekend(demand, save_dir):
    """工作日和周末的日内需求曲线对比."""
    hour_slots, dow = compute_time_indices(len(demand))
    is_weekend = (dow == 0) | (dow == 6)  # 0=Sun, 6=Sat

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for feat_idx, (feat_label, ax) in enumerate(
            zip(['Pickup (上车)', 'Dropoff (下车)'], axes)):

        for label, mask, color, ls in [
            ('Weekday (工作日)', ~is_weekend, COLORS['weekday'], '-'),
            ('Weekend (周末)',    is_weekend,  COLORS['weekend'], '--'),
        ]:
            means = np.zeros(SLOTS_PER_DAY)
            counts = np.zeros(SLOTS_PER_DAY)
            for t in range(len(demand)):
                if mask[t]:
                    s = hour_slots[t]
                    means[s] += demand[t, :, feat_idx].sum()
                    counts[s] += 1
            counts[counts == 0] = 1
            means /= counts
            ax.plot(range(SLOTS_PER_DAY), means, ls, color=color,
                    linewidth=2, label=label)

        ax.set_xticks(np.arange(0, 48, 4))
        ax.set_xticklabels([slot_to_time(s) for s in range(0, 48, 4)], fontsize=8)
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Total Demand')
        ax.set_title(feat_label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Weekday vs Weekend Demand Pattern', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_weekday_vs_weekend.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 04_weekday_vs_weekend.png")


# ============================================================
# 图6: OD 流量分析
# ============================================================
def plot_od_analysis(od_hourly, save_dir):
    """OD 矩阵热力图 + 高峰 vs 深夜 OD 对比."""
    if od_hourly is None:
        print("  [SKIP] No OD data")
        return

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (a) 全天平均 OD
    ax1 = fig.add_subplot(gs[0, 0])
    od_avg = od_hourly.mean(axis=0)
    im1 = ax1.imshow(np.log1p(od_avg), cmap='hot', aspect='equal')
    ax1.set_title('Daily Average OD\n(log scale)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Destination node')
    ax1.set_ylabel('Origin node')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # (b) 早高峰 OD
    ax2 = fig.add_subplot(gs[0, 1])
    s, e = PERIODS['morning_peak']
    od_morning = od_hourly[s:e].mean(axis=0)
    im2 = ax2.imshow(np.log1p(od_morning), cmap='hot', aspect='equal')
    ax2.set_title('Morning Peak OD\n(7:00-10:00, log)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Destination node')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # (c) 深夜 OD
    ax3 = fig.add_subplot(gs[0, 2])
    s, e = PERIODS['late_night']
    od_night = od_hourly[s:e].mean(axis=0)
    im3 = ax3.imshow(np.log1p(od_night), cmap='hot', aspect='equal')
    ax3.set_title('Late Night OD\n(0:00-4:00, log)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Destination node')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # (d) 各时段总 OD 流量
    ax4 = fig.add_subplot(gs[1, 0:2])
    total_flow = od_hourly.sum(axis=(1, 2))
    x_slots = np.arange(SLOTS_PER_DAY)
    ax4.bar(x_slots, total_flow, color='#E74C3C', alpha=0.7, width=0.8)
    ax4.set_xticks(np.arange(0, 48, 4))
    ax4.set_xticklabels([slot_to_time(s) for s in range(0, 48, 4)])
    ax4.set_xlabel('Time of Day')
    ax4.set_ylabel('Total OD Flow')
    ax4.set_title('Total OD Flow by Half-Hour Slot', fontsize=11, fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # (e) Top-10 OD 对
    ax5 = fig.add_subplot(gs[1, 2])
    flat = od_avg.flatten()
    top_k = 10
    top_idx = np.argsort(flat)[-top_k:][::-1]
    labels = []
    values = []
    for idx in top_idx:
        i, j = divmod(idx, NUM_NODES)
        ri, ci = divmod(i, GRID_COLS)
        rj, cj = divmod(j, GRID_COLS)
        labels.append(f'({ri},{ci})→({rj},{cj})')
        values.append(flat[idx])

    y_pos = np.arange(top_k)
    ax5.barh(y_pos, values, color='#3498DB', alpha=0.8)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels, fontsize=8)
    ax5.set_xlabel('Avg trips / 30min')
    ax5.set_title('Top-10 OD Pairs', fontsize=11, fontweight='bold')
    ax5.invert_yaxis()
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    plt.savefig(os.path.join(save_dir, '05_od_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 05_od_analysis.png")


# ============================================================
# 图7: 需求值分布（稀疏性 + 长尾）
# ============================================================
def plot_demand_distribution(demand, save_dir):
    """需求值的分布直方图 — 展示数据稀疏性和长尾特性."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) 全部值的分布
    all_vals = demand.flatten()
    ax = axes[0]
    ax.hist(all_vals, bins=100, color='#34495E', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Demand Value')
    ax.set_ylabel('Count')
    ax.set_title('Overall Distribution', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) 非零值的分布
    nonzero = all_vals[all_vals > 0]
    ax = axes[1]
    ax.hist(nonzero, bins=100, color='#E74C3C', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Demand Value (non-zero only)')
    ax.set_ylabel('Count')
    ax.set_title(f'Non-Zero Distribution (n={len(nonzero):,})',
                 fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) 各节点的平均需求排名
    ax = axes[2]
    node_avg = demand.mean(axis=0)  # (200, 2)
    node_total = node_avg.sum(axis=1)  # (200,)
    sorted_idx = np.argsort(node_total)[::-1]

    ax.bar(range(NUM_NODES), node_total[sorted_idx], color='#9B59B6', alpha=0.7, width=1)
    ax.set_xlabel('Node Rank (by avg demand)')
    ax.set_ylabel('Avg total demand / 30min')
    ax.set_title('Node Demand Ranking (long-tail)', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注热点占比
    top20_share = node_total[sorted_idx[:40]].sum() / node_total.sum() * 100
    ax.annotate(f'Top-20% nodes carry\n{top20_share:.0f}% of demand',
                xy=(40, node_total[sorted_idx[40]]),
                xytext=(80, node_total[sorted_idx[0]] * 0.7),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '06_demand_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 06_demand_distribution.png")


# ============================================================
# 图8: 热点区域的时序变化（动态迁移）
# ============================================================
def plot_hotspot_dynamics(demand, save_dir):
    """展示不同时段哪些区域是热点 — 对应"挑战二: 出行热点动态迁移"."""
    hour_slots, _ = compute_time_indices(len(demand))

    # 选 6 个代表性时段
    sample_slots = [2, 14, 20, 28, 36, 42]  # 1am, 7am, 10am, 2pm, 6pm, 9pm
    slot_labels = [slot_to_time(s) for s in sample_slots]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for idx, (slot, label) in enumerate(zip(sample_slots, slot_labels)):
        mask = hour_slots == slot
        if mask.sum() == 0:
            continue
        avg = demand[mask, :, 0].mean(axis=0)  # pickup
        grid = avg.reshape(GRID_ROWS, GRID_COLS)

        row, col = divmod(idx, 3)
        ax = axes[row, col]
        im = ax.imshow(grid, cmap=CMAP_HEAT, aspect='auto', origin='lower',
                       extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Pickup Hotspot Migration Through the Day\n'
                 '(出行热点的动态迁移)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '07_hotspot_dynamics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 07_hotspot_dynamics.png")


# ============================================================
# 图9: 输入窗口结构可视化（解释35步的含义）
# ============================================================
def plot_input_structure(splits, save_dir):
    """可视化一个样本的35步输入，展示其非连续性（含3天前context）."""
    X = splits['train']['X']
    # 取一个需求较高的样本
    sample_sums = X.sum(axis=(1, 2, 3))
    sample_idx = np.argsort(sample_sums)[-len(sample_sums)//2]  # 取中位附近

    sample = X[sample_idx]  # (35, 200, 2)
    total_demand = sample.sum(axis=(1, 2))  # (35,) 每个时间步的总需求

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(35)

    # 根据 ST-SSL 格式，前面部分是 3 天前的 context，后面是近期的
    # 一般是: [3天前若干步] + [2天前若干步] + [1天前若干步] + [近期若干步]
    ax.bar(x, total_demand, color='#3498DB', alpha=0.8, edgecolor='white')

    ax.set_xlabel('Input Time Step Index', fontsize=11)
    ax.set_ylabel('Total Demand', fontsize=11)
    ax.set_title(f'Input Window Structure (35 steps, sample #{sample_idx})\n'
                 'Note: Steps are NOT consecutive — includes 3-day periodic context',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注跳跃
    ax.annotate('← 3-day periodic context',
                xy=(5, total_demand[5]), xytext=(8, total_demand.max() * 0.9),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
                color='red')
    ax.annotate('Recent history →',
                xy=(30, total_demand[30]), xytext=(22, total_demand.max() * 0.9),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='green'),
                color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '08_input_structure.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 08_input_structure.png")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(SAVE_DIR)
    print("=" * 70)
    print("  NYC Taxi 出行需求数据探索")
    print("=" * 70)

    # 加载数据
    print("\n[Loading data...]")
    splits, adj, od_hourly = load_all_data()

    # 重建需求时间序列
    demand = reconstruct_demand_series(splits)
    print(f"\n  Reconstructed demand: {demand.shape}")

    # 打印概览
    print_data_overview(splits, demand, od_hourly)

    # 生成图表
    print(f"\n[Generating figures to {SAVE_DIR}]")

    print("\n  [1/8] Spatial heatmap...")
    plot_spatial_heatmap(demand, SAVE_DIR)

    print("  [2/8] Temporal pattern...")
    plot_temporal_pattern(demand, SAVE_DIR)

    print("  [3/8] Peak comparison heatmap...")
    plot_peak_comparison_heatmap(demand, SAVE_DIR)

    print("  [4/8] Weekday vs weekend...")
    plot_weekday_vs_weekend(demand, SAVE_DIR)

    print("  [5/8] OD analysis...")
    plot_od_analysis(od_hourly, SAVE_DIR)

    print("  [6/8] Demand distribution...")
    plot_demand_distribution(demand, SAVE_DIR)

    print("  [7/8] Hotspot dynamics...")
    plot_hotspot_dynamics(demand, SAVE_DIR)

    print("  [8/8] Input structure...")
    plot_input_structure(splits, SAVE_DIR)

    # 总结
    print("\n" + "=" * 70)
    print(f"  Done! {8} figures saved to: {SAVE_DIR}")
    print("=" * 70)
    print("\n关键发现提示 (跑完看图后确认):")
    print("  1. 空间分布: 曼哈顿中城是否为最大热点？")
    print("  2. 时间模式: 早高峰(7-10) 和晚高峰(17-20) 是否显著？")
    print("  3. 高峰 vs 深夜: 需求量差多少倍？热点位置是否迁移？")
    print("  4. 工作日 vs 周末: 模式差异大吗？周末是否缺早高峰？")
    print("  5. 稀疏性: 多少比例的格子需求为0？长尾效应多严重？")
    print("  6. OD流量: 哪些OD对最热？不同时段OD模式差异大吗？")


if __name__ == '__main__':
    main()