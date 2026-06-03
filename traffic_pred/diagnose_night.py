"""
深夜误差诊断 — 找出 Late Night MAE=51 的真正原因

分析维度:
  1. 深夜 vs 白天的实际需求分布 (是不是接近零?)
  2. 模型是高估还是低估? (signed error)
  3. 88 个深夜样本逐个的 MAE (有没有离群点/节日?)
  4. 哪些节点贡献了最大误差? (是热点节点还是冷区?)
  5. 深夜热点节点的 pred vs true 散点图

Usage:
    python diagnose_night.py
    python diagnose_night.py --config configs/nyctaxi_cluster.yaml
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.utils import load_config, get_project_path, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/nyctaxi_cluster.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    fig_dir = get_project_path('figures', 'night_diagnosis')
    ensure_dir(fig_dir)

    # Load predictions
    pred_path = get_project_path('figures', 'test_predictions.npz')
    if not os.path.exists(pred_path):
        print("ERROR: Run evaluate.py first")
        return

    data = np.load(pred_path)
    preds = data['predictions']   # (546, 1, 200, 2)
    trues = data['ground_truth']  # (546, 1, 200, 2)

    # Reconstruct approximate time indices for test set
    n_train, n_val = 1912, 274
    offset = 144 + n_train + n_val
    n_test = len(preds)
    test_hours = (np.arange(n_test) + offset) % 48      # 0-47
    test_days = ((np.arange(n_test) + offset) // 48)     # absolute day index
    test_dow = (test_days + 3) % 7                        # 0=Mon, 6=Sun

    # Define periods
    night_mask = test_hours < 8                           # slots 0-7 = 0:00-4:00
    morning_mask = (test_hours >= 14) & (test_hours < 20) # 7:00-10:00
    evening_mask = (test_hours >= 34) & (test_hours < 40) # 17:00-20:00
    day_mask = (test_hours >= 8) & (test_hours < 40)      # 4:00-20:00

    # Squeeze to (samples, N, C)
    p = preds[:, 0, :, :]   # (546, 200, 2)
    t = trues[:, 0, :, :]

    print("=" * 70)
    print("  深夜误差诊断")
    print("=" * 70)

    # ================================================================
    # 1. 深夜 vs 白天的实际需求分布
    # ================================================================
    night_true = t[night_mask]  # (88, 200, 2)
    day_true = t[day_mask]

    print(f"\n[1] 实际需求统计:")
    print(f"  深夜 (0-4am): mean={night_true.mean():.2f}, "
          f"median={np.median(night_true):.2f}, "
          f"max={night_true.max():.1f}, "
          f"std={night_true.std():.2f}")
    print(f"  白天 (4-8pm): mean={day_true.mean():.2f}, "
          f"median={np.median(day_true):.2f}, "
          f"max={day_true.max():.1f}, "
          f"std={day_true.std():.2f}")
    print(f"  深夜均值 / 白天均值 = {night_true.mean() / day_true.mean():.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Night demand distribution (only non-zero values)
    night_flat = night_true.flatten()
    day_flat = day_true.flatten()

    axes[0].hist(night_flat[night_flat > 0], bins=50, alpha=0.7,
                 color='#2C3E50', label=f'Night (n={len(night_true)})')
    axes[0].hist(day_flat[day_flat > 0], bins=50, alpha=0.5,
                 color='#E67E22', label=f'Day (n={len(day_true)})')
    axes[0].set_xlabel('Demand value (non-zero)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Demand distribution: Night vs Day')
    axes[0].legend()
    axes[0].set_xlim(0, 200)

    # Night: per-node mean demand
    night_node_mean = night_true.mean(axis=0).sum(axis=-1)  # (200,)
    day_node_mean = day_true.mean(axis=0).sum(axis=-1)
    axes[1].scatter(day_node_mean, night_node_mean, s=15, alpha=0.6,
                    color='#2C3E50')
    axes[1].plot([0, day_node_mean.max()], [0, day_node_mean.max()],
                 'r--', alpha=0.5, label='y=x (no change)')
    axes[1].set_xlabel('Node mean demand (Day)')
    axes[1].set_ylabel('Node mean demand (Night)')
    axes[1].set_title('Per-node demand: Day vs Night')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '1_demand_distribution.png'))
    plt.close()

    # ================================================================
    # 2. 模型是高估还是低估?
    # ================================================================
    night_pred = p[night_mask]
    night_error = night_pred - night_true  # signed: positive = 高估

    print(f"\n[2] 深夜误差方向:")
    print(f"  Signed error mean = {night_error.mean():.2f} "
          f"(正=高估, 负=低估)")
    print(f"  高估比例: {(night_error > 0).mean():.1%}")
    print(f"  |高估|均值: {night_error[night_error > 0].mean():.2f}")
    print(f"  |低估|均值: {abs(night_error[night_error < 0].mean()):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Signed error histogram
    err_flat = night_error.flatten()
    axes[0].hist(err_flat[err_flat != 0], bins=100, color='#E74C3C', alpha=0.7)
    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].axvline(err_flat.mean(), color='blue', linewidth=2,
                    label=f'Mean={err_flat.mean():.1f}')
    axes[0].set_xlabel('Signed error (pred - true)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Night: Error direction (>0 = over-predict)')
    axes[0].legend()
    axes[0].set_xlim(-100, 200)

    # Scatter: pred vs true at night (sample of points)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(night_true.flatten()), size=5000, replace=False)
    axes[1].scatter(night_true.flatten()[idx], night_pred.flatten()[idx],
                    s=3, alpha=0.3, color='#2C3E50')
    lim = max(night_true.flatten()[idx].max(), night_pred.flatten()[idx].max())
    axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5)
    axes[1].set_xlabel('Actual demand')
    axes[1].set_ylabel('Predicted demand')
    axes[1].set_title('Night: Pred vs Actual scatter')
    axes[1].set_xlim(-10, 200)
    axes[1].set_ylim(-10, 200)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '2_error_direction.png'))
    plt.close()

    # ================================================================
    # 3. 逐样本 MAE — 找离群深夜
    # ================================================================
    night_sample_mae = np.abs(night_error).mean(axis=(1, 2))  # (88,)
    night_indices = np.where(night_mask)[0]

    print(f"\n[3] 深夜逐样本 MAE:")
    print(f"  Min={night_sample_mae.min():.2f}, "
          f"Median={np.median(night_sample_mae):.2f}, "
          f"Max={night_sample_mae.max():.2f}")

    # Top-10 worst nights
    worst_order = np.argsort(night_sample_mae)[::-1]
    print(f"\n  Top-10 最差深夜样本:")
    print(f"  {'Rank':>4s} {'TestIdx':>8s} {'MAE':>8s} {'Hour':>6s} "
          f"{'DayIdx':>7s} {'DoW':>5s} {'TrueMean':>10s}")
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for rank, wi in enumerate(worst_order[:10]):
        tidx = night_indices[wi]
        hr = test_hours[tidx]
        day = test_days[tidx]
        dow = test_dow[tidx]
        true_mean = t[tidx].mean()
        print(f"  {rank+1:4d} {tidx:8d} {night_sample_mae[wi]:8.2f} "
              f"{hr/2:5.1f}h {day:7d} {dow_names[dow]:>5s} "
              f"{true_mean:10.2f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(night_sample_mae)), night_sample_mae[np.argsort(night_indices)],
           color='#2C3E50', alpha=0.8)
    ax.axhline(np.median(night_sample_mae), color='red', linewidth=1,
               linestyle='--', label=f'Median={np.median(night_sample_mae):.1f}')
    ax.set_xlabel('Night sample index (chronological)')
    ax.set_ylabel('MAE')
    ax.set_title('Per-sample MAE for all 88 late-night test samples')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '3_per_sample_mae.png'))
    plt.close()

    # ================================================================
    # 4. 哪些节点贡献了最大误差?
    # ================================================================
    night_node_mae = np.abs(night_error).mean(axis=(0, 2))  # (200,)
    node_demand = night_true.mean(axis=(0, 2))              # (200,)

    # Top-10 worst nodes at night
    worst_nodes = np.argsort(night_node_mae)[::-1][:10]
    print(f"\n[4] 深夜 Top-10 高误差节点:")
    print(f"  {'Node':>6s} {'NightMAE':>10s} {'NightDemand':>13s} "
          f"{'NightPred':>11s} {'DayDemand':>11s}")
    for nid in worst_nodes:
        nd = night_true[:, nid, :].mean()
        np_ = night_pred[:, nid, :].mean()
        dd = day_true[:, nid, :].mean()
        nm = night_node_mae[nid]
        print(f"  {nid:6d} {nm:10.2f} {nd:13.2f} {np_:11.2f} {dd:11.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Node MAE vs node demand (night)
    axes[0].scatter(node_demand, night_node_mae, s=15, alpha=0.6,
                    color='#2C3E50')
    for nid in worst_nodes[:5]:
        axes[0].annotate(f'N{nid}', (node_demand[nid], night_node_mae[nid]),
                         fontsize=8, color='red')
    axes[0].set_xlabel('Node mean demand (Night)')
    axes[0].set_ylabel('Node MAE (Night)')
    axes[0].set_title('Night error vs Night demand per node')

    # Compare: node demand ratio (night/day) vs error
    ratio = np.zeros(200)
    for n in range(200):
        d = day_true[:, n, :].mean()
        ratio[n] = night_true[:, n, :].mean() / d if d > 1 else 0
    axes[1].scatter(ratio, night_node_mae, s=15, alpha=0.6, color='#E67E22')
    axes[1].set_xlabel('Demand ratio (Night / Day)')
    axes[1].set_ylabel('Node MAE (Night)')
    axes[1].set_title('Night error vs demand drop ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '4_node_error_analysis.png'))
    plt.close()

    # ================================================================
    # 5. 热点节点深夜的详细对比
    # ================================================================
    hot_node = worst_nodes[0]  # node with worst night error
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full test series for this node
    axes[0].plot(t[:, hot_node, 0], label='Actual', color='#185FA5',
                 linewidth=1)
    axes[0].plot(p[:, hot_node, 0], label='Predicted', color='#D85A30',
                 linewidth=1, alpha=0.8)
    # Shade night regions
    for i in range(n_test):
        if night_mask[i]:
            axes[0].axvspan(i - 0.5, i + 0.5, alpha=0.1, color='navy')
    axes[0].set_title(f'Node {hot_node}: Full test series '
                      f'(shaded = late night)')
    axes[0].set_ylabel('Pickup demand')
    axes[0].legend()

    # Night-only: zoomed
    night_t = t[night_mask, hot_node, 0]
    night_p = p[night_mask, hot_node, 0]
    axes[1].bar(range(len(night_t)), night_t, color='#185FA5', alpha=0.6,
                label='Actual', width=0.4)
    axes[1].bar([x + 0.4 for x in range(len(night_t))], night_p,
                color='#D85A30', alpha=0.6, label='Predicted', width=0.4)
    axes[1].set_title(f'Node {hot_node}: Night-only samples '
                      f'(actual vs predicted)')
    axes[1].set_xlabel('Night sample index')
    axes[1].set_ylabel('Pickup demand')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '5_hotnode_night_detail.png'))
    plt.close()

    # ================================================================
    # 6. 总结
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  诊断总结")
    print(f"{'=' * 70}")

    over_ratio = (night_error > 0).mean()
    mean_signed = night_error.mean()

    if over_ratio > 0.6 and mean_signed > 10:
        print(f"  → 模型在深夜严重【高估】需求 "
              f"({over_ratio:.0%} 高估, 平均偏差 +{mean_signed:.1f})")
        print(f"  → 原因: 热点节点白天需求 ~{day_true[:, worst_nodes[0], :].mean():.0f}, "
              f"深夜降到 ~{night_true[:, worst_nodes[0], :].mean():.0f}")
        print(f"    模型没有学会这个急剧下降")
    elif over_ratio < 0.4:
        print(f"  → 模型在深夜【低估】需求")
    else:
        print(f"  → 误差方向混合 (高估 {over_ratio:.0%})")

    # Check if outlier samples dominate
    q75 = np.percentile(night_sample_mae, 75)
    q25 = np.percentile(night_sample_mae, 25)
    outlier_threshold = q75 + 1.5 * (q75 - q25)
    n_outliers = (night_sample_mae > outlier_threshold).sum()
    if n_outliers > 0:
        outlier_total_err = night_sample_mae[night_sample_mae > outlier_threshold].sum()
        all_err = night_sample_mae.sum()
        print(f"  → 有 {n_outliers} 个离群深夜样本, "
              f"贡献了 {outlier_total_err/all_err:.0%} 的总误差")
        print(f"    (可能是节日/特殊事件夜晚)")
    else:
        print(f"  → 没有明显离群样本, 误差是系统性的")

    # Top contributor nodes
    top5_err = night_node_mae[worst_nodes[:5]].sum()
    total_err = night_node_mae.sum()
    print(f"  → Top-5 节点贡献了 {top5_err/total_err:.0%} 的深夜总误差")
    print(f"    这些节点白天均需求: "
          f"{[f'{day_true[:, n, :].mean():.0f}' for n in worst_nodes[:5]]}")
    print(f"    深夜均需求: "
          f"{[f'{night_true[:, n, :].mean():.0f}' for n in worst_nodes[:5]]}")

    print(f"\n  Figures saved to: {fig_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()