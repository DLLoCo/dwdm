"""
Visualization script for traffic demand prediction.

Generates presentation-ready figures:
  1. Prediction vs ground truth heatmaps
  2. Baseline comparison bar chart
  3. Training loss curve
  4. Time series comparison (selected nodes)
  5. Per-volume error analysis
  6. Spatial error heatmap

Usage:
    python visualize.py
    python visualize.py --save_dir figures/
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lib.utils import load_config, get_project_path, ensure_dir

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nyctaxi_memory.yaml')
    p.add_argument('--save_dir', default='figures/')
    return p.parse_args()


# ============================================================
# 1. Prediction vs Ground Truth Heatmap
# ============================================================

def plot_heatmaps(preds, trues, save_dir, grid_rows=10, grid_cols=20,
                  sample_indices=None):
    """
    Show spatial heatmaps: ground truth vs prediction vs error.
    Picks a few representative samples.
    """
    if sample_indices is None:
        n = len(preds)
        sample_indices = [0, n // 4, n // 2, 3 * n // 4]

    for idx in sample_indices:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        for c, name in enumerate(['Pickup (inflow)', 'Dropoff (outflow)']):
            true_map = trues[idx, 0, :, c].reshape(grid_rows, grid_cols)
            pred_map = preds[idx, 0, :, c].reshape(grid_rows, grid_cols)
            err_map = np.abs(pred_map - true_map)

            vmax = max(true_map.max(), pred_map.max())

            ax1 = axes[c, 0]
            im1 = ax1.imshow(true_map, cmap='YlOrRd', vmin=0, vmax=vmax)
            ax1.set_title(f'{name} - Ground truth')
            plt.colorbar(im1, ax=ax1, shrink=0.8)

            ax2 = axes[c, 1]
            im2 = ax2.imshow(pred_map, cmap='YlOrRd', vmin=0, vmax=vmax)
            ax2.set_title(f'{name} - Prediction')
            plt.colorbar(im2, ax=ax2, shrink=0.8)

            ax3 = axes[c, 2]
            im3 = ax3.imshow(err_map, cmap='Reds', vmin=0)
            ax3.set_title(f'{name} - |Error|')
            plt.colorbar(im3, ax=ax3, shrink=0.8)

        for ax in axes.flat:
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

        fig.suptitle(f'Sample {idx}: Spatial prediction heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'heatmap_sample_{idx}.png'))
        plt.close()
        print(f"  Saved heatmap_sample_{idx}.png")


# ============================================================
# 2. Baseline Comparison Bar Chart
# ============================================================

def plot_baseline_comparison(save_dir):
    """Bar chart comparing all models on MAE, RMSE, MAPE."""
    # Hardcoded results — update with your actual numbers
    models = {
        'HA':          {'MAE': 36.14, 'RMSE': 77.99, 'MAPE': 96.88},
        'STGCN':       {'MAE': 30.98, 'RMSE': 70.14, 'MAPE': 109.73},
        'CCRNN':       {'MAE': 26.29, 'RMSE': 61.64, 'MAPE': 100.26},
        'TCN':         {'MAE': 25.05, 'RMSE': 58.81, 'MAPE': 93.35},
        'Transformer': {'MAE': 23.84, 'RMSE': 58.72, 'MAPE': 96.96},
        'MLP':         {'MAE': 23.64, 'RMSE': 57.77, 'MAPE': 96.93},
        'LSTM':        {'MAE': 22.55, 'RMSE': 54.40, 'MAPE': 90.20},
        'Ours':        {'MAE': 17.14, 'RMSE': 37.62, 'MAPE': 66.86},
        'Last-value':  {'MAE': 11.44, 'RMSE': 24.83, 'MAPE': 26.68},
    }

    names = list(models.keys())
    x = np.arange(len(names))

    # Highlight our model
    colors = ['#B4B2A9'] * len(names)
    ours_idx = names.index('Ours')
    colors[ours_idx] = '#534AB7'

    for metric in ['MAE', 'RMSE', 'MAPE']:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [models[n][metric] for n in names]
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel(metric if metric != 'MAPE' else 'MAPE (%)')
        ax.set_title(f'Model Comparison — {metric}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{metric}.png'))
        plt.close()
        print(f"  Saved comparison_{metric}.png")


# ============================================================
# 3. Training Loss Curve
# ============================================================

def plot_training_curve(save_dir):
    """Plot training and validation loss curves."""
    results_path = get_project_path('checkpoints/v7_memory', 'train_history.json')
    if not os.path.exists(results_path):
        print("  [SKIP] No results.json found")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    history = results
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history['train_loss'], label='Train loss', color='#534AB7')
    ax1.plot(epochs, history['val_loss'], label='Val loss', color='#D85A30')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and validation loss')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Metrics curve
    ax2.plot(epochs, history['val_MAE'], label='MAE', color='#534AB7')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, history['val_MAPE'], label='MAPE', color='#D85A30',
                  linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2_twin.set_ylabel('MAPE (%)')
    ax2.set_title('Validation metrics')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.close()
    print("  Saved training_curve.png")


# ============================================================
# 4. Time Series: Prediction vs Actual
# ============================================================

def plot_time_series(preds, trues, save_dir, node_indices=None, max_steps=100):
    """
    Plot predicted vs actual time series for selected nodes.
    Shows consecutive test samples as a continuous series.
    """
    # Flatten: (samples, 1, N, C) -> (samples, N, C)
    p = preds[:max_steps, 0, :, :]  # (steps, N, C)
    t = trues[:max_steps, 0, :, :]

    if node_indices is None:
        # Pick nodes with varied demand levels
        mean_demand = t.mean(axis=0).sum(axis=-1)  # (N,)
        sorted_idx = np.argsort(mean_demand)
        node_indices = [
            sorted_idx[len(sorted_idx) // 5],       # low demand
            sorted_idx[len(sorted_idx) // 2],        # medium demand
            sorted_idx[4 * len(sorted_idx) // 5],    # high demand
        ]

    fig, axes = plt.subplots(len(node_indices), 1, figsize=(14, 4 * len(node_indices)))
    if len(node_indices) == 1:
        axes = [axes]

    steps = np.arange(len(p))

    for ax, nid in zip(axes, node_indices):
        # Plot pickup (channel 0)
        ax.plot(steps, t[:, nid, 0], label='Actual (pickup)',
                color='#185FA5', linewidth=1.5)
        ax.plot(steps, p[:, nid, 0], label='Predicted (pickup)',
                color='#D85A30', linewidth=1.2, linestyle='--')

        demand_level = t[:, nid, 0].mean()
        ax.set_title(f'Node {nid} (avg demand: {demand_level:.1f})')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Demand')
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Prediction vs actual — selected nodes', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series.png'))
    plt.close()
    print("  Saved time_series.png")


# ============================================================
# 5. Error by Demand Volume Level
# ============================================================

def plot_volume_error(preds, trues, save_dir):
    """Bar chart: MAE grouped by demand volume level."""
    flat_p = preds.flatten()
    flat_t = trues.flatten()

    bins = [0, 5, 10, 20, 50, 100, 200, float('inf')]
    labels_list = ['0-5', '5-10', '10-20', '20-50', '50-100', '100-200', '200+']
    maes = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (flat_t >= bins[i]) & (flat_t < bins[i + 1])
        if mask.sum() > 0:
            maes.append(np.mean(np.abs(flat_p[mask] - flat_t[mask])))
            counts.append(mask.sum())
        else:
            maes.append(0)
            counts.append(0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(labels_list))
    bars = ax1.bar(x, maes, color='#534AB7', alpha=0.8, label='MAE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_list)
    ax1.set_xlabel('Demand volume range')
    ax1.set_ylabel('MAE')
    ax1.set_title('Prediction error by demand volume level')

    # Add sample count on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, counts, color='#D85A30', marker='o', linewidth=1.5,
             label='Sample count')
    ax2.set_ylabel('Sample count')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_by_volume.png'))
    plt.close()
    print("  Saved error_by_volume.png")


# ============================================================
# 6. Spatial Error Heatmap (average over all test samples)
# ============================================================

def plot_spatial_error(preds, trues, save_dir, grid_rows=10, grid_cols=20):
    """Average absolute error per region, shown as a heatmap."""
    # Average error per node across all samples and channels
    err = np.abs(preds - trues).mean(axis=(0, 1, 3))  # (N,)
    err_map = err.reshape(grid_rows, grid_cols)

    # Also show average demand for reference
    demand = trues.mean(axis=(0, 1, 3)).reshape(grid_rows, grid_cols)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(demand, cmap='YlOrRd')
    ax1.set_title('Average demand per region')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(err_map, cmap='Reds')
    ax2.set_title('Average prediction error (MAE)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

    fig.suptitle('Spatial distribution of demand and prediction error', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_error.png'))
    plt.close()
    print("  Saved spatial_error.png")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    save_dir = get_project_path(args.save_dir)
    ensure_dir(save_dir)

    print("=" * 60)
    print("Generating visualizations")
    print("=" * 60)
    print(f"Output: {save_dir}\n")

    # Load predictions
    pred_path = get_project_path('figures', 'test_predictions.npz')
    if not os.path.exists(pred_path):
        print("[ERROR] No predictions found. Run evaluate.py first:")
        print("  python evaluate.py")
        return

    data = np.load(pred_path)
    preds = data['predictions']
    trues = data['ground_truth']
    print(f"Loaded predictions: {preds.shape}")

    grid_rows = config['data']['grid_rows']
    grid_cols = config['data']['grid_cols']

    # 1. Heatmaps
    print("\n[1/6] Prediction heatmaps...")
    plot_heatmaps(preds, trues, save_dir, grid_rows, grid_cols)

    # 2. Baseline comparison
    print("\n[2/6] Baseline comparison chart...")
    plot_baseline_comparison(save_dir)

    # 3. Training curve
    print("\n[3/6] Training loss curve...")
    plot_training_curve(save_dir)

    # 4. Time series
    print("\n[4/6] Time series comparison...")
    plot_time_series(preds, trues, save_dir)

    # 5. Error by volume
    print("\n[5/6] Error by demand volume...")
    plot_volume_error(preds, trues, save_dir)

    # 6. Spatial error
    print("\n[6/6] Spatial error heatmap...")
    plot_spatial_error(preds, trues, save_dir, grid_rows, grid_cols)

    print("\n" + "=" * 60)
    print(f"All figures saved to {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
