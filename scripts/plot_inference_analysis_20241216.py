"""
Plotting script for IceCube inference analysis.
Creates histograms of pulse counts and angular loss distributions.

Usage:
    python scripts/plot_inference_analysis_20241216.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
INFERENCE_DIR = Path('/groups/pheno/inar/icecube_kaggle/v2/inference_results')
TRAIN_PREDICTIONS = INFERENCE_DIR / 'train_predictions.npz'
TRAIN_META = Path('/groups/pheno/inar/icecube_kaggle/train_meta.parquet')


def plot_pulse_count_comparison(n_events: int = 10_000_000):
    """
    Plot pulse count distributions as overlaid lines with log scale.
    Shows: all events, both-upgoing, both-upgoing with loss < 15°, < 10°.
    """
    print("Loading train predictions...")
    train_preds = np.load(TRAIN_PREDICTIONS)

    print("Loading train metadata...")
    meta = pd.read_parquet(TRAIN_META)
    meta['true_pulse_count'] = meta['last_pulse_index'] - meta['first_pulse_index'] + 1

    # Get data for first n_events
    event_indices = train_preds['event_indices'][:n_events]
    true_pulse_counts = meta['true_pulse_count'].values[event_indices]
    angular_loss = np.degrees(train_preds['angular_loss'][:n_events])
    pred_cos_zenith = train_preds['pred_cos_zenith'][:n_events]
    true_cos_zenith = train_preds['true_cos_zenith'][:n_events]

    # Define masks
    true_upgoing = true_cos_zenith < 0
    both_upgoing = (pred_cos_zenith < 0) & (true_cos_zenith < 0)
    both_upgoing_10 = both_upgoing & (angular_loss < 10)

    print(f"All events: {n_events:,}")
    print(f"True upgoing: {np.sum(true_upgoing):,}")
    print(f"Both upgoing: {np.sum(both_upgoing):,}")
    print(f"Both upgoing, loss < 10°: {np.sum(both_upgoing_10):,}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Histogram parameters - log-spaced bins from 1 to 1e5
    bins = np.logspace(0, 5, 101)  # 1 to 100,000
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean for log scale

    # Compute histograms (normalized to 1)
    hist_all, _ = np.histogram(true_pulse_counts, bins=bins, density=True)
    hist_true_upgoing, _ = np.histogram(true_pulse_counts[true_upgoing], bins=bins, density=True)
    hist_both_upgoing, _ = np.histogram(true_pulse_counts[both_upgoing], bins=bins, density=True)
    hist_upgoing_10, _ = np.histogram(true_pulse_counts[both_upgoing_10], bins=bins, density=True)

    # Plot as lines
    ax.plot(bin_centers, hist_all, label=f'All events (n={n_events:,})', linewidth=2, color='blue')
    ax.plot(bin_centers, hist_true_upgoing, label=f'True upgoing (n={np.sum(true_upgoing):,})', linewidth=2, color='purple')
    ax.plot(bin_centers, hist_both_upgoing, label=f'Both upgoing (n={np.sum(both_upgoing):,})', linewidth=2, color='green')
    ax.plot(bin_centers, hist_upgoing_10, label=f'Both upgoing, loss < 10° (n={np.sum(both_upgoing_10):,})', linewidth=2, color='red')

    ax.set_xlabel('Pulse Count', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Pulse Count Distribution ({n_events:,} events)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 1e5)
    ax.axvline(x=127, color='gray', linestyle='--', alpha=0.5, label='Model truncation (127)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = INFERENCE_DIR / 'pulse_count_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to {output_path}")


def plot_true_pulse_count_histogram(n_events: int = 10_000_000):
    """Plot true pulse count histogram from metadata."""
    print("Loading train predictions...")
    train_preds = np.load(TRAIN_PREDICTIONS)
    print(f"Total train events with predictions: {len(train_preds['event_indices']):,}")

    print("Loading train metadata...")
    meta = pd.read_parquet(TRAIN_META)
    print(f"Total events in metadata: {len(meta):,}")

    # Compute true pulse counts
    meta['true_pulse_count'] = meta['last_pulse_index'] - meta['first_pulse_index'] + 1

    # Use first n_events
    event_indices = train_preds['event_indices'][:n_events]
    true_pulse_counts = meta['true_pulse_count'].values[event_indices]
    angular_loss = np.degrees(train_preds['angular_loss'][:n_events])

    # Filter for both-upgoing events
    pred_cos_zenith = train_preds['pred_cos_zenith'][:n_events]
    true_cos_zenith = train_preds['true_cos_zenith'][:n_events]
    both_upgoing = (pred_cos_zenith < 0) & (true_cos_zenith < 0)

    print(f"\nUsing {n_events:,} events")
    print(f"Both upgoing: {np.sum(both_upgoing):,} ({100*np.mean(both_upgoing):.1f}%)")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. True pulse count distribution - All events
    ax1 = axes[0]
    ax1.hist(true_pulse_counts, bins=100, range=(0, 500), density=True, alpha=0.7, color='blue')
    ax1.set_xlabel('True Pulse Count')
    ax1.set_ylabel('Density')
    ax1.set_title(f'True Pulse Count Distribution - All Events (n={n_events:,})')
    ax1.axvline(x=127, color='red', linestyle='--', label='Model truncation (127)')
    ax1.axvline(x=np.median(true_pulse_counts), color='orange', linestyle='--',
                label=f'Median: {np.median(true_pulse_counts):.0f}')
    ax1.legend()

    # 2. True pulse count distribution - Both upgoing
    ax2 = axes[1]
    ax2.hist(true_pulse_counts[both_upgoing], bins=100, range=(0, 500), density=True, alpha=0.7, color='green')
    ax2.set_xlabel('True Pulse Count')
    ax2.set_ylabel('Density')
    ax2.set_title(f'True Pulse Count - Both Upgoing (n={np.sum(both_upgoing):,})')
    ax2.axvline(x=127, color='red', linestyle='--', label='Model truncation (127)')
    ax2.axvline(x=np.median(true_pulse_counts[both_upgoing]), color='orange', linestyle='--',
                label=f'Median: {np.median(true_pulse_counts[both_upgoing]):.0f}')
    ax2.legend()

    plt.tight_layout()
    output_path = INFERENCE_DIR / 'true_pulse_count_histogram.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to {output_path}")

    # Print statistics
    print(f"\n--- Pulse Count Statistics ({n_events:,} events) ---")
    print(f"All events - Mean: {np.mean(true_pulse_counts):.1f}, Median: {np.median(true_pulse_counts):.1f}")
    print(f"Both upgoing - Mean: {np.mean(true_pulse_counts[both_upgoing]):.1f}, Median: {np.median(true_pulse_counts[both_upgoing]):.1f}")
    print(f"Events with >127 pulses: {np.sum(true_pulse_counts > 127):,} ({100*np.mean(true_pulse_counts > 127):.1f}%)")
    print(f"Both upgoing with >127 pulses: {np.sum(true_pulse_counts[both_upgoing] > 127):,} ({100*np.mean(true_pulse_counts[both_upgoing] > 127):.1f}%)")

    return true_pulse_counts, both_upgoing


def plot_full_analysis(n_events: int = 10_000_000):
    """Plot full analysis: pulse counts, angular loss, and scatter."""
    print("Loading train predictions...")
    train_preds = np.load(TRAIN_PREDICTIONS)

    print("Loading train metadata...")
    meta = pd.read_parquet(TRAIN_META)
    meta['true_pulse_count'] = meta['last_pulse_index'] - meta['first_pulse_index'] + 1

    # Use first n_events
    event_indices = train_preds['event_indices'][:n_events]
    true_pulse_counts = meta['true_pulse_count'].values[event_indices]
    angular_loss = np.degrees(train_preds['angular_loss'][:n_events])

    # Filter for both-upgoing events
    pred_cos_zenith = train_preds['pred_cos_zenith'][:n_events]
    true_cos_zenith = train_preds['true_cos_zenith'][:n_events]
    both_upgoing = (pred_cos_zenith < 0) & (true_cos_zenith < 0)

    print(f"\nUsing {n_events:,} events")
    print(f"Both upgoing: {np.sum(both_upgoing):,} ({100*np.mean(both_upgoing):.1f}%)")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. True pulse count distribution - All events
    ax1 = axes[0, 0]
    ax1.hist(true_pulse_counts, bins=100, range=(0, 500), density=True, alpha=0.7, color='blue')
    ax1.set_xlabel('True Pulse Count')
    ax1.set_ylabel('Density')
    ax1.set_title(f'True Pulse Count Distribution - All Events (n={n_events:,})')
    ax1.axvline(x=127, color='red', linestyle='--', label='Model truncation (127)')
    ax1.legend()

    # 2. True pulse count distribution - Both upgoing
    ax2 = axes[0, 1]
    ax2.hist(true_pulse_counts[both_upgoing], bins=100, range=(0, 500), density=True, alpha=0.7, color='green')
    ax2.set_xlabel('True Pulse Count')
    ax2.set_ylabel('Density')
    ax2.set_title(f'True Pulse Count - Both Upgoing (n={np.sum(both_upgoing):,})')
    ax2.axvline(x=127, color='red', linestyle='--', label='Model truncation (127)')
    ax2.legend()

    # 3. Angular loss distribution - Both upgoing
    ax3 = axes[1, 0]
    ax3.hist(angular_loss[both_upgoing], bins=100, range=(0, 180), density=True, alpha=0.7, color='orange')
    ax3.set_xlabel('Angular Loss (degrees)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Angular Loss - Both Upgoing (n={np.sum(both_upgoing):,})')
    for thresh in [15, 20, 25, 30]:
        ax3.axvline(x=thresh, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=np.median(angular_loss[both_upgoing]), color='red', linestyle='--',
                label=f'Median: {np.median(angular_loss[both_upgoing]):.1f}°')
    ax3.legend()

    # 4. Pulse count vs angular loss scatter (subsample for visualization)
    ax4 = axes[1, 1]
    both_upgoing_idx = np.where(both_upgoing)[0]
    subsample_size = min(50000, len(both_upgoing_idx))
    subsample_idx = np.random.choice(both_upgoing_idx, subsample_size, replace=False)
    ax4.scatter(true_pulse_counts[subsample_idx], angular_loss[subsample_idx],
                alpha=0.1, s=1, c='purple')
    ax4.set_xlabel('True Pulse Count')
    ax4.set_ylabel('Angular Loss (degrees)')
    ax4.set_title(f'Pulse Count vs Angular Loss - Both Upgoing (subsample n={subsample_size:,})')
    ax4.set_xlim(0, 500)
    ax4.set_ylim(0, 180)

    plt.tight_layout()
    output_path = INFERENCE_DIR / 'train_10M_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to {output_path}")

    # Print statistics
    print(f"\n--- Statistics for {n_events:,} events ---")
    print(f"True pulse count - Mean: {np.mean(true_pulse_counts):.1f}, Median: {np.median(true_pulse_counts):.1f}")
    print(f"Events with >127 pulses: {np.sum(true_pulse_counts > 127):,} ({100*np.mean(true_pulse_counts > 127):.1f}%)")

    print(f"\n--- Both upgoing events ---")
    print(f"Angular loss - Mean: {np.mean(angular_loss[both_upgoing]):.2f}°, Median: {np.median(angular_loss[both_upgoing]):.2f}°")
    for thresh in [10, 15, 20, 25, 30]:
        passing = angular_loss[both_upgoing] < thresh
        print(f"Loss < {thresh}°: {np.sum(passing):,} events ({100*np.mean(passing):.1f}%)")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating pulse count comparison (log scale)...")
    print("=" * 60)
    plot_pulse_count_comparison(n_events=10_000_000)

    print("\n" + "=" * 60)
    print("Generating true pulse count histogram...")
    print("=" * 60)
    plot_true_pulse_count_histogram(n_events=10_000_000)

    print("\n" + "=" * 60)
    print("Generating full analysis plots...")
    print("=" * 60)
    plot_full_analysis(n_events=10_000_000)
