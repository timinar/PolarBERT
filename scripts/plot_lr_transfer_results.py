#!/usr/bin/env python3
"""
Plot LR Transfer Verification Results from W&B.

Creates a plot with 4 lines showing val/loss vs LR for:
- Base model (seed 42)
- Base model (seed 123)
- Wide model (N=1024, L=2)
- Deep model (N=256, L=8)
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# W&B project
ENTITY = "polargeese"
PROJECT = "completep-verification"

# LR values used in the sweep
LR_VALUES = [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3]


def fetch_results():
    """Fetch results from W&B."""
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    results = {
        'base_s42': {},
        'base_s123': {},
        'wide': {},
        'deep': {}
    }

    for run in runs:
        name = run.name

        # Parse run name: lrt_{model_tag}_s{seed}_lr{lr_str}_{timestamp}
        if not name.startswith('lrt_'):
            continue

        parts = name.split('_')
        if len(parts) < 4:
            continue

        model_tag = parts[1]  # base, wide, or deep
        seed_part = parts[2]  # s42 or s123
        lr_part = parts[3]    # lr1em4, lr2em4, etc.

        # Extract seed
        if seed_part.startswith('s'):
            seed = int(seed_part[1:])
        else:
            continue

        # Extract LR from lr_part (format: lr1em4 -> 1e-4)
        if lr_part.startswith('lr'):
            lr_str = lr_part[2:]
            # Convert format: 1em4 -> 1e-4, 2em3 -> 2e-3
            lr_str = lr_str.replace('m', '-').replace('p', '+')
            # Handle cases like "1e-04" vs "1e-4"
            try:
                lr = float(lr_str)
            except ValueError:
                print(f"Could not parse LR from: {lr_part}")
                continue
        else:
            continue

        # Get final validation loss
        summary = run.summary
        if 'val/loss' in summary:
            val_loss = summary['val/loss']
        elif 'val_loss' in summary:
            val_loss = summary['val_loss']
        else:
            # Try to get from history
            try:
                history = run.history(keys=['val/loss'])
                if len(history) > 0:
                    val_loss = history['val/loss'].dropna().iloc[-1]
                else:
                    print(f"No val/loss found for {name}")
                    continue
            except:
                print(f"Could not get val/loss for {name}")
                continue

        # Store result
        if model_tag == 'base':
            if seed == 42:
                results['base_s42'][lr] = val_loss
            elif seed == 123:
                results['base_s123'][lr] = val_loss
        elif model_tag == 'wide':
            results['wide'][lr] = val_loss
        elif model_tag == 'deep':
            results['deep'][lr] = val_loss

        print(f"  {name}: LR={lr:.1e}, val/loss={val_loss:.4f}")

    return results


def plot_results(results):
    """Create the plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors and markers
    styles = {
        'base_s42': {'color': 'blue', 'marker': 'o', 'label': 'Base (N=256, L=2) seed=42'},
        'base_s123': {'color': 'cyan', 'marker': 's', 'label': 'Base (N=256, L=2) seed=123'},
        'wide': {'color': 'green', 'marker': '^', 'label': 'Wide (N=1024, L=2)'},
        'deep': {'color': 'red', 'marker': 'D', 'label': 'Deep (N=256, L=8)'}
    }

    for key, style in styles.items():
        data = results[key]
        if not data:
            print(f"No data for {key}")
            continue

        lrs = sorted(data.keys())
        losses = [data[lr] for lr in lrs]

        ax.plot(lrs, losses,
                color=style['color'],
                marker=style['marker'],
                linewidth=2,
                markersize=8,
                label=style['label'])

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('CompleteP LR Transfer Verification\n(10M events, 5 epochs)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks to match LR values
    ax.set_xticks(LR_VALUES)
    ax.set_xticklabels([f'{lr:.0e}' for lr in LR_VALUES], rotation=45)

    plt.tight_layout()

    # Save figure
    output_path = '/lustre/hpc/pheno/inar/PolarBERT/checkpoints/lr_transfer_10M/lr_transfer_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.show()


def main():
    print("Fetching results from W&B...")
    print(f"Project: {ENTITY}/{PROJECT}\n")

    results = fetch_results()

    print("\n" + "="*50)
    print("Summary:")
    for key, data in results.items():
        if data:
            best_lr = min(data, key=data.get)
            print(f"  {key}: {len(data)} runs, best LR={best_lr:.1e} (loss={data[best_lr]:.4f})")
        else:
            print(f"  {key}: No data")
    print("="*50)

    plot_results(results)


if __name__ == "__main__":
    main()
