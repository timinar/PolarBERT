#!/usr/bin/env python3
"""
Analyze results from the Pretraining Duration Study.

Fetches experiment results from W&B and generates:
1. CSV file with all results
2. Plot: val_loss vs pretraining_steps for each dataset size
3. Summary statistics

Usage:
    python scripts/analyze_pretraining_duration.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
RESULTS_DIR = PROJECT_DIR / "checkpoints" / "pretraining_duration_study"
WANDB_PROJECT = "PolarBERT-pretraining-duration-study"

# Checkpoint step mapping
CHECKPOINT_STEPS = {
    "scratch": 0,
    "step015868": 15868,
    "step063473": 63473,
    "step126947": 126947,
    "step190421": 190421,
    "step253895": 253895,
}

# Dataset size mapping to numeric
DATASET_SIZES = {
    "100k": 100000,
    "1M": 1000000,
    "10M": 10000000,
}


def fetch_wandb_results() -> pd.DataFrame:
    """Fetch all experiment results from W&B."""
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        print("wandb not installed. Please install with: pip install wandb")
        return pd.DataFrame()

    runs = api.runs(WANDB_PROJECT)

    data = []
    for run in runs:
        if run.state != "finished":
            continue

        name = run.name

        # Skip LR sweep runs for main analysis
        if name.startswith("lr_sweep_"):
            continue

        # Parse checkpoint and dataset size from name
        parts = name.split("_")
        if len(parts) < 2:
            continue

        checkpoint = parts[0]
        dataset_size = parts[1]

        if checkpoint not in CHECKPOINT_STEPS or dataset_size not in DATASET_SIZES:
            continue

        summary = run.summary
        config = run.config

        data.append({
            "run_name": name,
            "checkpoint": checkpoint,
            "pretraining_steps": CHECKPOINT_STEPS[checkpoint],
            "dataset_size": dataset_size,
            "train_events": DATASET_SIZES[dataset_size],
            "val_loss": summary.get("val/loss"),
            "best_val_loss": summary.get("best_val_loss", summary.get("val/loss")),
            "train_loss": summary.get("train/loss"),
            "epochs": summary.get("epoch"),
            "max_lr": config.get("max_lr") or config.get("training", {}).get("max_lr"),
            "run_id": run.id,
        })

    return pd.DataFrame(data)


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate analysis plots."""
    if df.empty:
        print("No data to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Val loss vs pretraining steps for each dataset size
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(DATASET_SIZES)))

    for (size_name, size_events), color in zip(DATASET_SIZES.items(), colors):
        subset = df[df["dataset_size"] == size_name].sort_values("pretraining_steps")
        if not subset.empty:
            ax.plot(
                subset["pretraining_steps"],
                subset["val_loss"],
                marker="o",
                label=size_name,
                color=color,
                linewidth=2,
                markersize=8
            )

    ax.set_xlabel("Pretraining Steps", fontsize=12)
    ax.set_ylabel("Validation Loss (Angular Distance)", fontsize=12)
    ax.set_title("Finetuning Performance vs Pretraining Duration", fontsize=14)
    ax.legend(title="Dataset Size", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add x-axis ticks at checkpoint positions
    ax.set_xticks(list(CHECKPOINT_STEPS.values()))
    ax.set_xticklabels([f"{s//1000}k" if s > 0 else "0" for s in CHECKPOINT_STEPS.values()], rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "pretraining_duration_results.png", dpi=150)
    plt.savefig(output_dir / "pretraining_duration_results.pdf")
    print(f"Saved plot to {output_dir / 'pretraining_duration_results.png'}")

    # Plot 2: Heatmap of val_loss
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create pivot table
    pivot = df.pivot_table(
        values="val_loss",
        index="dataset_size",
        columns="checkpoint",
        aggfunc="first"
    )

    # Reorder columns by pretraining steps
    checkpoint_order = sorted(CHECKPOINT_STEPS.keys(), key=lambda x: CHECKPOINT_STEPS[x])
    pivot = pivot.reindex(columns=[c for c in checkpoint_order if c in pivot.columns])

    # Reorder rows by dataset size
    size_order = sorted(DATASET_SIZES.keys(), key=lambda x: DATASET_SIZES[x])
    pivot = pivot.reindex(index=[s for s in size_order if s in pivot.index])

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Validation Loss")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Checkpoint (Pretraining Steps)")
    ax.set_ylabel("Finetuning Dataset Size")
    ax.set_title("Validation Loss Heatmap")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "pretraining_duration_heatmap.png", dpi=150)
    plt.savefig(output_dir / "pretraining_duration_heatmap.pdf")
    print(f"Saved heatmap to {output_dir / 'pretraining_duration_heatmap.png'}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    if df.empty:
        print("No data available")
        return

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Best result for each dataset size
    print("\nBest results by dataset size:")
    for size_name in DATASET_SIZES:
        subset = df[df["dataset_size"] == size_name]
        if not subset.empty:
            best_row = subset.loc[subset["val_loss"].idxmin()]
            print(f"  {size_name}: checkpoint={best_row['checkpoint']}, val_loss={best_row['val_loss']:.6f}")

    # Improvement from scratch to best pretrained
    print("\nImprovement from scratch:")
    for size_name in DATASET_SIZES:
        subset = df[df["dataset_size"] == size_name]
        if not subset.empty:
            scratch_loss = subset[subset["checkpoint"] == "scratch"]["val_loss"].values
            if len(scratch_loss) > 0:
                best_pretrained = subset[subset["checkpoint"] != "scratch"]["val_loss"].min()
                improvement = (scratch_loss[0] - best_pretrained) / scratch_loss[0] * 100
                print(f"  {size_name}: {improvement:.1f}% improvement")

    # Effect of pretraining (average across dataset sizes)
    print("\nAverage val_loss by checkpoint:")
    for ckpt in sorted(CHECKPOINT_STEPS.keys(), key=lambda x: CHECKPOINT_STEPS[x]):
        subset = df[df["checkpoint"] == ckpt]
        if not subset.empty:
            avg_loss = subset["val_loss"].mean()
            print(f"  {ckpt} (step {CHECKPOINT_STEPS[ckpt]}): {avg_loss:.6f}")


def main():
    """Main entry point."""
    print("="*60)
    print("Pretraining Duration Study - Analysis")
    print("="*60)

    # Fetch results from W&B
    print("\nFetching results from W&B...")
    df = fetch_wandb_results()

    if df.empty:
        print("No results found in W&B. Make sure experiments have completed.")
        print("You can also manually create results from the checkpoint directories.")
        return

    print(f"Found {len(df)} completed experiments")

    # Save to CSV
    csv_path = RESULTS_DIR / "analysis" / "pretraining_duration_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Print summary
    print_summary(df)

    # Generate plots
    print("\nGenerating plots...")
    plot_results(df, RESULTS_DIR / "analysis")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
