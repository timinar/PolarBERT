#!/usr/bin/env python3
"""
Pretraining Duration Study Pipeline

This script runs a systematic study of how pretraining duration affects finetuning performance.

Phase 1: LR Sweep - Find optimal learning rate using early checkpoint + 100k events
Phase 2: Main Experiments - Run all checkpoint x dataset size combinations

Usage:
    screen -S pretraining_study
    cd /lustre/hpc/pheno/inar/PolarBERT
    python scripts/run_pretraining_duration_study.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

# Configuration
CHECKPOINT_BASE = Path("/groups/pheno/inar/PolarBERT/src/polarbert/checkpoints/annealed_results")
CONFIG_DIR = PROJECT_DIR / "configs" / "pretraining_duration_study"
RESULTS_DIR = PROJECT_DIR / "checkpoints" / "pretraining_duration_study"

# Checkpoints to test (name -> path)
CHECKPOINTS = {
    "scratch": "new",
    "step015868": str(CHECKPOINT_BASE / "annealed_epoch=00-step=015868_1000steps" / "annealed_epoch=00-step=015868_1000steps" / "last.ckpt"),
    "step063473": str(CHECKPOINT_BASE / "annealed_epoch=01-step=063473_1000steps" / "annealed_epoch=01-step=063473_1000steps" / "last.ckpt"),
    "step126947": str(CHECKPOINT_BASE / "annealed_epoch=03-step=126947_1000steps" / "annealed_epoch=03-step=126947_1000steps" / "last.ckpt"),
    "step190421": str(CHECKPOINT_BASE / "annealed_epoch=05-step=190421_1000steps" / "annealed_epoch=05-step=190421_1000steps" / "last.ckpt"),
    "step253895": str(CHECKPOINT_BASE / "annealed_epoch=07-step=253895_1000steps" / "annealed_epoch=07-step=253895_1000steps" / "last.ckpt"),
}

# Dataset sizes and their configs
DATASET_CONFIGS = {
    "100k": str(CONFIG_DIR / "main_100k.yaml"),
    "1M": str(CONFIG_DIR / "main_1M.yaml"),
    "10M": str(CONFIG_DIR / "main_10M.yaml"),
}

# Learning rates to sweep
LR_VALUES = [3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

# Early checkpoint for LR sweep
SWEEP_CHECKPOINT = "step015868"
SWEEP_CONFIG = str(CONFIG_DIR / "sweep_100k.yaml")


def run_finetuning(config_path: str, checkpoint_path: str, name: str, max_lr: float = None) -> int:
    """Run a single finetuning experiment."""
    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", config_path,
        "--task", "direction",
        "--dataset_type", "kaggle",
        "--checkpoint_path", checkpoint_path,
        "--name", name,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR / "src") + ":" + env.get("PYTHONPATH", "")

    # If max_lr is specified, we need to override it in the config
    # This is done via wandb config override mechanism
    if max_lr is not None:
        # Set wandb config override via environment
        env["WANDB_CONFIG_OVERRIDE"] = json.dumps({"max_lr": max_lr})

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    if max_lr:
        print(f"LR: {max_lr}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    return result.returncode


def run_finetuning_with_lr_override(config_path: str, checkpoint_path: str, name: str, max_lr: float) -> int:
    """Run finetuning with learning rate override via temporary config modification."""
    import yaml

    # Read base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override max_lr
    config['training']['max_lr'] = max_lr

    # Write temporary config
    temp_config_path = RESULTS_DIR / f"temp_config_{name}.yaml"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", str(temp_config_path),
        "--task", "direction",
        "--dataset_type", "kaggle",
        "--checkpoint_path", checkpoint_path,
        "--name", name,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR / "src") + ":" + env.get("PYTHONPATH", "")

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config_path} (LR override: {max_lr})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)

    # Clean up temp config
    if temp_config_path.exists():
        temp_config_path.unlink()

    return result.returncode


def phase1_lr_sweep() -> float:
    """
    Phase 1: Learning Rate Sweep

    Run finetuning with different learning rates using the early checkpoint
    and 100k training events. Returns the best learning rate.
    """
    print("\n" + "="*80)
    print("PHASE 1: Learning Rate Sweep")
    print("="*80)

    results = {}
    checkpoint_path = CHECKPOINTS[SWEEP_CHECKPOINT]

    for lr in LR_VALUES:
        lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
        name = f"lr_sweep_{lr_str}"

        returncode = run_finetuning_with_lr_override(
            config_path=SWEEP_CONFIG,
            checkpoint_path=checkpoint_path,
            name=name,
            max_lr=lr
        )

        results[lr] = {"returncode": returncode, "name": name}

        if returncode != 0:
            print(f"WARNING: LR sweep run with lr={lr} failed with code {returncode}")

    # Save sweep results
    sweep_results_path = RESULTS_DIR / "phase1_sweep" / "sweep_results.json"
    sweep_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_results_path, 'w') as f:
        json.dump({"lr_values": LR_VALUES, "results": {str(k): v for k, v in results.items()}}, f, indent=2)

    print("\n" + "="*80)
    print("Phase 1 Complete!")
    print(f"Results saved to: {sweep_results_path}")
    print("="*80)

    # Return the middle value as default - user should check W&B for actual best
    return extract_best_lr()


def extract_best_lr() -> float:
    """
    Extract the best learning rate from W&B sweep results.
    Falls back to default if W&B is not available.
    """
    try:
        import wandb
        api = wandb.Api()

        # Find sweep runs
        runs = api.runs(
            "PolarBERT-pretraining-duration-study",
            filters={"display_name": {"$regex": "lr_sweep_.*"}}
        )

        best_lr = None
        best_val_loss = float('inf')

        for run in runs:
            if run.state == "finished":
                summary = run.summary
                if 'val/loss' in summary:
                    val_loss = summary['val/loss']
                    config_lr = run.config.get('max_lr') or run.config.get('training', {}).get('max_lr')
                    if val_loss < best_val_loss and config_lr is not None:
                        best_val_loss = val_loss
                        best_lr = config_lr

        if best_lr is not None:
            print(f"\nBest LR from W&B: {best_lr} (val/loss: {best_val_loss:.6f})")

            # Save to file
            best_lr_path = RESULTS_DIR / "phase1_sweep" / "best_lr.json"
            with open(best_lr_path, 'w') as f:
                json.dump({"best_lr": best_lr, "best_val_loss": best_val_loss}, f, indent=2)

            return best_lr
    except Exception as e:
        print(f"Could not extract best LR from W&B: {e}")

    # Check if we have a saved result
    best_lr_path = RESULTS_DIR / "phase1_sweep" / "best_lr.json"
    if best_lr_path.exists():
        with open(best_lr_path, 'r') as f:
            data = json.load(f)
            return data['best_lr']

    # Default fallback
    print("WARNING: Using default LR 1e-4. Please check W&B results and update best_lr.json")
    return 1e-4


def phase2_main_experiments(best_lr: float):
    """
    Phase 2: Main Experiments

    Run finetuning for all checkpoint x dataset size combinations
    using the optimal learning rate from Phase 1.
    """
    print("\n" + "="*80)
    print("PHASE 2: Main Experiments")
    print(f"Using optimal LR: {best_lr}")
    print("="*80)

    results = {}
    total_runs = len(CHECKPOINTS) * len(DATASET_CONFIGS)
    current_run = 0

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        for size_name, config_path in DATASET_CONFIGS.items():
            current_run += 1
            name = f"{ckpt_name}_{size_name}"

            print(f"\n[{current_run}/{total_runs}] Running: {name}")

            returncode = run_finetuning_with_lr_override(
                config_path=config_path,
                checkpoint_path=ckpt_path,
                name=name,
                max_lr=best_lr
            )

            results[name] = {
                "checkpoint": ckpt_name,
                "dataset_size": size_name,
                "returncode": returncode
            }

            if returncode != 0:
                print(f"WARNING: Run {name} failed with code {returncode}")

    # Save results
    main_results_path = RESULTS_DIR / "phase2_main" / "experiment_results.json"
    main_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(main_results_path, 'w') as f:
        json.dump({
            "best_lr": best_lr,
            "checkpoints": list(CHECKPOINTS.keys()),
            "dataset_sizes": list(DATASET_CONFIGS.keys()),
            "results": results
        }, f, indent=2)

    print("\n" + "="*80)
    print("Phase 2 Complete!")
    print(f"Results saved to: {main_results_path}")
    print("="*80)


def main():
    """Main entry point."""
    start_time = datetime.now()

    print("="*80)
    print("Pretraining Duration Study")
    print(f"Started: {start_time}")
    print("="*80)

    # Create results directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase1_sweep").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase2_main").mkdir(parents=True, exist_ok=True)

    # Check if we should skip Phase 1
    best_lr_path = RESULTS_DIR / "phase1_sweep" / "best_lr.json"
    if best_lr_path.exists():
        print("\nFound existing best_lr.json - skipping Phase 1")
        with open(best_lr_path, 'r') as f:
            best_lr = json.load(f)['best_lr']
        print(f"Using saved best LR: {best_lr}")
    else:
        # Phase 1: LR Sweep
        best_lr = phase1_lr_sweep()

    # Phase 2: Main Experiments
    phase2_main_experiments(best_lr)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("Study Complete!")
    print(f"Total duration: {duration}")
    print(f"Results directory: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
