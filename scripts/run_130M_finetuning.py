#!/usr/bin/env python3
"""
Run 130M finetuning experiments for the Pretraining Duration Study.

This runs 6 experiments: scratch + 5 pretrained checkpoints
All with max_lr=1e-3, 4 epochs, 130M events.

Usage:
    screen -S finetuning_130M
    cd /lustre/hpc/pheno/inar/PolarBERT
    python scripts/run_130M_finetuning.py
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
CHECKPOINT_BASE = Path("/groups/pheno/inar/PolarBERT/src/polarbert/checkpoints/annealed_results")
CONFIG_DIR = PROJECT_DIR / "configs" / "pretraining_duration_study"
RESULTS_DIR = PROJECT_DIR / "checkpoints" / "pretraining_duration_study"

# Checkpoints to test
CHECKPOINTS = {
    "scratch": "new",
    "step015868": str(CHECKPOINT_BASE / "annealed_epoch=00-step=015868_1000steps" / "annealed_epoch=00-step=015868_1000steps" / "last.ckpt"),
    "step063473": str(CHECKPOINT_BASE / "annealed_epoch=01-step=063473_1000steps" / "annealed_epoch=01-step=063473_1000steps" / "last.ckpt"),
    "step126947": str(CHECKPOINT_BASE / "annealed_epoch=03-step=126947_1000steps" / "annealed_epoch=03-step=126947_1000steps" / "last.ckpt"),
    "step190421": str(CHECKPOINT_BASE / "annealed_epoch=05-step=190421_1000steps" / "annealed_epoch=05-step=190421_1000steps" / "last.ckpt"),
    "step253895": str(CHECKPOINT_BASE / "annealed_epoch=07-step=253895_1000steps" / "annealed_epoch=07-step=253895_1000steps" / "last.ckpt"),
}

CONFIG_PATH = str(CONFIG_DIR / "main_130M.yaml")
MAX_LR = 1e-3


def run_finetuning(checkpoint_path: str, name: str) -> int:
    """Run a single finetuning experiment."""
    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", CONFIG_PATH,
        "--task", "direction",
        "--dataset_type", "kaggle",
        "--checkpoint_path", checkpoint_path,
        "--name", name,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR / "src") + ":" + env.get("PYTHONPATH", "")

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LR: {MAX_LR}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    return result.returncode


def main():
    start_time = datetime.now()

    print("="*60)
    print("130M Finetuning Experiments")
    print(f"Started: {start_time}")
    print(f"Config: {CONFIG_PATH}")
    print(f"LR: {MAX_LR}")
    print("="*60)

    results = {}
    total = len(CHECKPOINTS)

    for i, (ckpt_name, ckpt_path) in enumerate(CHECKPOINTS.items(), 1):
        name = f"{ckpt_name}_130M"
        print(f"\n[{i}/{total}] Running: {name}")

        returncode = run_finetuning(ckpt_path, name)
        results[name] = returncode

        if returncode != 0:
            print(f"WARNING: {name} failed with code {returncode}")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("130M Experiments Complete!")
    print(f"Duration: {duration}")
    print(f"Results: {results}")
    print("="*60)


if __name__ == "__main__":
    main()
