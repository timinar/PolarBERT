#!/usr/bin/env python3
"""
Activation Function Comparison Script

Runs multiple seeds for GELU vs ReLU to compare performance and variance.
"""

import subprocess
import yaml
import os
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
BASE_CONFIG = PROJECT_DIR / "configs" / "completep_verification" / "base_A_N256_L2.yaml"

# Experiment settings
ACTIVATIONS = ["gelu", "relu"]
NUM_SEEDS = 3  # Number of runs per activation
SEEDS = [42, 123, 456]  # Fixed seeds for reproducibility


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_experiment(activation: str, seed: int, run_idx: int):
    """Run a single experiment with given activation and seed."""
    config = load_config(BASE_CONFIG)

    # Set activation
    config['model']['activation'] = activation

    # Add seed to config for logging
    config['training']['seed'] = seed

    # Create run name
    timestamp = datetime.now().strftime('%H%M%S')
    run_name = f"activation_{activation}_seed{seed}_run{run_idx}_{timestamp}"
    config['model']['model_name'] = run_name

    # Save temporary config
    temp_dir = PROJECT_DIR / "checkpoints" / "activation_comparison" / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / f"{run_name}.yaml"
    save_config(config, temp_config_path)

    print(f"\n{'='*60}")
    print(f"Running: {activation.upper()} - Seed {seed} - Run {run_idx}")
    print(f"Config: {temp_config_path}")
    print(f"{'='*60}\n")

    # Set seed via environment variable and PL_GLOBAL_SEED
    env = os.environ.copy()
    env['PL_GLOBAL_SEED'] = str(seed)

    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", str(temp_config_path),
        "--name", run_name,
        "--model_type", "flash",
        "--dataset_type", "kaggle",
        "--checkpoint_path", "new",
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    return result.returncode == 0


def main():
    print("\n" + "="*70)
    print("ACTIVATION FUNCTION COMPARISON: GELU vs ReLU")
    print("="*70)
    print(f"\nRunning {NUM_SEEDS} seeds for each activation function")
    print(f"Activations: {ACTIVATIONS}")
    print(f"Seeds: {SEEDS[:NUM_SEEDS]}")
    print("="*70 + "\n")

    results = {act: [] for act in ACTIVATIONS}

    for activation in ACTIVATIONS:
        for i, seed in enumerate(SEEDS[:NUM_SEEDS]):
            success = run_experiment(activation, seed, i)
            results[activation].append((seed, success))

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nResults:")
    for activation, runs in results.items():
        print(f"\n{activation.upper()}:")
        for seed, success in runs:
            status = "OK" if success else "FAILED"
            print(f"  Seed {seed}: {status}")

    print("\n" + "="*70)
    print("Check W&B dashboard: completep-verification project")
    print("Compare runs with tags: activation_gelu_* vs activation_relu_*")
    print("="*70)


if __name__ == "__main__":
    main()
