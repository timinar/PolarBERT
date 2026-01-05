#!/usr/bin/env python3
"""
CompleteP LR Transfer Verification (10M events, 5 epochs)

Tests whether optimal learning rate transfers across model sizes with CompleteP.
- Base model: multiple seeds for stability verification
- Wide/Deep models: single seed (slower training)
"""

import subprocess
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
CONFIG_DIR = PROJECT_DIR / "configs" / "completep_verification"

# LR grid - focusing on promising range from div factor sweep
LR_VALUES = [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3]

# Seeds
BASE_SEEDS = [42, 123]  # Multiple seeds for base model
SINGLE_SEED = 42        # Single seed for wide/deep

SHUFFLE_SEED = 1337


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_experiment(base_config_path: Path, lr: float, model_tag: str, seed: int):
    """Run a single experiment with given LR and seed."""
    config = load_config(base_config_path)

    # Set LR
    config['training']['max_lr'] = lr
    config['training']['completep']['lr_base'] = lr

    # Enable torch.compile for faster training with RMSNorm/QK Norm
    config['training']['torch_compile'] = True

    # Create run name
    timestamp = datetime.now().strftime('%H%M%S')
    lr_str = f"{lr:.0e}".replace('-', 'm').replace('+', 'p')
    run_name = f"lrt_{model_tag}_s{seed}_lr{lr_str}_{timestamp}"
    config['model']['model_name'] = run_name

    # Save temporary config
    temp_dir = PROJECT_DIR / "checkpoints" / "lr_transfer_10M" / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / f"{run_name}.yaml"
    save_config(config, temp_config_path)

    print(f"\n{'='*60}")
    print(f"Running: {model_tag} - seed={seed} - LR={lr:.1e}")
    print(f"Config: {temp_config_path}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env['PL_GLOBAL_SEED'] = str(seed)

    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", str(temp_config_path),
        "--name", run_name,
        "--model_type", "flash",
        "--dataset_type", "kaggle",
        "--checkpoint_path", "new",
        "--seed", str(seed),
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    return result.returncode == 0


def run_base_model():
    """Run base model with multiple seeds for stability check."""
    print("\n" + "#"*70)
    print("# BASE MODEL (N=256, L=2) - Multiple Seeds")
    print("#"*70)

    config_path = CONFIG_DIR / "base_A_N256_L2.yaml"
    results = []

    for seed in BASE_SEEDS:
        print(f"\n--- Seed {seed} ---")
        for lr in LR_VALUES:
            success = run_experiment(config_path, lr, "base", seed)
            results.append(('base', seed, lr, success))

    return results


def run_wide_model():
    """Run wide model with single seed."""
    print("\n" + "#"*70)
    print("# WIDE MODEL (N=1024, L=2) - Single Seed")
    print("#"*70)

    config_path = CONFIG_DIR / "wide_B_N1024_L2.yaml"
    results = []

    for lr in LR_VALUES:
        success = run_experiment(config_path, lr, "wide", SINGLE_SEED)
        results.append(('wide', SINGLE_SEED, lr, success))

    return results


def run_deep_model():
    """Run deep model with single seed."""
    print("\n" + "#"*70)
    print("# DEEP MODEL (N=256, L=8) - Single Seed")
    print("#"*70)

    config_path = CONFIG_DIR / "deep_C_N256_L8.yaml"
    results = []

    for lr in LR_VALUES:
        success = run_experiment(config_path, lr, "deep", SINGLE_SEED)
        results.append(('deep', SINGLE_SEED, lr, success))

    return results


def main():
    print("\n" + "="*70)
    print("COMPLETEP LR TRANSFER VERIFICATION")
    print("10M events, 5 epochs, final_div_factor=1")
    print("="*70)
    print(f"\nLR values: {[f'{lr:.1e}' for lr in LR_VALUES]}")
    print(f"Base model seeds: {BASE_SEEDS}")
    print(f"Wide/Deep seed: {SINGLE_SEED}")
    print(f"Total runs: {len(LR_VALUES) * len(BASE_SEEDS) + 2 * len(LR_VALUES)}")
    print("="*70)

    all_results = []

    # Run all models
    all_results.extend(run_base_model())
    all_results.extend(run_wide_model())
    all_results.extend(run_deep_model())

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults summary:")
    for model, seed, lr, success in all_results:
        status = "OK" if success else "FAILED"
        print(f"  {model} s{seed} LR={lr:.1e}: {status}")

    print("\nCheck W&B: lrt_base_*, lrt_wide_*, lrt_deep_*")
    print("="*70)


if __name__ == "__main__":
    main()
