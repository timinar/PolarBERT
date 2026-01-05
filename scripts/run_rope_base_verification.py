#!/usr/bin/env python3
"""
RoPE Verification - Base Model Only

Tests RoPE with the base model configuration from CompleteP verification.
"""

import subprocess
import yaml
import os
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
CONFIG_DIR = PROJECT_DIR / "configs" / "completep_verification"

# LR grid - same as original verification
LR_VALUES = [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3]

# Single seed for initial RoPE test
SEED = 42


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_experiment(base_config_path: Path, lr: float, seed: int):
    """Run a single experiment with given LR and seed."""
    config = load_config(base_config_path)

    # Set LR
    config['training']['max_lr'] = lr
    config['training']['completep']['lr_base'] = lr

    # Enable torch.compile for faster training
    config['training']['torch_compile'] = True

    # Enable RoPE
    config['model']['use_rope'] = True
    config['model']['rope_theta'] = 10000.0
    config['model']['rope_max_seq_len'] = 512

    # Update project name
    config['training']['project'] = 'rope-verification'

    # Create run name
    timestamp = datetime.now().strftime('%H%M%S')
    lr_str = f"{lr:.0e}".replace('-', 'm').replace('+', 'p')
    run_name = f"rope_base_s{seed}_lr{lr_str}_{timestamp}"
    config['model']['model_name'] = run_name

    # Save temporary config
    temp_dir = PROJECT_DIR / "checkpoints" / "rope_verification" / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / f"{run_name}.yaml"
    save_config(config, temp_config_path)

    print(f"\n{'='*60}")
    print(f"Running: RoPE Base - seed={seed} - LR={lr:.1e}")
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


def main():
    print("\n" + "="*70)
    print("ROPE VERIFICATION - BASE MODEL")
    print("10M events, 5 epochs, RoPE enabled")
    print("="*70)
    print(f"\nLR values: {[f'{lr:.1e}' for lr in LR_VALUES]}")
    print(f"Seed: {SEED}")
    print(f"Total runs: {len(LR_VALUES)}")
    print("="*70)

    config_path = CONFIG_DIR / "base_A_N256_L2.yaml"
    results = []

    for lr in LR_VALUES:
        success = run_experiment(config_path, lr, SEED)
        results.append((lr, success))

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults summary:")
    for lr, success in results:
        status = "OK" if success else "FAILED"
        print(f"  LR={lr:.1e}: {status}")

    print("\nCheck W&B project: rope-verification")
    print("="*70)


if __name__ == "__main__":
    main()
