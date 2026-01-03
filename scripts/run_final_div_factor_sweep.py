#!/usr/bin/env python3
"""
Final Div Factor Sweep for Deep Model

Tests different final_div_factor values with multiple init seeds.
Same init seed for each group of div factors, then repeat with different seeds.
"""

import subprocess
import yaml
import os
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
BASE_CONFIG = PROJECT_DIR / "configs" / "completep_verification" / "deep_C_10M_5ep.yaml"

# Final div factor values to test
FINAL_DIV_FACTORS = [0.1, 1, 1000]

# Init seeds to test
INIT_SEEDS = [42, 123, 456]

SHUFFLE_SEED = 1337


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_experiment(final_div_factor: float, seed: int):
    """Run experiment with given final_div_factor and seed."""
    config = load_config(BASE_CONFIG)

    # Set final_div_factor
    config['training']['final_div_factor'] = final_div_factor

    # Create run name
    timestamp = datetime.now().strftime('%H%M%S')
    fdf_str = str(final_div_factor).replace('.', 'p')
    run_name = f"fdf_sweep_s{seed}_fdf{fdf_str}_{timestamp}"
    config['model']['model_name'] = run_name

    # Save temporary config
    temp_dir = PROJECT_DIR / "checkpoints" / "div_factor_sweep" / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / f"{run_name}.yaml"
    save_config(config, temp_config_path)

    print(f"\n{'='*60}")
    print(f"Running: seed={seed}, final_div_factor={final_div_factor}")
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
    print("FINAL DIV FACTOR SWEEP - DEEP MODEL")
    print("="*70)
    print(f"\nfinal_div_factor values: {FINAL_DIV_FACTORS}")
    print(f"Init seeds: {INIT_SEEDS}")
    print(f"Total runs: {len(FINAL_DIV_FACTORS) * len(INIT_SEEDS)}")
    print("="*70)

    results = []

    # For each seed, run all div factors
    for seed in INIT_SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        for fdf in FINAL_DIV_FACTORS:
            success = run_experiment(fdf, seed)
            results.append((seed, fdf, success))

    print("\n" + "="*70)
    print("SWEEP COMPLETE")
    print("="*70)
    print(f"\n{'Seed':<8} {'FDF':<10} {'Status'}")
    print("-"*30)
    for seed, fdf, success in results:
        status = "OK" if success else "FAILED"
        print(f"{seed:<8} {fdf:<10} {status}")

    print("\nCheck W&B: fdf_sweep_s*")
    print("="*70)


if __name__ == "__main__":
    main()
