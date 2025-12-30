#!/usr/bin/env python3
"""
CompleteP Verification Script

Tests hyperparameter transfer by running the same base LR across different model sizes.
If CompleteP is implemented correctly, the optimal LR for the base model should
transfer to larger models without retuning.

Expected behavior:
1. Model A (base N=256, L=2): Optimal at lr_base
2. Model B (4x width N=1024, L=2): Same loss trajectory as A with transferred LR
3. Model C (4x depth N=256, L=8): Same loss trajectory as A with transferred LR

Usage:
    python scripts/run_completep_verification.py --experiment 1  # LR transfer verification
    python scripts/run_completep_verification.py --experiment 2  # Duration transfer
    python scripts/run_completep_verification.py --model A --lr 3e-4  # Single run
"""

import argparse
import subprocess
import tempfile
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/lustre/hpc/pheno/inar/PolarBERT")
CONFIG_DIR = PROJECT_DIR / "configs" / "completep_verification"

# Model configurations
MODELS = {
    "A": {
        "config": "base_A_N256_L2.yaml",
        "description": "Base model (N=256, L=2)",
        "width": 256,
        "depth": 2,
    },
    "B": {
        "config": "wide_B_N1024_L2.yaml",
        "description": "Wide model (N=1024, L=2) - 4x width",
        "width": 1024,
        "depth": 2,
    },
    "C": {
        "config": "deep_C_N256_L8.yaml",
        "description": "Deep model (N=256, L=8) - 4x depth",
        "width": 256,
        "depth": 8,
    },
}

# LR grid for verification
LR_VALUES = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path):
    """Save config to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_single_experiment(model_name: str, lr_base: float, run_name: str = None):
    """Run a single training experiment."""
    model_info = MODELS[model_name]
    config_path = CONFIG_DIR / model_info["config"]

    # Load and modify config
    config = load_config(config_path)
    config['training']['completep']['lr_base'] = lr_base

    # Create temporary config with modified LR
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    if run_name is None:
        run_name = f"completep_{model_name}_lr{lr_base:.0e}_{timestamp}"

    config['model']['model_name'] = run_name

    # Save temporary config
    temp_config_dir = PROJECT_DIR / "checkpoints" / "completep_verification" / "temp_configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"{run_name}.yaml"
    save_config(config, temp_config_path)

    print(f"\n{'='*60}")
    print(f"Running: {model_info['description']}")
    print(f"LR base: {lr_base:.2e}")
    print(f"Config: {temp_config_path}")
    print(f"{'='*60}\n")

    # Run training
    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", str(temp_config_path),
        "--name", run_name,
        "--model_type", "flash",
        "--dataset_type", "kaggle",
        "--checkpoint_path", "new",
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode == 0


def run_experiment_1():
    """
    Experiment 1: LR Transfer Verification

    Run LR grid search across all three model sizes to verify that
    the optimal LR transfers from Model A to Models B and C.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: LR Transfer Verification")
    print("="*70)
    print("\nRunning LR grid search across model sizes:")
    print(f"  LR values: {LR_VALUES}")
    print(f"  Models: A (base), B (4x width), C (4x depth)")
    print("\nExpected result: Optimal LR should be the same for all models")
    print("="*70 + "\n")

    results = {}

    for model_name in ["A", "B", "C"]:
        results[model_name] = {}
        for lr in LR_VALUES:
            print(f"\n>>> Training Model {model_name} with LR={lr:.2e}")
            success = run_single_experiment(model_name, lr)
            results[model_name][lr] = success

    print("\n" + "="*70)
    print("EXPERIMENT 1 COMPLETE")
    print("="*70)
    print("\nResults summary:")
    for model_name, model_results in results.items():
        print(f"\nModel {model_name} ({MODELS[model_name]['description']}):")
        for lr, success in model_results.items():
            status = "OK" if success else "FAILED"
            print(f"  LR={lr:.2e}: {status}")

    print("\n" + "="*70)
    print("Check W&B dashboard for loss curves comparison")
    print("Success metric: Loss vs LR curves should align across models")
    print("="*70)


def run_experiment_2():
    """
    Experiment 2: Duration Transfer

    Verify the 1/sqrt(kappa) LR scaling rule by training Model A
    at 100k and 1M events. The LR for 1M events should be automatically
    scaled by 1/sqrt(10) = 0.316.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Duration Transfer Verification")
    print("="*70)
    print("\nTesting duration scaling (kappa = T_target / T_base):")
    print("  Model A at 100k events (kappa=1)")
    print("  Model A at 1M events (kappa=10, LR scaled by 1/sqrt(10))")
    print("\nExpected result: Loss curves should overlap when plotted vs % complete")
    print("="*70 + "\n")

    # Run Model A at 100k events
    print(">>> Training Model A at 100k events (base duration)")
    run_single_experiment("A", 3e-4, run_name=f"completep_duration_100k_{datetime.now().strftime('%H%M%S')}")

    # Run Model A at 1M events (uses duration_1M.yaml config)
    print("\n>>> Training Model A at 1M events (10x duration)")
    config_path = CONFIG_DIR / "duration_1M.yaml"
    config = load_config(config_path)
    run_name = f"completep_duration_1M_{datetime.now().strftime('%H%M%S')}"
    config['model']['model_name'] = run_name

    temp_config_dir = PROJECT_DIR / "checkpoints" / "completep_verification" / "temp_configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"{run_name}.yaml"
    save_config(config, temp_config_path)

    cmd = [
        "python", "-m", "polarbert.finetuning",
        "--config", str(temp_config_path),
        "--name", run_name,
        "--model_type", "flash",
        "--dataset_type", "kaggle",
        "--checkpoint_path", "new",
    ]
    subprocess.run(cmd, cwd=str(PROJECT_DIR))

    print("\n" + "="*70)
    print("EXPERIMENT 2 COMPLETE")
    print("="*70)
    print("\nCheck W&B dashboard for:")
    print("  - Plot loss vs '% of training complete' for both runs")
    print("  - Curves should overlap if duration scaling is correct")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="CompleteP Verification Script")
    parser.add_argument("--experiment", type=int, choices=[1, 2],
                        help="Experiment to run: 1=LR transfer, 2=Duration transfer")
    parser.add_argument("--model", type=str, choices=["A", "B", "C"],
                        help="Single model to run (for debugging)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="LR base value (default: 3e-4)")
    args = parser.parse_args()

    if args.model:
        # Single model run
        run_single_experiment(args.model, args.lr)
    elif args.experiment == 1:
        run_experiment_1()
    elif args.experiment == 2:
        run_experiment_2()
    else:
        print("Please specify --experiment 1 or --experiment 2, or --model A/B/C")
        print("\nExamples:")
        print("  python scripts/run_completep_verification.py --experiment 1")
        print("  python scripts/run_completep_verification.py --experiment 2")
        print("  python scripts/run_completep_verification.py --model A --lr 3e-4")


if __name__ == "__main__":
    main()
