"""
Inference script for filtering IceCube Kaggle events.

Runs a finetuned DirectionalHead model on all dataset splits and saves
per-event predictions and angular losses for subsequent filtering.

Usage:
    python scripts/run_inference_for_filtering.py \
        --checkpoint /groups/pheno/inar/PolarBERT/checkpoints/results/dom_coord_100M_finetuned_251125-160742/last.ckpt \
        --config /groups/pheno/inar/PolarBERT/configs/2027_07_polarbert_IT_finetuning.yaml \
        --dataset all \
        --batch_size 2048 \
        --output_dir /groups/pheno/inar/icecube_kaggle/v2/inference_results/
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional

from polarbert.finetuning import DirectionalHead, SimpleTransformerCls
from polarbert.icecube_dataset import IceCubeDataset
from polarbert.loss_functions import (
    angles_to_unit_vector,
    unit_vector_to_angles,
)
from polarbert.utils.config import load_and_process_config
from polarbert.utils.data import default_transform

# Dataset paths
DATASETS = {
    'train': '/groups/pheno/inar/icecube_kaggle/v2/memmapped_train_130M_127',
    'eval': '/groups/pheno/inar/icecube_kaggle/v2/memmapped_eval_1.2M_127',
    'test': '/groups/pheno/inar/icecube_kaggle/v2/memmapped_test_0.6M_127',
}


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> DirectionalHead:
    """Load the finetuned DirectionalHead model from checkpoint."""
    print(f"Initializing model architecture...")

    # Initialize backbone
    backbone = SimpleTransformerCls(config)
    model = DirectionalHead(config, backbone)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    print("Successfully loaded model weights.")

    model.to(device)
    model.eval()
    return model


def target_transform_kaggle(y, c):
    """Standard target transform for Kaggle dataset."""
    return y.astype(np.float32), c.astype(np.float32)


def compute_per_event_loss(y_truth_vec: torch.Tensor, y_pred_vec: torch.Tensor) -> torch.Tensor:
    """Compute angular distance per event (not mean)."""
    scalar_prod = torch.sum(y_truth_vec * y_pred_vec, dim=1)
    scalar_prod = torch.clamp(scalar_prod, -1.0 + 1e-4, 1.0 - 1e-4)
    return torch.abs(torch.arccos(scalar_prod))  # Returns per-event loss in radians


def run_inference(
    model: DirectionalHead,
    dataset_path: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference sequentially through the dataset.

    Returns dict with all prediction arrays.
    """
    print(f"Loading dataset from: {dataset_path}")

    dataset = IceCubeDataset(
        data_dir=dataset_path,
        batch_size=batch_size,
        transform=default_transform,
        target_transform=target_transform_kaggle,
    )

    n_events = dataset.num_events
    n_batches = len(dataset)
    print(f"Dataset has {n_events:,} events in {n_batches:,} batches")

    # Collect results
    results = {
        'event_indices': [],
        'angular_loss': [],
        'pred_azimuth': [],
        'pred_zenith': [],
        'true_azimuth': [],
        'true_zenith': [],
        'pred_cos_zenith': [],
        'true_cos_zenith': [],
    }

    with torch.no_grad():
        for (x, l), (y, c), batch_start_idx in tqdm(dataset.iter_sequential(),
                                                      total=n_batches,
                                                      desc="Processing"):
            actual_batch_size = x['features'].shape[0]

            # Move inputs to device
            x_device = {
                'features': torch.tensor(x['features']).to(device),
                'dom_id': torch.tensor(x['dom_id']).to(device),
            }
            l_device = torch.tensor(l).to(device)

            # Convert target angles to tensor
            y_target_angles = torch.tensor(y).to(device)

            # Forward pass - get unit vector predictions
            y_pred_vec = model((x_device, l_device))  # [batch, 3]

            # Convert truth angles to unit vectors
            y_truth_vec = angles_to_unit_vector(
                y_target_angles[:, 0],  # azimuth
                y_target_angles[:, 1]   # zenith
            )

            # Compute per-event angular loss
            per_event_loss = compute_per_event_loss(y_truth_vec, y_pred_vec)

            # Convert predictions to angles
            pred_angles = unit_vector_to_angles(y_pred_vec)  # [batch, 2] = [azimuth, zenith]

            # Store results
            event_indices = np.arange(batch_start_idx, batch_start_idx + actual_batch_size)
            results['event_indices'].append(event_indices)
            results['angular_loss'].append(per_event_loss.cpu().numpy())
            results['pred_azimuth'].append(pred_angles[:, 0].cpu().numpy())
            results['pred_zenith'].append(pred_angles[:, 1].cpu().numpy())
            results['true_azimuth'].append(y_target_angles[:, 0].cpu().numpy())
            results['true_zenith'].append(y_target_angles[:, 1].cpu().numpy())
            # z-component = cos(zenith) for quick upgoing filtering
            results['pred_cos_zenith'].append(y_pred_vec[:, 2].cpu().numpy())
            results['true_cos_zenith'].append(y_truth_vec[:, 2].cpu().numpy())

    # Concatenate all results
    return {k: np.concatenate(v, axis=0) for k, v in results.items()}


def print_summary(results: Dict[str, np.ndarray], dataset_name: str):
    """Print summary statistics of the inference results."""
    angular_loss_deg = np.degrees(results['angular_loss'])
    pred_upgoing = results['pred_cos_zenith'] < 0
    true_upgoing = results['true_cos_zenith'] < 0
    both_upgoing = pred_upgoing & true_upgoing

    print(f"\n--- Summary for {dataset_name} ---")
    print(f"Total events: {len(results['event_indices']):,}")
    print(f"Angular loss - Mean: {np.mean(angular_loss_deg):.2f} deg, "
          f"Median: {np.median(angular_loss_deg):.2f} deg")
    print(f"True upgoing (zenith > pi/2): {np.sum(true_upgoing):,} ({100*np.mean(true_upgoing):.1f}%)")
    print(f"Predicted upgoing: {np.sum(pred_upgoing):,} ({100*np.mean(pred_upgoing):.1f}%)")
    print(f"Both upgoing: {np.sum(both_upgoing):,} ({100*np.mean(both_upgoing):.1f}%)")

    # For upgoing events
    if np.sum(both_upgoing) > 0:
        upgoing_loss_deg = angular_loss_deg[both_upgoing]
        print(f"\nFor both-upgoing events:")
        print(f"  Angular loss - Mean: {np.mean(upgoing_loss_deg):.2f} deg, "
              f"Median: {np.median(upgoing_loss_deg):.2f} deg")
        for threshold in [10, 15, 20, 25, 30]:
            passing = upgoing_loss_deg < threshold
            print(f"  Loss < {threshold} deg: {np.sum(passing):,} events ({100*np.mean(passing):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Run inference for filtering IceCube events')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the finetuned model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--dataset', type=str, choices=['train', 'eval', 'test', 'all'], default='all',
                        help='Which dataset to process')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str,
                        default='/groups/pheno/inar/icecube_kaggle/v2/inference_results/',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    print(f"Loading config from: {args.config}")
    config = load_and_process_config(args.config)

    # Ensure directional config exists
    if 'directional' not in config['model']:
        config['model']['directional'] = {'hidden_size': 1024}

    model = load_model(config, args.checkpoint, device)

    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = [args.dataset]

    # Process each dataset
    for dataset_name in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'='*60}")

        dataset_path = DATASETS[dataset_name]
        output_file = output_dir / f"{dataset_name}_predictions.npz"

        # Check if output already exists
        if output_file.exists():
            print(f"Output file already exists: {output_file}")
            user_input = input("Overwrite? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Skipping...")
                continue

        try:
            results = run_inference(
                model=model,
                dataset_path=dataset_path,
                batch_size=args.batch_size,
                device=device,
            )

            # Save results
            np.savez_compressed(output_file, **results)
            print(f"\nSaved {len(results['event_indices']):,} predictions to {output_file}")

            # Print summary
            print_summary(results, dataset_name)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nCUDA OOM error! Try reducing batch_size (current: {args.batch_size})")
                print("Suggested: --batch_size 1024 or --batch_size 512")
                raise
            raise

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
