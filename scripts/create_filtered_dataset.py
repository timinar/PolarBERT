"""
Create filtered IceCube dataset with only "good" upgoing events.

Filters events where:
- True neutrino is upgoing (zenith > pi/2, i.e., cos_zenith < 0)
- Model prediction is upgoing (cos_zenith < 0)
- Angular loss < threshold (default 10 degrees)

Creates memmapped arrays compatible with IceCubeDataset.

Usage:
    python scripts/create_filtered_dataset.py --threshold 10
    python scripts/create_filtered_dataset.py --threshold 10 --dataset train
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Source directories
V2_DIR = Path('/groups/pheno/inar/icecube_kaggle/v2')
INFERENCE_DIR = Path('/groups/pheno/inar/icecube_kaggle/v2/inference_results')

DATASETS = {
    'train': {
        'source': V2_DIR / 'memmapped_train_130M_127',
        'predictions': INFERENCE_DIR / 'train_predictions.npz',
    },
    'eval': {
        'source': V2_DIR / 'memmapped_eval_1.2M_127',
        'predictions': INFERENCE_DIR / 'eval_predictions.npz',
    },
    'test': {
        'source': V2_DIR / 'memmapped_test_0.6M_127',
        'predictions': INFERENCE_DIR / 'test_predictions.npz',
    },
}


def load_memmap_properties(data_dir: Path) -> dict:
    """Load memmap properties from JSON file."""
    with open(data_dir / 'memmap_properties.json', 'r') as f:
        return json.load(f)


def get_dtype(dtype_spec):
    """Convert dtype specification to numpy dtype."""
    if isinstance(dtype_spec, list):
        # Structured dtype
        return np.dtype([(name, dtype) for name, dtype in dtype_spec])
    return np.dtype(dtype_spec)


def get_filtered_indices(predictions_path: Path, threshold_deg: float) -> np.ndarray:
    """
    Get indices of events that pass the filter criteria.

    Criteria:
    - True upgoing (true_cos_zenith < 0)
    - Predicted upgoing (pred_cos_zenith < 0)
    - Angular loss < threshold
    """
    print(f"Loading predictions from {predictions_path}")
    preds = np.load(predictions_path)

    angular_loss_deg = np.degrees(preds['angular_loss'])
    pred_cos_zenith = preds['pred_cos_zenith']
    true_cos_zenith = preds['true_cos_zenith']
    event_indices = preds['event_indices']

    # Filter criteria
    both_upgoing = (pred_cos_zenith < 0) & (true_cos_zenith < 0)
    good_loss = angular_loss_deg < threshold_deg
    mask = both_upgoing & good_loss

    filtered_indices = event_indices[mask]

    print(f"  Total events: {len(event_indices):,}")
    print(f"  Both upgoing: {np.sum(both_upgoing):,} ({100*np.mean(both_upgoing):.1f}%)")
    print(f"  Loss < {threshold_deg}°: {np.sum(good_loss):,} ({100*np.mean(good_loss):.1f}%)")
    print(f"  Passing filter: {len(filtered_indices):,} ({100*np.mean(mask):.1f}%)")

    return filtered_indices


def create_filtered_dataset(
    source_dir: Path,
    output_dir: Path,
    filtered_indices: np.ndarray,
    chunk_size: int = 100000,
):
    """
    Create filtered memmapped dataset from source.

    Args:
        source_dir: Path to source memmapped directory
        output_dir: Path to output directory
        filtered_indices: Array of event indices to include
        chunk_size: Number of events to process at a time (for memory efficiency)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source properties
    props = load_memmap_properties(source_dir)
    n_filtered = len(filtered_indices)
    seq_length = props['x']['shape'][1]

    print(f"\nCreating filtered dataset with {n_filtered:,} events")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")

    # Create new memmap properties
    new_props = {
        'x': {
            'shape': [n_filtered, seq_length],
            'dtype': props['x']['dtype'],
        },
        'l': {
            'shape': [n_filtered],
            'dtype': props['l']['dtype'],
        },
        'c': {
            'shape': [n_filtered],
            'dtype': props['c']['dtype'],
        },
        'y': {
            'shape': [n_filtered, 2],
            'dtype': props['y']['dtype'],
        },
    }

    # Save properties
    with open(output_dir / 'memmap_properties.json', 'w') as f:
        json.dump(new_props, f, indent=4)
    print("  Saved memmap_properties.json")

    # Load source memmaps
    x_dtype = get_dtype(props['x']['dtype'])
    x_src = np.memmap(source_dir / 'x.npy', dtype=x_dtype, mode='r',
                      shape=tuple(props['x']['shape']))
    l_src = np.memmap(source_dir / 'l.npy', dtype=props['l']['dtype'], mode='r',
                      shape=tuple(props['l']['shape']))
    c_src = np.memmap(source_dir / 'c.npy', dtype=props['c']['dtype'], mode='r',
                      shape=tuple(props['c']['shape']))
    y_src = np.memmap(source_dir / 'y.npy', dtype=props['y']['dtype'], mode='r',
                      shape=tuple(props['y']['shape']))

    # Create output memmaps
    x_dst = np.memmap(output_dir / 'x.npy', dtype=x_dtype, mode='w+',
                      shape=(n_filtered, seq_length))
    l_dst = np.memmap(output_dir / 'l.npy', dtype=props['l']['dtype'], mode='w+',
                      shape=(n_filtered,))
    c_dst = np.memmap(output_dir / 'c.npy', dtype=props['c']['dtype'], mode='w+',
                      shape=(n_filtered,))
    y_dst = np.memmap(output_dir / 'y.npy', dtype=props['y']['dtype'], mode='w+',
                      shape=(n_filtered, 2))

    # Copy data in chunks
    print("  Copying filtered events...")
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size

    for i in tqdm(range(n_chunks), desc="  Processing"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_filtered)
        idx_chunk = filtered_indices[start:end]

        x_dst[start:end] = x_src[idx_chunk]
        l_dst[start:end] = l_src[idx_chunk]
        c_dst[start:end] = c_src[idx_chunk]
        y_dst[start:end] = y_src[idx_chunk]

    # Flush to disk
    x_dst.flush()
    l_dst.flush()
    c_dst.flush()
    y_dst.flush()

    print(f"  Done! Created {n_filtered:,} events")

    # Verify
    print("  Verifying...")
    x_verify = np.memmap(output_dir / 'x.npy', dtype=x_dtype, mode='r',
                         shape=(n_filtered, seq_length))
    assert x_verify.shape == (n_filtered, seq_length), "x shape mismatch"
    print("  Verification passed!")


def main():
    parser = argparse.ArgumentParser(description='Create filtered upgoing dataset')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Angular loss threshold in degrees (default: 10)')
    parser.add_argument('--dataset', type=str, choices=['train', 'eval', 'test', 'all'],
                        default='all', help='Which dataset to process')
    parser.add_argument('--output_base', type=str,
                        default='/groups/pheno/inar/icecube_kaggle/v4_upgoing',
                        help='Base output directory')
    parser.add_argument('--chunk_size', type=int, default=100000,
                        help='Chunk size for processing')
    args = parser.parse_args()

    threshold = args.threshold
    output_base = Path(args.output_base)

    print("=" * 60)
    print(f"Creating v4_upgoing dataset")
    print(f"  Threshold: {threshold}°")
    print(f"  Output base: {output_base}")
    print("=" * 60)

    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = ['train', 'eval', 'test']
    else:
        datasets_to_process = [args.dataset]

    # Process each dataset
    for dataset_name in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'='*60}")

        config = DATASETS[dataset_name]
        source_dir = config['source']
        predictions_path = config['predictions']

        # Check if predictions exist
        if not predictions_path.exists():
            print(f"  WARNING: Predictions not found at {predictions_path}")
            print(f"  Skipping {dataset_name}...")
            continue

        # Get filtered indices
        filtered_indices = get_filtered_indices(predictions_path, threshold)

        if len(filtered_indices) == 0:
            print(f"  WARNING: No events pass the filter!")
            continue

        # Determine output directory name based on number of events
        n_events = len(filtered_indices)
        if n_events >= 1_000_000:
            size_str = f"{n_events / 1_000_000:.1f}M"
        else:
            size_str = f"{n_events / 1_000:.0f}k"

        output_dir = output_base / f"memmapped_{dataset_name}_{size_str}_127"

        # Create filtered dataset
        create_filtered_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            filtered_indices=filtered_indices,
            chunk_size=args.chunk_size,
        )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")

    # Print summary
    print("\nCreated datasets:")
    for p in sorted(output_base.glob("memmapped_*")):
        props_file = p / 'memmap_properties.json'
        if props_file.exists():
            with open(props_file) as f:
                props = json.load(f)
            n = props['x']['shape'][0]
            print(f"  {p.name}: {n:,} events")


if __name__ == '__main__':
    main()
