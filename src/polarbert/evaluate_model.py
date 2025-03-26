import torch
import torch.nn as nn
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import math # Added for ceiling division
from typing import Dict, Any, Tuple

# --- Add Matplotlib Imports ---
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For LogNorm
# --- End Matplotlib Imports ---

# --- Assume these imports work correctly from your project structure ---
try:
    from polarbert.finetuning import DirectionalHead
    from polarbert.finetuning import SimpleTransformerCls as PretrainedBackboneCls 
    from polarbert.pretraining import load_and_process_config, get_dataloaders, MODEL_CLASSES
    from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors, unit_vector_to_angles
except ImportError as e:
    print(f"Error importing polarbert modules: {e}")
    print("Please ensure 'polarbert' is installed or accessible in your PYTHONPATH.")
    exit(1)
# --- End Imports ---


def load_model_from_checkpoint(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> DirectionalHead:
    """Loads the DirectionalHead model and weights from a checkpoint file."""
    print("Initializing model architecture...")
    try:
        backbone_model = PretrainedBackboneCls(config) 
    except Exception as e:
        print(f"Error initializing backbone model (SimpleTransformerCls/PretrainedBackboneCls): {e}")
        print("Check if the class name and config structure are correct.")
        exit(1)

    model = DirectionalHead(config, backbone_model)

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) 
    except Exception as e:
         print(f"Error loading checkpoint file: {e}")
         print("Attempting to load with weights_only=True...")
         try:
              checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
         except Exception as e_alt:
              print(f"Fallback loading failed: {e_alt}")
              exit(1)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Adjust keys if necessary (e.g., removing 'model.' prefix)
        state_dict = {k.partition('model.')[2] if k.startswith('model.') else k: v for k, v in state_dict.items()}
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        print("Error: Checkpoint dictionary structure not recognized or state_dict not found.")
        exit(1)

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True) 
        if missing_keys: print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys: {unexpected_keys}")
        print("Successfully loaded model weights.")
    except RuntimeError as e:
         print(f"Error loading state_dict: {e}")
         exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading the state dict: {e}")
         exit(1)
         
    model.to(device)
    model.eval()

    if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
        opt_state = checkpoint['optimizer_states'][0] 
        opt_class_name = opt_state.get('__class__', '') 
        if 'ScheduleFree' in opt_class_name or 'schedulefree' in str(opt_state).lower():
             print("Warning: Model appears trained with ScheduleFree optimizer. Standalone evaluation might differ slightly.")

    return model


def evaluate_loop(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Runs the evaluation loop and returns loss, predictions, and truths."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds_list = []
    all_truths_list = []

    print("Note: You may see warnings about IterableDataset length mismatch. This is expected and handled.")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                inp, yc = batch 
                y_target_angles, c_target = yc 
                
                batch_size = 0
                processed_inp = None
                if isinstance(inp, torch.Tensor):
                     batch_size = inp.size(0)
                     processed_inp = inp.to(device)
                elif isinstance(inp, (list, tuple)):
                     processed_inp = []
                     for item in inp:
                          if isinstance(item, torch.Tensor):
                               if batch_size == 0: batch_size = item.size(0)
                               processed_inp.append(item.to(device))
                          else: processed_inp.append(item)
                     processed_inp = tuple(processed_inp) if isinstance(inp, tuple) else processed_inp
                else: raise TypeError(f"Unsupported input type: {type(inp)}")
                     
                if batch_size == 0: continue

                y_target_angles = y_target_angles.to(device)
                y_pred_vectors = model(processed_inp) 
                y_truth_vectors = angles_to_unit_vector(y_target_angles[:, 0], y_target_angles[:, 1]).to(device)
                loss = angular_dist_score_unit_vectors(y_truth_vectors, y_pred_vectors, epsilon=1e-4)

                total_loss += loss.item() * batch_size
                num_samples += batch_size
                all_preds_list.append(y_pred_vectors.cpu().numpy())
                all_truths_list.append(y_truth_vectors.cpu().numpy())

            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue

    if num_samples == 0: return 0.0, np.array([]), np.array([])

    avg_loss = total_loss / num_samples
    all_preds = np.concatenate(all_preds_list, axis=0)
    all_truths = np.concatenate(all_truths_list, axis=0)
    
    print(f"Processed {num_samples} samples in total.")
    return avg_loss, all_preds, all_truths


def generate_plots(true_vectors: np.ndarray, pred_vectors: np.ndarray, angular_dists_deg: np.ndarray, output_dir: Path):
    """Generates and saves evaluation plots (angular error hist, 2D angle hists)."""
    print("Generating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Histogram of Angular Errors ---
    plt.figure(figsize=(10, 6))
    plt.hist(angular_dists_deg, bins=50, alpha=0.7, label='Angular Error Distribution')
    plt.xlabel('Angular Distance (degrees)')
    plt.ylabel('Count')
    plt.title('Distribution of Angular Errors')
    median_error = np.median(angular_dists_deg)
    mean_error = np.mean(angular_dists_deg)
    plt.axvline(median_error, color='r', linestyle='--', label=f'Median: {median_error:.2f}°')
    plt.axvline(mean_error, color='g', linestyle='--', label=f'Mean: {mean_error:.2f}°')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Use log scale for count if distribution is very peaked
    plot_path = output_dir / 'angular_error_distribution.png'
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Failed to save angular error plot: {e}")
    plt.close()
    print(f"Saved angular error distribution plot to {plot_path}")

    # --- Calculate Angles for 2D Histograms ---
    try:
        true_angles = unit_vector_to_angles(torch.from_numpy(true_vectors)).numpy()
        pred_angles = unit_vector_to_angles(torch.from_numpy(pred_vectors)).numpy()
        true_azimuth_deg = np.degrees(true_angles[:, 0])
        true_zenith_deg = np.degrees(true_angles[:, 1])
        pred_azimuth_deg = np.degrees(pred_angles[:, 0])
        pred_zenith_deg = np.degrees(pred_angles[:, 1])
    except Exception as e:
        print(f"Warning: Could not convert vectors to angles for plotting. Skipping angle plots. Error: {e}")
        return # Skip remaining plots if conversion fails

    # --- Plot 2: True vs Predicted Zenith (2D Histogram) ---
    plt.figure(figsize=(10, 8))
    try:
        # Use LogNorm for better visibility across density ranges
        counts, xbins, ybins, im = plt.hist2d(
            true_zenith_deg, pred_zenith_deg, 
            bins=50, cmap='viridis', 
            norm=mcolors.LogNorm(vmin=1) # vmin=1 avoids log(0) warnings/errors
        )
        plt.colorbar(im, label='Count') # Pass the image artist to colorbar
    except ValueError as ve: # Handle case with few data points causing LogNorm issues
        print(f"ValueError plotting zenith hist2d (possibly too few points): {ve}. Trying without LogNorm.")
        try:
            counts, xbins, ybins, im = plt.hist2d(true_zenith_deg, pred_zenith_deg, bins=50, cmap='viridis')
            plt.colorbar(im, label='Count')
        except Exception as e_fallback:
            print(f"Fallback plotting failed: {e_fallback}")
            plt.close() # Close figure if plotting failed
            return # Skip remaining plots for zenith

    min_val = min(np.min(true_zenith_deg), np.min(pred_zenith_deg))
    max_val = max(np.max(true_zenith_deg), np.max(pred_zenith_deg))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('True Zenith (degrees)')
    plt.ylabel('Predicted Zenith (degrees)')
    plt.title('True vs Predicted Zenith Angle (2D Histogram)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal') # Ensure aspect ratio is equal for angle plots
    plot_path = output_dir / 'zenith_comparison_hist2d.png'
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Failed to save zenith comparison plot: {e}")
    plt.close()
    print(f"Saved zenith comparison plot to {plot_path}")

    # --- Plot 3: True vs Predicted Azimuth (2D Histogram) ---
    plt.figure(figsize=(10, 8))
    try:
        counts, xbins, ybins, im = plt.hist2d(
            true_azimuth_deg, pred_azimuth_deg, 
            bins=50, cmap='viridis', 
            norm=mcolors.LogNorm(vmin=1)
        )
        plt.colorbar(im, label='Count')
    except ValueError as ve:
        print(f"ValueError plotting azimuth hist2d (possibly too few points): {ve}. Trying without LogNorm.")
        try:
            counts, xbins, ybins, im = plt.hist2d(true_azimuth_deg, pred_azimuth_deg, bins=50, cmap='viridis')
            plt.colorbar(im, label='Count')
        except Exception as e_fallback:
            print(f"Fallback plotting failed: {e_fallback}")
            plt.close()
            return # Skip remaining plots for azimuth

    min_val = min(np.min(true_azimuth_deg), np.min(pred_azimuth_deg))
    max_val = max(np.max(true_azimuth_deg), np.max(pred_azimuth_deg))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('True Azimuth (degrees)')
    plt.ylabel('Predicted Azimuth (degrees)')
    plt.title('True vs Predicted Azimuth Angle (2D Histogram)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plot_path = output_dir / 'azimuth_comparison_hist2d.png'
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Failed to save azimuth comparison plot: {e}")
    plt.close()
    print(f"Saved azimuth comparison plot to {plot_path}")


def calculate_and_save_results(true_vectors: np.ndarray, pred_vectors: np.ndarray, avg_loss: float, output_dir: Path, eval_set_name: str, args: argparse.Namespace):
    """Calculates metrics, saves them, saves details, and generates plots."""
    print("\nCalculating metrics...")
    dot_products = np.sum(true_vectors * pred_vectors, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_dists_rad = np.arccos(dot_products)
    angular_dists_deg = np.degrees(angular_dists_rad)

    metrics = {
        'avg_loss (angular_dist)': avg_loss,
        'mean_angular_error_deg': np.mean(angular_dists_deg),
        'median_angular_error_deg': np.median(angular_dists_deg),
        'std_angular_error_deg': np.std(angular_dists_deg),
        'angular_error_deg_90percentile': np.percentile(angular_dists_deg, 90),
        'angular_error_deg_95percentile': np.percentile(angular_dists_deg, 95),
    }

    print(f"\n--- Evaluation Results ({eval_set_name} set) ---")
    for name, value in metrics.items(): print(f"  {name}: {value:.6f}")
    print("--- End Results ---")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f'evaluation_metrics_{eval_set_name}.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation on {eval_set_name} set\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n\n")
        for name, value in metrics.items(): f.write(f"  {name}: {value:.6f}\n")
    print(f"Metrics saved to: {metrics_file}")

    try:
        print("Saving detailed results to CSV...")
        true_angles = unit_vector_to_angles(torch.from_numpy(true_vectors)).numpy()
        pred_angles = unit_vector_to_angles(torch.from_numpy(pred_vectors)).numpy()
        results_df = pd.DataFrame({
            'true_x': true_vectors[:, 0], 'true_y': true_vectors[:, 1], 'true_z': true_vectors[:, 2],
            'pred_x': pred_vectors[:, 0], 'pred_y': pred_vectors[:, 1], 'pred_z': pred_vectors[:, 2],
            'true_azimuth_rad': true_angles[:, 0], 'true_zenith_rad': true_angles[:, 1],
            'pred_azimuth_rad': pred_angles[:, 0], 'pred_zenith_rad': pred_angles[:, 1],
            'angular_error_deg': angular_dists_deg
        })
        csv_file = output_dir / f'evaluation_details_{eval_set_name}.csv'
        results_df.to_csv(csv_file, index=False, float_format='%.6f')
        print(f"Detailed results saved to: {csv_file}")
    except Exception as e: print(f"Warning: Could not save detailed CSV results. Error: {e}")

    # --- Generate Plots ---
    generate_plots(true_vectors, pred_vectors, angular_dists_deg, output_dir)


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a finetuned directional prediction model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file used for finetuning.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation outputs.')
    parser.add_argument('--use_test_set', action='store_true', help='Evaluate on the test set instead of the validation set.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda", "cuda:0", "cpu").')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading configuration from: {args.config}")
    try:
        config = load_and_process_config(args.config)
        config.setdefault('model', {}).setdefault('directional', {'hidden_size': 1024})
    except Exception as e:
         print(f"Error loading or processing config file '{args.config}': {e}")
         exit(1)

    model = load_model_from_checkpoint(config, args.checkpoint, device)

    print("Loading data...")
    try:
        # Assuming get_dataloaders returns (train, val) by default
        # Modify if your function returns test loader differently
        train_loader, val_loader = get_dataloaders(config) 
        test_loader = None # Placeholder, adapt if test loader is available
        # Add logic here if get_dataloaders CAN return a test_loader, e.g.,
        # if 'test_dataset' in config['data']:
        #     try:
        #         _, _, test_loader = get_dataloaders(config, include_test=True) # Fictional flag
        #     except: pass # Ignore if test loading fails or not implemented
            
    except Exception as e:
        print(f"Error getting dataloaders: {e}")
        exit(1)

    if args.use_test_set:
        if test_loader is not None:
            eval_loader = test_loader
            eval_set_name = "test"
            print("Using TEST set for evaluation.")
        else:
            print("Warning: --use_test_set specified, but test loader is not available. Using VALIDATION set.")
            eval_loader = val_loader
            eval_set_name = "validation"
            print("Using VALIDATION set for evaluation.")
    else:
        eval_loader = val_loader
        eval_set_name = "validation"
        print("Using VALIDATION set for evaluation.")

    if eval_loader is None:
         print(f"Error: Could not obtain {eval_set_name} dataloader.")
         exit(1)

    avg_loss, all_preds, all_truths = evaluate_loop(model, eval_loader, device)

    if all_preds.size > 0 and all_truths.size > 0:
        calculate_and_save_results(all_truths, all_preds, avg_loss, output_dir, eval_set_name, args)
    else:
        print("Skipping metric calculation and plotting as no results were generated.")

    print("\nEvaluation script finished.")