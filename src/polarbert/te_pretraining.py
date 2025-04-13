import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback # Added Callback
# import yaml # No longer needed directly here if using config class
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Optional, Callable # Removed Dict as config object used
import math
import os
import warnings

torch.set_float32_matmul_precision('high')

# --- Import project modules ---
# Assuming your classes are structured like this
try:
    from polarbert.config import PolarBertConfig # Import the main config class
    from polarbert.time_embed_polarbert import PolarBertModel # Import the new model class
    # Dataloader depends on dataset type, handled in get_dataloaders
    # Embedding is part of PolarBertModel
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure classes are defined and accessible.")
    raise e

# --- Dataloader Function (Adapted) ---

# Keep default transforms for now, could be moved to config later if needed
# def default_transform(x, l):
#     # Minimal transform, just ensure float type if needed by model
#     return x.astype(np.float32), l.astype(np.int32) # Ensure lengths are int

#  to fix this warning:
# UserWarning: The given NumPy array is not writable... writing to this tensor will result in undefined behavior.
def default_transform(x, l):
    # Ensure a writable copy is returned
    return x.astype(np.float32).copy(), l.astype(np.int32).copy()

def default_target_transform(y, c):
    # Ensure charge is float, y might be complex (dict/structured array) or simple array
    # Return None for y if no specific target transform needed for pretraining labels
    return y, c.astype(np.float32)


def get_dataloaders(
        config: PolarBertConfig, # Use PolarBertConfig object
        dataset_type: str,
        transform=default_transform,
        target_transform=default_target_transform,
    ) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation dataloaders."""

    print(f"Using dataset type: {dataset_type}")
    if dataset_type == 'prometheus':
        # Ensure prometheus dataset class is importable
        from polarbert.prometheus_dataset import IceCubeDataset
        print("Imported Prometheus IceCubeDataset.")
    elif dataset_type == 'kaggle':
        # Ensure kaggle dataset class is importable
        from polarbert.icecube_dataset import IceCubeDataset
        print("Imported Kaggle IceCubeDataset.")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if config.training.per_device_batch_size is None:
        raise ValueError("per_device_batch_size must be calculated before calling get_dataloaders.")

    print(f"Instantiating train dataset from: {config.data.train_dir}")
    full_train_dataset = IceCubeDataset(
        data_dir=config.data.train_dir,
        # Use the per_device_batch_size calculated based on logical and max_per_device
        batch_size=config.training.per_device_batch_size,
        transform=transform,
        target_transform=target_transform
    )
    print(f"Instantiating validation dataset from: {config.data.val_dir}")
    full_val_dataset = IceCubeDataset(
        data_dir=config.data.val_dir,
        batch_size=config.training.per_device_batch_size, # Use same device batch size for validation
        transform=transform,
        target_transform=target_transform
    )

    # Slicing logic based on event counts
    train_events = config.data.train_events
    val_events = config.data.val_events

    # Select appropriate slicing based on dataset type if needed
    # (Using simple slicing for now, adjust if datasets have different structures)
    print(f"Slicing train dataset to {train_events} events.")
    train_dataset = full_train_dataset.slice(0, train_events)
    print(f"Slicing validation dataset to {val_events} events.")
    val_dataset = full_val_dataset.slice(0, val_events)

    # Dataloader arguments
    loader_kwargs = {
        'batch_size': None, # Handled by IceCubeDataset internal batching
        'num_workers': config.data.num_workers,
        'pin_memory': config.data.pin_memory,
        # Persistent workers require num_workers > 0
        'persistent_workers': config.data.persistent_workers and config.data.num_workers > 0
    }
    print(f"Creating DataLoaders with num_workers={loader_kwargs['num_workers']}, "
          f"pin_memory={loader_kwargs['pin_memory']}, "
          f"persistent_workers={loader_kwargs['persistent_workers']}")

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    return train_loader, val_loader


# --- Callback Setup Function (Adapted) ---

def setup_callbacks(config: PolarBertConfig, run_name: str) -> list:
    """Sets up PyTorch Lightning callbacks."""
    callbacks = [LearningRateMonitor(logging_interval='step')]

    # Get checkpoint config from the training config
    checkpoint_cfg = config.training.checkpoint

    # Setup checkpoint directory using run_name
    checkpoint_dir = Path(checkpoint_cfg.dirpath) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    # Create ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch:02d}-{step:06d}", # TODO: Consider adding val_loss to filename
        save_top_k=checkpoint_cfg.save_top_k,
        monitor=checkpoint_cfg.monitor,
        mode=checkpoint_cfg.mode,
        save_last=checkpoint_cfg.save_last,
        # save_weights_only=False # Keep default to save optimizer state etc.
    )
    callbacks.append(checkpoint_callback)

    # --- Callback to save config YAML with the checkpoint ---
    class SaveConfigCallback(Callback):
        def __init__(self, config_to_save: PolarBertConfig):
            self.config_to_save = config_to_save

        # Using on_validation_end, assuming checkpoints are saved based on validation
        # Or use on_train_epoch_end if saving more frequently
        def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
             # Save config whenever a checkpoint is potentially saved
             # This might save duplicates, but ensures config exists with checkpoints
             if trainer.is_global_zero: # Ensure only rank 0 saves config
                 try:
                     # Save in the same directory as checkpoints for this run
                     config_save_path = Path(trainer.checkpoint_callback.dirpath) / "config.yaml"
                     self.config_to_save.save_yaml(str(config_save_path))
                     # print(f"Config saved to {config_save_path}") # Can be verbose
                 except Exception as e:
                     warnings.warn(f"Failed to save config YAML: {e}")

        # Optional: Save config at the very end of training
        def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            if trainer.is_global_zero:
                try:
                    config_save_path = Path(trainer.checkpoint_callback.dirpath) / "final_config.yaml"
                    self.config_to_save.save_yaml(str(config_save_path))
                    print(f"Final config saved to {config_save_path}")
                except Exception as e:
                    warnings.warn(f"Failed to save final config YAML: {e}")

    callbacks.append(SaveConfigCallback(config)) # Add the callback


    # Callback for final model (optional, if save_final=True)
    # Note: PL usually saves the final state via save_last=True if training completes.
    # This explicit callback might be redundant unless save_final behaviour is different.
    # Commenting out for now, rely on save_last=True.
    # if checkpoint_cfg.save_final:
    #     class FinalModelCallback(pl.Callback):
    #         def on_train_end(self, trainer, pl_module):
    #             if trainer.is_global_zero:
    #                 save_path = checkpoint_dir / "final_model_state_dict.pth"
    #                 torch.save(pl_module.state_dict(), save_path)
    #                 print(f"Final model state_dict saved to {save_path}")
    #     callbacks.append(FinalModelCallback())

    return callbacks

# --- Main Training Function ---

def main():
    parser = argparse.ArgumentParser(description="Train PolarBERT Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument('--name', type=str, default=None, help="Custom name for the training run.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID (e.g., SLURM ID) for naming.")
    # parser.add_argument("--model_type", type=str, ...) # No longer needed, model is PolarBertModel
    parser.add_argument("--dataset_type", type=str, choices=['kaggle', 'prometheus'], required=True)
    # parser.add_argument("--random_time_offset", type=float, default=None) # Add back if needed
    args = parser.parse_args()

    # 1. Load Configuration Object
    config = PolarBertConfig.from_yaml(args.config)

    # 2. Determine Run Name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    run_name = f"{args.name or config.model.model_name}_{suffix}"
    # Update config object with potentially overridden name for logging/saving
    config.model.model_name = run_name
    print(f"Starting run: {run_name}")


    # 3. Setup Logging (WandB)
    print("Setting up WandB logger...")
    wandb_logger = WandbLogger(
        project=config.training.logging.project,
        name=run_name,
        config=config.to_dict(), # Log the initial static config
        # entity="your_wandb_entity", # Optional: specify WandB entity
        # log_model="all", # Optional: Log checkpoints to WandB
    )

    # 4. Calculate Per-Device Batch Size and Grad Accumulation Steps
    # This step is needed *before* creating the dataloader
    logical_batch = config.training.logical_batch_size
    max_per_device = config.data.max_per_device_batch_size
    per_device_batch_size = min(max_per_device, logical_batch)
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    # Store these in the config object for access later (e.g., by Trainer)
    config.training.per_device_batch_size = per_device_batch_size
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    print(f"Batch parameters: Per-Device Size={per_device_batch_size}, Grad Accum Steps={gradient_accumulation_steps}")


    # 5. Get Dataloaders
    print("Creating dataloaders...")
    # Add transform logic back here if needed (e.g., random time offset)
    # transform_fn = add_random_time_offset(args.random_time_offset) if args.random_time_offset else default_transform
    train_loader, val_loader = get_dataloaders(
        config,
        dataset_type=args.dataset_type,
        transform=default_transform, # Use basic transform for now
        target_transform=default_target_transform
    )

    # 6. Calculate Runtime Training Parameters (Total Steps, Final pct_start)
    print("Calculating runtime parameters (steps, etc.)...")
    # Estimate train_loader_len (IMPORTANT: Adjust if using sample limits or different dataset structures)
    if config.data.train_events is None:
         warnings.warn("config.data.train_events is not set. Cannot accurately estimate total steps.")
         # Default to a large number or raise error if scheduler requires total_steps
         num_batches_per_epoch = 10000 # Fallback estimate
    else:
         num_batches_per_epoch = math.ceil(config.data.train_events / config.training.per_device_batch_size)

    print(f"Estimated batches per epoch: {num_batches_per_epoch}")
    # Calculate and store total_steps, updated pct_start etc. in the config object
    config.calculate_runtime_params(num_batches_per_epoch) # Uses stored per_device_batch_size

    # Log the updated config (including calculated steps) to WandB
    # wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True) # Update if needed


    # 7. Initialize Model
    print(f"Initializing PolarBertModel...")
    model = PolarBertModel(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Parameters: {param_count:,}')


    # 8. Setup Callbacks
    print("Setting up callbacks...")
    callbacks = setup_callbacks(config, run_name)


    # 9. Setup Trainer
    print("Setting up PyTorch Lightning Trainer...")
    trainer = Trainer(
        accelerator='gpu', # Assumes GPU
        devices=config.training.gpus,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        gradient_clip_val=config.training.gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=config.training.val_check_interval,
        accumulate_grad_batches=config.training.gradient_accumulation_steps, # Use calculated value
        # deterministic=True, # Optional: for reproducibility, might slow down
        # benchmark=True, # Optional: might speed up if input sizes are constant
    )

    # 10. Start Training
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTraining finished.")


if __name__ == '__main__':
    main()