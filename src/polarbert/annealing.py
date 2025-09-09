# filepath: /groups/pheno/inar/PolarBERT/src/polarbert/annealing.py
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

import argparse
from datetime import datetime
import logging
import os
import glob
import re
import csv
from pathlib import Path

from polarbert.utils.config import load_and_process_config
from polarbert.utils.data import (
    get_dataloaders, 
    add_random_time_offset, 
    default_transform
)
from polarbert.utils.training import update_training_steps, compute_batch_params
from polarbert.utils.callbacks import setup_callbacks
from polarbert.utils.sweep_params import update_config_for_wandb_sweep
from polarbert.utils.activation_logging import ActivationLoggingCallback
from polarbert.utils.custom_lr_scheduler import TrapezoidalLR

from polarbert.flash_model import FlashTransformer
from polarbert.swiglu_model import SwiGLUTransformer
from polarbert.base_model import SimpleTransformer

MODEL_CLASSES = {
    'flash': (FlashTransformer, "Flash Transformer"),
    'swiglu': (SwiGLUTransformer, "SwiGLU Transformer"),
    'base': (SimpleTransformer, "Base Transformer")
}





def extract_step_from_checkpoint(checkpoint_path):
    """Extract the step number from checkpoint filename like 'epoch=00-step=015868.ckpt'"""
    filename = os.path.basename(checkpoint_path)
    match = re.search(r'step=(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract step number from checkpoint: {checkpoint_path}")


def get_checkpoint_files(checkpoint_dir):
    """Get all checkpoint files and sort them by step number"""
    pattern = os.path.join(checkpoint_dir, "epoch=*-step=*.ckpt")
    checkpoint_files = glob.glob(pattern)
    
    # Sort by step number
    checkpoint_files.sort(key=extract_step_from_checkpoint)
    
    return checkpoint_files


def create_annealing_scheduler(optimizer, num_anneal_steps, current_lr):
    """Create a scheduler that anneals from current_lr to 0 over num_anneal_steps"""
    class AnnealingLR(torch.optim.lr_scheduler.LRScheduler):
        def __init__(self, optimizer, num_steps, initial_lr):
            self.num_steps = num_steps
            self.initial_lr = initial_lr
            super().__init__(optimizer)
        
        def get_lr(self):
            if self.last_epoch >= self.num_steps:
                return [0.0 for _ in self.base_lrs]
            
            # Linear decay from initial_lr to 0
            decay_factor = 1.0 - (self.last_epoch / self.num_steps)
            return [self.initial_lr * decay_factor for _ in self.base_lrs]
    
    return AnnealingLR(optimizer, num_anneal_steps, current_lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/basic_transformer.yaml')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoints to anneal')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save annealed models')
    parser.add_argument('--num_anneal', type=int, default=1000,
                        help='Number of annealing steps')
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='flash')
    parser.add_argument("--dataset_type", type=str, choices=['kaggle', 'prometheus'], default='kaggle')
    parser.add_argument("--watch", action='store_true')
    parser.add_argument("--log_frequency", type=int, default=100, 
                        help="Logging frequency (in steps) for W&B and activation logging")
    args = parser.parse_args()

    # Load and process config
    config = load_and_process_config(args.config)
    
    # Compute and update batch parameters (same as pretraining.py)
    batch_params = compute_batch_params(config)
    config['training'].update(batch_params)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all checkpoint files
    checkpoint_files = get_checkpoint_files(args.checkpoint_dir)
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {args.checkpoint_dir}")
    
    print(f"Found {len(checkpoint_files)} checkpoints to anneal")
    
    # Setup transform for data loaders (will create fresh loaders for each checkpoint)
    random_time_offset_std = config['training'].get('random_time_offset')
    if random_time_offset_std is not None:
        logging.info(f"Applying random time offset with std: {random_time_offset_std}")
        transform = add_random_time_offset(random_time_offset_std)
    else:
        transform = default_transform
    
    # Prepare CSV file for results
    csv_path = os.path.join(args.output_dir, f'annealing_results_{args.num_anneal}steps.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['number_of_steps', 'val_dom_loss', 'val_total_loss', 'val_charge_loss'])
    
    try:
        # Process each checkpoint
        for checkpoint_path in checkpoint_files:
            checkpoint_step = extract_step_from_checkpoint(checkpoint_path)
            final_steps = checkpoint_step + args.num_anneal
            
            print(f"Processing checkpoint: {os.path.basename(checkpoint_path)} (step {checkpoint_step})")
            
            # Setup model name for this annealing run
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            model_name = f"annealed_{checkpoint_name}_{args.num_anneal}steps"
            
            # Ensure any previous W&B run is finished
            if wandb.run is not None:
                wandb.finish()
            
            # Setup W&B logging with unique run
            torch.set_float32_matmul_precision('high')
            wandb_logger = WandbLogger(
                project=config['training']['project'] + '_annealing',
                name=model_name,
                config={**config, 'annealing': {'num_steps': args.num_anneal, 'original_checkpoint': checkpoint_path}}
            )
            
            # Update config for this specific run
            run_config = config.copy()
            run_config['training']['total_steps'] = args.num_anneal
            run_config['training']['lr_scheduler'] = 'constant'
            run_config['training']['initial_lr'] = config['training']['max_lr']
            run_config['model']['model_name'] = model_name
            
            # Setup callbacks - only save final checkpoint
            checkpoint_callback_config = {
                'dirpath': os.path.join(args.output_dir, model_name),
                'save_top_k': 0,  # Don't save intermediate checkpoints
                'monitor': 'val/full_loss',
                'mode': 'min',
                'save_last': True,
                'save_final': True
            }
            run_config['training']['checkpoint'] = checkpoint_callback_config
            
            # Initialize model with updated config
            model_class, model_name_desc = MODEL_CLASSES[args.model_type]
            model = model_class(run_config)
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            
            print(f"Loaded checkpoint from step {checkpoint_step}")
            
            callbacks = setup_callbacks(run_config, model_name)
            
            # Add activation logging if watch is enabled
            if args.watch:
                activation_callback = ActivationLoggingCallback(
                    log_frequency=args.log_frequency,
                    csv_path=args.save_activations_to,
                )
                callbacks.append(activation_callback)
            
            # Create fresh dataloaders for this checkpoint to avoid multiprocessing queue issues
            train_loader, val_loader = get_dataloaders(run_config, dataset_type=args.dataset_type, transform=transform)
            
            # Create limited dataloader wrapper to fix progress bar
            class LimitedStepsWrapper:
                def __init__(self, dataloader, max_steps):
                    self.dataloader = dataloader
                    self.max_steps = max_steps
                
                def __iter__(self):
                    step_count = 0
                    for batch in self.dataloader:
                        if step_count >= self.max_steps:
                            break
                        yield batch
                        step_count += 1
                
                def __len__(self):
                    return self.max_steps
                
                def __getattr__(self, name):
                    return getattr(self.dataloader, name)
            
            limited_train_loader = LimitedStepsWrapper(train_loader, args.num_anneal)
            
            # Setup trainer for annealing with proper progress tracking
            trainer = Trainer(
                max_steps=args.num_anneal,
                max_epochs=-1,
                callbacks=callbacks,
                accelerator='gpu',
                devices=config['training']['gpus'],
                precision=config['training']['precision'],
                gradient_clip_val=config['training']['gradient_clip_val'],
                logger=wandb_logger,
                val_check_interval=min(100, args.num_anneal // 2),
            )
            
            # Load optimizer and scheduler states from checkpoint
            optimizer_states = checkpoint.get('optimizer_states', [])
            lr_scheduler_states = checkpoint.get('lr_schedulers', [])
            
            # Get the current learning rate from checkpoint or use max_lr as fallback
            if optimizer_states and len(optimizer_states) > 0:
                # Extract learning rate from the optimizer state
                current_lr = optimizer_states[0]['param_groups'][0]['lr']
                print("Loaded optimizer state from checkpoint")
            else:
                current_lr = config['training']['max_lr']
            
            # Override the model's configure_optimizers to use annealing scheduler
            original_configure_optimizers = model.configure_optimizers
            def annealing_configure_optimizers():
                optimizer_config = original_configure_optimizers()
                if isinstance(optimizer_config, tuple):
                    optimizer, _ = optimizer_config  # Ignore original scheduler
                else:
                    optimizer = optimizer_config
                
                # Load optimizer state if available
                if optimizer_states:
                    optimizer.load_state_dict(optimizer_states[0])
                
                # Create annealing scheduler
                annealing_scheduler = create_annealing_scheduler(optimizer, args.num_anneal, current_lr)
                
                return [optimizer], [{"scheduler": annealing_scheduler, "interval": "step", "frequency": 1}]
            
            model.configure_optimizers = annealing_configure_optimizers
            
            print(f"Current LR: {current_lr}, will anneal to 0 over {args.num_anneal} steps")
            print(f"Trainer max_steps: {args.num_anneal}, max_epochs: -1")
            
            # Run annealing
            print(f"Starting annealing training for {args.num_anneal} steps...")
            trainer.fit(model, limited_train_loader, val_loader)
            print(f"Training completed after {args.num_anneal} steps")
            
            # Get final validation metrics
            val_results = trainer.validate(model, val_loader, verbose=False)
            if val_results:
                val_metrics = val_results[0]
                val_dom_loss = val_metrics.get('val/dom_loss', 0.0)
                val_total_loss = val_metrics.get('val/full_loss', 0.0)
                val_charge_loss = val_metrics.get('val/charge_loss', 0.0)
            else:
                val_dom_loss = val_total_loss = val_charge_loss = 0.0
            
            # Write results to CSV
            csv_writer.writerow([final_steps, val_dom_loss, val_total_loss, val_charge_loss])
            csv_file.flush()
            
            print(f"Completed annealing for {checkpoint_name}")
            print(f"Final validation losses - DOM: {val_dom_loss:.6f}, Total: {val_total_loss:.6f}, Charge: {val_charge_loss:.6f}")
            
                        # Clean up W&B
            if args.watch:
                wandb_logger.experiment.unwatch(model)
            
            # Finish the W&B run to avoid conflicts with the next run
            wandb_logger.experiment.finish()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()

    finally:
        csv_file.close()
    
    print(f"Annealing completed! Results saved to {csv_path}")


if __name__ == '__main__':
    main()