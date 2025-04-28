# ---- te_finetuning.py (Modified for Mean Pooling & Fixed total_steps Error) ----
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Optional
import math
import os
import warnings

# --- Import project modules ---
try:
    from polarbert.config import PolarBertConfig
    from polarbert.time_embed_polarbert import PolarBertModel # The main pre-trained model architecture
    from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors
    # Import get_dataloaders and setup_callbacks from pretraining script (or move to shared utils)
    from polarbert.te_pretraining import get_dataloaders, setup_callbacks, default_transform
    # Add specific target transforms
    # Note: Importing these from finetuning might cause circular dependencies if not careful
    # It's better if target transforms are defined separately or within the dataset/utils
    from polarbert.finetuning import DirectionalHead as OldDirectionalHead # For target transforms only
    from polarbert.finetuning import EnergyRegressionHead as OldEnergyRegressionHead # For target transforms only
except ImportError as e:
    print(f"Error importing project modules: {e}")
    raise e

# --- Fine-tuning Lightning Module ---

class PolarBertFinetuner(pl.LightningModule):
    """
    LightningModule for fine-tuning a pre-trained PolarBertModel.
    Loads backbone weights and adds a task-specific prediction head.
    Uses MEAN POOLING of sequence embeddings for prediction input.
    """
    def __init__(self, config: PolarBertConfig, pretrained_checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config # Keep config reference if needed, but rely on hparams mainly
        # Save hyperparams, make sure config includes any necessary info for this class
        # e.g., task, freeze_backbone, relevant head configs, AND calculated total_steps
        hparams_to_save = config.to_dict()
        # Add the pretrained_checkpoint_path argument to hparams for traceability
        hparams_to_save['training']['pretrained_checkpoint_path'] = pretrained_checkpoint_path
        self.save_hyperparameters(hparams_to_save)

        # 1. Instantiate the Backbone (full model structure)
        # Use masking=False as we don't need masking logic during fine-tuning inference
        # The underlying embedding layer should handle its state (train/eval)
        self.backbone = PolarBertModel(config) # Contains embedding, blocks, final_norm

        # 2. Load Pre-trained Weights into Backbone
        # Use the passed argument, not hparams, for the path to load from
        if pretrained_checkpoint_path and pretrained_checkpoint_path.lower() != 'new':
            print(f"Loading backbone weights from: {pretrained_checkpoint_path}")
            try:
                # Set weights_only=True for security unless you trust the source completely
                # For typical PL checkpoints, weights_only=False is needed to load the structure/hparams
                # Add the warning suppression if you understand the risk or manage checkpoints carefully
                # warnings.filterwarnings("ignore", message=".*weights_only=False is deprecated.*")
                checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu') # weights_only=False (default)
                state_dict = checkpoint.get('state_dict', checkpoint)

                # Clean keys (remove 'model.' prefix if PL added it during pre-training save)
                # And potentially handle 'backbone.' if the finetuner itself was saved
                cleaned_state_dict = {}
                prefixes_to_remove = ['model.', 'backbone.'] # Check for both common prefixes
                for k, v in state_dict.items():
                    key_modified = False
                    for prefix in prefixes_to_remove:
                        if k.startswith(prefix):
                            cleaned_state_dict[k[len(prefix):]] = v
                            key_modified = True
                            break
                    if not key_modified:
                         # TEMPORARY FIX for loading old state dict without prefix (REMOVE LATER)
                         if k.startswith('embedding.') or k.startswith('transformer_blocks.') or k.startswith('final_norm.'):
                             cleaned_state_dict[k] = v
                         # END TEMPORARY FIX

                # Load into the backbone PART of the model
                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)

                print("Backbone weights loaded.")
                # Filter expected missing/unexpected keys more carefully
                expected_missing = {'dom_head.weight', 'dom_head.bias', 'charge_head.weight', 'charge_head.bias'}
                filtered_missing = [k for k in missing_keys if k not in expected_missing]
                # Unexpected keys often include optimizer states, lr schedulers etc. from the checkpoint
                filtered_unexpected = [k for k in unexpected_keys if not k.startswith('optimizer.') and not k.startswith('lr_scheduler') and not k.startswith('_forward_module')]

                if filtered_missing:
                    print("  Warning: Missing unexpected keys in backbone:", filtered_missing)
                if filtered_unexpected:
                    print("  Warning: Unexpected keys found and ignored:", filtered_unexpected)

            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Proceeding with untrained backbone.")
                # raise e # Option to raise error instead of continuing
        else:
            print("No pretrained checkpoint provided or 'new' specified. Training backbone from scratch.")


        # 3. Define Task-Specific Prediction Head
        # Use hparams which includes potential command-line overrides AND calculated steps
        task = self.hparams.training['task'].lower() # Access task from saved hparams
        backbone_embed_dim = self.hparams.model['embedding_dim'] # Access from hparams
        head_hidden_size = None # Initialize

        # Safely access head configuration based on task from hparams
        if task == 'direction':
            head_config_dict = self.hparams.model.get('directional_head', None)
            if head_config_dict is None:
                 raise ValueError(f"Task is '{task}', but 'directional_head' section is missing in the model configuration saved in hparams.")
            head_hidden_size = head_config_dict.get('hidden_size', None)
            if head_hidden_size is None:
                 raise ValueError(f"Task is '{task}', but 'hidden_size' key is missing within the 'directional_head' configuration saved in hparams.")
            head_hidden_size = int(head_hidden_size)

            self.prediction_head = nn.Sequential(
                nn.Linear(backbone_embed_dim, head_hidden_size),
                nn.ReLU(),
                # Add another hidden layer
                nn.Linear(head_hidden_size, head_hidden_size // 2), # Example: reduce size
                nn.ReLU(),
                # Final output layer
                nn.Linear(head_hidden_size // 2, 3)
            )
            self.loss_fn = self.angular_distance_loss

        elif task == 'energy':
            head_config_dict = self.hparams.model.get('energy_head', None)
            if head_config_dict is None:
                raise ValueError(f"Task is '{task}', but 'energy_head' section is missing in the model configuration saved in hparams.")
            head_hidden_size = head_config_dict.get('hidden_size', None)
            if head_hidden_size is None:
                raise ValueError(f"Task is '{task}', but 'hidden_size' key is missing within the 'energy_head' configuration saved in hparams.")
            head_hidden_size = int(head_hidden_size)

            self.prediction_head = nn.Sequential(
                nn.Linear(backbone_embed_dim, head_hidden_size),
                nn.ReLU(),
                nn.Linear(head_hidden_size, 1) # Predict scalar energy
            )
            self.loss_fn = self.mse_loss
        else:
            raise ValueError(f"Unsupported fine-tuning task: {task}")

        print(f"Initialized head for task: {task} with hidden size: {head_hidden_size}")


        # 4. Handle Backbone Freezing
        # Access freeze_backbone from hparams which includes potential command-line overrides
        if self.hparams.training.get('freeze_backbone', False): # Use .get for safety
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Ensure prediction head is trainable (it should be by default)
            for param in self.prediction_head.parameters():
                 param.requires_grad = True


    def forward(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Any]) -> torch.Tensor:
        """
        Forward pass for fine-tuning. Extracts MEAN POOLED embedding from backbone sequence output.
        """
        (x, l), _ = batch

        # 1. Pass through embedding, transformer blocks, and final norm of the backbone
        # The backbone's embedding layer should handle its masking state internally (eval mode usually means no masking)
        hidden_states, final_padding_mask, _ = self.backbone.embedding((x, l)) # output_mask not needed
        attn_key_padding_mask = final_padding_mask

        # --- Optional: Add RoPE or Positional Embeddings if configured ---
        # This should typically be handled within the backbone structure if enabled via config

        for block in self.backbone.transformer_blocks:
            hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)
        hidden_states = self.backbone.final_norm(hidden_states)
        # hidden_states shape: (B, L_full, E), L_full = 1 (CLS) + L_orig

        # --- Mean Pooling Calculation ---
        sequence_embeds = hidden_states[:, 1:, :] # (B, L_orig, E) - Exclude CLS token
        seq_padding_mask = final_padding_mask[:, 1:] # (B, L_orig) - Get mask for sequence part
        valid_token_mask = ~seq_padding_mask # Shape: (B, L_orig), True for valid tokens

        # Expand mask for broadcasting: (B, L_orig) -> (B, L_orig, 1) -> (B, L_orig, E)
        valid_token_mask_expanded = valid_token_mask.unsqueeze(-1).expand_as(sequence_embeds).float()

        # Zero out embeddings corresponding to padding tokens
        masked_sequence_embeds = sequence_embeds * valid_token_mask_expanded

        # Sum embeddings across the sequence dimension
        summed_embeds = masked_sequence_embeds.sum(dim=1) # Shape: (B, E)

        # Count the number of valid (non-padding) tokens per sequence
        num_valid_tokens = valid_token_mask.sum(dim=1, keepdim=True).float() + 1e-6 # Shape: (B, 1)

        # Calculate the mean pooled embedding
        mean_pooled_embed = summed_embeds / num_valid_tokens # Shape: (B, E)
        # --- End Mean Pooling ---

        # 3. Pass the mean pooled embedding through the task-specific head
        predictions = self.prediction_head(mean_pooled_embed)
        return predictions # (B, 3) for direction, (B, 1) for energy

    def angular_distance_loss(self, y_pred_vectors, y_target_angles):
        """ Calculates angular distance loss """
        y_truth_vectors = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1])
        norm = torch.linalg.vector_norm(y_pred_vectors, dim=1, keepdim=True)
        y_pred_unit_vectors = y_pred_vectors / (norm + 1e-8)
        loss = angular_dist_score_unit_vectors(y_truth_vectors, y_pred_unit_vectors, epsilon=1e-4)
        return loss

    def mse_loss(self, y_pred_energy, y_target_log_energy):
         """ Calculates MSE loss for energy regression """
         return F.mse_loss(y_pred_energy.squeeze(-1), y_target_log_energy)

    def shared_step(self, batch):
        (x, l), y_data = batch
        if y_data is None:
             raise ValueError("Targets (y_data) are missing in the batch. Ensure dataset provides labels for fine-tuning.")
        y_target = y_data[0]
        predictions = self.forward(batch)
        loss = self.loss_fn(predictions, y_target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """ Set up optimizer and scheduler for fine-tuning. """
        freeze_backbone = self.hparams.training.get('freeze_backbone', False)

        if freeze_backbone:
             print("Configuring optimizer only for prediction head.")
             parameters_to_optimize = self.prediction_head.parameters()
        else:
             print("Configuring optimizer for full model (backbone + head).")
             parameters_to_optimize = self.parameters()

        # --- Create Optimizer ---
        optimizer_name = self.hparams.training['optimizer'].lower()
        lr = self.hparams.training['max_lr']
        weight_decay = self.hparams.training['weight_decay']
        print(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")

        optimizer_kwargs = {
            'lr': lr,
            'betas': (self.hparams.training['adam_beta1'], self.hparams.training['adam_beta2']),
            'eps': self.hparams.training['adam_eps'],
            'weight_decay': weight_decay,
            'amsgrad': self.hparams.training['amsgrad']
        }

        if optimizer_name == 'adamw':
             optimizer = torch.optim.AdamW(parameters_to_optimize, **optimizer_kwargs)
        else:
             raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # --- Create Scheduler ---
        scheduler_name = self.hparams.training['lr_scheduler'].lower()
        if scheduler_name == 'onecycle':
            # Access total_steps FROM HPARAMS where it should now exist
            if 'total_steps' not in self.hparams.training or self.hparams.training['total_steps'] is None:
                 # This case should ideally not happen if main() is structured correctly
                 # Maybe fallback to trainer estimation if available as a last resort?
                 if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches is not None:
                     total_steps = self.trainer.estimated_stepping_batches
                     warnings.warn(f"total_steps not found in hparams, using trainer's estimate: {total_steps}")
                 else:
                     raise ValueError("total_steps not found in hparams and could not be estimated from trainer.")
            else:
                  total_steps = self.hparams.training['total_steps']

            print(f"Scheduler: OneCycleLR with total_steps={total_steps}, "
                  f"pct_start={self.hparams.training['pct_start']}")

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.training['max_lr'], # Use max_lr from hparams
                total_steps=total_steps,
                pct_start=self.hparams.training['pct_start'], # Use pct_start from hparams
                div_factor=self.hparams.training['div_factor'],
                final_div_factor=self.hparams.training['final_div_factor'],
                anneal_strategy='cos'
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        elif scheduler_name in ['none', None, 'constant']:
            print(f"Scheduler: None (constant LR = {lr})")
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")


# --- Main Fine-tuning Script Logic ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PolarBERT Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the FINE-TUNING configuration YAML file.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the PRE-TRAINED model checkpoint (.ckpt). Use 'new' to train from scratch.")
    parser.add_argument('--task', type=str, choices=['direction', 'energy'], help="Fine-tuning task (overrides config if specified).")
    parser.add_argument('--dataset_type', type=str, choices=['kaggle', 'prometheus'], required=True)
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, help="Freeze weights of the pre-trained backbone (overrides config).")
    parser.add_argument('--name', type=str, default=None, help="Custom name for the training run.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID for naming.")
    args = parser.parse_args()

    # 1. Load Fine-tuning Configuration
    config = PolarBertConfig.from_yaml(args.config)

    # --- Sanity Checks & Overrides from Command Line ---
    if args.task and args.task != config.training.task:
        warnings.warn(f"Task mismatch: CLI '--task {args.task}' overrides config task '{config.training.task}'.")
        config.training.task = args.task
    elif not config.training.task:
         raise ValueError("Fine-tuning 'task' must be defined in the config file or via the --task argument.")
    current_task = config.training.task

    if args.freeze_backbone is not None:
        if args.freeze_backbone != config.training.freeze_backbone:
             warnings.warn(f"Freeze backbone mismatch: CLI '--{'no-' if not args.freeze_backbone else ''}freeze-backbone' overrides config value '{config.training.freeze_backbone}'.")
             config.training.freeze_backbone = args.freeze_backbone

    config.training.pretrained_checkpoint_path = args.checkpoint_path # Store for record keeping

    print("--- Fine-tuning Configuration ---")
    print(f"Task: {current_task}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Pretrained Checkpoint: {args.checkpoint_path}")
    print(f"Freeze Backbone: {config.training.freeze_backbone}")
    print("---------------------------------")

    # 2. Determine Run Name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    base_name = args.name or f"finetune_{current_task}_{args.dataset_type}"
    run_name = f"{base_name}_{suffix}"
    print(f"Starting run: {run_name}")

    # 3. Setup Logging (WandB)
    print("Setting up WandB logger...")
    wandb_logger = WandbLogger(
        project=config.training.logging.project,
        name=run_name,
        config=config.to_dict(), # Log initial config before runtime calcs
    )

    # 4. Calculate Per-Device Batch Size and Grad Accumulation Steps
    # Needed before dataloader AND before runtime calc
    logical_batch = config.training.logical_batch_size
    max_per_device = config.data.max_per_device_batch_size
    per_device_batch_size = min(max_per_device, logical_batch)
    if per_device_batch_size == 0:
         raise ValueError("Calculated per_device_batch_size is zero. Check config.")
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    config.training.per_device_batch_size = per_device_batch_size
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    print(f"Batch parameters: Per-Device Size={per_device_batch_size}, Grad Accum Steps={gradient_accumulation_steps}")

    # ------------------------------------------------------------------
    # STEP 5: Get Dataloaders (Moved BEFORE Runtime Param Calculation)
    # ------------------------------------------------------------------
    print("Creating dataloaders...")
    if current_task == 'direction':
         try: target_transform = OldDirectionalHead.target_transform_kaggle if args.dataset_type == 'kaggle' else OldDirectionalHead.target_transform_prometheus
         except AttributeError as e: raise ImportError(f"Error getting target transform: {e}") from e
    elif current_task == 'energy':
         if args.dataset_type == 'kaggle': raise ValueError("Energy task not supported for Kaggle dataset.")
         try: target_transform = OldEnergyRegressionHead.target_transform_prometheus
         except AttributeError as e: raise ImportError(f"Error getting target transform: {e}") from e
    else: raise ValueError(f"Unknown task '{current_task}'")

    train_loader, val_loader = get_dataloaders(
        config,
        dataset_type=args.dataset_type,
        transform=default_transform,
        target_transform=target_transform
    )

    # ----------------------------------------------------------------------
    # STEP 6: Calculate Runtime Training Parameters (Moved BEFORE Model Init)
    # ----------------------------------------------------------------------
    print("Calculating runtime parameters (total steps, final pct_start)...")
    # Estimate train_loader_len
    if hasattr(train_loader.dataset, '__len__') and len(train_loader.dataset) > 0:
         num_batches_per_epoch = len(train_loader) # Dataloader length is num batches
         print(f"Using dataloader length for batches per epoch: {num_batches_per_epoch}")
    elif config.data.train_events is not None and config.training.per_device_batch_size > 0:
         num_batches_per_epoch = math.ceil(config.data.train_events / config.training.per_device_batch_size)
         print(f"Estimating batches per epoch from train_events: {num_batches_per_epoch}")
    else:
         warnings.warn("Cannot accurately estimate train_loader length. Using fallback estimate of 1000 batches/epoch.")
         num_batches_per_epoch = 1000 # Fallback estimate

    # Calculate and store total_steps, updated pct_start etc. in the config object
    # This MODIFIES the config object IN-PLACE
    config.calculate_runtime_params(num_batches_per_epoch)
    print(f"Calculated total_steps: {config.training.total_steps}")
    print(f"Final pct_start for scheduler: {config.training.pct_start:.4f}")

    # Log the final, updated config (including calculated steps) to WandB
    if wandb_logger.experiment:
        try:
             wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True)
             print("Updated WandB config with runtime parameters.")
        except Exception as e:
             warnings.warn(f"Could not update WandB config after runtime calculations: {e}")

    # --------------------------------------------------------------------
    # STEP 7: Initialize Fine-tuning Model (Moved AFTER Runtime Calc)
    # --------------------------------------------------------------------
    print(f"Initializing PolarBertFinetuner...")
    # The config object now contains calculated total_steps, pct_start etc.
    # Pass the command-line checkpoint path for loading, config for structure/hparams
    model = PolarBertFinetuner(config, pretrained_checkpoint_path=args.checkpoint_path)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Parameters: {param_count:,}')

    # --------------------------------------------------------------------
    # STEP 8: Setup Callbacks
    # --------------------------------------------------------------------
    print("Setting up callbacks...")
    callbacks = setup_callbacks(config, run_name) # Uses config for checkpoint settings

    # --------------------------------------------------------------------
    # STEP 9: Setup Trainer
    # --------------------------------------------------------------------
    print("Setting up PyTorch Lightning Trainer...")
    trainer = Trainer(
        accelerator='gpu',
        devices=config.training.gpus,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        # max_steps=config.training.total_steps, # Optional: Use total_steps calculated
        gradient_clip_val=config.training.gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=config.training.val_check_interval,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        # log_every_n_steps=50 # Optional
    )

    # --------------------------------------------------------------------
    # STEP 10: Start Fine-tuning
    # --------------------------------------------------------------------
    print("\nStarting fine-tuning...")
    trainer.fit(model, train_loader, val_loader) # Pass ckpt_path here if resuming fine-tuning run

    print("\nFine-tuning finished.")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high') # Or 'medium'
    main()