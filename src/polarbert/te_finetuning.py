# ---- te_finetuning.py (Modified for Pooling Choice + DOM Loss) ----
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
    from polarbert.time_embed_polarbert import PolarBertModel # Backbone
    from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors
    from polarbert.te_pretraining import setup_callbacks, default_transform # Keep setup_callbacks, default_transform
    from polarbert.dataloader_utils import (
        get_dataloaders,
        target_transform_prometheus,
        target_transform_kaggle,
        default_transform # Already imported from te_pretraining, but explicit is fine
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    raise e

# --- Fine-tuning Lightning Module ---

class PolarBertFinetuner(pl.LightningModule):
    """
    LightningModule for fine-tuning PolarBertModel using MULTI-TASK learning (Dir + DOM).
    Allows choosing 'mean' or 'cls' pooling for the directional head input.
    """
    def __init__(self, config: PolarBertConfig, pretrained_checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config # Keep config reference if needed
        hparams_to_save = config.to_dict()
        # Add the specific checkpoint path used for this instance to hparams
        hparams_to_save['training']['pretrained_checkpoint_path_runtime'] = pretrained_checkpoint_path
        self.save_hyperparameters(hparams_to_save)

        # Store pooling mode and lambda_dom from hparams (which now have defaults from config.py)
        self.pooling_mode = self.hparams.training.get('directional_pooling_mode', 'mean') # Default just in case
        self.lambda_dom = self.hparams.training['lambda_dom'] # Should exist due to config.py

        print(f"Directional head pooling mode: {self.pooling_mode}")
        print(f"Using DOM loss weight (lambda_dom): {self.lambda_dom}")


        # 1. Instantiate the Backbone
        self.backbone = PolarBertModel(config)

        # 2. Load Pre-trained Weights into Backbone
        if pretrained_checkpoint_path and pretrained_checkpoint_path.lower() != 'new':
            print(f"Loading backbone weights from: {pretrained_checkpoint_path}")
            # --- (Checkpoint loading logic remains the same as previous version) ---
            try:
                checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                cleaned_state_dict = {}
                prefixes_to_remove = ['model.', 'backbone.']
                for k, v in state_dict.items():
                    key_modified = False
                    for prefix in prefixes_to_remove:
                        if k.startswith(prefix):
                            cleaned_state_dict[k[len(prefix):]] = v
                            key_modified = True
                            break
                    if not key_modified:
                         cleaned_state_dict[k] = v # Keep unprefixed keys

                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
                print("Backbone weights loaded.")
                filtered_missing = [k for k in missing_keys]
                filtered_unexpected = [k for k in unexpected_keys if not k.startswith('optimizer.') and not k.startswith('lr_scheduler') and not k.startswith('_forward_module')]
                if filtered_missing: print("  Warning: Missing keys in backbone:", filtered_missing)
                if filtered_unexpected: print("  Warning: Unexpected keys found and ignored:", filtered_unexpected)
            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Proceeding with untrained backbone.")
        else:
            print("No pretrained checkpoint provided or 'new' specified. Training backbone from scratch.")


        # 3. Define the NEW Task-Specific Prediction Head (Directional Head)
        task = self.hparams.training['task'].lower()
        if task != 'direction':
             raise ValueError(f"This multi-task setup currently requires primary task 'direction', got '{task}'")

        backbone_embed_dim = self.hparams.model['embedding_dim']
        head_config_dict = self.hparams.model.get('directional_head', {}) # Use {} default
        head_hidden_size = int(head_config_dict.get('hidden_size', 1024)) # Default hidden size

        self.directional_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 3)
        )
        print(f"Initialized NEW directional head with hidden size: {head_hidden_size}")

        # 4. Handle Backbone Freezing
        if self.hparams.training.get('freeze_backbone', False):
            print("Freezing backbone parameters (including pre-trained DOM/Charge heads).")
            for param in self.backbone.parameters(): param.requires_grad = False
            for param in self.directional_head.parameters(): param.requires_grad = True
        else:
             print("Training full model (backbone + DOM head + new Directional head).")


    def forward(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Any]) \
            -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-task fine-tuning (Dir + DOM).
        Uses pooling_mode specified in hparams ('mean' or 'cls') for directional head.
        Returns:
            dom_logits (Tensor): Logits from the backbone's DOM head.
            dir_pred (Tensor): Predictions from the new directional head.
            output_mask (Tensor | None): Mask for calculating DOM loss.
        """
        (x, l), _ = batch

        # 1. Run embedding layer
        hidden_states, final_padding_mask, output_mask = self.backbone.embedding((x, l))
        attn_key_padding_mask = final_padding_mask

        # 2. Pass through transformer blocks
        for block in self.backbone.transformer_blocks:
            hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)

        # 3. Final Normalization
        hidden_states = self.backbone.final_norm(hidden_states)
        # hidden_states shape: (B, L_full, E)

        # 4. Get predictions from BOTH heads

        # --- DOM Head Prediction (always uses sequence embeddings) ---
        sequence_embeds = hidden_states[:, 1:, :] # (B, L_orig, E)
        dom_logits = self.backbone.dom_head(sequence_embeds) # (B, L_orig, num_dom_classes)

        # --- Directional Head Prediction (Input depends on pooling_mode) ---
        if self.pooling_mode == 'cls':
            dir_head_input = hidden_states[:, 0, :] # Use CLS token embedding (B, E)
        elif self.pooling_mode == 'mean':
            # --- Mean Pooling Calculation ---
            seq_padding_mask = final_padding_mask[:, 1:] # (B, L_orig)
            valid_token_mask = ~seq_padding_mask # (B, L_orig)
            valid_token_mask_expanded = valid_token_mask.unsqueeze(-1).expand_as(sequence_embeds).float()
            masked_sequence_embeds = sequence_embeds * valid_token_mask_expanded
            summed_embeds = masked_sequence_embeds.sum(dim=1) # (B, E)
            num_valid_tokens = valid_token_mask.sum(dim=1, keepdim=True).float() + 1e-6 # (B, 1)
            dir_head_input = summed_embeds / num_valid_tokens # Use mean pooled embedding (B, E)
            # --- End Mean Pooling ---
        else:
            # This should be caught by config validation, but belts and suspenders
            raise ValueError(f"Invalid pooling_mode: {self.pooling_mode}")

        # Use the selected input for the directional head
        dir_pred = self.directional_head(dir_head_input) # (B, 3)

        return dom_logits, dir_pred, output_mask

    # --- Loss functions (angular_distance_loss, calculate_dom_loss) remain the same ---
    def angular_distance_loss(self, y_pred_vectors, y_target_angles):
        # ... (previous implementation) ...
        y_truth_vectors = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1])
        norm = torch.linalg.vector_norm(y_pred_vectors, dim=1, keepdim=True)
        y_pred_unit_vectors = y_pred_vectors / (norm + 1e-8)
        loss = angular_dist_score_unit_vectors(y_truth_vectors, y_pred_unit_vectors, epsilon=1e-4)
        return loss

    def calculate_dom_loss(self, dom_logits, true_dom_ids, output_mask):
        # ... (previous implementation) ...
        dom_loss = torch.tensor(0.0, device=dom_logits.device)
        if output_mask is not None and output_mask.sum() > 0:
             dom_targets = true_dom_ids - 1
             masked_logits = dom_logits[output_mask]
             masked_targets = dom_targets[output_mask]
             # Check if logits dimension matches target range implicitly
             # num_classes = dom_logits.shape[-1] # Get num_classes from logits
             dom_loss = F.cross_entropy(masked_logits, masked_targets, ignore_index=-1)
        return dom_loss

    def shared_step(self, batch):
        (x, l), y_data = batch
        if y_data is None: raise ValueError("Targets (y_data) are missing.")

        y_target_angles = y_data[0]
        true_dom_ids = x[:, :, 3].long()

        dom_logits, dir_pred, output_mask = self.forward(batch)

        direction_loss = self.angular_distance_loss(dir_pred, y_target_angles)
        dom_loss = self.calculate_dom_loss(dom_logits, true_dom_ids, output_mask)

        # Combine losses using lambda_dom from hparams
        combined_loss = direction_loss + self.hparams.training['lambda_dom'] * dom_loss

        return combined_loss, direction_loss, dom_loss

    # --- training_step, validation_step remain the same (log all 3 losses) ---
    def training_step(self, batch, batch_idx):
        combined_loss, direction_loss, dom_loss = self.shared_step(batch)
        self.log('train/loss', combined_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/dir_loss', direction_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/dom_loss', dom_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        combined_loss, direction_loss, dom_loss = self.shared_step(batch)
        self.log('val/loss', combined_loss, prog_bar=True, sync_dist=True)
        self.log('val/dir_loss', direction_loss, prog_bar=False, sync_dist=True)
        self.log('val/dom_loss', dom_loss, prog_bar=False, sync_dist=True)
        return combined_loss

    # --- configure_optimizers remains the same ---
    def configure_optimizers(self):
        # ... (previous implementation - logic based on freeze_backbone and hparams is correct) ...
        freeze_backbone = self.hparams.training.get('freeze_backbone', False)
        if freeze_backbone:
             print("Configuring optimizer only for the new directional head.")
             parameters_to_optimize = self.directional_head.parameters()
        else:
             print("Configuring optimizer for full model (backbone + heads).")
             parameters_to_optimize = self.parameters() # Includes backbone, dom_head, directional_head

        optimizer_name = self.hparams.training['optimizer'].lower()
        lr = self.hparams.training['max_lr']
        weight_decay = self.hparams.training['weight_decay']
        print(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")
        optimizer_kwargs = {'lr': lr, 'betas': (self.hparams.training['adam_beta1'], self.hparams.training['adam_beta2']), 'eps': self.hparams.training['adam_eps'], 'weight_decay': weight_decay, 'amsgrad': self.hparams.training['amsgrad']}
        if optimizer_name == 'adamw': optimizer = torch.optim.AdamW(parameters_to_optimize, **optimizer_kwargs)
        else: raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler_name = self.hparams.training['lr_scheduler'].lower()
        if scheduler_name == 'onecycle':
            if 'total_steps' not in self.hparams.training or self.hparams.training['total_steps'] is None:
                 if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches is not None:
                     total_steps = self.trainer.estimated_stepping_batches
                     warnings.warn(f"total_steps not found in hparams, using trainer's estimate: {total_steps}")
                 else: raise ValueError("total_steps not found in hparams and could not be estimated.")
            else: total_steps = self.hparams.training['total_steps']
            print(f"Scheduler: OneCycleLR with total_steps={total_steps}, pct_start={self.hparams.training['pct_start']}")
            scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=self.hparams.training['max_lr'], total_steps=total_steps, pct_start=self.hparams.training['pct_start'], div_factor=self.hparams.training['div_factor'], final_div_factor=self.hparams.training['final_div_factor'], anneal_strategy='cos')
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif scheduler_name in ['none', None, 'constant']:
            print(f"Scheduler: None (constant LR = {lr})")
            return optimizer
        else: raise ValueError(f"Unsupported scheduler: {scheduler_name}")


# --- Main Fine-tuning Script Logic (Unchanged from previous version) ---

def main():
    # --- (Argument parsing remains the same) ---
    parser = argparse.ArgumentParser(description="Fine-tune PolarBERT Model (Multi-Task Optional)")
    parser.add_argument('--config', type=str, required=True, help="Path to the FINE-TUNING configuration YAML file.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the PRE-TRAINED model checkpoint (.ckpt). Use 'new' to train from scratch.")
    # parser.add_argument('--task', type=str, choices=['direction'], default='direction', help="Primary task.") # Task is now mainly from config
    parser.add_argument('--dataset_type', type=str, choices=['kaggle', 'prometheus'], required=True)
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, help="Freeze backbone (overrides config).")
    parser.add_argument('--name', type=str, default=None, help="Custom run name.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID for naming.")
    args = parser.parse_args()

    # 1. Load Fine-tuning Configuration (will include new defaults)
    config = PolarBertConfig.from_yaml(args.config)

    # --- (Sanity Checks & Overrides remain the same) ---
    if config.training.task != 'direction':
        warnings.warn(f"Config task is '{config.training.task}'. Overriding to 'direction' for multi-task setup.")
        config.training.task = 'direction'
    current_task = config.training.task

    if args.freeze_backbone is not None:
        if args.freeze_backbone != config.training.freeze_backbone:
             warnings.warn(f"Freeze backbone CLI arg overrides config value.")
             config.training.freeze_backbone = args.freeze_backbone
    config.training.pretrained_checkpoint_path_runtime = args.checkpoint_path # Store runtime path for info

    print("--- Multi-Task Fine-tuning Configuration ---")
    print(f"Primary Task: {current_task}")
    # Access values directly from the config object now
    print(f"Auxiliary Task: DOM Prediction (lambda_dom={config.training.lambda_dom})")
    print(f"Directional Pooling: {config.training.directional_pooling_mode}") # Print the mode
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Pretrained Checkpoint: {args.checkpoint_path}")
    print(f"Freeze Backbone: {config.training.freeze_backbone}")
    print("------------------------------------------")

    # --- (Steps 2-6: Run Name, Logging, Batch Calcs, Dataloaders, Runtime Params remain the same) ---
    # 2. Determine Run Name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    base_name = args.name or f"multitask_{current_task}-DOM_{config.training.directional_pooling_mode}_{args.dataset_type}" # Add pooling mode to name
    run_name = f"{base_name}_{suffix}"
    print(f"Starting run: {run_name}")

    # 3. Setup Logging (WandB) - will log config including lambda_dom/pooling_mode defaults
    print("Setting up WandB logger...")
    wandb_logger = WandbLogger(project=config.training.logging.project, name=run_name, config=config.to_dict())

    # 4. Calculate Per-Device Batch Size & Grad Accum Steps
    logical_batch = config.training.logical_batch_size
    max_per_device = config.data.max_per_device_batch_size
    per_device_batch_size = min(max_per_device, logical_batch)
    if per_device_batch_size == 0: raise ValueError("Calculated per_device_batch_size is zero.")
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    config.training.per_device_batch_size = per_device_batch_size
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    print(f"Batch parameters: Per-Device Size={per_device_batch_size}, Grad Accum Steps={gradient_accumulation_steps}")

    # 5. Get Dataloaders
    print("Creating dataloaders...")
    # Select target transform based on dataset type using imported functions
    if args.dataset_type == 'kaggle':
        target_transform_fn = target_transform_kaggle
    elif args.dataset_type == 'prometheus':
        target_transform_fn = target_transform_prometheus
    else:
        # Should not happen due to argparse choices
        raise ValueError(f"Invalid dataset_type: {args.dataset_type}")

    # Use the imported get_dataloaders and selected transform
    train_loader, val_loader = get_dataloaders(
        config,
        dataset_type=args.dataset_type,
        transform=default_transform, # Use the basic transform
        target_transform=target_transform_fn # Pass the selected function
    )

    # 6. Calculate Runtime Training Parameters
    print("Calculating runtime parameters...")
    if hasattr(train_loader.dataset, '__len__') and len(train_loader.dataset) > 0: num_batches_per_epoch = len(train_loader)
    elif config.data.train_events is not None and config.training.per_device_batch_size > 0: num_batches_per_epoch = math.ceil(config.data.train_events / config.training.per_device_batch_size)
    else: num_batches_per_epoch = 1000; warnings.warn("Using fallback estimate for batches/epoch.")
    config.calculate_runtime_params(num_batches_per_epoch) # Modifies config in-place
    print(f"Calculated total_steps: {config.training.total_steps}")
    print(f"Final pct_start for scheduler: {config.training.pct_start:.4f}")

    # Update Wandb config (optional, as initial log now includes defaults)
    if wandb_logger.experiment:
        try: wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True); print("Updated WandB config with runtime parameters.")
        except Exception as e: warnings.warn(f"Could not update WandB config: {e}")

    # 7. Initialize Model (config object now contains runtime params)
    print(f"Initializing Multi-Task PolarBertFinetuner...")
    model = PolarBertFinetuner(config, pretrained_checkpoint_path=args.checkpoint_path) # Pass runtime checkpoint path
    param_count_total = sum(p.numel() for p in model.parameters())
    param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {param_count_total:,}'); print(f'Trainable Parameters: {param_count_trainable:,}')

    # --- (Steps 8-10: Callbacks, Trainer, Training remain the same) ---
    # 8. Setup Callbacks
    print("Setting up callbacks...")
    callbacks = setup_callbacks(config, run_name)

    # 9. Setup Trainer
    print("Setting up PyTorch Lightning Trainer...")
    trainer = Trainer( accelerator='gpu', devices=config.training.gpus, precision=config.training.precision, max_epochs=config.training.max_epochs, gradient_clip_val=config.training.gradient_clip_val, logger=wandb_logger, callbacks=callbacks, val_check_interval=config.training.val_check_interval, accumulate_grad_batches=config.training.gradient_accumulation_steps)

    # 10. Start Fine-tuning
    print("\nStarting multi-task fine-tuning...")
    trainer.fit(model, train_loader, val_loader)
    print("\nMulti-task fine-tuning finished.")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()