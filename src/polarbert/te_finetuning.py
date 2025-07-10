# ---- te_finetuning.py (Modified for Pooling Choice + DOM Loss + Scheduler Fix) ----
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
    # Import get_dataloaders from dataloader_utils instead of te_pretraining
    from polarbert.dataloader_utils import (
        get_dataloaders,
        target_transform_prometheus,
        target_transform_kaggle,
        default_transform
    )
    # Keep setup_callbacks from te_pretraining as it includes config saving
    from polarbert.te_pretraining import setup_callbacks
    # Import old heads only for target transforms if needed, otherwise remove
    # from polarbert.finetuning import DirectionalHead as OldDirectionalHead
    # from polarbert.finetuning import EnergyRegressionHead as OldEnergyRegressionHead
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
            try:
                # Use weights_only=True for safety if checkpoint structure is known
                checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=True)
                # If it's a Lightning checkpoint, state_dict might be nested
                state_dict = checkpoint.get('state_dict', checkpoint)
                cleaned_state_dict = {}
                prefixes_to_remove = ['model.', 'backbone.'] # Prefixes to strip
                for k, v in state_dict.items():
                    key_modified = False
                    for prefix in prefixes_to_remove:
                        if k.startswith(prefix):
                            cleaned_state_dict[k[len(prefix):]] = v
                            key_modified = True
                            break
                    # Keep keys that don't start with the prefixes AND are not optimizer/scheduler states
                    if not key_modified and not k.startswith(('optimizer.', 'lr_scheduler', '_forward_module')):
                         cleaned_state_dict[k] = v

                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
                print("Backbone weights loaded.")
                # Filter warnings for keys related to heads we might replace/ignore
                filtered_missing = [k for k in missing_keys if not k.startswith(('dom_head.', 'charge_head.'))]
                filtered_unexpected = [k for k in unexpected_keys if not k.startswith(('dom_head.', 'charge_head.'))]
                if filtered_missing: print("  Warning: Missing keys in backbone:", filtered_missing)
                if filtered_unexpected: print("  Warning: Unexpected keys found and ignored:", filtered_unexpected)
            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Checkpoint format might be incompatible or file corrupted. Proceeding with untrained backbone.")
        else:
            print("No pretrained checkpoint provided or 'new' specified. Training backbone from scratch.")


        # 3. Define the NEW Task-Specific Prediction Head (Directional Head)
        task = self.hparams.training['task'].lower()
        if task != 'direction':
             raise ValueError(f"This multi-task setup currently requires primary task 'direction', got '{task}'")

        backbone_embed_dim = self.hparams.model['embedding_dim']
        # Ensure 'directional_head' exists in the model config section of hparams
        head_config_dict = self.hparams.model.get('directional_head', {})
        head_hidden_size = int(head_config_dict.get('hidden_size', 1024)) # Default hidden size

        self.directional_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 3) # Output is 3D unit vector prediction
        )
        print(f"Initialized NEW directional head with hidden size: {head_hidden_size}")

        # 4. Handle Backbone Freezing
        if self.hparams.training.get('freeze_backbone', False):
            print("Freezing backbone parameters (including pre-trained DOM/Charge heads).")
            for param in self.backbone.parameters(): param.requires_grad = False
            # Ensure the new head is trainable
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
            output_mask (Tensor | None): Mask for calculating DOM loss (relevant during training if backbone uses masking).
        """
        (x, l), _ = batch

        # 1. Run embedding layer
        # Pass masking=False to embedding during fine-tuning forward pass if masking is only for pre-training
        # Or rely on the backbone's internal masking state if it's controlled differently
        hidden_states, final_padding_mask, output_mask = self.backbone.embedding((x, l)) # Assumes embedding handles masking flag internally
        attn_key_padding_mask = final_padding_mask

        # 2. Pass through transformer blocks
        # Ensure backbone is in eval mode if frozen, or train mode if training
        # PyTorch Lightning handles setting the mode based on trainer state.
        with torch.set_grad_enabled(not self.hparams.training.get('freeze_backbone', False)):
             for block in self.backbone.transformer_blocks:
                 hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)

        # 3. Final Normalization
        hidden_states = self.backbone.final_norm(hidden_states)

        # 4. Get predictions from BOTH heads

        # --- DOM Head Prediction (always uses sequence embeddings) ---
        sequence_embeds = hidden_states[:, 1:, :]
        # Make sure the backbone has the dom_head attribute
        if not hasattr(self.backbone, 'dom_head'):
             raise AttributeError("Backbone model is missing the 'dom_head' attribute.")
        dom_logits = self.backbone.dom_head(sequence_embeds)

        # --- Directional Head Prediction (Input depends on pooling_mode) ---
        if self.pooling_mode == 'cls':
            dir_head_input = hidden_states[:, 0, :]
        elif self.pooling_mode == 'mean':
            seq_padding_mask = final_padding_mask[:, 1:]
            valid_token_mask = ~seq_padding_mask
            valid_token_mask_expanded = valid_token_mask.unsqueeze(-1).expand_as(sequence_embeds).float()
            masked_sequence_embeds = sequence_embeds * valid_token_mask_expanded
            summed_embeds = masked_sequence_embeds.sum(dim=1)
            num_valid_tokens = valid_token_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-6) # Avoid division by zero
            dir_head_input = summed_embeds / num_valid_tokens
        else:
            raise ValueError(f"Invalid pooling_mode: {self.pooling_mode}")

        dir_pred = self.directional_head(dir_head_input)

        # output_mask comes from the embedding layer, used for DOM loss calculation
        return dom_logits, dir_pred, output_mask

    def angular_distance_loss(self, y_pred_vectors, y_target_angles):
        """Calculates the mean angular distance loss."""
        y_truth_vectors = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1])
        # Normalize predicted vectors to ensure they are unit vectors before loss calculation
        norm = torch.linalg.vector_norm(y_pred_vectors, dim=1, keepdim=True)
        y_pred_unit_vectors = y_pred_vectors / (norm + 1e-8) # Add epsilon for numerical stability
        # Calculate loss using unit vectors
        loss = angular_dist_score_unit_vectors(y_truth_vectors, y_pred_unit_vectors, epsilon=1e-4)
        return loss

    def calculate_dom_loss(self, dom_logits, true_dom_ids, output_mask):
        """Calculates the masked DOM prediction loss."""
        dom_loss = torch.tensor(0.0, device=dom_logits.device, dtype=dom_logits.dtype)
        # Masking logic: Use output_mask during training if available (from embedding),
        # otherwise mask based on non-padding tokens during validation/testing.
        mask_to_use = output_mask if self.training and output_mask is not None else (true_dom_ids != 0) # 0 is PAD_IDX for dom_ids

        if mask_to_use.sum() > 0:
             # Target DOM IDs are 1-indexed in input, need 0-indexed for CrossEntropyLoss
             # PAD index (0 in input) becomes -1, which is ignored by default.
             dom_targets = true_dom_ids - 1
             # Ensure mask shape matches logits/targets shape (B, L_orig)
             if mask_to_use.shape == dom_targets.shape:
                 masked_logits = dom_logits[mask_to_use]
                 masked_targets = dom_targets[mask_to_use]
                 if masked_logits.nelement() > 0: # Check if anything is left after masking
                     dom_loss = F.cross_entropy(masked_logits, masked_targets, ignore_index=-1) # ignore_index=-1 handles PAD
             else:
                  warnings.warn(f"DOM Loss: Mask shape {mask_to_use.shape} incompatible with target shape {dom_targets.shape}. Skipping loss calculation.")
        return dom_loss

    def shared_step(self, batch):
        """Common logic for training and validation steps."""
        (x, l), y_data = batch
        if y_data is None or y_data[0] is None:
            # Handle cases where target data might be missing (e.g., during inference or if dataloader setup allows it)
            warnings.warn("Target data (y_data) is missing in shared_step.")
            # Return dummy losses or handle appropriately
            return torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)

        y_target_angles = y_data[0].to(x.device) # Ensure targets are on the correct device
        true_dom_ids = x[:, :, 3].long() # Assuming DOM ID is the 4th feature (index 3)

        dom_logits, dir_pred, output_mask = self.forward(batch)

        direction_loss = self.angular_distance_loss(dir_pred, y_target_angles)
        dom_loss = self.calculate_dom_loss(dom_logits, true_dom_ids, output_mask)

        # Combine losses using lambda_dom from hparams
        combined_loss = direction_loss + self.hparams.training['lambda_dom'] * dom_loss

        # Handle potential NaN losses
        if torch.isnan(direction_loss):
            warnings.warn("NaN detected in direction_loss.")
            direction_loss = torch.tensor(0.0, device=combined_loss.device) # Replace NaN with 0 for logging/combination
        if torch.isnan(dom_loss):
             warnings.warn("NaN detected in dom_loss.")
             dom_loss = torch.tensor(0.0, device=combined_loss.device) # Replace NaN with 0
        if torch.isnan(combined_loss):
             warnings.warn("NaN detected in combined_loss. Replacing with direction_loss component if available.")
             combined_loss = direction_loss # Fallback to direction loss if combined is NaN

        return combined_loss, direction_loss, dom_loss

    def training_step(self, batch, batch_idx):
        combined_loss, direction_loss, dom_loss = self.shared_step(batch)
        self.log('train/loss', combined_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/dir_loss', direction_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/dom_loss', dom_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        combined_loss, direction_loss, dom_loss = self.shared_step(batch)
        self.log('val/loss', combined_loss, prog_bar=True, sync_dist=True)
        self.log('val/dir_loss', direction_loss, prog_bar=False, sync_dist=True)
        self.log('val/dom_loss', dom_loss, prog_bar=False, sync_dist=True)
        return combined_loss

    def configure_optimizers(self):
        freeze_backbone = self.hparams.training.get('freeze_backbone', False)
        if freeze_backbone:
             print("Configuring optimizer only for the new directional head.")
             parameters_to_optimize = self.directional_head.parameters()
        else:
             print("Configuring optimizer for full model (backbone + heads).")
             parameters_to_optimize = self.parameters()

        optimizer_name = self.hparams.training['optimizer'].lower()
        lr = self.hparams.training['max_lr']
        weight_decay = self.hparams.training['weight_decay']
        print(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")
        optimizer_kwargs = {'lr': lr, 'betas': (self.hparams.training['adam_beta1'], self.hparams.training['adam_beta2']), 'eps': self.hparams.training['adam_eps'], 'weight_decay': weight_decay, 'amsgrad': self.hparams.training['amsgrad']}

        # --- Add Weight Decay Handling ---
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, getattr(torch.nn, 'RMSNorm', type(None))) # Add RMSNorm if available

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if fpn not in param_dict: continue

                if pn.endswith('bias'): no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules): decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules): no_decay.add(fpn)
                elif 'norm' in pn.lower(): no_decay.add(fpn) # Catch norm layer parameters by name

        # Handle any potentially missed parameters (e.g. custom layers)
        unassigned_params = param_dict.keys() - (decay | no_decay)
        if unassigned_params:
            warnings.warn(f"Assigning parameters to no_decay by default: {unassigned_params}")
            no_decay.update(unassigned_params)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # --- End Weight Decay Handling ---

        if optimizer_name == 'adamw':
            # Pass optim_groups instead of parameters_to_optimize
            optimizer = torch.optim.AdamW(optim_groups, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler_name = self.hparams.training['lr_scheduler'].lower()
        if scheduler_name == 'onecycle':
            total_steps = self.hparams.training.get('total_steps') # Get pre-calculated total_steps
            if total_steps is None:
                 # Try to estimate again if not found (should have been calculated in main)
                 if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
                     total_steps = self.trainer.estimated_stepping_batches
                     warnings.warn(f"configure_optimizers: total_steps not found in hparams, using trainer's estimate: {total_steps}")
                 else:
                      raise ValueError("total_steps is required for OneCycleLR and could not be found or estimated.")
            print(f"Scheduler: OneCycleLR with total_steps={total_steps}, pct_start={self.hparams.training['pct_start']}")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                 optimizer,
                 max_lr=self.hparams.training['max_lr'],
                 total_steps=int(total_steps), # Ensure total_steps is int
                 pct_start=self.hparams.training['pct_start'],
                 div_factor=self.hparams.training['div_factor'],
                 final_div_factor=self.hparams.training['final_div_factor'],
                 anneal_strategy='cos'
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif scheduler_name in ['none', None, 'constant']:
            print(f"Scheduler: None (constant LR = {lr})")
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PolarBERT Model (Multi-Task Optional)")
    parser.add_argument('--config', type=str, required=True, help="Path to the FINE-TUNING configuration YAML file.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the PRE-TRAINED model checkpoint (.ckpt). Use 'new' to train from scratch.")
    parser.add_argument('--dataset_type', type=str, choices=['kaggle', 'prometheus'], required=True)
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, help="Freeze backbone (overrides config).")
    parser.add_argument('--name', type=str, default=None, help="Custom run name.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID for naming.")
    args = parser.parse_args()

    config = PolarBertConfig.from_yaml(args.config)

    if config.training.task != 'direction':
        warnings.warn(f"Config task is '{config.training.task}'. Overriding to 'direction' for multi-task setup.")
        config.training.task = 'direction'
    current_task = config.training.task

    if args.freeze_backbone is not None:
        if args.freeze_backbone != config.training.freeze_backbone:
             warnings.warn(f"Freeze backbone CLI arg overrides config value.")
             config.training.freeze_backbone = args.freeze_backbone
    config.training.pretrained_checkpoint_path_runtime = args.checkpoint_path

    print("--- Multi-Task Fine-tuning Configuration ---")
    print(f"Primary Task: {current_task}")
    print(f"Auxiliary Task: DOM Prediction (lambda_dom={config.training.lambda_dom})")
    print(f"Directional Pooling: {config.training.directional_pooling_mode}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Pretrained Checkpoint: {args.checkpoint_path}")
    print(f"Freeze Backbone: {config.training.freeze_backbone}")
    print("------------------------------------------")

    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    base_name = args.name or f"multitask_{current_task}-DOM_{config.training.directional_pooling_mode}_{args.dataset_type}"
    run_name = f"{base_name}_{suffix}"
    print(f"Starting run: {run_name}")

    print("Setting up WandB logger...")
    wandb_logger = WandbLogger(project=config.training.logging.project, name=run_name, config=config.to_dict())

    logical_batch = config.training.logical_batch_size
    max_per_device = config.data.max_per_device_batch_size
    per_device_batch_size = min(max_per_device, logical_batch)
    if per_device_batch_size <= 0: raise ValueError("Calculated per_device_batch_size must be positive.")
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    config.training.per_device_batch_size = per_device_batch_size
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    print(f"Batch parameters: Per-Device Size={per_device_batch_size}, Grad Accum Steps={gradient_accumulation_steps}")

    print("Creating dataloaders...")
    if args.dataset_type == 'kaggle':
        target_transform_fn = target_transform_kaggle
    elif args.dataset_type == 'prometheus':
        target_transform_fn = target_transform_prometheus
    else:
        raise ValueError(f"Invalid dataset_type: {args.dataset_type}")

    train_loader, val_loader = get_dataloaders(
        config,
        dataset_type=args.dataset_type,
        transform=default_transform,
        target_transform=target_transform_fn,
        # Pass per_device_batch_size to ensure dataloader uses the calculated size
        override_batch_size=per_device_batch_size
    )

    print("Calculating runtime parameters...")
    try:
        # Use the actual length of the dataloader if possible
        num_batches_per_epoch = len(train_loader)
        if num_batches_per_epoch == 0: raise TypeError # Force fallback if length is 0
    except TypeError:
        # Fallback estimation if dataloader has no __len__
        if config.data.train_events is not None and config.training.per_device_batch_size > 0:
            num_batches_per_epoch = math.ceil(config.data.train_events / config.training.per_device_batch_size)
            warnings.warn(f"Using estimated batches/epoch based on train_events: {num_batches_per_epoch}")
        else:
            num_batches_per_epoch = 1000 # Final fallback
            warnings.warn(f"Using fallback estimate for batches/epoch: {num_batches_per_epoch}")

    # --- FIX: Calculate total device steps across all epochs ---
    if config.training.max_epochs is None or config.training.max_epochs <= 0:
         raise ValueError("config.training.max_epochs must be a positive integer.")
    total_device_steps = num_batches_per_epoch * config.training.max_epochs
    print(f"Estimated batches/epoch: {num_batches_per_epoch}, Max Epochs: {config.training.max_epochs}, Total Device Steps: {total_device_steps}")
    # --- End FIX ---

    config.calculate_runtime_params(total_device_steps) # Pass correct total steps
    print(f"Calculated total optimizer steps (for scheduler): {config.training.total_steps}")
    print(f"Final pct_start for scheduler: {config.training.pct_start:.4f}")

    if wandb_logger.experiment:
        try: wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True); print("Updated WandB config with runtime parameters.")
        except Exception as e: warnings.warn(f"Could not update WandB config: {e}")

    print(f"Initializing Multi-Task PolarBertFinetuner...")
    model = PolarBertFinetuner(config, pretrained_checkpoint_path=args.checkpoint_path)
    param_count_total = sum(p.numel() for p in model.parameters())
    param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {param_count_total:,}'); print(f'Trainable Parameters: {param_count_trainable:,}')

    print("Setting up callbacks...")
    callbacks = setup_callbacks(config, run_name)

    print("Setting up PyTorch Lightning Trainer...")
    trainer = Trainer(
        accelerator='gpu',
        devices=config.training.gpus,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        gradient_clip_val=config.training.gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=config.training.val_check_interval,
        accumulate_grad_batches=config.training.gradient_accumulation_steps
    )

    print("\nStarting multi-task fine-tuning...")
    trainer.fit(model, train_loader, val_loader)
    print("\nMulti-task fine-tuning finished.")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()
