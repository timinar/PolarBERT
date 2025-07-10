#!/usr/bin/env python
import argparse
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# --- Project Imports ---
# Assuming these imports work from your environment
from polarbert.config import PolarBertConfig
from polarbert.time_embed_polarbert import PolarBertModel, RMSNorm # Backbone
from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors
from polarbert.dataloader_utils import (get_dataloaders, target_transform_prometheus,
                                        target_transform_kaggle, default_transform)
from polarbert.prometheus_dataset import IceCubeDataset as PrometheusDataset
from polarbert.icecube_dataset import IceCubeDataset as KaggleDataset
from polarbert.te_pretraining import setup_callbacks # Includes config saving


class PolarBertWeightedFinetuner(pl.LightningModule):
    """
    LightningModule for fine-tuning on Prometheus using MDM-loss-weighted
    directional loss. Validates on both Kaggle and Prometheus.
    """
    def __init__(self, config: PolarBertConfig,
                 pretrained_checkpoint_path: Optional[str] = None,
                 weight_constant: float = 2.0):
        super().__init__()
        self.config = config
        self.weight_constant = weight_constant
        hparams_to_save = config.to_dict()
        hparams_to_save['training']['pretrained_checkpoint_path_runtime'] = pretrained_checkpoint_path
        hparams_to_save['training']['weight_constant'] = weight_constant # Log constant
        self.save_hyperparameters(hparams_to_save)

        self.pooling_mode = self.config.training.directional_pooling_mode
        # Loss lambdas are not used for weighting in this script's training_step,
        # but keep them if config structure requires them.
        # self.lambda_dom_k = self.config.training.lambda_dom_kaggle
        # self.lambda_dom_p = self.config.training.lambda_dom_prometheus
        # self.lambda_dir_p = self.config.training.lambda_dir_prometheus

        print(f"Directional head pooling mode: {self.pooling_mode}")
        print(f"Using MDM loss weighting for directional loss with C = {self.weight_constant}")

        # --- Backbone Instantiation & Weight Loading ---
        # Make sure the embedding layer has masking enabled for MDM loss calculation
        # config.model.embedding.masking_prob = config.training.get('mask_prob', 0.25) # Ensure mask prob is set
        self.backbone = PolarBertModel(config)
        if pretrained_checkpoint_path and pretrained_checkpoint_path.lower() != 'new':
            print(f"Loading backbone weights from: {pretrained_checkpoint_path}")
            try:
                # Use weights_only=True for safety if optimizer state isn't needed
                checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=True)
                # Handle different checkpoint structures
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                     state_dict = checkpoint
                else: raise TypeError("Unrecognized checkpoint format")

                cleaned_state_dict = {}
                prefixes_to_remove = ['model.', 'backbone.'] # Add other prefixes if needed
                for k, v in state_dict.items():
                    key_modified = False
                    for prefix in prefixes_to_remove:
                        if k.startswith(prefix):
                            cleaned_state_dict[k[len(prefix):]] = v
                            key_modified = True; break
                    if not key_modified and not k.startswith(('optimizer.', 'lr_scheduler', '_forward_module')):
                         cleaned_state_dict[k] = v # Keep relevant unprefixed keys

                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
                print("Backbone weights loaded.")
                # Filter potential heads from pre-training checkpoint
                filtered_missing = [k for k in missing_keys if not k.startswith(('dom_head', 'charge_head'))]
                filtered_unexpected = [k for k in unexpected_keys if not k.startswith(('dom_head', 'charge_head'))]
                if filtered_missing: print("  Warning: Missing keys in backbone:", filtered_missing)
                if filtered_unexpected: print("  Warning: Unexpected keys in backbone state_dict:", filtered_unexpected)

            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Proceeding with untrained backbone.")
                import traceback
                traceback.print_exc()
        else:
            print("No pretrained checkpoint provided or 'new' specified. Training backbone from scratch.")

        # --- Directional Head Definition ---
        backbone_embed_dim = self.config.model.embedding_dim
        # Ensure directional_head config exists
        head_config_dict = self.config.model.directional_head or {'hidden_size': 1024}
        head_hidden_size = int(head_config_dict.get('hidden_size', 1024))
        self.directional_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 3)
        )
        print(f"Initialized Directional head with hidden size: {head_hidden_size}")

        # --- Backbone Freezing ---
        if self.config.training.freeze_backbone:
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters(): param.requires_grad = False
            for param in self.directional_head.parameters(): param.requires_grad = True
        else:
            print("Training full model (backbone + heads).")

        # Init lists for validation outputs
        self.validation_step_outputs_kaggle = []
        self.validation_step_outputs_prometheus = []

    # --- Forward methods (can reuse from mixed_finetuning) ---
    def forward_features(self, batch_input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Runs backbone embedding and transformer blocks."""
        (x, l) = batch_input
        # Embedding layer performs masking based on its config
        hidden_states, final_padding_mask, output_mask = self.backbone.embedding((x, l))
        attn_key_padding_mask = final_padding_mask
        for block in self.backbone.transformer_blocks:
            hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)
        hidden_states = self.backbone.final_norm(hidden_states)
        seq_padding_mask = final_padding_mask[:, 1:] # Mask for original sequence tokens
        return hidden_states, output_mask, seq_padding_mask

    def forward_heads(self, hidden_states: torch.Tensor, seq_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs prediction heads."""
        cls_embed = hidden_states[:, 0, :]
        sequence_embeds = hidden_states[:, 1:, :]
        if not hasattr(self.backbone, 'dom_head'): raise AttributeError("Backbone missing 'dom_head'.")
        dom_logits = self.backbone.dom_head(sequence_embeds) # Needed for MDM loss calc

        # Determine input for directional head based on pooling mode
        if self.pooling_mode == 'cls':
            dir_head_input = cls_embed
        elif self.pooling_mode == 'mean':
            valid_token_mask = ~seq_padding_mask
            valid_token_mask_expanded = valid_token_mask.unsqueeze(-1).expand_as(sequence_embeds).float()
            masked_sequence_embeds = sequence_embeds * valid_token_mask_expanded
            summed_embeds = masked_sequence_embeds.sum(dim=1)
            num_valid_tokens = valid_token_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-6)
            dir_head_input = summed_embeds / num_valid_tokens
        else: raise ValueError(f"Invalid pooling_mode: {self.pooling_mode}")

        dir_pred = self.directional_head(dir_head_input) # Needs gradients
        return dom_logits, dir_pred

    # --- Loss Calculation Helpers ---
    def calculate_per_event_dom_loss_no_grad(self, dom_logits, true_dom_ids, output_mask, seq_padding_mask, epsilon=1e-8):
        """ Calculates per-event MDM loss, returns detached tensor. Handles NaNs. """
        with torch.no_grad():
            dom_targets = true_dom_ids - 1 # Target IDs: -1=PAD, 0=DOM_0, ...
            # Mask for loss: use output_mask from forward pass & ignore padding
            effective_mask = output_mask & ~seq_padding_mask if output_mask is not None else ~seq_padding_mask

            if effective_mask.shape != dom_targets.shape:
                warnings.warn(f"MDM Loss: Mask shape {effective_mask.shape} mismatch target {dom_targets.shape}. Returning NaNs.")
                return torch.full((dom_logits.size(0),), float('nan'), device=dom_logits.device)

            loss_per_token = F.cross_entropy(
                dom_logits.permute(0, 2, 1), dom_targets, ignore_index=-1, reduction='none'
            )
            masked_loss_per_token = loss_per_token * effective_mask.float()
            summed_loss_per_event = masked_loss_per_token.sum(dim=1)
            num_masked_tokens_per_event = effective_mask.sum(dim=1).float()

            avg_loss_per_event = summed_loss_per_event / (num_masked_tokens_per_event + epsilon)
            # Set loss to NaN if no tokens contributed
            avg_loss_per_event[num_masked_tokens_per_event == 0] = float('nan')
            return avg_loss_per_event.detach() # Ensure it's detached

    def calculate_per_event_dir_loss(self, dir_pred, batch_target, epsilon=1e-4):
        """ Calculates per-event angular distance loss. Needs gradients. """
        if batch_target is None or batch_target[0] is None:
             # Return NaNs with grad if needed, though targets should exist for training/val
             return torch.full((dir_pred.size(0),), float('nan'), device=dir_pred.device, requires_grad=dir_pred.requires_grad)

        y_target_angles = batch_target[0]
        y_truth_vectors = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1]).to(dir_pred.device) # Ensure device match
        norm = torch.linalg.vector_norm(dir_pred, dim=1, keepdim=True)
        y_pred_unit_vectors = dir_pred / (norm + 1e-8) # Normalize predictions

        # Calculate dot product per event
        scalar_prod = torch.sum(y_truth_vectors * y_pred_unit_vectors, dim=1)
        # Clamp for numerical stability
        scalar_prod = torch.clamp(scalar_prod, -1.0 + epsilon, 1.0 - epsilon)
        # Calculate angular distance per event
        loss_per_event = torch.abs(torch.arccos(scalar_prod))
        return loss_per_event

    # --- Training Step ---
    def training_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Any], batch_idx: int):
        """ Processes one training batch from Prometheus using weighted loss. """
        # Note: Assumes trainer is configured to only provide Prometheus batches here
        prometheus_input, prometheus_target = batch

        # 1. Forward pass to get logits and predictions
        hidden_states, output_mask, seq_padding_mask = self.forward_features(prometheus_input)
        dom_logits, dir_pred = self.forward_heads(hidden_states, seq_padding_mask)

        # 2. Calculate per-event Directional Loss (requires grad)
        dir_loss_per_event = self.calculate_per_event_dir_loss(dir_pred, prometheus_target) # Shape (B,)

        # 3. Calculate per-event MDM Loss (NO grad)
        true_dom_ids = prometheus_input[0][:, :, 3].long()
        mdm_loss_per_event = self.calculate_per_event_dom_loss_no_grad(
            dom_logits, true_dom_ids, output_mask, seq_padding_mask
        ) # Shape (B,)

        # 4. Calculate Weights based on MDM loss
        # Handle potential NaNs in MDM loss: assign a neutral weight (e.g., 1.0)
        # Clamp mdm_loss >= 0 before adding C
        weights = torch.where(
            torch.isnan(mdm_loss_per_event),
            torch.tensor(1.0, device=self.device), # Neutral weight for NaN MDM loss
            self.weight_constant / (self.weight_constant + mdm_loss_per_event.clamp(min=0.0))
        )
        weights = weights.detach() # Ensure weights don't carry gradients

        # 5. Calculate Weighted Average Directional Loss
        # Also handle potential NaNs in directional loss (if targets were bad)
        valid_dir_loss_mask = ~torch.isnan(dir_loss_per_event)
        weighted_dir_loss = dir_loss_per_event[valid_dir_loss_mask] * weights[valid_dir_loss_mask]
        # Average over only the valid samples
        total_loss = weighted_dir_loss.sum() / valid_dir_loss_mask.sum().clamp(min=1.0)

        # Logging
        log_dict = {
            'train_step/final_loss': total_loss.item(),
            'train_step/mean_mdm_loss': mdm_loss_per_event[~torch.isnan(mdm_loss_per_event)].mean().item() if not torch.all(torch.isnan(mdm_loss_per_event)) else float('nan'),
            'train_step/mean_dir_loss': dir_loss_per_event[valid_dir_loss_mask].mean().item() if valid_dir_loss_mask.any() else float('nan'),
            'train_step/mean_weight': weights[valid_dir_loss_mask].mean().item() if valid_dir_loss_mask.any() else float('nan'),
        }
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)

        return total_loss

    # --- Validation Step ---
    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Any], batch_idx: int, dataloader_idx: int):
        """ Processes a validation batch from either Kaggle (idx 0) or Prometheus (idx 1)."""
        batch_input, batch_target = batch
        true_dom_ids = batch_input[0][:, :, 3].long()

        # Forward pass
        hidden_states, output_mask, seq_padding_mask = self.forward_features(batch_input)
        dom_logits, dir_pred = self.forward_heads(hidden_states, seq_padding_mask)

        # Calculate MEAN losses for logging (no weighting needed here)
        # MDM loss (no grad implicitly handled by no_grad context in validation)
        # Use a simplified mean calculation for validation logging
        val_mdm_loss = self.backbone._calculate_loss(batch=((batch_input[0], batch_input[1]), batch_target))[1] # Get dom_loss part

        # Directional loss
        val_dir_loss = angular_dist_score_unit_vectors(
             angles_to_unit_vector(batch_target[0][:,0], batch_target[0][:,1]).to(dir_pred.device),
             dir_pred / (torch.linalg.vector_norm(dir_pred, dim=1, keepdim=True) + 1e-8),
             epsilon=1e-4
        )

        # Store results based on dataloader index
        if dataloader_idx == 0: # Kaggle Validation
            losses = {
                'val/dom_loss_kaggle': val_mdm_loss,
                'val/dir_loss_kaggle': val_dir_loss
            }
            self.validation_step_outputs_kaggle.append(losses)
        elif dataloader_idx == 1: # Prometheus Validation
             losses = {
                'val/dom_loss_prometheus': val_mdm_loss,
                'val/dir_loss_prometheus': val_dir_loss
            }
             self.validation_step_outputs_prometheus.append(losses)
        else:
            warnings.warn(f"Unexpected dataloader_idx in validation_step: {dataloader_idx}")

    # --- Validation Epoch End ---
    def on_validation_epoch_end(self):
        """ Aggregates stored validation step outputs and logs epoch metrics. """
        avg_losses = {}

        # Process Kaggle Outputs
        if self.validation_step_outputs_kaggle:
            kaggle_outputs = self.validation_step_outputs_kaggle
            # Aggregate, handling potential NaNs from skipped batches or calculation issues
            dom_k_valid = [x['val/dom_loss_kaggle'] for x in kaggle_outputs if not torch.isnan(x.get('val/dom_loss_kaggle', torch.nan))]
            dir_k_valid = [x['val/dir_loss_kaggle'] for x in kaggle_outputs if not torch.isnan(x.get('val/dir_loss_kaggle', torch.nan))]
            dom_k_epoch = torch.stack(dom_k_valid).mean() if dom_k_valid else torch.tensor(float('nan'), device=self.device)
            dir_k_epoch = torch.stack(dir_k_valid).mean() if dir_k_valid else torch.tensor(float('nan'), device=self.device)
            avg_losses['val_epoch/dom_loss_kaggle'] = dom_k_epoch
            avg_losses['val_epoch/dir_loss_kaggle'] = dir_k_epoch

        # Process Prometheus Outputs
        if self.validation_step_outputs_prometheus:
             prometheus_outputs = self.validation_step_outputs_prometheus
             dom_p_valid = [x['val/dom_loss_prometheus'] for x in prometheus_outputs if not torch.isnan(x.get('val/dom_loss_prometheus', torch.nan))]
             dir_p_valid = [x['val/dir_loss_prometheus'] for x in prometheus_outputs if not torch.isnan(x.get('val/dir_loss_prometheus', torch.nan))]
             dom_p_epoch = torch.stack(dom_p_valid).mean() if dom_p_valid else torch.tensor(float('nan'), device=self.device)
             dir_p_epoch = torch.stack(dir_p_valid).mean() if dir_p_valid else torch.tensor(float('nan'), device=self.device)
             avg_losses['val_epoch/dom_loss_prometheus'] = dom_p_epoch
             avg_losses['val_epoch/dir_loss_prometheus'] = dir_p_epoch

        # Log all aggregated metrics
        # Add primary metric for checkpointing (e.g., Prometheus dir loss)
        primary_metric = avg_losses.get('val_epoch/dir_loss_prometheus', torch.tensor(float('nan')))
        self.log('val_primary_metric', primary_metric, prog_bar=True, sync_dist=True)
        self.log_dict(avg_losses, prog_bar=False, sync_dist=True) # Log others without prog bar

        # Clear the stored outputs for the next epoch
        self.validation_step_outputs_kaggle.clear()
        self.validation_step_outputs_prometheus.clear()

    # --- Configure Optimizers ---
    def configure_optimizers(self):
        """ Sets up optimizer and scheduler using self.config. (Reused logic) """
        # ... (Copy the configure_optimizers logic from mixed_finetuning.py) ...
        # ... (It correctly handles freezing and parameter groups) ...
        parameters_to_optimize = []
        param_dict = {}
        freeze_backbone = self.config.training.freeze_backbone
        if freeze_backbone:
             print("Optimizing only the Directional Head.")
             parameters_to_optimize = list(self.directional_head.parameters())
             param_dict = {f'directional_head.{pn}': p for pn, p in self.directional_head.named_parameters() if p.requires_grad}
        else:
             print("Optimizing the full model.")
             parameters_to_optimize = list(self.parameters())
             param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if not parameters_to_optimize:
             warnings.warn("No parameters selected for optimization.")
             return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-8) # Return dummy optimizer

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, RMSNorm, torch.nn.Embedding)
        # Determine modules to consider based on freezing
        relevant_modules = self.directional_head.named_modules() if freeze_backbone else self.named_modules()

        for mn, m in relevant_modules:
            for pn, p in m.named_parameters():
                if not p.requires_grad: continue
                # Construct full parameter name relative to the top-level model being optimized
                fpn_rel = f'{mn}.{pn}' if mn else pn # Name relative to relevant_modules start
                # Prepend correct prefix if only optimizing head
                fpn = f'directional_head.{fpn_rel}' if freeze_backbone else fpn_rel

                if fpn not in param_dict: continue # Ensure we only consider parameters being optimized

                # Apply decay rules
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif isinstance(m, (RMSNorm, torch.nn.LayerNorm)): # Catch norm parameters by type
                    no_decay.add(fpn)
                # Add specific parameters like cls_token if needed and not covered
                # elif fpn == 'backbone.embedding.cls_embedding': decay.add(fpn) # Example if cls token is decayed

        # Validate grouping
        union_params = decay | no_decay
        unassigned_params = set(param_dict.keys()) - union_params
        if len(unassigned_params) > 0:
            print(f"WARNING: Assigning parameters to no_decay group by default: {unassigned_params}")
            no_decay.update(unassigned_params)

        print(f"Optimizing {len(decay)} decaying parameter tensors and {len(no_decay)} non-decaying tensors.")
        # Create optimizer groups, ensuring no empty groups are passed
        optim_groups = []
        if decay:
            optim_groups.append({"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.training.weight_decay})
        if no_decay:
             optim_groups.append({"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0})

        if not optim_groups:
             warnings.warn("No parameter groups created for optimizer.")
             return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-8)

        # Create Optimizer
        optimizer = torch.optim.AdamW(
             optim_groups, lr=self.config.training.max_lr,
             betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
             eps=self.config.training.adam_eps, amsgrad=self.config.training.amsgrad)

        # Create Scheduler
        scheduler_name = self.config.training.lr_scheduler.lower()
        if scheduler_name == 'onecycle':
            if self.config.training.total_steps is None:
                # Need trainer reference, but it might not be available yet.
                # Rely on total_steps being calculated in main() before trainer.fit()
                raise ValueError("Total steps required for OneCycleLR not calculated in config.")
            print(f"Using OneCycleLR scheduler with total_steps={self.config.training.total_steps}")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                 optimizer, max_lr=self.config.training.max_lr,
                 total_steps=self.config.training.total_steps,
                 pct_start=self.config.training.pct_start,
                 div_factor=self.config.training.div_factor,
                 final_div_factor=self.config.training.final_div_factor,
                 anneal_strategy='cos')
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif scheduler_name in ['none', None, 'constant']:
            print(f"Using constant learning rate: {self.config.training.max_lr}")
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")


# --- Main Training Script ---

def main():
    parser = argparse.ArgumentParser(description="Weighted Fine-tune PolarBERT on Prometheus")
    parser.add_argument('--config', type=str, required=True, help="Path to the fine-tuning config YAML.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to PRE-TRAINED backbone checkpoint (.ckpt or .pth). Use 'new' for scratch.")
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, help="Freeze backbone (overrides config value).")
    parser.add_argument('--name', type=str, default=None, help="Custom name for the WandB run.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID (e.g., SLURM ID) for run naming.")
    parser.add_argument('--weight_constant', type=float, default=2.0, help="Constant C for loss weighting: C / (C + mdm_loss).")
    # Remove lambda args as they are not used for training weighting here
    # parser.add_argument('--lambda_dom_k', type=float, default=None, help="Override lambda_dom_kaggle.")
    # parser.add_argument('--lambda_dom_p', type=float, default=None, help="Override lambda_dom_prometheus.")
    # parser.add_argument('--lambda_dir_p', type=float, default=None, help="Override lambda_dir_prometheus.")
    args = parser.parse_args()

    # 1. Load Configuration
    config = PolarBertConfig.from_yaml(args.config)

    # 2. Apply Command-Line Overrides
    if args.freeze_backbone is not None: config.training.freeze_backbone = args.freeze_backbone
    # Store weight constant in config for logging/reference (though passed to init)
    config.training.weight_constant = args.weight_constant
    config.training.pretrained_checkpoint_path_runtime = args.checkpoint_path

    # 3. Determine Run Name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    freeze_str = "_frz" if config.training.freeze_backbone else ""
    base_name = args.name or f"weighted{freeze_str}_{config.training.directional_pooling_mode}_C{args.weight_constant}"
    run_name = f"{base_name}_{suffix}"
    config.model.model_name = run_name # Store run name in config for saving
    print(f"Run Name: {run_name}")

    # 4. Setup Logging
    print("Setting up WandB logger...")
    # Ensure the updated config dict is logged
    wandb_logger = WandbLogger(
        project=config.training.logging.project,
        name=run_name,
        entity=config.training.logging.entity,
        config=config.to_dict(), # Log final config state including weight_constant
    )

    # 5. Calculate Batch Params
    logical_batch = config.training.logical_batch_size
    max_per_device = config.data.max_per_device_batch_size
    per_device_batch_size = min(max_per_device, logical_batch)
    if per_device_batch_size == 0: raise ValueError("Calculated per_device_batch_size is zero.")
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    config.training.per_device_batch_size = per_device_batch_size
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    print(f"Batch parameters: Per-Device Size={per_device_batch_size}, Grad Accum Steps={gradient_accumulation_steps}")

    # 6. Instantiate Dataloaders
    print("Creating dataloaders...")
    # --- Training Loader: Prometheus ONLY ---
    prom_train_events = config.data.prometheus_train_events
    prom_val_events = config.data.prometheus_val_events # Needed to find start index
    if prom_train_events is None or prom_val_events is None:
        raise ValueError("Prometheus train/val event counts must be set in config.")

    full_prometheus_dataset = PrometheusDataset(
        data_dir=config.data.prometheus_dir,
        batch_size=per_device_batch_size, # Dataset batch size
        transform=default_transform,
        target_transform=target_transform_prometheus # Use standard transform for targets
    )
    # Slice dataset correctly for training (after validation part)
    try: total_prometheus_events = len(full_prometheus_dataset.x)
    except: total_prometheus_events = prom_train_events + prom_val_events; warnings.warn("Could not get exact Prometheus dataset size.")

    val_end_idx = min(prom_val_events, total_prometheus_events)
    train_end_idx = min(val_end_idx + prom_train_events, total_prometheus_events)
    print(f"Prometheus splitting for Training: Start Idx={val_end_idx}, End Idx={train_end_idx}")
    prometheus_train_dataset = full_prometheus_dataset.slice(val_end_idx, train_end_idx)

    loader_kwargs = {'batch_size': None, 'num_workers': config.data.num_workers, 'pin_memory': config.data.pin_memory, 'persistent_workers': config.data.persistent_workers and config.data.num_workers > 0}
    prometheus_train_loader = DataLoader(prometheus_train_dataset, **loader_kwargs)
    print(f"Prometheus Training Loader created.")

    # --- Validation Loaders: Kaggle AND Prometheus ---
    print("Creating Validation Loaders...")
    # Use get_dataloaders to easily get Kaggle val loader
    _, kaggle_val_loader = get_dataloaders(
        config, dataset_type='kaggle', transform=default_transform,
        target_transform=target_transform_kaggle,
        override_batch_size=per_device_batch_size # Use eval batch size for validation
    )
    # Reuse the Prometheus dataset object, but slice for validation
    prometheus_val_dataset = full_prometheus_dataset.slice(0, val_end_idx)
    prometheus_val_loader = DataLoader(prometheus_val_dataset, **loader_kwargs)
    print(f"Kaggle and Prometheus Validation Loaders created.")
    validation_loaders = [kaggle_val_loader, prometheus_val_loader]

    # 7. Calculate Runtime Training Parameters (Total Steps) based on Prometheus Train Loader
    print("Calculating runtime scheduler parameters...")
    def estimate_loader_len(loader, configured_events, batch_size):
        # Try getting length, fallback to estimation
        try: return len(loader)
        except TypeError: # IterableDataset may not have __len__
             return math.ceil(configured_events / batch_size) if configured_events and batch_size > 0 else 1000 # Fallback estimate

    # Use only prometheus train loader length for steps calculation
    effective_device_batches_per_epoch = estimate_loader_len(
        prometheus_train_loader, config.data.prometheus_train_events, per_device_batch_size
    )
    if effective_device_batches_per_epoch <= 0: raise ValueError("Prometheus training loader appears empty or event count is zero.")

    total_device_steps = effective_device_batches_per_epoch * config.training.max_epochs
    config.calculate_runtime_params(total_device_steps) # Calculates total_steps etc. in config
    print(f"Effective device batches/epoch (Prometheus): {effective_device_batches_per_epoch}")
    print(f"Total optimizer steps: {config.training.total_steps}")
    # Log the final calculated config state
    if wandb_logger.experiment:
         try: wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True)
         except Exception as e: warnings.warn(f"Could not update WandB config: {e}")

    # 8. Initialize Model
    print(f"Initializing Weighted Finetuning Model...")
    model = PolarBertWeightedFinetuner(
        config,
        pretrained_checkpoint_path=args.checkpoint_path,
        weight_constant=args.weight_constant
    )
    param_count_total = sum(p.numel() for p in model.parameters())
    param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {param_count_total:,}'); print(f'Trainable Parameters: {param_count_trainable:,}')

    # 9. Setup Callbacks
    print("Setting up callbacks...")
    # Monitor the primary validation metric (e.g., Prometheus directional loss)
    # Make sure the monitor key matches what's logged in on_validation_epoch_end
    config.training.checkpoint.monitor = 'val_primary_metric' # Or 'val_epoch/dir_loss_prometheus'
    callbacks = setup_callbacks(config, run_name)

    # 10. Setup Trainer
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
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        enable_model_summary=True,
        # Add deterministic=False if needed
    )

    # 11. Start Fine-tuning
    print("\nStarting Weighted Fine-tuning...")
    trainer.fit(
        model,
        # Only Prometheus data for training
        train_dataloaders=prometheus_train_loader,
        # Both Kaggle and Prometheus for validation
        val_dataloaders=validation_loaders
    )
    print("\nWeighted Fine-tuning finished.")

if __name__ == '__main__':
    # Set matmul precision globally if desired
    torch.set_float32_matmul_precision('high')
    main()