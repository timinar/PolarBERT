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
from polarbert.config import PolarBertConfig
from polarbert.time_embed_polarbert import PolarBertModel, RMSNorm # Backbone
from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors
from polarbert.dataloader_utils import (get_dataloaders, target_transform_prometheus,
                                        target_transform_kaggle, default_transform)
from polarbert.prometheus_dataset import IceCubeDataset as PrometheusDataset
from polarbert.icecube_dataset import IceCubeDataset as KaggleDataset
from polarbert.te_pretraining import setup_callbacks # Includes config saving


class PolarBertMixedFinetuner(pl.LightningModule):
    """
    LightningModule for mixed fine-tuning of PolarBertModel using
    Kaggle (DOM loss) and Prometheus (DOM + Direction loss) datasets.
    """
    def __init__(self, config: PolarBertConfig, pretrained_checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config
        hparams_to_save = config.to_dict()
        hparams_to_save['training']['pretrained_checkpoint_path_runtime'] = pretrained_checkpoint_path
        self.save_hyperparameters(hparams_to_save)

        self.pooling_mode = self.config.training.directional_pooling_mode
        self.lambda_dom_k = self.config.training.lambda_dom_kaggle
        self.lambda_dom_p = self.config.training.lambda_dom_prometheus
        self.lambda_dir_p = self.config.training.lambda_dir_prometheus

        print(f"Directional head pooling mode: {self.pooling_mode}")
        print(f"Loss Lambdas: dom_k={self.lambda_dom_k}, dom_p={self.lambda_dom_p}, dir_p={self.lambda_dir_p}")

        # --- Backbone Instantiation & Weight Loading ---
        self.backbone = PolarBertModel(config)
        if pretrained_checkpoint_path and pretrained_checkpoint_path.lower() != 'new':
            print(f"Loading backbone weights from: {pretrained_checkpoint_path}")
            try:
                checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=True)
                state_dict = checkpoint.get('state_dict', checkpoint)
                cleaned_state_dict = {}
                prefixes_to_remove = ['model.', 'backbone.']
                for k, v in state_dict.items():
                    key_modified = False
                    for prefix in prefixes_to_remove:
                        if k.startswith(prefix):
                            cleaned_state_dict[k[len(prefix):]] = v
                            key_modified = True; break
                    if not key_modified and not k.startswith(('optimizer.', 'lr_scheduler', '_forward_module')):
                         cleaned_state_dict[k] = v
                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
                print("Backbone weights loaded.")
                filtered_missing = [k for k in missing_keys if not k.startswith(('dom_head', 'charge_head'))]
                filtered_unexpected = [k for k in unexpected_keys if not k.startswith(('dom_head', 'charge_head'))]
                if filtered_missing: print("  Warning: Missing keys in backbone:", filtered_missing)
                if filtered_unexpected: print("  Warning: Unexpected keys in backbone state_dict:", filtered_unexpected)
            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Proceeding with untrained backbone.")
        else:
            print("No pretrained checkpoint provided or 'new' specified. Training backbone from scratch.")

        # --- Directional Head Definition ---
        backbone_embed_dim = self.config.model.embedding_dim
        head_config_dict = self.config.model.directional_head
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

    def forward_features(self, batch_input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Runs backbone embedding and transformer blocks."""
        (x, l) = batch_input
        hidden_states, final_padding_mask, output_mask = self.backbone.embedding((x, l))
        attn_key_padding_mask = final_padding_mask
        for block in self.backbone.transformer_blocks:
            hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)
        hidden_states = self.backbone.final_norm(hidden_states)
        seq_padding_mask = final_padding_mask[:, 1:]
        return hidden_states, output_mask, seq_padding_mask

    def forward_heads(self, hidden_states: torch.Tensor, seq_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs prediction heads."""
        cls_embed = hidden_states[:, 0, :]
        sequence_embeds = hidden_states[:, 1:, :]
        if not hasattr(self.backbone, 'dom_head'): raise AttributeError("Backbone missing 'dom_head'.")
        dom_logits = self.backbone.dom_head(sequence_embeds)

        if self.pooling_mode == 'cls': dir_head_input = cls_embed
        elif self.pooling_mode == 'mean':
            valid_token_mask = ~seq_padding_mask
            valid_token_mask_expanded = valid_token_mask.unsqueeze(-1).expand_as(sequence_embeds).float()
            masked_sequence_embeds = sequence_embeds * valid_token_mask_expanded
            summed_embeds = masked_sequence_embeds.sum(dim=1)
            num_valid_tokens = valid_token_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-6)
            dir_head_input = summed_embeds / num_valid_tokens
        else: raise ValueError(f"Invalid pooling_mode: {self.pooling_mode}")
        dir_pred = self.directional_head(dir_head_input)
        return dom_logits, dir_pred

    def calculate_dom_loss(self, dom_logits, batch_input, output_mask):
        """Calculates masked DOM prediction loss."""
        (x, l) = batch_input
        true_dom_ids = x[:, :, 3].long()
        dom_loss = torch.tensor(0.0, device=dom_logits.device, dtype=dom_logits.dtype)
        mask_to_use = output_mask if self.training else (true_dom_ids != 0)

        if mask_to_use is not None and mask_to_use.sum() > 0:
            dom_targets = true_dom_ids - 1
            if mask_to_use.shape == dom_targets.shape:
                 masked_logits = dom_logits[mask_to_use]
                 masked_targets = dom_targets[mask_to_use]
                 if masked_logits.nelement() > 0:
                      dom_loss = F.cross_entropy(masked_logits, masked_targets, ignore_index=-1)
            else:
                 warnings.warn(f"DOM Loss: Mask shape {mask_to_use.shape} incompatible with data shape {dom_targets.shape}. Skipping loss calculation.")
        return dom_loss

    def calculate_dir_loss(self, dir_pred, batch_target):
        """Calculates angular distance loss for direction."""
        if batch_target is None or batch_target[0] is None:
             return torch.tensor(float('nan'), device=dir_pred.device, dtype=dir_pred.dtype)
        y_target_angles = batch_target[0]
        y_truth_vectors = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1])
        norm = torch.linalg.vector_norm(dir_pred, dim=1, keepdim=True)
        y_pred_unit_vectors = dir_pred / (norm + 1e-8)
        loss = angular_dist_score_unit_vectors(y_truth_vectors, y_pred_unit_vectors, epsilon=1e-4)
        return loss

    def training_step(self, batch: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], Any]], batch_idx: int):
        """ Processes one combined batch containing data from Kaggle and Prometheus. """
        kaggle_batch = batch.get('kaggle')
        prometheus_batch = batch.get('prometheus')
        dom_loss_k = torch.tensor(0.0, device=self.device)
        dom_loss_p = torch.tensor(0.0, device=self.device)
        dir_loss_p = torch.tensor(0.0, device=self.device)

        # Process Kaggle
        if kaggle_batch:
            kaggle_input, _ = kaggle_batch
            hidden_states_k, output_mask_k, seq_padding_mask_k = self.forward_features(kaggle_input)
            dom_logits_k, _ = self.forward_heads(hidden_states_k, seq_padding_mask_k)
            dom_loss_k = self.calculate_dom_loss(dom_logits_k, kaggle_input, output_mask_k)

        # Process Prometheus
        if prometheus_batch:
            prometheus_input, prometheus_target = prometheus_batch
            hidden_states_p, output_mask_p, seq_padding_mask_p = self.forward_features(prometheus_input)
            dom_logits_p, dir_pred_p = self.forward_heads(hidden_states_p, seq_padding_mask_p)
            dom_loss_p = self.calculate_dom_loss(dom_logits_p, prometheus_input, output_mask_p)
            dir_loss_p = self.calculate_dir_loss(dir_pred_p, prometheus_target)

        # Combine losses
        total_loss = (self.lambda_dom_k * dom_loss_k +
                      self.lambda_dom_p * dom_loss_p +
                      self.lambda_dir_p * dir_loss_p)

        # Logging
        log_dict = {}
        if kaggle_batch: log_dict['train_step/dom_loss_k'] = dom_loss_k.item()
        if prometheus_batch:
            log_dict['train_step/dom_loss_p'] = dom_loss_p.item()
            log_dict['train_step/dir_loss_p'] = dir_loss_p.item()
        if kaggle_batch or prometheus_batch:
             log_dict['train_step/loss_combined'] = total_loss.item()
             lr = self.optimizers().param_groups[0]['lr']
             self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Any], batch_idx: int, dataloader_idx: int):
        """ Processes a validation batch and stores the output. """
        batch_input, batch_target = batch
        hidden_states, _, seq_padding_mask = self.forward_features(batch_input)
        dom_logits, dir_pred = self.forward_heads(hidden_states, seq_padding_mask)

        losses = {}
        if dataloader_idx == 0: # Kaggle Validation
            dom_loss_k = self.calculate_dom_loss(dom_logits, batch_input, output_mask=None)
            losses['val/dom_loss_kaggle'] = dom_loss_k
            dir_loss_k = self.calculate_dir_loss(dir_pred, batch_target)
            losses['val/dir_loss_kaggle'] = dir_loss_k
            self.validation_step_outputs_kaggle.append(losses)
        elif dataloader_idx == 1: # Prometheus Validation
             dom_loss_p = self.calculate_dom_loss(dom_logits, batch_input, output_mask=None)
             losses['val/dom_loss_prometheus'] = dom_loss_p
             dir_loss_p = self.calculate_dir_loss(dir_pred, batch_target)
             losses['val/dir_loss_prometheus'] = dir_loss_p
             self.validation_step_outputs_prometheus.append(losses)
        else:
            warnings.warn(f"Unexpected dataloader_idx in validation_step: {dataloader_idx}")

    def on_validation_epoch_end(self):
        """ Aggregates stored validation step outputs and logs epoch metrics. """
        avg_losses = {}
        combined_val_loss_terms = []

        # Process Kaggle Outputs
        if self.validation_step_outputs_kaggle:
            kaggle_outputs = self.validation_step_outputs_kaggle
            dom_k_valid = [x['val/dom_loss_kaggle'] for x in kaggle_outputs if not torch.isnan(x.get('val/dom_loss_kaggle', torch.nan))]
            dir_k_valid = [x['val/dir_loss_kaggle'] for x in kaggle_outputs if not torch.isnan(x.get('val/dir_loss_kaggle', torch.nan))]
            dom_k_epoch = torch.stack(dom_k_valid).mean() if dom_k_valid else torch.tensor(float('nan'), device=self.device)
            dir_k_epoch = torch.stack(dir_k_valid).mean() if dir_k_valid else torch.tensor(float('nan'), device=self.device)
            avg_losses['val_epoch/dom_loss_kaggle'] = dom_k_epoch
            avg_losses['val_epoch/dir_loss_kaggle'] = dir_k_epoch
            if not torch.isnan(dom_k_epoch): combined_val_loss_terms.append(self.lambda_dom_k * dom_k_epoch)

        # Process Prometheus Outputs
        if self.validation_step_outputs_prometheus:
             prometheus_outputs = self.validation_step_outputs_prometheus
             dom_p_valid = [x['val/dom_loss_prometheus'] for x in prometheus_outputs if not torch.isnan(x.get('val/dom_loss_prometheus', torch.nan))]
             dir_p_valid = [x['val/dir_loss_prometheus'] for x in prometheus_outputs if not torch.isnan(x.get('val/dir_loss_prometheus', torch.nan))]
             dom_p_epoch = torch.stack(dom_p_valid).mean() if dom_p_valid else torch.tensor(float('nan'), device=self.device)
             dir_p_epoch = torch.stack(dir_p_valid).mean() if dir_p_valid else torch.tensor(float('nan'), device=self.device)
             avg_losses['val_epoch/dom_loss_prometheus'] = dom_p_epoch
             avg_losses['val_epoch/dir_loss_prometheus'] = dir_p_epoch
             if not torch.isnan(dom_p_epoch): combined_val_loss_terms.append(self.lambda_dom_p * dom_p_epoch)
             if not torch.isnan(dir_p_epoch): combined_val_loss_terms.append(self.lambda_dir_p * dir_p_epoch)

        # Calculate final combined validation loss for checkpointing
        avg_losses['val/loss_combined'] = torch.stack(combined_val_loss_terms).sum() if combined_val_loss_terms else torch.tensor(float('nan'), device=self.device)
        # Log all aggregated metrics
        self.log_dict(avg_losses, prog_bar=True, sync_dist=True)

        # Clear the stored outputs for the next epoch
        self.validation_step_outputs_kaggle.clear()
        self.validation_step_outputs_prometheus.clear()

    def configure_optimizers(self):
        """Sets up optimizer and scheduler using self.config."""
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
             return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-8)

        # Parameter Grouping Logic
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, RMSNorm, torch.nn.Embedding)
        relevant_modules = self.directional_head.named_modules() if freeze_backbone else self.named_modules()

        for mn, m in relevant_modules:
            for pn, p in m.named_parameters():
                if not p.requires_grad: continue
                fpn = f'{mn}.{pn}' if mn else pn
                if freeze_backbone: fpn = f'directional_head.{fpn}'
                if fpn not in param_dict: continue # Only consider params being optimized

                if pn.endswith('bias'): no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules): decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules): no_decay.add(fpn)
                elif isinstance(m, (RMSNorm, torch.nn.LayerNorm)): no_decay.add(fpn)

        union_params = decay | no_decay
        unassigned_params = set(param_dict.keys()) - union_params
        if len(unassigned_params) > 0: no_decay.update(unassigned_params)

        print(f"Optimizing {len(decay)} decaying parameter tensors and {len(no_decay)} non-decaying tensors.")
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": self.config.training.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0}, ]
        optim_groups = [group for group in optim_groups if group["params"]]

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
            if self.config.training.total_steps is None: raise ValueError("Total steps required for OneCycleLR.")
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
            print("Using constant learning rate.")
            return optimizer
        else: raise ValueError(f"Unsupported scheduler: {scheduler_name}")

# --- Main Training Script ---

def main():
    parser = argparse.ArgumentParser(description="Mixed Fine-tune PolarBERT")
    parser.add_argument('--config', type=str, required=True, help="Path to the MIXED fine-tuning config YAML.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to PRE-TRAINED backbone checkpoint (.ckpt or .pth). Use 'new' for scratch.")
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, help="Freeze backbone (overrides config value).")
    parser.add_argument('--name', type=str, default=None, help="Custom name for the WandB run.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID (e.g., SLURM ID) for run naming.")
    parser.add_argument('--lambda_dom_k', type=float, default=None, help="Override lambda_dom_kaggle.")
    parser.add_argument('--lambda_dom_p', type=float, default=None, help="Override lambda_dom_prometheus.")
    parser.add_argument('--lambda_dir_p', type=float, default=None, help="Override lambda_dir_prometheus.")
    args = parser.parse_args()

    # 1. Load Configuration
    config = PolarBertConfig.from_yaml(args.config)

    # 2. Apply Command-Line Overrides
    if args.freeze_backbone is not None: config.training.freeze_backbone = args.freeze_backbone
    if args.lambda_dom_k is not None: config.training.lambda_dom_kaggle = args.lambda_dom_k
    if args.lambda_dom_p is not None: config.training.lambda_dom_prometheus = args.lambda_dom_p
    if args.lambda_dir_p is not None: config.training.lambda_dir_prometheus = args.lambda_dir_p
    config.training.pretrained_checkpoint_path_runtime = args.checkpoint_path

    # 3. Determine Run Name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    base_name = args.name or f"mixed_{config.training.directional_pooling_mode}"
    run_name = f"{base_name}_{suffix}"
    config.model.model_name = run_name # Store run name in config for saving
    print(f"Run Name: {run_name}")

    # 4. Setup Logging
    print("Setting up WandB logger...")
    wandb_logger = WandbLogger(
        project=config.training.logging.project,
        name=run_name,
        entity=config.training.logging.entity, # Optional
        config=config.to_dict(), # Log final config state
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
    if gradient_accumulation_steps < 2: warnings.warn("Grad accum steps < 2; loss averaging might be uneven.")

    # 6. Instantiate Dataloaders
    print("Creating dataloaders...")
    # Kaggle (uses config.data.train_events / val_events)
    kaggle_train_loader, kaggle_val_loader = get_dataloaders(
        config, dataset_type='kaggle', transform=default_transform,
        target_transform=target_transform_kaggle,
        override_batch_size=per_device_batch_size
    )
    # Prometheus (manual split uses config.data.prometheus_*_events)
    prom_train_events = config.data.prometheus_train_events
    prom_val_events = config.data.prometheus_val_events
    if prom_train_events is None or prom_val_events is None: raise ValueError("Prometheus event counts must be set.")

    full_prometheus_dataset = PrometheusDataset(
        data_dir=config.data.prometheus_dir, batch_size=per_device_batch_size,
        transform=default_transform, target_transform=target_transform_prometheus
    )
    # Correctly get total size if dataset supports it, otherwise estimate
    try: total_prometheus_events = len(full_prometheus_dataset.x)
    except: total_prometheus_events = prom_train_events + prom_val_events; warnings.warn("Could not get exact Prometheus dataset size.")

    val_end_idx = min(prom_val_events, total_prometheus_events)
    train_end_idx = min(val_end_idx + prom_train_events, total_prometheus_events)
    print(f"Prometheus splitting: Val End Idx={val_end_idx}, Train End Idx={train_end_idx}")

    prometheus_val_dataset = full_prometheus_dataset.slice(0, val_end_idx)
    prometheus_train_dataset = full_prometheus_dataset.slice(val_end_idx, train_end_idx)

    loader_kwargs = {'batch_size': None, 'num_workers': config.data.num_workers, 'pin_memory': config.data.pin_memory, 'persistent_workers': config.data.persistent_workers and config.data.num_workers > 0}
    prometheus_train_loader = DataLoader(prometheus_train_dataset, **loader_kwargs)
    prometheus_val_loader = DataLoader(prometheus_val_dataset, **loader_kwargs)

    # 7. Calculate Runtime Training Parameters (Total Steps)
    print("Calculating runtime scheduler parameters...")
    # Estimate loader lengths carefully
    def estimate_loader_len(loader, configured_events, batch_size):
        try: return len(loader)
        except TypeError: return configured_events // batch_size if configured_events and batch_size > 0 else 0

    len_kaggle_train = estimate_loader_len(kaggle_train_loader, config.data.train_events, per_device_batch_size)
    len_prometheus_train = estimate_loader_len(prometheus_train_loader, config.data.prometheus_train_events, per_device_batch_size)

    if len_kaggle_train == 0 and len_prometheus_train == 0: raise ValueError("Both training loaders appear empty.")
    effective_device_batches_per_epoch = max(len_kaggle_train, len_prometheus_train)
    total_device_steps = effective_device_batches_per_epoch * config.training.max_epochs

    config.calculate_runtime_params(total_device_steps)
    print(f"Effective device batches/epoch (max_loader_len): {effective_device_batches_per_epoch}")
    print(f"Total optimizer steps: {config.training.total_steps}")
    wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True)

    # 8. Initialize Model
    print(f"Initializing Mixed Finetuning Model...")
    model = PolarBertMixedFinetuner(config, pretrained_checkpoint_path=args.checkpoint_path)
    param_count_total = sum(p.numel() for p in model.parameters())
    param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {param_count_total:,}'); print(f'Trainable Parameters: {param_count_trainable:,}')

    # 9. Setup Callbacks
    print("Setting up callbacks...")
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
    )

    # 11. Start Fine-tuning
    print("\nStarting Mixed Fine-tuning...")
    trainer.fit(
        model,
        train_dataloaders={'kaggle': kaggle_train_loader, 'prometheus': prometheus_train_loader},
        val_dataloaders=[kaggle_val_loader, prometheus_val_loader] # List for validation
    )
    print("\nMixed Fine-tuning finished.")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()