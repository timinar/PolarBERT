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
    """
    def __init__(self, config: PolarBertConfig, pretrained_checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict()) # Save config

        # 1. Instantiate the Backbone (full model structure)
        # Use masking=False as we don't need masking logic during fine-tuning inference
        # Note: If dropout is desired during fine-tuning, need config flag
        self.backbone = PolarBertModel(config) # Contains embedding, blocks, final_norm

        # 2. Load Pre-trained Weights into Backbone
        if pretrained_checkpoint_path and pretrained_checkpoint_path.lower() != 'new':
            print(f"Loading backbone weights from: {pretrained_checkpoint_path}")
            try:
                checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)

                # Clean keys (remove 'model.' prefix if PL added it during pre-training save)
                cleaned_state_dict = {}
                prefix_to_remove = 'model.' # Adjust if your pre-training save had a different structure
                for k, v in state_dict.items():
                    if k.startswith(prefix_to_remove):
                         cleaned_state_dict[k[len(prefix_to_remove):]] = v
                    #--- TEMPORARY FIX FOR LOADING OLD STATE DICT ---
                    # REMOVE THIS BLOCK IF PRETRAINING USED THE FINAL PolarBertModel STRUCTURE
                    elif k.startswith('embedding.'): # Map old embedding keys if needed
                        cleaned_state_dict[k] = v # Assumes embedding module named 'embedding'
                    elif k.startswith('transformer_blocks.'):
                        cleaned_state_dict[k] = v
                    elif k.startswith('final_norm.'):
                         cleaned_state_dict[k] = v
                    #--- END TEMPORARY FIX ---
                    # else:
                    #     cleaned_state_dict[k] = v # Keep if no prefix needed

                # Load into the backbone PART of the model
                # We expect missing keys (prediction heads) and possibly unexpected (optimizer states)
                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)

                print("Backbone weights loaded.")
                if missing_keys:
                     # Expect dom_head and charge_head weights to be missing
                     print("  Missing keys (expected for fine-tuning):", [k for k in missing_keys if not k.endswith('head.weight') and not k.endswith('head.bias')])
                if unexpected_keys:
                     print("  Warning: Unexpected keys found:", unexpected_keys)

            except Exception as e:
                print(f"ERROR loading checkpoint: {e}. Proceeding with untrained backbone.")
                # raise e # Option to raise error instead of continuing
        else:
            print("No pretrained checkpoint provided or 'new' specified. Training backbone from scratch.")


        # 3. Define Task-Specific Prediction Head
        task = config.training.task.lower()
        backbone_embed_dim = config.model.embedding_dim
        head_hidden_size = None # Initialize

        # Safely access head configuration based on task
        if task == 'direction':
            head_config_dict = config.model.directional_head # Get the dictionary (or None)
            if head_config_dict is None:
                # You could provide a default hidden size here instead of raising an error
                # head_hidden_size = 1024 # Example default
                # warnings.warn("Directional head config missing, using default hidden size.")
                raise ValueError(f"Task is '{task}', but 'directional_head' section is missing in the model configuration.")

            # Use .get() for safe access within the dictionary, provide a default or check if None
            head_hidden_size = head_config_dict.get('hidden_size')
            if head_hidden_size is None:
                # Again, consider a default or raise error
                raise ValueError(f"Task is '{task}', but 'hidden_size' key is missing within the 'directional_head' configuration.")
            head_hidden_size = int(head_hidden_size) # Ensure it's an integer

            self.prediction_head = nn.Sequential(
                nn.Linear(backbone_embed_dim, head_hidden_size), # Use the retrieved size
                nn.ReLU(),
                nn.Linear(head_hidden_size, 3)
            )
            self.loss_fn = self.angular_distance_loss

        elif task == 'energy':
            head_config_dict = config.model.energy_head # Get the dictionary (or None)
            if head_config_dict is None:
                raise ValueError(f"Task is '{task}', but 'energy_head' section is missing in the model configuration.")

            head_hidden_size = head_config_dict.get('hidden_size')
            if head_hidden_size is None:
                raise ValueError(f"Task is '{task}', but 'hidden_size' key is missing within the 'energy_head' configuration.")
            head_hidden_size = int(head_hidden_size) # Ensure it's an integer

            self.prediction_head = nn.Sequential(
                nn.Linear(backbone_embed_dim, head_hidden_size),
                nn.ReLU(),
                nn.Linear(head_hidden_size, 1)
            )
            # Ensure you have defined self.mse_loss or the correct loss function
            self.loss_fn = self.mse_loss
        else:
            raise ValueError(f"Unsupported fine-tuning task: {task}")

        print(f"Initialized head for task: {task} with hidden size: {head_hidden_size}")


        # 4. Handle Backbone Freezing
        if config.training.freeze_backbone:
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Ensure prediction head is trainable (it should be by default)
            for param in self.prediction_head.parameters():
                 param.requires_grad = True


    def forward(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Any]) -> torch.Tensor:
        """ Forward pass for fine-tuning. Extracts CLS token embedding from backbone."""
        (x, l), _ = batch
        # Pass through embedding, transformer blocks, and final norm of the backbone
        # Set masking=False in embedding call if backbone's embedding has that arg
        # Assuming backbone.embedding is IceCubeTimeEmbedding
        hidden_states, final_padding_mask, _ = self.backbone.embedding((x, l)) # output_mask not needed
        attn_key_padding_mask = final_padding_mask

        for block in self.backbone.transformer_blocks:
            hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)
        hidden_states = self.backbone.final_norm(hidden_states)

        # Get CLS token embedding
        cls_embed = hidden_states[:, 0, :] # (B, E)

        # Pass CLS token through the task-specific head
        predictions = self.prediction_head(cls_embed)
        return predictions # (B, 3) for direction, (B, 1) or (B,) for energy

    def angular_distance_loss(self, y_pred_vectors, y_target_angles):
        """ Calculates angular distance loss """
        y_truth_vectors = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1])
        # Normalize predicted vector (important if head doesn't guarantee unit norm)
        norm = torch.linalg.vector_norm(y_pred_vectors, dim=1, keepdim=True)
        y_pred_unit_vectors = y_pred_vectors / (norm + 1e-8)
        # Calculate loss
        return angular_dist_score_unit_vectors(y_truth_vectors, y_pred_unit_vectors, epsilon=1e-4)

    def mse_loss(self, y_pred_energy, y_target_log_energy):
         """ Calculates MSE loss for energy regression """
         # Ensure correct shape (B,) vs (B, 1)
         return F.mse_loss(y_pred_energy.squeeze(-1), y_target_log_energy)


    def shared_step(self, batch):
        (x, l), y_data = batch
        y_target = y_data[0] # First element is usually angles or energy target
        # y_data structure depends on target_transform used in dataloader

        predictions = self.forward(batch) # Get predictions from forward pass

        # Calculate loss based on the task
        loss = self.loss_fn(predictions, y_target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """ Set up optimizer and scheduler for fine-tuning. """
        # Separate parameters: backbone (potentially lower LR, maybe frozen) vs head
        if self.config.training.freeze_backbone:
             print("Configuring optimizer only for prediction head.")
             parameters = self.prediction_head.parameters()
        else:
             # Option: Differential learning rates (not implemented here, simpler: use one LR)
             print("Configuring optimizer for full model (backbone + head).")
             parameters = self.parameters() # Train all parameters

        # Create optimizer using fine-tuning LR and settings
        optimizer_name = self.config.training.optimizer.lower()
        # Use fine-tuning LR!
        lr = self.config.training.max_lr

        # Reuse weight decay logic if desired, but apply to fewer params if frozen
        # Simplified: apply decay to all trainable params for now
        optimizer_kwargs = {
             'lr': lr,
             'betas': (self.config.training.adam_beta1, self.config.training.adam_beta2),
             'eps': self.config.training.adam_eps,
             'weight_decay': self.config.training.weight_decay,
             'amsgrad': self.config.training.amsgrad
        }

        if optimizer_name == 'adamw':
             optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
        else: raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Create Scheduler using fine-tuning settings
        scheduler_name = self.config.training.lr_scheduler.lower()
        if scheduler_name == 'onecycle':
             if self.config.training.total_steps is None:
                 raise ValueError("total_steps must be calculated before setting up OneCycleLR.")
             scheduler = torch.optim.lr_scheduler.OneCycleLR(
                 optimizer, max_lr=self.config.training.max_lr, total_steps=self.config.training.total_steps,
                 pct_start=self.config.training.pct_start, div_factor=self.config.training.div_factor,
                 final_div_factor=self.config.training.final_div_factor, anneal_strategy='cos'
             )
             return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif scheduler_name in ['none', None]: return optimizer
        else: raise ValueError(f"Unsupported scheduler: {scheduler_name}")


# --- Main Fine-tuning Script Logic ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PolarBERT Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the FINE-TUNING configuration YAML file.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the PRE-TRAINED model checkpoint (.ckpt). Use 'new' to train from scratch.")
    parser.add_argument('--task', type=str, choices=['direction', 'energy'], required=True, help="Fine-tuning task.")
    parser.add_argument('--dataset_type', type=str, choices=['kaggle', 'prometheus'], required=True)
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze weights of the pre-trained backbone.")
    parser.add_argument('--name', type=str, default=None, help="Custom name for the training run.")
    parser.add_argument("--job_id", type=str, default=None, help="Job ID for naming.")
    # Add other necessary args like random_time_offset if used
    # parser.add_argument("--random_time_offset", type=float, default=None)
    args = parser.parse_args()

    # 1. Load Fine-tuning Configuration
    config = PolarBertConfig.from_yaml(args.config)

    # --- Sanity Checks & Overrides ---
    if args.task != config.training.task:
        warnings.warn(f"Task mismatch: Arg specifies '{args.task}', config specifies '{config.training.task}'. Using arg '{args.task}'.")
        config.training.task = args.task # Override config with command line arg

    # Override freeze_backbone and checkpoint_path from command line
    config.training.freeze_backbone = args.freeze_backbone
    config.training.pretrained_checkpoint_path = args.checkpoint_path
    print(f"Task: {config.training.task}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Pretrained Checkpoint: {config.training.pretrained_checkpoint_path}")
    print(f"Freeze Backbone: {config.training.freeze_backbone}")


    # 2. Determine Run Name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    base_name = args.name or f"finetune_{config.training.task}_{args.dataset_type}"
    run_name = f"{base_name}_{suffix}"
    print(f"Starting run: {run_name}")


    # 3. Setup Logging (WandB)
    print("Setting up WandB logger...")
    wandb_logger = WandbLogger(
        project=config.training.logging.project, # Project name from config
        name=run_name,
        config=config.to_dict(), # Log the fine-tuning config
    )

    # 4. Calculate Per-Device Batch Size and Grad Accumulation Steps
    logical_batch = config.training.logical_batch_size
    max_per_device = config.data.max_per_device_batch_size
    per_device_batch_size = min(max_per_device, logical_batch)
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    config.training.per_device_batch_size = per_device_batch_size
    config.training.gradient_accumulation_steps = gradient_accumulation_steps


    # 5. Get Dataloaders with correct Target Transform
    print("Creating dataloaders...")
    if args.task == 'direction':
         target_transform = OldDirectionalHead.target_transform_kaggle if args.dataset_type == 'kaggle' else OldDirectionalHead.target_transform_prometheus
    elif args.task == 'energy':
         if args.dataset_type == 'kaggle': raise ValueError("Energy task not supported for Kaggle dataset.")
         target_transform = OldEnergyRegressionHead.target_transform_prometheus
    else: raise ValueError(f"Unknown task: {args.task}")

    # Add other transforms if needed (e.g., random time offset)
    # transform_fn = ...
    train_loader, val_loader = get_dataloaders(
        config,
        dataset_type=args.dataset_type,
        transform=default_transform,
        target_transform=target_transform
    )

    # 6. Calculate Runtime Training Parameters (Total Steps, etc.)
    print("Calculating runtime parameters...")
    # Estimate train_loader_len
    if config.data.train_events is None:
         warnings.warn("config.data.train_events not set. Cannot accurately estimate total steps.")
         num_batches_per_epoch = 1000 # Fallback, adjust as needed
    else:
         num_batches_per_epoch = math.ceil(config.data.train_events / config.training.per_device_batch_size)
    print(f"Estimated batches per epoch: {num_batches_per_epoch}")
    config.calculate_runtime_params(num_batches_per_epoch)

    # Log updated config to WandB
    # wandb_logger.experiment.config.update(config.to_dict(), allow_val_change=True)

    # 7. Initialize Fine-tuning Model
    print(f"Initializing PolarBertFinetuner...")
    # Pass the checkpoint path directly to the finetuner's init
    model = PolarBertFinetuner(config, pretrained_checkpoint_path=config.training.pretrained_checkpoint_path)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Parameters: {param_count:,}')


    # 8. Setup Callbacks
    print("Setting up callbacks...")
    # Pass the fine-tuning run name
    callbacks = setup_callbacks(config, run_name) # setup_callbacks needs config access to checkpoint section


    # 9. Setup Trainer
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
    )

    # 10. Start Fine-tuning
    print("\nStarting fine-tuning...")
    trainer.fit(model, train_loader, val_loader)

    print("\nFine-tuning finished.")


if __name__ == '__main__':
    main()