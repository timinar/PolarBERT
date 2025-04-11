import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable # Added Callable
import schedulefree
import math
import numpy as np # Added for numpy operations in target transforms

# Ensure necessary imports from pretraining module
from polarbert.pretraining import (
    load_and_process_config,
    setup_callbacks,
    get_dataloaders,
    update_training_steps,
    MODEL_CLASSES,
    compute_batch_params,
    default_transform,     # Keep default for input features
    # default_target_transform # We won't use this one for finetuning
)
from polarbert.embedding import IceCubeEmbedding
from polarbert.flash_model import TransformerBlock
from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors


class ScheduleFreeOptimizerCallback(Callback):
    """Callback to handle train/eval mode of Schedule-Free optimizer."""
    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if isinstance(optimizer, list): # Handle potential list wrapper
                optimizer = optimizer[0]
            if hasattr(optimizer, 'train'):
                print("Setting ScheduleFree optimizer to TRAIN mode.")
                optimizer.train()
            elif hasattr(optimizer, '_eval_mode'):
                 optimizer._eval_mode = False
                 print("Setting ScheduleFree optimizer internal state to TRAIN mode.")

    def on_validation_start(self, trainer, pl_module):
        if hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            if hasattr(optimizer, 'eval'):
                print("Setting ScheduleFree optimizer to EVAL mode.")
                optimizer.eval()
            elif hasattr(optimizer, '_eval_mode'):
                 optimizer._eval_mode = True
                 print("Setting ScheduleFree optimizer internal state to EVAL mode.")

    def on_validation_end(self, trainer, pl_module):
        if hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            if hasattr(optimizer, 'train'):
                print("Setting ScheduleFree optimizer back to TRAIN mode after validation.")
                optimizer.train()
            elif hasattr(optimizer, '_eval_mode'):
                 optimizer._eval_mode = False
                 print("Setting ScheduleFree optimizer internal state back to TRAIN mode after validation.")

    def on_test_start(self, trainer, pl_module):
         if hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            if hasattr(optimizer, 'eval'):
                print("Setting ScheduleFree optimizer to EVAL mode for test.")
                optimizer.eval()
            elif hasattr(optimizer, '_eval_mode'):
                 optimizer._eval_mode = True
                 print("Setting ScheduleFree optimizer internal state to EVAL mode for test.")


class SimpleTransformerCls(pl.LightningModule):
    """Backbone Transformer model used for extracting features (CLS token)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        embedding_config = config.copy()
        embedding_config.setdefault('training', {})['mask_prob'] = 0.0
        self.embedding = IceCubeEmbedding(embedding_config, masking=False)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['model']['num_layers'])
        ])

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        embeddings, padding_mask, _ = self.embedding(x)
        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask)
        return embeddings[:, 0, :]


class DirectionalHead(pl.LightningModule):
    """Head for directional prediction task using ScheduleFree optimizer."""
    def __init__(self, config: Dict[str, Any], pretrained_model: nn.Module):
        super().__init__()
        self.save_hyperparameters(config, ignore=['pretrained_model'])
        self.pretrained_model = pretrained_model
        self.config = config

        if self.hparams.get('pretrained', {}).get('freeze_backbone', False):
            print("Freezing backbone parameters.")
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        else:
             print("Finetuning backbone parameters.")

        directional_config = self.hparams.model.get('directional', {'hidden_size': 1024})
        hidden_size = directional_config['hidden_size']
        embedding_dim = self.hparams.model['embedding_dim']

        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, inp: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        freeze_backbone = self.hparams.get('pretrained', {}).get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone):
            cls_embed = self.pretrained_model(inp)

        x = self.fc1(cls_embed)
        x = self.relu(x)
        x = self.fc2(x)

        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = x / (norm + 1e-7)
        return x

    def shared_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        inp, yc = batch # yc could be None if target_transform failed or data is missing

        # <<< MODIFICATION: Check if yc is None >>>
        if yc is None:
             raise ValueError("Target data (yc) is None in shared_step. Check dataset and target_transform.")

        y_target_angles, _ = yc # Unpack targets

        # <<< MODIFICATION: Check if y_target_angles is None >>>
        if y_target_angles is None:
             # This case indicates the target_transform likely returned None for angles
             raise ValueError("y_target_angles is None in shared_step. Check target_transform function.")

        # Ensure y_target_angles is a tensor before proceeding
        if not isinstance(y_target_angles, torch.Tensor):
             # Attempt conversion if it's numpy, otherwise raise error
             if isinstance(y_target_angles, np.ndarray):
                 y_target_angles = torch.from_numpy(y_target_angles)
             else:
                 raise TypeError(f"y_target_angles must be a Tensor or NumPy array, got {type(y_target_angles)}")

        y_pred_vector = self(inp) # Get predicted vector

        # Ensure targets are on the same device as predictions
        y_target_angles = y_target_angles.to(y_pred_vector.device)

        # Convert target angles to unit vector
        try:
            # Add shape check for safety
            if y_target_angles.ndim != 2 or y_target_angles.shape[1] != 2:
                 raise ValueError(f"Expected y_target_angles shape (batch, 2), got {y_target_angles.shape}")
            y_target_vector = angles_to_unit_vector(y_target_angles[:,0], y_target_angles[:,1])
        except IndexError as e:
             print(f"IndexError during angles_to_unit_vector. y_target_angles shape: {y_target_angles.shape}")
             raise e
        except Exception as e:
             print(f"Error during angles_to_unit_vector: {e}")
             raise e


        # Calculate angular distance loss
        loss = angular_dist_score_unit_vectors(y_target_vector, y_pred_vector, epsilon=1e-4)
        return loss

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, batch_idx)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0][0].size(0)) # Add batch_size
        return loss

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, batch_idx)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0][0].size(0)) # Add batch_size
        return loss

    def configure_optimizers(self):
        """Configure the ScheduleFree optimizer."""
        training_config = self.hparams.training
        if 'max_lr' in training_config and 'div_factor' in training_config:
            lr = float(training_config['max_lr'])
            print(f"Using max_lr for ScheduleFree: {lr:.2e}")
        else:
            lr = 1e-4
            print(f"Warning: Using fallback initial_lr for ScheduleFree: {lr:.2e}")

        weight_decay = float(training_config.get('weight_decay', 0.0))
        adam_beta1 = float(training_config.get('adam_beta1', 0.9))
        adam_beta2 = float(training_config.get('adam_beta2', 0.999))
        adam_eps = float(training_config.get('adam_eps', 1e-8))

        parameters = filter(lambda p: p.requires_grad, self.parameters())

        print(f"Initializing AdamWScheduleFree with lr={lr:.2e}, weight_decay={weight_decay}, "
              f"betas=({adam_beta1}, {adam_beta2}), eps={adam_eps}")

        optimizer = schedulefree.AdamWScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            warmup_steps=training_config.get('warmup_steps', 1000) # Allow configuration
        )
        return optimizer

    # <<< MODIFICATION: Added static methods for target transforms >>>
    @staticmethod
    def target_transform_prometheus(y, c):
        """Transforms Prometheus target labels."""
        # IMPORTANT: Ensure these field names match your actual y.npy structure for Prometheus
        try:
            y_transformed = np.vstack([
                y['initial_state_azimuth'].astype(np.float32),
                y['initial_state_zenith'].astype(np.float32)
            ]).T
            return y_transformed, c.astype(np.float32)
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error in target_transform_prometheus: {e}. Check y.npy structure and dtypes.")
            # Return None or raise error depending on desired behavior
            return None, None # Example: return None if transform fails

    @staticmethod
    def target_transform_kaggle(y, c):
        """Transforms Kaggle target labels."""
        # Assumes y is already numpy array [batch, 2] with [azimuth, zenith]
        if y is None:
            print("Warning: Received None for 'y' in target_transform_kaggle.")
            return None, None
        try:
            # Basic check for shape, might need more robust checks
            if y.ndim != 2 or y.shape[1] != 2:
                print(f"Warning: Unexpected shape for 'y' in target_transform_kaggle: {y.shape}. Expected (batch, 2).")
                # Attempt conversion anyway, or return None
            return y.astype(np.float32), c.astype(np.float32)
        except (AttributeError, ValueError, TypeError) as e:
             print(f"Error in target_transform_kaggle: {e}. Check input y and c types/values.")
             return None, None # Example: return None if transform fails
    # <<< END MODIFICATION >>>


def load_pretrained_model(config: Dict[str, Any]) -> nn.Module:
    """Load and prepare pretrained model backbone."""
    print("Initializing backbone model...")
    backbone_model = SimpleTransformerCls(config)
    pretrained_config = config.get('pretrained', {})
    checkpoint_path_str = pretrained_config.get('checkpoint_path', 'new')

    if checkpoint_path_str.strip().lower() == 'new':
        print("Training from scratch ('new' specified). Backbone weights are random.")
        return backbone_model

    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained weights from: {checkpoint_path}")
    try:
        loaded_data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception as e:
         raise IOError(f"Failed to load checkpoint file {checkpoint_path}: {e}")

    if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
         pretrained_state = loaded_data['state_dict']
         print("Loaded state_dict from PyTorch Lightning checkpoint.")
    elif isinstance(loaded_data, dict):
         pretrained_state = loaded_data
         print("Loaded raw state_dict from checkpoint file.")
    else:
        raise TypeError(f"Unrecognized checkpoint format in {checkpoint_path}. Expected dict.")

    filtered_state = {}
    loaded_keys_count = 0
    for key, value in pretrained_state.items():
        adjusted_key = key.partition('model.')[2] if key.startswith('model.') else key
        if adjusted_key.startswith('embedding.') or adjusted_key.startswith('transformer_blocks.'):
            filtered_state[adjusted_key] = value
            loaded_keys_count += 1

    if loaded_keys_count == 0:
         print(f"Warning: No backbone keys ('embedding.*', 'transformer_blocks.*') found in checkpoint {checkpoint_path}. "
               "Backbone weights will remain randomly initialized.")
         return backbone_model

    print(f"Attempting to load {loaded_keys_count} filtered keys into backbone...")
    missing_keys, unexpected_keys = backbone_model.load_state_dict(filtered_state, strict=False)
    if missing_keys:
        # Filter out head keys if we expect them to be missing
        head_prefix = ('fc1.', 'relu.', 'fc2.')
        missing_backbone_keys = [k for k in missing_keys if not k.startswith(head_prefix)]
        if missing_backbone_keys:
            print(f"Warning: Missing backbone keys when loading pretrained state: {missing_backbone_keys}")
    if unexpected_keys:
         print(f"Warning: Unexpected keys found in checkpoint state_dict (might be from optimizer or different model structure): {unexpected_keys}")

    print(f"Successfully loaded {loaded_keys_count} pretrained weights into backbone.")
    return backbone_model


def main():
    """Main function to run ScheduleFree finetuning."""
    parser = argparse.ArgumentParser(description="Finetune PolarBERT with ScheduleFree optimizer.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--name', type=str, default=None, help="Optional base name for the run.")
    parser.add_argument("--job_id", type=str, default=None, help="Optional job ID for suffix.")
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='flash', help="Base architecture type (e.g., flash, swiglu, base).")
    parser.add_argument("--dataset_type", type=str, choices=['kaggle', 'prometheus'], required=True, help="Type of dataset to use (kaggle or prometheus).")
    args = parser.parse_args()

    print("Loading configuration...")
    config = load_and_process_config(args.config)

    config.setdefault('model', {})
    config.setdefault('training', {})
    config.setdefault('data', {})
    config.setdefault('pretrained', {})

    config['model']['model_type'] = args.model_type

    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    base_name = args.name or 'sf_finetuned'
    model_name = f"{base_name}_{args.model_type}_{args.dataset_type}_{suffix}"
    config['model']['model_name'] = model_name
    print(f"Effective model name: {model_name}")

    config['model'].setdefault('directional', {'hidden_size': 1024})

    print("Calculating batch parameters...")
    config['training'].setdefault('logical_batch_size', 1024)
    config['data'].setdefault('max_per_device_batch_size', config['training']['logical_batch_size'])
    try:
        batch_params = compute_batch_params(config)
        config['training'].update(batch_params)
        print(f"  per_device_batch_size: {config['training']['per_device_batch_size']}")
        print(f"  gradient_accumulation_steps: {config['training']['gradient_accumulation_steps']}")
        print(f"  effective_batch_size: {config['training']['batch_size']}")
    except KeyError as e:
        print(f"Error: Missing key required for compute_batch_params: {e}. Check config: {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error during compute_batch_params: {e}")
        exit(1)

    backbone_model = load_pretrained_model(config)

    print("Initializing finetuning head...")
    model = DirectionalHead(config, backbone_model)
    print(f"Total parameters in finetuning model: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # <<< MODIFICATION: Select and pass target_transform >>>
    target_transform: Optional[Callable] = None # Initialize
    if args.dataset_type == 'kaggle':
        target_transform = DirectionalHead.target_transform_kaggle
        print("Using Kaggle target transform.")
    elif args.dataset_type == 'prometheus':
        target_transform = DirectionalHead.target_transform_prometheus
        print("Using Prometheus target transform.")
    else:
        print(f"Warning: Unknown dataset_type '{args.dataset_type}', using no target transform.")
        # Or raise ValueError("Unsupported dataset_type...")

    print(f"Loading data for dataset type: {args.dataset_type}...")
    try:
        train_loader, val_loader = get_dataloaders(
            config,
            dataset_type=args.dataset_type,
            transform=default_transform,
            target_transform=target_transform # Pass the selected transform
        )
        print(f"Train dataloader length estimate: {len(train_loader)}")
        print(f"Validation dataloader length estimate: {len(val_loader)}")
    except Exception as e:
        print(f"Error getting dataloaders: {e}")
        exit(1)
    # <<< END MODIFICATION >>>

    print("Updating total training steps...")
    config = update_training_steps(config, train_loader)
    print(f"  Steps per epoch: {config['training']['steps_per_epoch']}")
    print(f"  Total training steps: {config['training']['total_steps']}")

    print("Setting up Wandb logger...")
    config['training'].setdefault('project', 'PolarBERT-ScheduleFree-Finetuning')
    wandb_logger = WandbLogger(
        project=config['training']['project'],
        name=model_name,
        config=config
    )

    print("Setting up PyTorch Lightning Trainer...")
    config['training'].setdefault('checkpoint', {})

    # Set matmul precision hint (as suggested by warning)
    torch.set_float32_matmul_precision('high') # Or 'medium'

    trainer = Trainer(
        max_epochs=config['training'].get('max_epochs', 10),
        callbacks=[
            *setup_callbacks(config, model_name),
            ScheduleFreeOptimizerCallback(),
            LearningRateMonitor(logging_interval='step')
        ],
        accelerator='gpu',
        devices=config['training'].get('gpus', 1),
        precision=config['training'].get('precision', '16-mixed'),
        gradient_clip_val=config['training'].get('gradient_clip_val', None),
        logger=wandb_logger,
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        enable_model_summary=True,
        deterministic=False,
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
        # gradient_clip_algorithm='norm', # Set below if needed
    )

    if config['training'].get('gradient_clip_val') is not None:
        trainer.gradient_clip_algorithm = 'norm'
        print(f"Using gradient clipping: value={config['training']['gradient_clip_val']}, algorithm='norm'")

    print("Starting training...")
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as e:
        print(f"\nAn error occurred during trainer.fit(): {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        # Optionally, try to save checkpoint on failure
        # try:
        #    trainer.save_checkpoint("emergency_checkpoint.ckpt")
        #    print("Emergency checkpoint saved.")
        # except Exception as save_e:
        #    print(f"Failed to save emergency checkpoint: {save_e}")
        exit(1) # Exit after error

    print("Training finished.")


if __name__ == '__main__':
    main()