import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from abc import abstractmethod

import logging

from polarbert.utils.config import load_and_process_config
from polarbert.utils.data import (
    get_dataloaders, 
    add_random_time_offset, 
    default_transform
)
from polarbert.utils.training import update_training_steps, compute_batch_params
from polarbert.utils.callbacks import setup_callbacks
from polarbert.utils.sweep_params import update_config_for_wandb_sweep

from polarbert.pretraining import MODEL_CLASSES

from polarbert.base_model import _configure_optimizers, _configure_optimizers_completep
from polarbert.embedding import IceCubeEmbedding
from polarbert.flash_model import TransformerBlock, precompute_freqs_cis
from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors
from polarbert.completep import (
    is_completep_enabled,
    get_completep_config,
    compute_multipliers,
    get_init_std,
    log_completep_info
)


class SimpleTransformerCls(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = IceCubeEmbedding(config, masking=False)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['model']['num_layers'])
        ])

        # RoPE: Precompute and cache cos/sin frequencies
        self.use_rope = config['model'].get('use_rope', False)
        if self.use_rope:
            self.rope_theta = config['model'].get('rope_theta', 10000.0)
            self.rope_max_seq_len = config['model'].get('rope_max_seq_len', 512)
            head_dim = config['model']['embedding_dim'] // config['model']['num_heads']
            cos, sin = precompute_freqs_cis(
                head_dim=head_dim,
                max_seq_len=self.rope_max_seq_len,
                theta=self.rope_theta
            )
            # Register as buffers so they move with the model to GPU (not saved in checkpoints)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

        # Log QK Norm and RoPE status
        use_qk_norm = self.transformer_blocks[0].attention.use_qk_norm
        print(f"SimpleTransformerCls: QK Norm = {use_qk_norm}, RoPE = {self.use_rope}, CompleteP = {is_completep_enabled(config)}")

        # Optional final RMSNorm (for compatibility with muP-trained models)
        self.use_final_layer_norm = config['model'].get('use_final_layer_norm', True)
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.RMSNorm(config['model']['embedding_dim'])

        # CompleteP initialization
        if is_completep_enabled(config):
            self._init_completep_weights()

    def forward(self, x):
        embeddings, padding_mask, _ = self.embedding(x)

        # Slice RoPE frequencies to actual sequence length and reshape for broadcasting
        if self.use_rope:
            actual_seq_len = embeddings.shape[1]
            # Shape: (seqlen, d//2) -> (1, seqlen, 1, d//2) for broadcasting with (bsz, seqlen, n_heads, head_dim)
            rope_cos = self.rope_cos[:actual_seq_len].unsqueeze(0).unsqueeze(2)
            rope_sin = self.rope_sin[:actual_seq_len].unsqueeze(0).unsqueeze(2)
        else:
            rope_cos, rope_sin = None, None

        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask, rope_cos, rope_sin)

        if self.use_final_layer_norm:
            embeddings = self.final_layer_norm(embeddings)

        return embeddings[:, 0, :]  # Return CLS token

    def _init_completep_weights(self):
        """Initialize weights according to CompleteP parameterization.

        Initialization rules (from CompleteP papers):
        - Learnable tokens (CLS, mask): std = init_std_base (fixed variance)
        - One-hot lookup (dom_embedding): std = init_std_base (fixed variance)
        - Dense input Linear (features, position): std = init_std_base / sqrt(fan_in)
        - Hidden weights (Q, K, V, O, FF): std = init_std_base / sqrt(m_N)
        - Biases: zero
        - RMSNorm: weight = 1

        Note: For hidden layers, fan_in scaling is implicit via the width multiplier m_N.
        Only input linear layers need explicit 1/sqrt(fan_in) since their fan_in is fixed.
        """
        std_input = get_init_std(self.config, 'input_embedding')
        std_hidden = get_init_std(self.config, 'hidden')
        # Dense input layers have fan_in=3 (xyz coordinates or features)
        std_input_linear = get_init_std(self.config, 'input_linear', fan_in=3)

        for name, param in self.named_parameters():
            # Zero all biases first (before other checks, since bias is 1D)
            if param.dim() == 1 and 'bias' in name:
                nn.init.zeros_(param)
                continue

            # Learnable tokens: fixed variance (no 1/sqrt(d_in) since not processing input)
            if 'embedding.cls_embedding' in name:
                nn.init.normal_(param, mean=0.0, std=std_input)
            elif 'embedding.mask_token_embedding' in name:
                nn.init.normal_(param, mean=0.0, std=std_input)

            # One-hot lookup table: fixed variance (one-hot input has no variance issue)
            elif 'embedding.dom_embedding' in name:
                if param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=std_input)

            # Dense input Linear layers: scale by 1/sqrt(fan_in) for stable signal variance
            elif 'embedding.features_embedding' in name:
                if param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=std_input_linear)
            elif 'embedding.position_embedding' in name:
                if param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=std_input_linear)

            # Hidden weights (Q, K, V, W_O, FF): scale by 1/sqrt(m_N)
            # fan_in scaling is implicit via width multiplier
            elif any(s in name for s in ['wq.weight', 'wk.weight', 'wv.weight', 'wo.weight',
                                          'feed_forward.0.weight', 'feed_forward.2.weight']):
                nn.init.normal_(param, mean=0.0, std=std_hidden)

        # RMSNorm: standard init (weight=1)
        for module in self.modules():
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
    

class PredictionHead(pl.LightningModule):
    """Generic head for multiple downstream tasks."""
    @abstractmethod
    def __init__(self, config: Dict[str, Any], pretrained_model: Optional[nn.Module] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.config = config
        self._lr_scales = None  # For CompleteP per-group LR scaling
        # Initialize a new pretrained model if none is provided
        self.pretrained_model = pretrained_model or SimpleTransformerCls(config)
        if config.get('pretrained', {}).get('freeze_backbone', False):
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

    @abstractmethod
    def forward(self, inp):
        pass

    @abstractmethod
    def shared_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if is_completep_enabled(self.config):
            optimizer, lr_scales, scheduler_dict = _configure_optimizers_completep(
                self.config, self.named_parameters()
            )
            self._lr_scales = lr_scales
            if scheduler_dict is None:
                return optimizer
            return [optimizer], [scheduler_dict]
        return _configure_optimizers(self.config, self.parameters())

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Apply CompleteP per-group LR scaling before optimizer step."""
        if self._lr_scales is not None:
            # Store current learning rates (as set by the scheduler)
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            # Apply the proper learning scaling to each parameter group
            for param_group, lr_scale in zip(optimizer.param_groups, self._lr_scales):
                param_group['lr'] *= lr_scale
            # Call the parent optimizer step
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
            # Restore the original learning rates (without scaling)
            for param_group, original_lr in zip(optimizer.param_groups, current_lrs):
                param_group['lr'] = original_lr
        else:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)


class DirectionalHead(PredictionHead):
    """Head for directional prediction task."""
    def __init__(self, config: Dict[str, Any], pretrained_model: Optional[nn.Module] = None):
        super().__init__(config, pretrained_model)

        # Directional prediction layers
        self.fc1 = nn.Linear(config['model']['embedding_dim'], config['model']['directional']['hidden_size'])
        activation_name = config['model'].get('activation', 'gelu').lower()
        self.activation = nn.GELU() if activation_name == 'gelu' else nn.ReLU()
        self.fc2 = nn.Linear(config['model']['directional']['hidden_size'], 3)

        # CompleteP readout initialization
        if is_completep_enabled(config):
            self._init_completep_readout()

    def forward(self, inp):
        # Handle the input tuple and get CLS embedding
        with torch.set_grad_enabled(not self.config.get('pretrained', {}).get('freeze_backbone', False)):
            cls_embed = self.pretrained_model(inp)
        
        x = self.fc1(cls_embed)
        x = self.activation(x)
        x = self.fc2(x)
        
        # Normalize to unit vector
        norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        x = x / (norm + 1e-8)  # Add small epsilon to prevent division by zero
        
        return x
    
    def shared_step(self, batch, batch_idx):
        inp, yc = batch
        y, c = yc
        y_pred = self(inp)
        y_truth = angles_to_unit_vector(y[:,0], y[:,1])
        loss = angular_dist_score_unit_vectors(y_truth, y_pred, epsilon=1e-4)
        return loss
    
    @staticmethod
    def target_transform_prometheus(y, c):
        y = np.vstack([y['initial_state_azimuth'].astype(np.float32), y['initial_state_zenith'].astype(np.float32)]).T
        return y, c.astype(np.float32)

    @staticmethod
    def target_transform_kaggle(y, c):
        return y.astype(np.float32), c.astype(np.float32)

    def _init_completep_readout(self):
        """Initialize readout weights according to CompleteP parameterization."""
        std_readout = get_init_std(self.config, 'readout')
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std_readout)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std_readout)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)


class EnergyRegressionHead(PredictionHead):
    """Head for energy regression task."""
    def __init__(self, config: Dict[str, Any], pretrained_model: Optional[nn.Module] = None):
        super().__init__(config, pretrained_model)
        
        # Energy regression layers
        self.fc1 = nn.Linear(config['model']['embedding_dim'], config['model']['directional']['hidden_size'])
        activation_name = config['model'].get('activation', 'gelu').lower()
        self.activation = nn.GELU() if activation_name == 'gelu' else nn.ReLU()
        self.fc2 = nn.Linear(config['model']['directional']['hidden_size'], 1)

    def forward(self, inp):
        # Handle the input tuple and get CLS embedding
        with torch.set_grad_enabled(not self.config.get('pretrained', {}).get('freeze_backbone', False)):
            cls_embed = self.pretrained_model(inp)
        
        x = self.fc1(cls_embed)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x.view(-1)
    
    def shared_step(self, batch, batch_idx):
        inp, yc = batch
        y_truth, _ = yc
        y_pred = self(inp)
        loss = nn.MSELoss()(y_truth, y_pred)
        return loss
    
    @staticmethod
    def target_transform_prometheus(y, c):
        y = np.log10(y['initial_state_energy'].astype(np.float32))
        return y, c.astype(np.float32)

    @staticmethod
    def target_transform_kaggle(y, c):
        raise(ValueError("Kaggle dataset does not contain energy targets"))


def load_pretrained_model(config: Dict[str, Any]):
    """Load and prepare pretrained model."""
    # Validate CompleteP usage - only allowed for training from scratch
    checkpoint_path = config['pretrained']['checkpoint_path'].strip().lower()
    if is_completep_enabled(config) and checkpoint_path != 'new':
        raise ValueError(
            "CompleteP can only be used for training from scratch. "
            "Set pretrained.checkpoint_path to 'new' or disable completep.enabled. "
            f"Current checkpoint_path: {config['pretrained']['checkpoint_path']}"
        )

    # Initialize new model for finetuning
    model = SimpleTransformerCls(config)

    if checkpoint_path == 'new':
        print("Training from scratch")
        if is_completep_enabled(config):
            log_completep_info(config)
        return model

    # Load pretrained weights from the full model
    checkpoint_path = Path(config['pretrained']['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pretrained_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'state_dict' in pretrained_state: # PyTorch Lightning checkpoints contain extra data in addition to the state dict
        pretrained_state = pretrained_state['state_dict']
    assert 'transformer_blocks.0.feed_forward.0.weight' in pretrained_state, "State dict does not contain the expected keys. Check the checkpoint format."

    # Filter state dict to only include embedding and transformer blocks
    filtered_state = {}
    use_final_layer_norm = config['model'].get('use_final_layer_norm', True)
    for key, value in pretrained_state.items():
        if key.startswith('embedding.') or key.startswith('transformer_blocks.'):
            filtered_state[key] = value
        elif key.startswith('final_layer_norm.') and use_final_layer_norm:
            filtered_state[key] = value

    # Load filtered state dict
    model.load_state_dict(filtered_state, strict=False)
    print("Loaded pretrained weights for embedding and transformer blocks")

    return model


def load_full_model(config: Dict[str, Any], task: str = 'direction'):
    """Load full finetuned model (backbone + head) for continued fine-tuning."""
    # Create the appropriate head model
    if task == 'direction':
        model = DirectionalHead(config)
    elif task == 'energy':
        model = EnergyRegressionHead(config)
    else:
        raise ValueError(f'Unsupported task: {task}')

    checkpoint_path = Path(config['pretrained']['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    pretrained_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']

    # Load the full state dict (including fc1, fc2, and pretrained_model.*)
    model.load_state_dict(pretrained_state, strict=True)
    print(f"Loaded full model from {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['direction', 'energy'], default='direction')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='flash')
    parser.add_argument("--dataset_type", type=str, choices=['kaggle', 'prometheus'], default='kaggle')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the pretrained model checkpoint. If 'new', the model will be trained from scratch.")
    parser.add_argument("--continue_finetuning", action="store_true", help="Continue fine-tuning from a full model checkpoint (backbone + head)")
    parser.add_argument("--schedule-free", action="store_true", help="Use schedule-free optimizer (overrides config lr_scheduler)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    import os
    seed = args.seed or int(os.environ.get('PL_GLOBAL_SEED', 42))
    pl.seed_everything(seed, workers=True)
    logging.info(f"Random seed set to {seed}")

    if args.dataset_type == 'kaggle' and args.task != 'direction':
        raise ValueError("Kaggle dataset only contains fine-tuning targets for directional reconstruction")

    # Load and process config
    config = load_and_process_config(args.config)

    # Apply schedule-free if requested via CLI
    if args.schedule_free:
        config['training']['lr_scheduler'] = 'schedule_free'
        config['training']['schedule_free'] = True

    # Setup model name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = f"{args.name or config['model']['model_name'] or 'finetuned'}_{suffix}"
    config['model']['model_name'] = model_name
    
    # Setup training
    torch.set_float32_matmul_precision('high')
    wandb_logger = WandbLogger(
        project=config['training'].get('project', 'PolarBERT-finetuning'),
        name=model_name,
        config=config
    )
    
    # Update config with parameters from wandb sweep
    update_config_for_wandb_sweep(config, wandb_logger.experiment.config)
    
    # Compute and update batch parameters
    batch_params = compute_batch_params(config)
    config['training'].update(batch_params)
    
    # Override checkpoint_path if provided in command line
    if args.checkpoint_path is not None:
        config.setdefault('pretrained', {})['checkpoint_path'] = args.checkpoint_path

    # Add model type to config if not present
    if 'model_type' not in config.get('pretrained', {}):
        config.setdefault('pretrained', {})['model_type'] = args.model_type
    
    # Setup directional config if not present
    if 'directional' not in config['model']:
        config['model']['directional'] = {
            'hidden_size': 1024,
        }
    
    # Initialize finetuning model
    if args.continue_finetuning:
        # Load full model for continued fine-tuning
        model = load_full_model(config, task=args.task)
    else:
        # Original behavior: load backbone and create new head
        pretrained_model = load_pretrained_model(config)
        if args.task == 'direction':
            model = DirectionalHead(config, pretrained_model)
        elif args.task == 'energy':
            model = EnergyRegressionHead(config, pretrained_model)
        else:
            raise ValueError(f'Unsupported task: {args.task}')

    # Optional torch.compile for faster training (especially with QK Norm)
    if config['training'].get('torch_compile', False):
        print("Compiling model with torch.compile...")
        model.pretrained_model = torch.compile(model.pretrained_model)
        print("Model compiled successfully")

    # Select the right target transform based on the dataset type
    if args.dataset_type == 'kaggle':
        target_transform = model.target_transform_kaggle
    elif args.dataset_type == 'prometheus':
        target_transform = model.target_transform_prometheus
    else:
        assert False
    
    # Get data loaders
    random_time_offset_std = config['training'].get('random_time_offset') # Read from config
    if random_time_offset_std is not None:
        logging.info(f"Applying random time offset with std: {random_time_offset_std}")
        transform = add_random_time_offset(random_time_offset_std)
    else:
        transform = default_transform
    train_loader, val_loader = get_dataloaders(config, dataset_type=args.dataset_type, transform=transform, target_transform=target_transform)
    
    # Update training steps
    config = update_training_steps(config, train_loader)

    # Ensure the full updated config is logged on W&B before we start training
    wandb_logger.experiment.config.update(config, allow_val_change=True)
    
    # Setup training with gradient scaling
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=[
            *setup_callbacks(config, model_name),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ],
        accelerator='gpu',
        devices=config['training']['gpus'],
        precision=config['training'].get('precision', '16-mixed'),
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=wandb_logger,
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        enable_model_summary=True,
        deterministic=False,  # Add this for better performance
        gradient_clip_algorithm='norm',  # Add this for better stability
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
