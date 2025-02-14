import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from polarbert.pretraining import (
    load_and_process_config,
    setup_callbacks,
    get_dataloaders,
    update_training_steps,
    compute_batch_params,
    MODEL_CLASSES,
    SWEEP_PARAMS,
)
from polarbert.base_model import _configure_optimizers
from polarbert.embedding import IceCubeEmbedding
from polarbert.flash_model import TransformerBlock
from polarbert.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors


class SimpleTransformerCls(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = IceCubeEmbedding(config, masking=False)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['model']['num_layers'])
        ])

    def forward(self, x):
        embeddings, padding_mask, _ = self.embedding(x)
        
        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask)
        
        return embeddings[:, 0, :]  # Return CLS token


class DirectionalHead(pl.LightningModule):
    """Head for directional prediction task."""
    def __init__(self, config: Dict[str, Any], pretrained_model: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.pretrained_model = pretrained_model
        self.config = config
        
        # Freeze pretrained model if specified
        if config.get('pretrained', {}).get('freeze_backbone', False):
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # Directional prediction layers
        self.fc1 = nn.Linear(config['model']['embedding_dim'], config['model']['directional']['hidden_size'])
        # TODO: allow for different activation functions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config['model']['directional']['hidden_size'], 3)

    def forward(self, inp):
        # Handle the input tuple and get CLS embedding
        with torch.set_grad_enabled(not self.config.get('pretrained', {}).get('freeze_backbone', False)):
            cls_embed = self.pretrained_model(inp)
        
        x = self.fc1(cls_embed)
        x = self.relu(x)
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

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return _configure_optimizers(self.config, self.parameters())

    @staticmethod
    def target_transform(y, c):
        y = np.vstack([y['initial_state_azimuth'].astype(np.float32), y['initial_state_zenith'].astype(np.float32)]).T
        return y, c.astype(np.float32)


class EnergyRegressionHead(pl.LightningModule):
    """Head for energy regression task."""
    def __init__(self, config: Dict[str, Any], pretrained_model: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.pretrained_model = pretrained_model
        self.config = config
        
        # Freeze pretrained model if specified
        if config.get('pretrained', {}).get('freeze_backbone', False):
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # Energy regression layers
        self.fc1 = nn.Linear(config['model']['embedding_dim'], config['model']['directional']['hidden_size'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config['model']['directional']['hidden_size'], 1)

    def forward(self, inp):
        # Handle the input tuple and get CLS embedding
        with torch.set_grad_enabled(not self.config.get('pretrained', {}).get('freeze_backbone', False)):
            cls_embed = self.pretrained_model(inp)
        
        x = self.fc1(cls_embed)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x.view(-1)
    
    def shared_step(self, batch, batch_idx):
        inp, yc = batch
        y_truth, _ = yc
        y_pred = self(inp)
        loss = nn.MSELoss()(y_truth, y_pred)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return _configure_optimizers(self.config, self.parameters())
        
    @staticmethod
    def target_transform(y, c):
        y = np.log10(y['initial_state_energy'].astype(np.float32))
        return y, c.astype(np.float32)


def load_pretrained_model(config: Dict[str, Any]):
    """Load and prepare pretrained model."""
    # Initialize new model for finetuning
    model = SimpleTransformerCls(config)
    
    # Load pretrained weights from the full model
    checkpoint_path = Path(config['pretrained']['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    pretrained_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Filter state dict to only include embedding and transformer blocks
    filtered_state = {}
    for key, value in pretrained_state.items():
        if key.startswith('embedding.') or key.startswith('transformer_blocks.'):
            filtered_state[key] = value
    
    # Load filtered state dict
    model.load_state_dict(filtered_state, strict=False)
    print("Loaded pretrained weights for embedding and transformer blocks")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['direction', 'energy'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='base')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional path for pretrained model checkpoint")
    args = parser.parse_args()

    # Load and process config
    config = load_and_process_config(args.config)
    
    # Setup model name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = f"{args.name or 'finetuned'}_{suffix}"
    
    # Setup training
    torch.set_float32_matmul_precision('high')
    wandb_logger = WandbLogger(
        project=config['training'].get('project', 'PolarBERT-finetuning'),
        name=model_name,
        config=config
    )
    
    # Update config with parameters from wandb sweep
    for param, (section, key) in SWEEP_PARAMS.items():
        if param in wandb_logger.experiment.config:
            config[section][key] = wandb_logger.experiment.config[param]
    
    # Compute dependent Adam parameters from sweep values
    if 'one_minus_adam_beta1' in wandb_logger.experiment.config:
        config['training']['adam_beta1'] = 1.0 - wandb_logger.experiment.config['one_minus_adam_beta1']
    if 'one_minus_adam_beta2' in wandb_logger.experiment.config:
        config['training']['adam_beta2'] = 1.0 - wandb_logger.experiment.config['one_minus_adam_beta2']
    
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
    
    # Load pretrained model
    pretrained_model = load_pretrained_model(config)
    
    # Initialize finetuning model
    if args.task == 'direction':
        model = DirectionalHead(config, pretrained_model)
    elif args.task == 'energy':
        model = EnergyRegressionHead(config, pretrained_model)
    else:
        assert False, f'Unsupported task: {args.task}'
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config, target_transform=model.target_transform)
    
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
        precision='16-mixed',
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=wandb_logger,
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        enable_model_summary=True,
        deterministic=False,  # Add this for better performance
        gradient_clip_algorithm='norm',  # Add this for better stability
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
