import torch
import torch.nn as nn
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
    MODEL_CLASSES
)
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config['training']['initial_lr']),
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=float(self.config['training']['weight_decay']),
            amsgrad=bool(self.config['training'].get('amsgrad', False))
        )

        if self.config['training']['lr_scheduler'] == 'constant':
            return optimizer
        elif self.config['training']['lr_scheduler'] == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.config['training']['max_lr']),
                total_steps=int(self.config['training']['total_steps']),
                pct_start=float(self.config['training']['pct_start']),
                final_div_factor=float(self.config['training']['final_div_factor']),
                anneal_strategy='cos'
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        else:
            raise ValueError(f"Unknown scheduler: {self.config['training']['lr_scheduler']}")

def load_pretrained_model(config: Dict[str, Any]):
    """Load and prepare pretrained model."""
    # Initialize new model for finetuning
    model = SimpleTransformerCls(config)
    
    # Load pretrained weights from the full model
    checkpoint_path = Path(config['pretrained']['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    pretrained_state = torch.load(checkpoint_path, map_location='cpu')
    
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
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='base')
    args = parser.parse_args()

    # Load and process config
    config = load_and_process_config(args.config)
    
    # Add model type to config if not present
    if 'model_type' not in config.get('pretrained', {}):
        config.setdefault('pretrained', {})['model_type'] = args.model_type
    
    # Setup model name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = f"{args.name or 'finetuned'}_{suffix}"
    
    # Setup directional config if not present
    if 'directional' not in config['model']:
        config['model']['directional'] = {
            'hidden_size': 1024,
        }
    
    # Load pretrained model
    pretrained_model = load_pretrained_model(config)
    
    # Initialize finetuning model
    model = DirectionalHead(config, pretrained_model)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Update training steps
    config = update_training_steps(config, train_loader)
    
    # Setup training
    wandb_logger = WandbLogger(
        project=config['training'].get('project', '2024-09-IceCube-finetuning'),
        name=model_name,
        config=config
    )
    
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
