import os
import torch
import numpy as np
import yaml
import argparse
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from icecube_pretraining.icecube_dataset import IceCubeDataset, train_validation_loaders
from icecube_pretraining.flash_model import SimpleTransformer as PretrainedModel
from icecube_pretraining.loss_functions import angles_to_unit_vector, angular_dist_score_unit_vectors

torch.set_float32_matmul_precision('high')

class SimpleTransformerCls(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = PretrainedModel(config).embedding
        self.transformer_blocks = PretrainedModel(config).transformer_blocks

    def forward(self, x):
        embeddings, padding_mask, mask = self.embedding(x)
        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask)
        return embeddings[:, 0, :]  # CLS token

class DirectionalModel(pl.LightningModule):
    def __init__(self, config, pretrained_path=None, from_polar_bert=False, freeze_pretrained=False):
        super().__init__()
        # Save all init parameters in hyperparameters including pretrained_path
        self.save_hyperparameters()
        self.config = config
        self.from_polar_bert = from_polar_bert
        self.pretrained_path = pretrained_path  # Explicitly store pretrained_path
        
        # Initialize pretrained model
        self.pretrained_model = SimpleTransformerCls(config)
        
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Keep everything but unembedding and charge prediction
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if not any(x in k for x in ['unembedding', 'charge_prediction'])
            }
            
            self.pretrained_model.load_state_dict(filtered_state_dict, strict=False)
            
            # Only freeze if explicitly requested
            if freeze_pretrained:
                print("Freezing pretrained model weights")
                for param in self.pretrained_model.parameters():
                    param.requires_grad = False
            else:
                print("Fine-tuning pretrained model weights")
        
        # Direction prediction layers
        self.fc1 = torch.nn.Linear(config['model']['embedding_dim'], config['directional']['hidden_size'])
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(config['directional']['hidden_size'], 3)

    def forward(self, x):
        if self.from_polar_bert:
            # Unpack the input tuple
            features, lengths = x
            # Create a copy and flip the sign of z coordinate
            features = features.clone()
            features[:,:,-2] = -features[:,:,-2]
            x = (features, lengths)

        # Use self.pretrained_path instead of hparams
        with torch.no_grad() if self.pretrained_path else torch.enable_grad():
            cls_embed = self.pretrained_model(x)
        
        x = self.fc1(cls_embed)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.nn.functional.normalize(x, dim=1)
    
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
            lr=self.config['training']['initial_lr'],
            weight_decay=self.config['training']['weight_decay'],
            amsgrad=self.config['training'].get('amsgrad', False)
        )

        if self.config['training']['lr_scheduler'] == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['training']['max_lr'],
                total_steps=self.config['training']['total_steps'],
                pct_start=self.config['training']['pct_start'],
                final_div_factor=self.config['training']['final_div_factor'],
                anneal_strategy='cos'
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

def cast_config_values(config):
    # Make a deep copy to avoid modifying nested dictionaries
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                if isinstance(value, str):
                    try:
                        # Try to convert string to float first
                        if 'e' in value.lower() or '.' in value:
                            config[section][key] = float(value)
                        else:
                            # If no decimal point or scientific notation, try int
                            config[section][key] = int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    config[section][key] = cast_config_values({None: value})[None]
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model checkpoint (optional)')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--job_id', type=str, default=None, help='(Optional) Job ID')
    parser.add_argument('--from_polar_bert', action='store_true', help='Enable PolarBERT compatibility mode')
    parser.add_argument('--freeze_pretrained', action='store_true', help='Freeze pretrained model weights')
    args = parser.parse_args()

    # Load and cast config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = cast_config_values(config)  # Add this line to cast values

    # Print config values for debugging
    print("Training config after casting:")
    print(f"initial_lr: {config['training']['initial_lr']} ({type(config['training']['initial_lr'])})")
    print(f"max_lr: {config['training']['max_lr']} ({type(config['training']['max_lr'])})")

    # Add directional config if not present
    if 'directional' not in config:
        config['directional'] = {'hidden_size': 1024}

    # Setup data
    full_dataset = IceCubeDataset(
        data_dir=config['data']['train_dir'],
        batch_size=config['data']['batch_size'],
        transform=lambda x: x.astype(np.float32),
        target_transform=lambda x: x.astype(np.float32)
    )

    train_loader, val_loader = train_validation_loaders(
        full_dataset,
        train_ratio=config['data']['train_ratio'],
        pin_memory=config['data']['pin_memory'],
    )

    # Calculate steps with gradient accumulation
    steps_per_epoch = len(train_loader) // config['training']['accumulate_grad_batches']
    total_steps = steps_per_epoch * config['training']['max_epochs']
    effective_batch_size = config['data']['batch_size'] * config['training']['accumulate_grad_batches']
    num_events = total_steps * effective_batch_size

    # Update config with calculated values
    config['training']['total_steps'] = total_steps
    config['training']['steps_per_epoch'] = steps_per_epoch
    config['training']['num_events'] = num_events
    config['training']['effective_batch_size'] = effective_batch_size

    # Setup model
    model = DirectionalModel(
        config, 
        args.pretrained_path, 
        from_polar_bert=args.from_polar_bert,
        freeze_pretrained=args.freeze_pretrained
    )
    print(f'Num params: {sum(p.numel() for p in model.parameters())}')

    # Print training mode
    if args.pretrained_path:
        print(f"Training with pretrained weights from: {args.pretrained_path}")
        print(f"Pretrained weights are {'frozen' if args.freeze_pretrained else 'being fine-tuned'}")
        if args.from_polar_bert:
            print("Using PolarBERT compatibility mode (flipping z coordinate)")
    else:
        print("Training from scratch (no pretrained weights)")

    # Setup experiment name
    if args.job_id:
        suffix = args.job_id
    else:
        suffix = datetime.now().strftime('%y%m%d-%H%M%S')
    exp_name = f"{args.name or 'directional'}_{suffix}"

    # Setup logging
    wandb_logger = WandbLogger(
        project="2024-09-IceCube-finetuning", 
        name=exp_name, 
        config=config
    )

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{exp_name}',
        filename='{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1
    )
    
    class FinalEvaluationCallback(pl.Callback):
        def on_train_end(self, trainer, pl_module):
            print("Performing final evaluation...")
            trainer.validate(model=pl_module, dataloaders=trainer.val_dataloaders)
            print("Saving final model...")
            torch.save(pl_module.state_dict(), f"checkpoints/{exp_name}/final_model.pth")

    final_eval_callback = FinalEvaluationCallback()

    # Setup trainer
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=[lr_monitor, checkpoint_callback, final_eval_callback],
        accelerator='gpu',
        devices=config['training']['gpus'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == '__main__':
    main()
