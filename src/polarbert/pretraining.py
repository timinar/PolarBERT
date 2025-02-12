import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import math

from polarbert.prometheus_dataset import IceCubeDataset
from polarbert.flash_model import FlashTransformer
from polarbert.swiglu_model import SwiGLUTransformer
from polarbert.base_model import SimpleTransformer

MODEL_CLASSES = {
    'flash': (FlashTransformer, "Flash Transformer"),
    'swiglu': (SwiGLUTransformer, "SwiGLU Transformer"),
    'base': (SimpleTransformer, "Base Transformer")
}

def load_and_process_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert string numbers to proper types
    for section in config:
        for key, value in config[section].items():
            if isinstance(value, str):
                try:
                    if '.' in value:
                        config[section][key] = float(value)
                    else:
                        config[section][key] = int(value)
                except ValueError:
                    pass
    return config

def setup_callbacks(config: Dict[str, Any], model_name: str) -> list:
    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    # Get checkpoint config with defaults
    checkpoint_config = config['training'].get('checkpoint', {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    
    # Set default checkpoint settings
    checkpoint_defaults = {
        'dirpath': 'checkpoints',
        'save_top_k': 1,
        'monitor': 'val/full_loss',
        'mode': 'min',
        'save_last': True,
        'save_final': True,
    }
    
    # Merge defaults with provided config
    checkpoint_config = {**checkpoint_defaults, **checkpoint_config}
    
    # Setup checkpoint directory
    checkpoint_dir = Path(checkpoint_config['dirpath']) / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint callback with cleaned config
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch:02d}-{step:06d}",
        save_top_k=checkpoint_config['save_top_k'],
        monitor=checkpoint_config['monitor'],
        mode=checkpoint_config['mode'],
        save_last=checkpoint_config['save_last']
    )
    callbacks.append(checkpoint_callback)
    
    if checkpoint_config.get('save_final', False):
        class FinalModelCallback(pl.Callback):
            def on_train_end(self, trainer, pl_module):
                save_path = checkpoint_dir / "final_model.pth"
                torch.save(pl_module.state_dict(), save_path)
                print(f"Final model saved to {save_path}")
        callbacks.append(FinalModelCallback())
    
    return callbacks

def get_dataloaders(config: Dict[str, Any]):
    def transform(x, l):
        return x.astype(np.float32), l.astype(np.float32)
    def target_transform(y, c):
        y = np.vstack([y['initial_state_azimuth'].astype(np.float32), y['initial_state_zenith'].astype(np.float32)]).T
        return y, c.astype(np.float32)
    
    # Training dataset
    full_dataset = IceCubeDataset(
        data_dir=config['data']['train_dir'], 
        batch_size=config['training']['per_device_batch_size'],
        transform=transform,
        target_transform=target_transform
    )
    val_dataset = full_dataset.slice(0, config['data']['val_events'])
    train_dataset = full_dataset.slice(config['data']['val_events'], config['data']['val_events'] + config['data']['train_events'])
    del full_dataset
    
    # # Validation dataset with optional subsampling
    # full_val_dataset = IceCubeDataset(
    #     data_dir=config['data']['val_dir'], 
    #     batch_size=config['data']['batch_size'],
    #     transform=transform,
    #     target_transform=transform
    # )
    
    # val_events = config['data'].get('val_events', None)
    # val_dataset = full_val_dataset.slice(0, val_events) if val_events else full_val_dataset
    # del full_val_dataset
    
    loader_kwargs = {
        'batch_size': None,
        'num_workers': config['data']['num_workers'],
        'pin_memory': config['data']['pin_memory'],
        'persistent_workers': config['data']['persistent_workers']
    }
    
    return (
        DataLoader(train_dataset, **loader_kwargs),
        DataLoader(val_dataset, **loader_kwargs)
    )

def update_training_steps(config: Dict[str, Any], train_loader: DataLoader) -> Dict[str, Any]:
    """Calculate and update training steps in config."""
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['training']['max_epochs']
    
    # Update config with calculated values
    config['training'].update({
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'num_events': total_steps * config['training']['batch_size']
    })
    
    return config

def compute_batch_params(config: Dict[str, Any]) -> Dict[str, Any]:
    logical_batch = config['training']['logical_batch_size']
    max_per_device = config['data'].get('max_per_device_batch_size', logical_batch)
    per_device_batch_size = min(max_per_device, logical_batch)
    gradient_accumulation_steps = math.ceil(logical_batch / per_device_batch_size)
    actual_batch_size = gradient_accumulation_steps * per_device_batch_size
    return {
        'per_device_batch_size': per_device_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'batch_size': actual_batch_size,
    }

SWEEP_PARAMS = {
    # Architecture
    'embedding_dim': ('model', 'embedding_dim'),
    'dom_embed_dim': ('model', 'dom_embed_dim'),
    'num_heads': ('model', 'num_heads'),
    'hidden_size': ('model', 'hidden_size'),
    'num_layers': ('model', 'num_layers'),
    'lambda_charge': ('model', 'lambda_charge'),
    # Optimizer
    'mask_prob': ('training', 'mask_prob'),
    'max_epochs': ('training', 'max_epochs'),
    'logical_batch_size': ('training', 'logical_batch_size'),
    'gradient_clip_val': ('training', 'gradient_clip_val'),
    'max_lr': ('training', 'max_lr'),
    'weight_decay': ('training', 'weight_decay'),
    'amsgrad': ('training', 'amsgrad'),
    'lr_scheduler': ('training', 'lr_scheduler'),
    'pct_start': ('training', 'pct_start'),
    'final_div_factor': ('training', 'final_div_factor'),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/basic_transformer.yaml')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='base')
    args = parser.parse_args()

    # Load and process config
    config = load_and_process_config(args.config)
    
    # Setup model name
    suffix = args.job_id or datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = f"{args.name or config['model']['model_name']}_{suffix}"
    config['model']['model_name'] = model_name

    # Setup training
    torch.set_float32_matmul_precision('high')
    wandb_logger = WandbLogger(
        project=config['training']['project'],
        name=model_name,
        config=config
    )
    
    # Update config with parameters from wandb sweep
    for param, (section, key) in SWEEP_PARAMS.items():
        if param in wandb_logger.experiment.config:
            config[section][key] = wandb_logger.experiment.config[param]

    # Compute and update batch parameters
    batch_params = compute_batch_params(config)
    config['training'].update(batch_params)

    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Update training steps in config
    config = update_training_steps(config, train_loader)

    # Initialize model
    model_class, model_name = MODEL_CLASSES[args.model_type]
    model = model_class(config)
    print(f"Using {model_name} model")
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Setup training with flexible validation interval
    val_interval = config['training'].get('val_check_interval', 1.0)
    
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=setup_callbacks(config, model_name),
        accelerator='gpu',
        devices=config['training']['gpus'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=wandb_logger,
        val_check_interval=val_interval,  # Can be float (fraction of epoch) or int (number of steps)
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
