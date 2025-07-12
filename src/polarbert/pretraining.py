import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import argparse
from datetime import datetime
import logging

from polarbert.utils.config import load_and_process_config
from polarbert.utils.data import (
    get_dataloaders, 
    add_random_time_offset, 
    default_transform
)
from polarbert.utils.training import update_training_steps, compute_batch_params
from polarbert.utils.callbacks import setup_callbacks
from polarbert.utils.sweep_params import SWEEP_PARAMS

from polarbert.flash_model import FlashTransformer
from polarbert.swiglu_model import SwiGLUTransformer
from polarbert.base_model import SimpleTransformer

MODEL_CLASSES = {
    'flash': (FlashTransformer, "Flash Transformer"),
    'swiglu': (SwiGLUTransformer, "SwiGLU Transformer"),
    'base': (SimpleTransformer, "Base Transformer")
}




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/basic_transformer.yaml')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CLASSES.keys()), default='flash')
    parser.add_argument("--dataset_type", type=str, choices=['kaggle', 'prometheus'], default='kaggle')
    parser.add_argument("--watch", action='store_true')
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
    
    # Compute dependent Adam parameters from sweep values
    if 'one_minus_adam_beta1' in wandb_logger.experiment.config:
        config['training']['adam_beta1'] = 1.0 - wandb_logger.experiment.config['one_minus_adam_beta1']
    if 'one_minus_adam_beta2' in wandb_logger.experiment.config:
        config['training']['adam_beta2'] = 1.0 - wandb_logger.experiment.config['one_minus_adam_beta2']

    # Compute and update batch parameters
    batch_params = compute_batch_params(config)
    config['training'].update(batch_params)

    # Get data loaders
    random_time_offset_std = config['training'].get('random_time_offset')
    if random_time_offset_std is not None:
        logging.info(f"Applying random time offset with std: {random_time_offset_std}")
        transform = add_random_time_offset(random_time_offset_std)
    else:
        transform = default_transform
    train_loader, val_loader = get_dataloaders(config, dataset_type=args.dataset_type, transform=transform)
    
    # Update training steps in config
    config = update_training_steps(config, train_loader)
    
    # Ensure the full updated config is logged on W&B before we start training
    wandb_logger.experiment.config.update(config, allow_val_change=True)

    # Initialize model
    model_class, model_name = MODEL_CLASSES[args.model_type]
    model = model_class(config)
    print(f"Using {model_name} model")
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    # Log gradient & parameter histograms, as well as model topology
    if args.watch:
        wandb_logger.watch(model, log='all')
    
    # Setup training with flexible validation interval
    val_interval = config['training'].get('val_check_interval', 1.0)
    
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=setup_callbacks(config, config['model']['model_name']),
        accelerator='gpu',
        devices=config['training']['gpus'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=wandb_logger,
        val_check_interval=val_interval,  # Can be float (fraction of epoch) or int (number of steps)
    )

    trainer.fit(model, train_loader, val_loader)

    # Remove W&B hook if the --watch option was passed
    if args.watch:
        wandb_logger.experiment.unwatch(model)

if __name__ == '__main__':
    main()
