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
from polarbert.utils.sweep_params import update_config_for_wandb_sweep
from polarbert.utils.activation_logging import ActivationLoggingCallback

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
    parser.add_argument("--log_frequency", type=int, default=100, help="Logging frequency (in steps) for W&B and activation logging")
    parser.add_argument("--save-activations-to", type=str, default=None, 
                        help="Path to CSV file for saving activation statistics")
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
    update_config_for_wandb_sweep(config, wandb_logger.experiment.config)

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

    # Setup callbacks
    callbacks = setup_callbacks(config, config['model']['model_name'])
    
    # Log gradient & parameter histograms, model topology, and add activation logging callback if watch is enabled
    if args.watch:
        wandb_logger.watch(model, log='all', log_freq=args.log_frequency)
        
        activation_callback = ActivationLoggingCallback(
            log_frequency=args.log_frequency,
            csv_path=args.save_activations_to,
        )
        callbacks.append(activation_callback)
    
    # Setup training with flexible validation interval
    val_interval = config['training'].get('val_check_interval', 1.0)
    
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        accelerator='gpu',
        devices=config['training']['gpus'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=wandb_logger,
        val_check_interval=val_interval,  # Can be float (fraction of epoch) or int (number of steps)
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
    )

    trainer.fit(model, train_loader, val_loader)

    # Remove W&B hook if the --watch option was passed
    if args.watch:
        wandb_logger.experiment.unwatch(model)

if __name__ == '__main__':
    main()
