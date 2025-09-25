from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pathlib import Path
from polarbert.utils.coordinate_checking import CoordinateCheckingCallback


def setup_callbacks(config: dict[str, Any], model_name: str) -> list:
    callbacks: list = [LearningRateMonitor(logging_interval='step')]
    
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

    early_stopping_config = config['training'].get('early_stopping')
    if early_stopping_config and early_stopping_config.get('enabled', False):
        early_stopping_callback = EarlyStopping(
            monitor=checkpoint_config['monitor'],
            mode='min',
            divergence_threshold=early_stopping_config.get('divergence_threshold', 10.),
            patience=early_stopping_config.get('patience', 3),
            check_finite=True,
        )
        callbacks.append(early_stopping_callback)

    if 'mup' in config['training'] and config['training']['mup'].get('check_coordinates', False):
        coordinate_checking_callback = CoordinateCheckingCallback(checkpoint_dir)
        callbacks.append(coordinate_checking_callback)

    return callbacks