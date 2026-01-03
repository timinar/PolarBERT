from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pathlib import Path
from polarbert.utils.coordinate_checking import CoordinateCheckingCallback


class BestValLossCallback(pl.Callback):
    """Tracks and logs the best validation loss to W&B."""
    def __init__(self):
        self.best_val_loss = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val/loss')
        if val_loss is not None:
            val_loss_value = val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)
            if val_loss_value < self.best_val_loss:
                self.best_val_loss = val_loss_value
            # Log during epoch end (allowed hook for logging)
            pl_module.log('val/best_loss', self.best_val_loss, prog_bar=False, sync_dist=True)

    def on_fit_end(self, trainer, pl_module):
        # Log final best loss directly to W&B to ensure it's in summary
        if trainer.logger:
            trainer.logger.log_metrics({'val/best_loss_final': self.best_val_loss})


class ScheduleFreeOptimizerCallback(pl.Callback):
    """Callback to handle train/eval mode of Schedule-Free optimizer.

    Schedule-free optimizers require switching between train() and eval() modes
    to properly interpolate between iterate and averaged parameters.
    """
    def on_train_start(self, trainer, pl_module):
        optimizer = pl_module.optimizers()
        if hasattr(optimizer, 'train'):
            optimizer.train()

    def on_validation_start(self, trainer, pl_module):
        optimizer = pl_module.optimizers()
        if hasattr(optimizer, 'eval'):
            optimizer.eval()

    def on_validation_end(self, trainer, pl_module):
        optimizer = pl_module.optimizers()
        if hasattr(optimizer, 'train'):
            optimizer.train()

    def on_test_start(self, trainer, pl_module):
        optimizer = pl_module.optimizers()
        if hasattr(optimizer, 'eval'):
            optimizer.eval()


def setup_callbacks(config: dict[str, Any], model_name: str) -> list:
    callbacks: list = [
        LearningRateMonitor(logging_interval='step'),
        BestValLossCallback(),
    ]
    
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

    if config['training'].get('schedule_free', False):
        callbacks.append(ScheduleFreeOptimizerCallback())

    return callbacks