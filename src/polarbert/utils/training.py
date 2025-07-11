from typing import Any
import math
from torch.utils.data import DataLoader


def update_training_steps(config: dict[str, Any], train_loader: DataLoader) -> dict[str, Any]:
    """Calculate and update training steps in config."""
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['training']['max_epochs']
    
    # Update config with calculated values
    config['training'].update({
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'num_events': total_steps * config['training']['batch_size']
    })
    
    # If warm_up_steps is provided, calculate pct_start based on total_steps
    warm_up_steps = config['training'].get('warm_up_steps')
    if warm_up_steps is not None:
        # Calculate pct_start as the ratio of warm_up_steps to total_steps
        pct_start = min(1.0, warm_up_steps / total_steps)
        config['training']['pct_start'] = pct_start
        print(f"Using warm_up_steps: {warm_up_steps}, calculated pct_start: {pct_start:.4f}")
    
    return config

def compute_batch_params(config: dict[str, Any]) -> dict[str, Any]:
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
