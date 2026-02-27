SWEEP_PARAMS = {
    # Architecture
    'embedding_dim': ['model', 'embedding_dim'],
    'dom_embed_dim': ['model', 'dom_embed_dim'],
    'num_heads': ['model', 'num_heads'],
    'hidden_size': ['model', 'hidden_size'],
    'num_layers': ['model', 'num_layers'],
    'lambda_charge': ['model', 'lambda_charge'],
    # Optimizer
    'mask_prob': ['training', 'mask_prob'],
    'max_epochs': ['training', 'max_epochs'],
    'logical_batch_size': ['training', 'logical_batch_size'],
    'gradient_clip_val': ['training', 'gradient_clip_val'],
    'max_lr': ['training', 'max_lr'],
    'one_minus_adam_beta1': ['training', 'one_minus_adam_beta1'],
    'one_minus_adam_beta2': ['training', 'one_minus_adam_beta2'],
    'adam_eps': ['training', 'adam_eps'],
    'weight_decay': ['training', 'weight_decay'],
    'amsgrad': ['training', 'amsgrad'],
    'lr_scheduler': ['training', 'lr_scheduler'],
    'pct_start': ['training', 'pct_start'],
    'div_factor': ['training', 'div_factor'],
    'final_div_factor': ['training', 'final_div_factor'],
    'warmup_steps': ['training', 'warmup_steps'],
    'decay_steps': ['training', 'decay_steps'],
    # muP
    'mup_init_std': ['training', 'mup', 'init_std'],
    'mup_input_alpha': ['training', 'mup', 'input_alpha'],
    'mup_output_alpha': ['training', 'mup', 'output_alpha'],
    # Finetuning
    'prediction_head_hidden_size': ['model', 'directional', 'hidden_size'],
}


def update_config_for_wandb_sweep(config: dict, wandb_config, sweep_params: dict[str, list[str]]=SWEEP_PARAMS):
    """Update config with parameters from W&B sweep, including nested values and Adam beta computation.
    
    Args:
        config: Configuration dictionary to update
        wandb_config: W&B experiment config containing sweep parameters
        sweep_params: Mapping of sweep parameter names to nested config paths
    """
    # Override parameters that are being swept over
    for param, keys in sweep_params.items():
        if param in wandb_config:
            _set_nested_config_value(config, keys, wandb_config[param])

    # Compute dependent Adam parameters from sweep values
    if 'one_minus_adam_beta1' in wandb_config:
        config['training']['adam_beta1'] = 1.0 - wandb_config['one_minus_adam_beta1']
    if 'one_minus_adam_beta2' in wandb_config:
        config['training']['adam_beta2'] = 1.0 - wandb_config['one_minus_adam_beta2']


def _set_nested_config_value(config, keys, value):
    """Set a value in a nested config dictionary using a tuple of keys."""
    assert keys, "Keys tuple cannot be empty"
    if len(keys) == 1:
        config[keys[0]] = value
    else:
        if keys[0] not in config:
            config[keys[0]] = {}
        elif not isinstance(config[keys[0]], dict):
            raise TypeError(f"Cannot set nested value: {keys[0]} is not a dictionary")
        _set_nested_config_value(config[keys[0]], keys[1:], value)
