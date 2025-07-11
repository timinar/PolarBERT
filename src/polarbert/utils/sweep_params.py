
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
    'one_minus_adam_beta1': ('training', 'one_minus_adam_beta1'),
    'one_minus_adam_beta2': ('training', 'one_minus_adam_beta2'),
    'adam_eps': ('training', 'adam_eps'),
    'weight_decay': ('training', 'weight_decay'),
    'amsgrad': ('training', 'amsgrad'),
    'lr_scheduler': ('training', 'lr_scheduler'),
    'pct_start': ('training', 'pct_start'),
    'div_factor': ('training', 'div_factor'),
    'final_div_factor': ('training', 'final_div_factor'),
}