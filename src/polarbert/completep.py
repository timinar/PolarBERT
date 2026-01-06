"""
CompleteP parameterization utilities for hyperparameter transfer.

CompleteP enables hyperparameters tuned on a small "base" model to transfer
to larger "target" models across Width, Depth, and Training Duration.

References:
- Paper 1 (CompleteP): "Don't be lazy: CompleteP enables compute-efficient deep transformers"
  https://arxiv.org/abs/2505.01618
- Paper 2 (Complete(d)P): "Completed Hyperparameter Transfer across Modules, Width, Depth, Batch & Duration"
  https://arxiv.org/abs/2512.22382
"""

import math
from typing import Dict, Any, Tuple


def is_completep_enabled(config: dict) -> bool:
    """Check if CompleteP is enabled in config."""
    return config.get('training', {}).get('completep', {}).get('enabled', False)


def get_completep_config(config: dict) -> Dict[str, Any]:
    """Get CompleteP config section with defaults."""
    defaults = {
        'enabled': False,
        'n_base': 256,        # Base embedding dimension
        'l_base': 2,          # Base number of layers
        't_base': 100000,     # Base training duration (events)
        'init_std_base': 0.02,  # Base initialization std
        'lr_base': 3e-4,      # Base learning rate
        'eps_base': 1e-8,     # Base Adam epsilon
    }
    cp_config = config.get('training', {}).get('completep', {})
    return {**defaults, **cp_config}


def compute_multipliers(config: dict) -> Tuple[float, float, float]:
    """
    Compute width, depth, and duration multipliers.

    Returns:
        m_N: width multiplier (embedding_dim / n_base)
        m_L: depth multiplier (num_layers / l_base)
        kappa: duration multiplier (train_events / t_base)
    """
    cp_config = get_completep_config(config)

    # Width multiplier
    embedding_dim = config['model']['embedding_dim']
    m_N = embedding_dim / cp_config['n_base']

    # Depth multiplier
    num_layers = config['model']['num_layers']
    m_L = num_layers / cp_config['l_base']

    # Duration multiplier
    train_events = config['data']['train_events']
    kappa = train_events / cp_config['t_base']

    return m_N, m_L, kappa


def get_residual_scale(config: dict) -> float:
    """
    Get residual branch scaling factor: 1/m_L for alpha=1.

    In CompleteP with alpha=1, residual branches are scaled by 1/m_L:
        x = x + (1/m_L) * F(x)

    This ensures feature learning transfers across depth.
    """
    _, m_L, _ = compute_multipliers(config)
    return 1.0 / m_L


def get_init_std(config: dict, param_type: str, fan_in: int = None) -> float:
    """
    Get initialization std for different parameter types.

    Args:
        config: Full config dict
        param_type: One of:
            - 'input_embedding': Fixed variance (CLS token, DOM embeddings, etc.)
            - 'input_linear': Dense input layer, scaled by 1/sqrt(fan_in) (CompleteP Paper 1)
            - 'hidden': Scaled by 1/sqrt(m_N) (Q, K, V, W_O, FF weights)
            - 'readout': Scaled by 1/sqrt(m_N) (fc1, fc2 in DirectionalHead)
        fan_in: Input dimension for 'input_linear' type (required for that type)

    Returns:
        Initialization standard deviation

    Note:
        For dense input layers, CompleteP Paper 1 states:
        "If the input data is dense, then we would require a pre-factor of 1/sqrt(d_in)"
        This keeps signal variance stable across different input dimensions.
    """
    cp_config = get_completep_config(config)
    m_N, _, _ = compute_multipliers(config)
    std_base = cp_config['init_std_base']

    if param_type == 'input_embedding':
        # Fixed variance for learnable tokens and lookup tables (one-hot inputs)
        return std_base
    elif param_type == 'input_linear':
        # Dense input layer: scale by 1/sqrt(fan_in) to stabilize signal variance
        if fan_in is None:
            raise ValueError("fan_in must be provided for 'input_linear' param_type")
        return std_base / math.sqrt(fan_in)
    elif param_type == 'hidden':
        # Scaled by 1/sqrt(m_N) for hidden weights
        return std_base / math.sqrt(m_N)
    elif param_type == 'readout':
        # Scaled by 1/sqrt(m_N) for readout weights
        return std_base / math.sqrt(m_N)
    else:
        raise ValueError(f"Unknown param_type: {param_type}. "
                        f"Expected one of: 'input_embedding', 'input_linear', 'hidden', 'readout'")


def get_lr_scale(config: dict, param_group: str) -> float:
    """
    Get learning rate scale for different parameter groups.

    The global LR is scaled by 1/sqrt(kappa) for duration transfer,
    then each group gets an additional scale factor.

    Args:
        config: Full config dict
        param_group: One of:
            - 'input_embedding': LR_base / m_N
            - 'hidden': LR_base / m_N
            - 'biases_norms': LR_base (fixed)
            - 'readout': LR_base / m_N

    Returns:
        Learning rate scale factor (to be applied via optimizer_step)
    """
    m_N, _, kappa = compute_multipliers(config)
    duration_scale = 1.0 / math.sqrt(kappa)  # Global LR scaling by duration

    if param_group == 'input_embedding':
        return duration_scale / m_N
    elif param_group == 'hidden':
        return duration_scale / m_N
    elif param_group == 'biases_norms':
        return duration_scale  # Fixed LR (no width scaling)
    elif param_group == 'readout':
        return duration_scale / m_N
    else:
        raise ValueError(f"Unknown param_group: {param_group}. "
                        f"Expected one of: 'input_embedding', 'hidden', 'biases_norms', 'readout'")


def get_eps_scale(config: dict, param_group: str) -> float:
    """
    Get Adam epsilon scale for different parameter groups.

    Args:
        config: Full config dict
        param_group: One of:
            - 'input_embedding': eps_base / m_N
            - 'hidden': eps_base / (m_N * m_L)
            - 'biases_norms': eps_base / (m_N * m_L)
            - 'readout': eps_base / m_N

    Returns:
        Epsilon scale factor
    """
    m_N, m_L, _ = compute_multipliers(config)

    if param_group == 'input_embedding':
        return 1.0 / m_N
    elif param_group == 'hidden':
        return 1.0 / (m_N * m_L)
    elif param_group == 'biases_norms':
        return 1.0 / (m_N * m_L)
    elif param_group == 'readout':
        return 1.0 / m_N
    else:
        raise ValueError(f"Unknown param_group: {param_group}. "
                        f"Expected one of: 'input_embedding', 'hidden', 'biases_norms', 'readout'")


def classify_parameter(name: str, param) -> str:
    """
    Classify a parameter into one of the CompleteP parameter groups.

    Args:
        name: Parameter name (from named_parameters())
        param: The parameter tensor

    Returns:
        One of: 'input_embedding', 'hidden', 'biases_norms', 'readout'
    """
    # Input embeddings (fixed variance)
    if any(s in name for s in ['embedding.dom_embedding', 'embedding.features_embedding',
                                'embedding.position_embedding', 'cls_embedding',
                                'mask_token_embedding']):
        return 'input_embedding'

    # Readout weights (DirectionalHead fc layers)
    if any(s in name for s in ['fc1.weight', 'fc2.weight']):
        return 'readout'

    # Readout biases go to biases_norms
    if any(s in name for s in ['fc1.bias', 'fc2.bias']):
        return 'biases_norms'

    # Hidden weights (transformer Q, K, V, O, FF)
    if any(s in name for s in ['wq.weight', 'wk.weight', 'wv.weight', 'wo.weight',
                                'feed_forward.0.weight', 'feed_forward.2.weight']):
        return 'hidden'

    # RMSNorms and biases (including QK Norm)
    if 'layer_norm' in name or 'final_layer_norm' in name or 'q_norm' in name or 'k_norm' in name:
        return 'biases_norms'

    # Any remaining 1D tensors (biases) go to biases_norms
    if param.dim() < 2:
        return 'biases_norms'

    # Any remaining 2D+ tensors default to hidden
    return 'hidden'


def log_completep_info(config: dict):
    """Log CompleteP configuration info for debugging."""
    if not is_completep_enabled(config):
        return

    cp_config = get_completep_config(config)
    m_N, m_L, kappa = compute_multipliers(config)

    print("\n" + "="*60)
    print("CompleteP Configuration")
    print("="*60)
    print(f"Base dimensions: N={cp_config['n_base']}, L={cp_config['l_base']}, T={cp_config['t_base']}")
    print(f"Target dimensions: N={config['model']['embedding_dim']}, "
          f"L={config['model']['num_layers']}, T={config['data']['train_events']}")
    print(f"Multipliers: m_N={m_N:.2f}, m_L={m_L:.2f}, kappa={kappa:.2f}")
    print(f"Residual scale: {get_residual_scale(config):.4f}")
    print(f"Duration LR scale: {1/math.sqrt(kappa):.4f}")
    print("\nPer-group scales:")
    for group in ['input_embedding', 'hidden', 'biases_norms', 'readout']:
        lr_scale = get_lr_scale(config, group)
        eps_scale = get_eps_scale(config, group)
        print(f"  {group:20s}: LR scale={lr_scale:.4f}, eps scale={eps_scale:.4f}")
    print("="*60 + "\n")
