import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from polarbert.embedding import IceCubeEmbedding
from polarbert.flash_model import FlashTransformer
from polarbert.utils.config import load_and_process_config
from polarbert.utils.data import (
    get_dataloaders, 
    add_random_time_offset, 
    default_transform
)
from polarbert.utils.training import update_training_steps, compute_batch_params


class AnalyzableEmbedding(IceCubeEmbedding):
    """Modified embedding class that allows manual control over masking."""
    
    def __init__(self, config, masking=False):
        super().__init__(config, masking)
        self.manual_mask = None
    
    def set_manual_mask(self, mask):
        """Set a manual mask to override random masking.
        
        Args:
            mask: Boolean tensor of shape (batch_size, seq_length) or None
        """
        self.manual_mask = mask
    
    def forward(self, input):
        x, l = input
        other_features = x['features']
        dom_ids = x['dom_id']
        batch_size, max_seq_len = other_features.shape[:2]
        device = other_features.device
        mask = None

        padding_mask = torch.arange(max_seq_len, device=device)[None, :] >= l[:, None]
        
        if not self.use_dom_positions:
            dom_embeds = self.dom_embedding(dom_ids)
        else:
            pos = self.dom_positions[dom_ids]
            dom_embeds = self.position_embedding(pos)

        if self.masking:
            auxiliary_mask = other_features[:, :, 2] < 0
            
            if self.manual_mask is not None:
                if self.manual_mask.shape != auxiliary_mask.shape:
                    raise ValueError(f"Manual mask shape {self.manual_mask.shape} doesn't match expected {auxiliary_mask.shape}")
                random_mask = self.manual_mask.to(device)
            else:
                mask_prob = self.mask_prob if self.training else self.val_mask_prob
                random_mask = torch.rand(auxiliary_mask.shape, device=device) < mask_prob
            
            mask = auxiliary_mask & random_mask & ~padding_mask
            dom_embeds[mask] = self.mask_token_embedding.to(dtype=dom_embeds.dtype)
        
        features_embeds = self.features_embedding(other_features)
        combined_embeds = torch.cat([dom_embeds, features_embeds], dim=2)
        full_embedding = torch.cat([self.cls_embedding.expand(batch_size, -1, -1), combined_embeds], dim=1)
        padding_mask = torch.cat([torch.zeros(batch_size, 1, device=device, dtype=torch.bool), padding_mask], dim=1)
        
        return full_embedding, padding_mask, mask


class FlashTransformerAnalyzable(FlashTransformer):
    """FlashTransformer with AnalyzableEmbedding for consistent masking analysis."""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace the embedding with AnalyzableEmbedding
        self.embedding = AnalyzableEmbedding(config, masking=True)


def load_config(config_path: str, batch_size: int = 32) -> Dict[str, Any]:
    """Load and process configuration for analysis.
    
    Args:
        config_path: Path to the configuration file
        batch_size: Batch size to use for analysis
        
    Returns:
        Processed configuration dictionary
    """
    config = load_and_process_config(config_path)
    
    # Force CPU-friendly settings for analysis
    config['training']['gpus'] = 0  # Use CPU
    config['data']['max_per_device_batch_size'] = batch_size
    config['training']['logical_batch_size'] = batch_size
    config['data']['num_workers'] = 1
    config['data']['pin_memory'] = False
    
    return config


def load_pretrained_model(
    config: Dict[str, Any], 
    checkpoint_path: str, 
    device: str = 'cpu', 
    from_scratch: bool = False
) -> FlashTransformerAnalyzable:
    """Load pretrained model from checkpoint.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to checkpoint file or 'new' for scratch model
        device: Device to load model on
        from_scratch: If True, return model with random weights
        
    Returns:
        FlashTransformerAnalyzable model instance
    """
    # Initialize new model with analyzable embedding
    model = FlashTransformerAnalyzable(config)
    
    if from_scratch or checkpoint_path.strip().lower() == 'new':
        logging.info("Using model from scratch")
        model = model.to(device)
        model.eval()
        return model
    
    # Load pretrained weights
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    pretrained_state = torch.load(checkpoint_path_obj, map_location=device, weights_only=True)
    
    if 'state_dict' in pretrained_state:  # PyTorch Lightning checkpoints
        pretrained_state = pretrained_state['state_dict']
    
    # Load the state dict
    model.load_state_dict(pretrained_state, strict=False)
    model = model.to(device)
    model.eval()
    
    logging.info("Loaded pretrained weights successfully!")
    return model


def create_dataloaders(
    config: Dict[str, Any], 
    dataset_type: str = 'kaggle',
    use_random_time_offset: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        config: Configuration dictionary
        dataset_type: Type of dataset ('kaggle' or 'prometheus')
        use_random_time_offset: Whether to apply random time offset augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Compute batch parameters
    batch_params = compute_batch_params(config)
    config['training'].update(batch_params)

    # Get data transformation
    if use_random_time_offset:
        random_time_offset_std = config['training'].get('random_time_offset')
        if random_time_offset_std is not None:
            logging.info(f"Applying random time offset with std: {random_time_offset_std}")
            transform = add_random_time_offset(random_time_offset_std)
        else:
            transform = default_transform
    else:
        transform = default_transform

    # Get data loaders
    train_loader, val_loader = get_dataloaders(
        config, 
        dataset_type=dataset_type, 
        transform=transform
    )
    
    # Update training steps in config
    config = update_training_steps(config, train_loader)
    
    logging.info(f"Train loader ready: {len(train_loader)} batches")
    logging.info(f"Val loader ready: {len(val_loader)} batches")
    logging.info(f"Logical batch size: {config['training']['logical_batch_size']}")
    
    return train_loader, val_loader


def batch_to_device(batch_data, device: str = 'cpu'):
    """Move batch data to specified device, handling the complex nested structure.
    
    Args:
        batch_data: Batch data from data loader (format: ((x, l), (y, c)) or ((x, l), None))
        device: Target device ('cpu', 'cuda', etc.)
        
    Returns:
        Batch tuple moved to device: (input_batch, targets) where input_batch = (x, l)
        
    Raises:
        ValueError: If batch_data format is unexpected
    """
    if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
        raise ValueError(f"Expected batch_data to be list/tuple of length 2, got {type(batch_data)} of length {len(batch_data) if hasattr(batch_data, '__len__') else 'unknown'}")
    
    # Unpack the batch: ((x, l), targets)
    input_data, targets = batch_data
    
    if not isinstance(input_data, (list, tuple)) or len(input_data) != 2:
        raise ValueError(f"Expected input_data to be list/tuple of length 2, got {type(input_data)} of length {len(input_data) if hasattr(input_data, '__len__') else 'unknown'}")
    
    x, l = input_data
    
    if not isinstance(x, dict):
        raise ValueError(f"Expected x to be dict, got {type(x)}")
    
    # Move input data to device
    x_device = {k: v.to(device) for k, v in x.items()}
    l_device = l.to(device)
    input_batch = (x_device, l_device)
    
    # Move targets to device if they exist
    targets_device = None
    if targets is not None:
        if isinstance(targets, (list, tuple)) and len(targets) == 2:
            y, c = targets
            if y is not None and hasattr(y, 'to'):
                y_device = y.to(device)
            else:
                y_device = y
            if c is not None and hasattr(c, 'to'):
                c_device = c.to(device)
            else:
                c_device = c
            targets_device = (y_device, c_device)
        else:
            # Single target or other format
            if hasattr(targets, 'to'):
                targets_device = targets.to(device)
            else:
                targets_device = targets
    
    return input_batch, targets_device


def validate_model(
    model: torch.nn.Module, 
    val_loader: torch.utils.data.DataLoader, 
    num_batches: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Validate model on a subset of validation data.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        num_batches: Number of batches to validate on
        device: Device to run validation on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_masked_tokens = 0
    total_correct_predictions = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            
            # Use batch_to_device helper function
            try:
                batch, targets = batch_to_device(batch_data, device)
            except ValueError as e:
                logging.warning(f"Batch format error at index {batch_idx}: {e}")
                continue
            
            # Forward pass
            try:
                logits, mask, charge, padding_mask = model(batch)
                
                if mask is not None and mask.sum() > 0:
                    # Calculate loss using the exact same method as base_model.masked_prediction_loss
                    target_dom_ids = batch[0]['dom_id']
                    
                    # Apply mask and padding mask (same as base model)
                    final_mask = mask & ~padding_mask
                    
                    # Calculate cross-entropy loss per position (same as base model)
                    loss_unreduced = torch.nn.functional.cross_entropy(
                        logits.permute(0, 2, 1), target_dom_ids, reduction='none'
                    )
                    
                    # Average per sample, then across batch (same as base model)
                    loss_per_sample = (loss_unreduced * final_mask).sum(dim=1) / (final_mask.sum(dim=1) + 1e-8)
                    loss = loss_per_sample.mean()
                    total_loss += loss.item()
                    
                    # Calculate accuracy for masked tokens only
                    masked_logits = logits[mask]
                    true_dom_ids_masked = batch[0]['dom_id'][mask]
                    predictions = torch.argmax(masked_logits, dim=-1)
                    correct = (predictions == true_dom_ids_masked).sum().item()
                    total_correct_predictions += correct
                    total_masked_tokens += mask.sum().item()
                    
            except Exception as e:
                logging.warning(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics
    avg_loss = total_loss / min(num_batches, len(val_loader))
    accuracy = total_correct_predictions / max(total_masked_tokens, 1)
    
    metrics = {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'total_masked_tokens': total_masked_tokens,
        'total_correct_predictions': total_correct_predictions,
        'batches_processed': min(num_batches, len(val_loader))
    }
    
    logging.info(f"Validation metrics: {metrics}")
    return metrics


def create_mask(
    sample_batch: Tuple,
    config: Dict[str, Any],
    seed: int = 42,
    mask_prob: Optional[float] = None
) -> torch.Tensor:
    """Creates a fixed mask to be used in analysis.
    
    Args:
        sample_batch: Batch data (features, lengths)
        config: Configuration dictionary
        seed: Random seed for reproducibility
        mask_prob: Override mask probability (uses config value if None)
        
    Returns:
        Boolean tensor indicating which tokens should be masked
    """
    torch.manual_seed(seed)
    
    features = sample_batch[0]['features']
    lengths = sample_batch[1]
    
    batch_size, seq_length = features.shape[:2]
    device = features.device
    
    # Create padding mask
    padding_mask = torch.arange(seq_length, device=device)[None, :] >= lengths[:, None]
    
    # Find tokens eligible for masking (aux < 0)
    auxiliary_mask = features[:, :, 2] < 0
    
    # Get mask probability
    if mask_prob is None:
        actual_mask_prob = config['training'].get('val_mask_prob', config['training']['mask_prob'])
    else:
        actual_mask_prob = mask_prob
    
    # Create random mask
    random_mask = torch.rand(auxiliary_mask.shape, device=device) < actual_mask_prob
    
    # Final mask combines all conditions (exactly like embedding.py)
    final_mask = auxiliary_mask & random_mask & ~padding_mask
    
    logging.info(f"Created mask with {final_mask.sum().item()} masked tokens out of {auxiliary_mask.sum().item()} eligible tokens")
    
    return final_mask