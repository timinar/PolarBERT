import torch
import torch.nn as nn
import logging
import pandas as pd
import numpy as np
from pathlib import Path

class IceCubeEmbedding(nn.Module):
    def __init__(self, config, masking=False):
        super().__init__()
        embedding_dim = config['model']['embedding_dim']
        dom_embed_dim = config['model']['dom_embed_dim']
        self.use_dom_positions = config['model'].get('use_dom_positions', False)
        
        self.mask_prob = config['training']['mask_prob']
        self.val_mask_prob = config['training'].get('val_mask_prob', self.mask_prob)
        if self.val_mask_prob != self.mask_prob:
            logging.info(f"Using different mask probabilities for training and validation: {self.mask_prob} and {self.val_mask_prob}")

        num_doms = 5160
        self.masking = masking
        self.padding_idx = 0

        # Initialize embeddings with proper scale (std=0.02) for CompleteP compatibility
        # This matches init_std_base and ensures stable training across seeds
        init_std = config.get('training', {}).get('completep', {}).get('init_std_base', 0.02)
        self.cls_embedding = nn.Parameter(torch.empty(1, 1, embedding_dim).normal_(mean=0.0, std=init_std))

        # Mask token is always a dedicated, trainable parameter if masking is enabled
        if self.masking:
            self.mask_token_embedding = nn.Parameter(torch.empty(1, dom_embed_dim).normal_(mean=0.0, std=init_std))

        if not self.use_dom_positions:
            # ID-based: Use padding_idx to freeze the padding embedding at zeros
            self.dom_embedding = nn.Embedding(num_doms + 1, dom_embed_dim, padding_idx=self.padding_idx)
        else:
            # Position-based: No need for a trainable padding vector
            sensor_geometry_path = config['data'].get('sensor_geometry_path')
            if sensor_geometry_path is None:
                raise ValueError("sensor_geometry_path must be provided when use_dom_positions is True.")
            
            positions = self._load_sensor_positions(sensor_geometry_path, num_doms)
            # to push on device automatically when model is moved
            self.register_buffer('dom_positions', positions)
            self.position_embedding = nn.Linear(3, dom_embed_dim)

        self.features_embedding = nn.Linear(3, embedding_dim - dom_embed_dim)

    def _load_sensor_positions(self, sensor_geometry_path, num_doms):
        """Loads and normalizes sensor positions by dividing by 500."""
        geo_path = Path(sensor_geometry_path)
        if not geo_path.exists():
            raise FileNotFoundError(f"Sensor geometry file not found at: {geo_path}")
        
        geometry = pd.read_csv(geo_path)
        
        # Tensor for padding (idx 0 -> [0,0,0]) and DOMs (idx 1 to num_doms)
        positions = torch.zeros(num_doms + 1, 3, dtype=torch.float32)
        pos_xyz = torch.from_numpy(geometry[['x', 'y', 'z']].values.astype(np.float32))
        positions[1:num_doms + 1] = pos_xyz / 500.0
        
        return positions

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
            # The lookup provides (0,0,0) for padding, which is fine since attention will ignore it.
            pos = self.dom_positions[dom_ids]
            dom_embeds = self.position_embedding(pos)

        if self.masking:
            auxiliary_mask = other_features[:, :, 2] < 0
            mask_prob = self.mask_prob if self.training else self.val_mask_prob
            random_mask = torch.rand(auxiliary_mask.shape, device=device) < mask_prob
            mask = auxiliary_mask & random_mask & ~padding_mask
            
            dom_embeds[mask] = self.mask_token_embedding.to(dtype=dom_embeds.dtype)
        
        features_embeds = self.features_embedding(other_features)
        combined_embeds = torch.cat([dom_embeds, features_embeds], dim=2)
        # Prepend CLS embedding
        full_embedding = torch.cat([self.cls_embedding.expand(batch_size, -1, -1), combined_embeds], dim=1)
        padding_mask = torch.cat([torch.zeros(batch_size, 1, device=device, dtype=torch.bool), padding_mask], dim=1)
        
        return full_embedding, padding_mask, mask