import torch
import torch.nn as nn
import logging
class IceCubeEmbedding(nn.Module):
    def __init__(self, config, masking=False):
        super().__init__()
        embedding_dim = config['model']['embedding_dim']
        dom_embed_dim = config['model']['dom_embed_dim']
        self.mask_prob = config['training']['mask_prob']
        self.val_mask_prob = config['training'].get('val_mask_prob', self.mask_prob)
        if self.val_mask_prob != self.mask_prob:
            logging.info(f"Using different mask probabilities for training and validation: {self.mask_prob} and {self.val_mask_prob}")
        num_doms = 5160
        self.dom_embedding = nn.Embedding(num_doms + 2, dom_embed_dim)
        self.features_embedding = nn.Linear(3, embedding_dim - dom_embed_dim)
        self.masking = masking
        self.padding_idx = 0
        self.mask_idx = num_doms + 1
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, input):
        x, l = input  # l is the sequence length for each sample in the batch
        other_features = x['features']
        dom_ids = x['dom_id']
        batch_size, max_seq_len = other_features.shape[:2]
        assert dom_ids.shape == (batch_size, max_seq_len)
        device = other_features.device
        assert dom_ids.device == device
        
        # Create padding mask
        padding_mask = torch.arange(max_seq_len, device=device)[None, :] >= l[:, None]
        
        # DOM embeddings
        dom_embeds = self.dom_embedding(dom_ids)
        
        # Masking
        if self.masking:
            auxiliary_mask = other_features[:, :, 2] < 0 # More robust than == -0.5. Values can only be ±1/2
            mask_prob = self.mask_prob if self.training else self.val_mask_prob
            random_mask = torch.rand(auxiliary_mask.shape, device=device) < mask_prob
            mask = auxiliary_mask & random_mask & ~padding_mask
            dom_embeds[mask] = self.dom_embedding(torch.tensor(self.mask_idx, device=device))
        
        # Other features embedding
        features_embeds = self.features_embedding(other_features)
        
        # Concatenate embeddings
        combined_embeds = torch.cat([dom_embeds, features_embeds], dim=2)
        
        # Prepend CLS embedding
        full_embedding = torch.cat([self.cls_embedding.expand(batch_size, -1, -1), combined_embeds], dim=1)
        
        # Update padding mask to account for CLS token
        padding_mask = torch.cat([torch.zeros(batch_size, 1, device=device, dtype=torch.bool), padding_mask], dim=1)
        
        return full_embedding, padding_mask, mask if self.masking else None
