import torch
import torch.nn as nn

class IceCubeEmbedding(nn.Module):
    def __init__(self, config, masking=False):
        super().__init__()
        embedding_dim = config['model']['embedding_dim']
        dom_embed_dim = config['model']['dom_embed_dim']
        self.mask_prob = config['training']['mask_prob']
        num_doms = 5160
        self.dom_embedding = nn.Embedding(num_doms + 2, dom_embed_dim)
        self.features_embedding = nn.Linear(3, embedding_dim - dom_embed_dim)
        self.masking = masking
        self.padding_idx = 0
        self.mask_idx = num_doms + 1
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, input):
        x, l = input  # l is the sequence length for each sample in the batch
        batch_size, max_seq_len, _ = x.shape
        
        # Create padding mask
        padding_mask = torch.arange(max_seq_len, device=x.device)[None, :] >= l[:, None]
        
        # DOM embeddings
        dom_embeds = self.dom_embedding(x[:, :, -1].long())
        
        # Masking
        if self.masking:
            auxiliary_mask = x[:, :, 2] == -0.5
            random_mask = torch.rand(auxiliary_mask.shape, device=x.device) < self.mask_prob
            mask = auxiliary_mask & random_mask & ~padding_mask
            dom_embeds[mask] = self.dom_embedding(torch.tensor(self.mask_idx, device=x.device))
        
        # Other features embedding
        other_features = x[:, :, :3]
        features_embeds = self.features_embedding(other_features)
        
        # Concatenate embeddings
        combined_embeds = torch.cat([dom_embeds, features_embeds], dim=2)
        
        # Prepend CLS embedding
        full_embedding = torch.cat([self.cls_embedding.expand(batch_size, -1, -1), combined_embeds], dim=1)
        
        # Update padding mask to account for CLS token
        padding_mask = torch.cat([torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool), padding_mask], dim=1)
        
        return full_embedding, padding_mask, mask if self.masking else None
