import torch
import torch.nn as nn
import numpy as np # Only needed for float('inf') if torch.inf unavailable in older torch
import warnings
from typing import Tuple, Optional # Added Optional for return type hint

# --- Define Constants calculated FROM Config ---
PAD_IDX = 0

class IceCubeTimeEmbedding(nn.Module):
    """
    Embedding layer using index replacement for masking specific features.
    Indices: 0=[PAD], LAST_IDX=[MASK]. Assumes sub-embedding dims sum
    to the final model embedding dimension (no projection layer).
    Quantizes charge based on normalized log values using pre-calculated buffers.
    """
    def __init__(self, config, masking=False):
        """
        Initializes the embedding layer.

        Args:
            config (PolarBertConfig): The main configuration object.
            masking (bool): Whether to enable masking during training.
        """
        super().__init__()
        self.config = config
        self.masking = masking
        embedding_cfg = config.model.embedding

        self.mask_prob = embedding_cfg.masking_prob if masking else 0.0

        # --- Calculate Vocab Sizes and MASK Indices ---
        self.dom_vocab_size = embedding_cfg.dom_vocab_size
        self.dom_mask_idx = self.dom_vocab_size - 1

        self.time_vocab_size = embedding_cfg.time_vocab_size
        self.time_mask_idx = self.time_vocab_size - 1
        self.max_time_duration = self.time_mask_idx - 1

        self.charge_vocab_size = embedding_cfg.charge_vocab_size
        self.charge_mask_idx = self.charge_vocab_size - 1
        self.num_charge_bins = self.charge_vocab_size - 2
        if self.num_charge_bins <= 0:
            raise ValueError("charge_vocab_size must be >= 2 for PAD and MASK indices.")

        # --- TODO: Define Aux Vocab Size and MASK index ---
        self.aux_num_cats = 2 # Example
        self.aux_vocab_size = self.aux_num_cats + 2
        self.aux_mask_idx = self.aux_vocab_size - 1

        # --- Embedding Layers (All use padding_idx=0) ---
        self.dom_embedding = nn.Embedding(
            self.dom_vocab_size, embedding_cfg.dom_embedding_dim, padding_idx=PAD_IDX
        )
        self.time_embedding = nn.Embedding(
            self.time_vocab_size, embedding_cfg.time_embedding_dim, padding_idx=PAD_IDX
        )
        self.charge_embedding = nn.Embedding(
             self.charge_vocab_size, embedding_cfg.charge_embedding_dim, padding_idx=PAD_IDX
        )
        self.aux_embedding = nn.Embedding( # Assuming aux embedding needed
             self.aux_vocab_size, embedding_cfg.aux_embedding_dim, padding_idx=PAD_IDX
        )

        # --- Define and Register Charge Bin Edges Buffer ---
        # Get binning range from config (assuming they are added)
        charge_bin_min = embedding_cfg.charge_bin_min # e.g., -0.6
        charge_bin_max = embedding_cfg.charge_bin_max # e.g., 0.9
        # Create tensor on CPU initially
        charge_bin_edges_tensor = torch.linspace(
            charge_bin_min,
            charge_bin_max,
            steps=self.num_charge_bins + 1 # N+1 edges for N bins
        )
        # Register as buffer - will be moved to device with the module
        self.register_buffer('charge_bin_edges', charge_bin_edges_tensor)


        # --- Optional Projection Layer ---
        self.embedding_dim = config.model.embedding_dim
        total_sub_embed_dim = (
            embedding_cfg.dom_embedding_dim + embedding_cfg.time_embedding_dim +
            embedding_cfg.charge_embedding_dim + embedding_cfg.aux_embedding_dim
        )
        if embedding_cfg.embedding_projection:
             print("INFO: Using projection layer after concatenating embeddings.")
             self.projection = nn.Linear(total_sub_embed_dim, self.embedding_dim)
             if total_sub_embed_dim != self.embedding_dim:
                 print(f"  Projecting from {total_sub_embed_dim} to {self.embedding_dim}")
        elif total_sub_embed_dim == self.embedding_dim:
             print("INFO: Concatenated embeddings directly match model embedding dim. No projection layer used.")
             self.projection = None
        else:
             # Sum doesn't match and projection is False
             raise ValueError(
                 f"[Config Error] embedding_projection is False, but sum of sub-embedding dims ({total_sub_embed_dim}) "
                 f"does not match model.embedding_dim ({self.embedding_dim}). "
                 f"Adjust dimensions or set embedding_projection to True."
             )

        # --- CLS Token ---
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, self.embedding_dim))


    def forward(self, input_batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """ Forward pass for the embedding layer. """
        x, l = input_batch
        batch_size, seq_len, _ = x.shape
        device = x.device
        embedding_cfg = self.config.model.embedding

        # --- 1. Get Original Padding Mask ---
        padding_mask = (x[:, :, 3] == PAD_IDX)

        # --- 2. Process Time Feature to Indices ---
        time_normalized = x[:, :, 0]
        time_float_approx = time_normalized * 3e4 + 1e4
        time_float_masked_for_min = torch.where(padding_mask, torch.full_like(time_float_approx, float('inf')), time_float_approx)
        t_min_per_event = torch.min(time_float_masked_for_min, dim=1, keepdim=True)[0]
        t_min_per_event = torch.where(torch.isinf(t_min_per_event), torch.zeros_like(t_min_per_event), t_min_per_event)
        time_relative_float = time_float_approx - t_min_per_event
        time_relative_int = torch.round(time_relative_float).long()
        time_relative_int_clipped = torch.clamp(time_relative_int, min=0, max=self.max_time_duration)
        time_indices = torch.where(padding_mask, PAD_IDX, time_relative_int_clipped + 1)

        # --- 3. Process DOM ID Feature to Indices ---
        dom_indices = x[:, :, 3].long()

        # --- 4. Process Charge Feature to Indices ---
        charge_normalized = x[:, :, 1]
        # Use the buffer for bin edges - it's already on the correct device
        # Note: self.charge_bin_edges is automatically moved by PyTorch's .to(device)
        bucket_indices = torch.bucketize(charge_normalized, self.charge_bin_edges, right=True)
        charge_indices_base = torch.clamp(bucket_indices, min=1, max=self.num_charge_bins)
        charge_indices = torch.where(padding_mask, PAD_IDX, charge_indices_base)

        # --- 5. Process Auxiliary Feature to Indices ---
        aux_normalized = x[:, :, 2]
        aux_base_idx = torch.round(aux_normalized + 0.5).long()
        aux_base_idx_clipped = torch.clamp(aux_base_idx, 0, self.aux_num_cats - 1)
        aux_indices = torch.where(padding_mask, PAD_IDX, aux_base_idx_clipped + 1)

        # --- 6. Apply Masking (Index Replacement) ---
        output_mask = None
        if self.masking:
            is_non_auxiliary = (x[:, :, 2] == -0.5)
            random_mask = torch.rand(is_non_auxiliary.shape, device=device) < self.mask_prob
            output_mask = is_non_auxiliary & random_mask & ~padding_mask

            if embedding_cfg.masking_doms:
                dom_indices = torch.where(output_mask, self.dom_mask_idx, dom_indices)
            if embedding_cfg.masking_times:
                time_indices = torch.where(output_mask, self.time_mask_idx, time_indices)
            if embedding_cfg.masking_charges:
                charge_indices = torch.where(output_mask, self.charge_mask_idx, charge_indices)
            # if embedding_cfg.masking_aux: aux_indices = torch.where(output_mask, self.aux_mask_idx, aux_indices)

        # --- 7. Embedding Lookups ---
        dom_embeds = self.dom_embedding(dom_indices)
        time_embeds = self.time_embedding(time_indices)
        charge_embeds = self.charge_embedding(charge_indices)
        aux_embeds = self.aux_embedding(aux_indices)

        # --- 8. Combine Embeddings ---
        combined_embeds = torch.cat([dom_embeds, time_embeds, charge_embeds, aux_embeds], dim=2)

        # --- 8b. Apply Optional Projection Layer ---
        if self.projection is not None:
            projected_embeds = self.projection(combined_embeds)
        else:
            projected_embeds = combined_embeds

        # --- 9. Prepend CLS Token ---
        cls_token_expanded = self.cls_embedding.expand(batch_size, -1, -1).to(projected_embeds.dtype)
        full_embedding = torch.cat([cls_token_expanded, projected_embeds], dim=1)

        # --- 10. Create Final Padding Mask ---
        final_padding_mask = torch.cat([
             torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
             padding_mask
        ], dim=1)

        # --- 11. Return ---
        return full_embedding, final_padding_mask, output_mask