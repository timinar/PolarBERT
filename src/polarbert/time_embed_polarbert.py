import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
import math
from typing import Tuple, Optional


from polarbert.config import PolarBertConfig # Import ModelConfig if separate
from polarbert.time_embedding import IceCubeTimeEmbedding

# --- RMSNorm Implementation ---
class RMSNorm(torch.nn.Module):
    """ Root Mean Square Layer Normalization """
    def __init__(self, dim: int, eps: float = 1e-6): # Default eps if not in config
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate sqrt(E[x^2] + eps)
        # Original: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Using separate steps for clarity:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x):
        # Normalize and scale by gamma
        output = self._norm(x.float()).type_as(x) # Ensure float for norm, then cast back
        return output * self.weight


# --- Updated Transformer Components ---

class PolarBertAttention(nn.Module):

    # ... (using nn.MultiheadAttention -- should use flash attention)
    """ Basic Multi-Head Attention using PyTorch's efficient implementation """
    def __init__(self, config: PolarBertConfig):
        super().__init__()
        model_cfg = config.model
        self.embed_dim = model_cfg.embedding_dim
        self.n_head = model_cfg.num_heads
        self.dropout = model_cfg.dropout
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by num_heads"
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, is_causal=False, need_weights=False)
        return attn_output


class PolarBertFeedForward(nn.Module):
    # ... (handling SwiGLU/MLP) ...
    """ FeedForward network with choice of SwiGLU or MLP """
    def __init__(self, config: PolarBertConfig):
        super().__init__()
        model_cfg = config.model
        emb_dim = model_cfg.embedding_dim
        hidden_dim = model_cfg.hidden_size
        dropout = model_cfg.dropout # TODO: remove completely?
        self.ffd_type = model_cfg.ffd_type.lower()

        if self.ffd_type == "swiglu":
            self.w1 = nn.Linear(emb_dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(emb_dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, emb_dim, bias=False)
            self.dropout = nn.Dropout(dropout)
        elif self.ffd_type == "mlp":
             self.ffn = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, emb_dim),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unsupported ffd_type: {model_cfg.ffd_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffd_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w3(x)) * self.w1(x)))
        else: # MLP
            return self.ffn(x)


class PolarBertBlock(nn.Module):
    """ A single Transformer block using RMSNorm (Pre-Normalization) """
    def __init__(self, config: PolarBertConfig):
        super().__init__()
        self.config = config
        emb_dim = config.model.embedding_dim
        norm_eps = config.model.norm_eps # Get eps from config
        dropout = config.model.dropout # TODO: Remove?

        # Use RMSNorm
        self.ln1 = RMSNorm(emb_dim, eps=norm_eps)
        self.attn = PolarBertAttention(config)
        # Use RMSNorm
        self.ln2 = RMSNorm(emb_dim, eps=norm_eps)
        self.ffn = PolarBertFeedForward(config)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-Norm: Norm -> Op -> Dropout -> Add Residual
        # Attention part
        residual = x
        x_norm = self.ln1(x)
        attn_output = self.attn(x_norm, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_output)

        # FeedForward part
        residual = x
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)
        return x

# --- Updated Main Model ---

class PolarBertModel(pl.LightningModule):
    """
    Main PolarBERT model using RMSNorm and custom components.
    """
    def __init__(self, config: PolarBertConfig):
        super().__init__()
        # Ensure config is validated if not done automatically by PolarBertConfig.__init__
        # config._validate() # Assuming validation happens in PolarBertConfig __init__
        self.config = config
        # Important: Convert config object to dict for save_hyperparameters
        # Ensure config.to_dict() method exists and works correctly
        self.save_hyperparameters(config.to_dict())

        # --- Modules ---
        # Use masking=True for pre-training by default
        self.embedding = IceCubeTimeEmbedding(config, masking=True)

        self.transformer_blocks = nn.ModuleList(
            [PolarBertBlock(config) for _ in range(config.model.num_layers)]
        )

        # Use RMSNorm
        self.final_norm = RMSNorm(config.model.embedding_dim, eps=config.model.norm_eps)

        # --- Prediction Heads ---
        # Predict original DOM ID (0 to N_DOMS-1)
        num_dom_classes = config.model.embedding.dom_vocab_size - 2 # Exclude PAD, MASK
        self.dom_head = nn.Linear(config.model.embedding_dim, num_dom_classes)

        self.charge_head = nn.Linear(config.model.embedding_dim, 1)

        # Store loss weight
        self.lambda_charge = config.model.lambda_charge


    def forward(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]) \
            -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """ Forward pass for pre-training. """
        (x, l), _ = batch
        # batch_size, seq_len_orig variables removed as they were unused

        # 1. Get Embeddings and Masks
        hidden_states, final_padding_mask, output_mask = self.embedding((x, l))
        attn_key_padding_mask = final_padding_mask # Use the mask covering CLS + Sequence

        # TODO: Add RoPE or Positional Embeddings here if config enables them

        # 2. Pass through Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, key_padding_mask=attn_key_padding_mask)

        # 3. Final Normalization
        hidden_states = self.final_norm(hidden_states)

        # 4. Separate CLS and Sequence Embeddings
        cls_embed = hidden_states[:, 0, :]         # (B, E)
        sequence_embeds = hidden_states[:, 1:, :] # (B, L_orig, E)

        # 5. Prediction Heads
        charge_pred = self.charge_head(cls_embed)             # (B, 1)
        dom_logits = self.dom_head(sequence_embeds)         # (B, L_orig, num_dom_classes)

        # 6. Prepare Padding Mask for Loss Calculation (needs shape B, L_orig)
        seq_padding_mask = final_padding_mask[:, 1:]

        return dom_logits, charge_pred, output_mask, seq_padding_mask

    def _calculate_loss(self, batch):
        # (Keep implementation from previous response)
        # ... (calculating dom_loss and charge_loss) ...
        (x, l), y_data = batch
        true_total_charge = y_data[1] if y_data is not None else None
        true_dom_ids = x[:, :, 3].long()

        dom_logits, charge_pred, output_mask, seq_padding_mask = self.forward(batch)

        dom_loss = torch.tensor(0.0, device=dom_logits.device)
        if output_mask is not None and output_mask.sum() > 0:
             dom_targets = true_dom_ids - 1 # Map 1..N -> 0..N-1; Pad 0 -> -1
             masked_logits = dom_logits[output_mask]
             masked_targets = dom_targets[output_mask]
             dom_loss = F.cross_entropy(masked_logits, masked_targets, ignore_index=-1)

        charge_loss = torch.tensor(0.0, device=charge_pred.device)
        if true_total_charge is not None:
             true_log_charge = torch.log10(torch.clamp(true_total_charge.float(), min=1e-6))
             charge_loss = F.mse_loss(charge_pred.squeeze(-1), true_log_charge)

        combined_loss = dom_loss + self.lambda_charge * charge_loss
        return combined_loss, dom_loss, charge_loss


    def training_step(self, batch, batch_idx):
        # (Keep implementation from previous response)
        # ... (logging losses) ...
        combined_loss, dom_loss, charge_loss = self._calculate_loss(batch)
        self.log('train/loss', combined_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/dom_loss', dom_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/charge_loss', charge_loss, on_step=True, on_epoch=False, sync_dist=True)
        return combined_loss


    def validation_step(self, batch, batch_idx):
        # (Keep implementation from previous response)
        # ... (logging losses) ...
        combined_loss, dom_loss, charge_loss = self._calculate_loss(batch)
        self.log('val/loss', combined_loss, prog_bar=True, sync_dist=True)
        self.log('val/dom_loss', dom_loss, sync_dist=True)
        self.log('val/charge_loss', charge_loss, sync_dist=True)
        return combined_loss


    def configure_optimizers(self):
        """Set up optimizer and learning rate scheduler with proper weight decay handling."""

        # --- Define parameter groups based on decay settings ---
        decay = set()
        no_decay = set()

        # Modules whose weights SHOULD be decayed (whitelist)
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        # Modules whose weights should NOT be decayed (blacklist)
        # Includes normalization layers and embedding layers
        blacklist_weight_modules = (RMSNorm, torch.nn.LayerNorm, torch.nn.Embedding)

        print("Classifying parameters for weight decay...")
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn

                if not p.requires_grad:
                    continue # Skip parameters that don't require gradients

                # --- Parameter Classification Logic ---
                if pn.endswith('bias'):
                    # All biases don't decay
                    # print(f"No Decay: {fpn} (bias)")
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # Whitelisted weights decay
                    # print(f"Decay:   {fpn} (whitelist weight)")
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # Blacklisted weights don't decay
                    # print(f"No Decay: {fpn} (blacklist weight)")
                    no_decay.add(fpn)
                # Catch normalization parameters explicitly by name/type if not covered above
                elif isinstance(m, (RMSNorm, torch.nn.LayerNorm)):
                    # print(f"No Decay: {fpn} (norm param)")
                    no_decay.add(fpn)
                # Special case specific parameters if needed
                elif pn in ['cls_embedding']: # Decay CLS token embedding
                    # print(f"Decay:   {fpn} (specific param)")
                    decay.add(fpn)

        # --- Validation ---
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"
        unassigned_params = param_dict.keys() - union_params
        if len(unassigned_params) > 0:
            print(f"WARNING: Assigning parameters to no_decay group by default: {unassigned_params}")
            no_decay.update(unassigned_params) # Assign leftovers to no_decay

        print(f"  {len(decay)} parameter tensors decaying.")
        print(f"  {len(no_decay)} parameter tensors NOT decaying.")

        # --- Create Optimizer Groups ---
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": self.config.training.weight_decay}, # Apply configured decay
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0}, # No decay for this group
        ]

        # --- Create Optimizer ---
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.max_lr

        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                optim_groups, lr=lr,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                eps=self.config.training.adam_eps,
                amsgrad=self.config.training.amsgrad
            )
        else: raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # --- Create Scheduler ---
        scheduler_name = self.config.training.lr_scheduler.lower()
        if scheduler_name == 'onecycle':
            if self.config.training.total_steps is None:
                # Attempt to retrieve from trainer if possible (might not be ready yet)
                # Or raise error more clearly here if estimation failed earlier
                raise ValueError("config.training.total_steps must be calculated before setting up OneCycleLR.")

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.training.max_lr,
                total_steps=self.config.training.total_steps,
                pct_start=self.config.training.pct_start, # Already adjusted by warm_up_steps if needed
                div_factor=self.config.training.div_factor,
                final_div_factor=self.config.training.final_div_factor,
                anneal_strategy='cos'
            )
            # Standard PL return format for optimizer + LR scheduler
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # Step scheduler every batch step
                    "frequency": 1
                }
            }
        elif scheduler_name in ['none', None, 'constant']: # Handle 'constant' if needed
            # If constant, just return optimizer, LR set initially
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")