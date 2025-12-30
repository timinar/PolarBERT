import torch
import torch.nn as nn
from polarbert.base_model import SimpleTransformer
from torch.optim.lr_scheduler import OneCycleLR
import inspect
from polarbert.utils.custom_lr_scheduler import TrapezoidalLR
from polarbert.completep import is_completep_enabled, get_residual_scale


def _is_mup_enabled(config: dict) -> bool:
    return config['training'].get('mup', {}).get('enabled', False)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['model']['num_heads']
        self.dim = config['model']['embedding_dim']
        self.head_dim = self.dim // self.n_heads
        self.is_mup_enabled = _is_mup_enabled(config)
        self.is_completep_enabled = is_completep_enabled(config)
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        bsz, seqlen, _ = x.shape
        
        # QKV projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape: (bsz, seqlen, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        # Transpose: (bsz, n_heads, seqlen, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        
        # Create attention mask from padding mask
        # Notice different logic for padding mask!
        attn_mask = padding_mask.logical_not().unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, seqlen)
        
        # Attention scaling factor
        # CompleteP and muP both use 1/d_head scaling instead of 1/sqrt(d_head)
        if self.is_mup_enabled or self.is_completep_enabled:
            attention_scale = 1.0 / xk.size(-1)
        else:
            attention_scale = 1.0 / xk.size(-1)**0.5
        
        # Flash attention (non-causal)
        output = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attn_mask, is_causal=False, scale=attention_scale
        )
        
        # Reshape: (bsz, seqlen, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # Final projection
        output = self.wo(output)
        
        return output
    


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config['model']['embedding_dim'], config['model']['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['model']['hidden_size'], config['model']['embedding_dim'])
        )
        self.layer_norm1 = nn.LayerNorm(config['model']['embedding_dim'])
        self.layer_norm2 = nn.LayerNorm(config['model']['embedding_dim'])

        # CompleteP residual scaling: 1/m_L for alpha=1
        self.residual_scale = 1.0
        if is_completep_enabled(config):
            self.residual_scale = get_residual_scale(config)

    def forward(self, x, padding_mask):
        # Attention block with residual scaling
        attn_output = self.attention(self.layer_norm1(x), padding_mask)
        x = x + self.residual_scale * attn_output

        # Feed-forward block with residual scaling
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.residual_scale * ff_output

        return x



class FlashTransformer(SimpleTransformer):
    def __init__(self, config):
        # Set flag to skip transformer creation in parent class
        self._skip_transformer = True
        super().__init__(config)
        self.is_mup_enabled = _is_mup_enabled(config)
        self._lr_scales = None
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['model']['num_layers'])
        ])
        
        # Optional final LayerNorm (can be disabled for ablation experiments)
        self.use_final_layer_norm = config['model'].get('use_final_layer_norm', True)
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config['model']['embedding_dim'])

        # Initialise weights for muP
        if self.is_mup_enabled:
            self.apply(self._init_mup_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('wq.weight') or pn.endswith('wk.weight') or pn.endswith('wv.weight') or pn.endswith('feed_forward.0.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.config['training']['mup']['init_std'] / self.config['training']['mup']['width_multiplier']**0.5)
                elif pn.endswith('wo.weight') or pn.endswith('feed_forward.2.weight'): # Both correspond to c_proj in the EleutherAI implementation
                    torch.nn.init.normal_(p, mean=0.0, std=self.config['training']['mup']['init_std'] / (2 * self.config['model']['num_layers'] * self.config['training']['mup']['width_multiplier'])**0.5)

    def _init_mup_weights(self, module):
        assert self.is_mup_enabled, "μP is not enabled"
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['training']['mup']['init_std'])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['training']['mup']['init_std'])

    def forward(self, x):
        embeddings, padding_mask, mask = self.embedding(x)
        
        # muP input scaling
        if self.is_mup_enabled:
            embeddings *= self.config['training']['mup']['input_alpha']

        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask)

        if self.use_final_layer_norm:
            embeddings = self.final_layer_norm(embeddings)

        # muP output scaling
        if self.is_mup_enabled:
            embeddings *= self.config['training']['mup']['output_alpha'] / self.config['training']['mup']['width_multiplier']
        
        cls_embed = embeddings[:, 0, :]
        charge = self.charge_prediction(cls_embed)
        logits = self.unembedding(embeddings[:, 1:, :])
        
        return logits, mask, charge, padding_mask[:, 1:]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        assert self._lr_scales is not None, "lr_scales not set. Call configure_optimizers() first."
        # Store current learning rates (as set by the scheduler)
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        # Apply the proper learning scaling to each parameter group
        for param_group, lr_scale in zip(optimizer.param_groups, self._lr_scales):
            param_group['lr'] *= lr_scale
        # Call the parent optimiser
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        # Restore the original learning rates (without scaling)
        for param_group, original_lr in zip(optimizer.param_groups, current_lrs):
            param_group['lr'] = original_lr

    def configure_optimizers(self):

        if self.config['training']['lr_scheduler'] == 'constant':
            initial_lr = float(self.config['training']['initial_lr'])
        elif self.config['training']['lr_scheduler'] == 'onecycle':
            initial_lr = float(self.config['training']['max_lr']) / float(self.config['training']['div_factor'])
        elif self.config['training']['lr_scheduler'] == 'trapezoidal':
            initial_lr = float(self.config['training']['max_lr'])
        else:
            raise ValueError(f"Unknown scheduler: {self.config['training']['lr_scheduler']}")
        weight_decay = float(self.config['training']['weight_decay'])

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.is_mup_enabled:
            mup_decay_params = []
            decay_params = []
            nodecay_params = []
            for n, p in param_dict.items():
                if p.dim() >= 2:
                    if (n.endswith('wq.weight') or n.endswith('wk.weight') or n.endswith('wv.weight') or n.endswith('wo.weight') or
                        n.endswith('feed_forward.0.weight') or n.endswith('feed_forward.2.weight')):
                        mup_decay_params.append(p)
                    else:
                        decay_params.append(p)
                else:
                    nodecay_params.append(p)
            optim_groups = [
                {'params': mup_decay_params, 'weight_decay': weight_decay},
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            self._lr_scales = [
                1/self.config['training']['mup']['width_multiplier'], # mup_decay_params
                1.0, # decay_params
                1.0, # nodecay_params
            ]
            num_mup_decay_params = sum(p.numel() for p in mup_decay_params)
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            self._lr_scales = [
                1.0, # decay_params
                1.0, # nodecay_params
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # TODO: refactor into a reusable function
        # Create AdamW optimizer and use the fused version if it is available
        use_fused_if_available = self.config['training'].get('use_fused_if_available', False)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = use_fused_if_available and fused_available and self.device.type == 'cuda'
        #TODO this is a temporary fix:
        use_fused = False
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=initial_lr,
            betas=(
                float(self.config['training'].get('adam_beta1', 0.9)),
                float(self.config['training'].get('adam_beta2', 0.999))
            ),
            eps=float(self.config['training'].get('adam_eps', 1e-8)),
            weight_decay=float(self.config['training']['weight_decay']),
            amsgrad=bool(self.config['training'].get('amsgrad', False)),
            fused=use_fused
        )

        # TODO: refactor into a reusable function
        total_steps = self.config['training'].get('total_steps')
        if self.config['training']['lr_scheduler'] == 'constant':
            return optimizer
        elif self.config['training']['lr_scheduler'] == 'onecycle':
            if total_steps is None:
                raise ValueError("total_steps must be specified in config for onecycle scheduler")
            # Use the pre-calculated total_steps from config
            scheduler = OneCycleLR(
                optimizer,
                max_lr=float(self.config['training']['max_lr']),
                total_steps=total_steps,
                pct_start=float(self.config['training']['pct_start']),
                div_factor=float(self.config['training']['div_factor']),
                final_div_factor=float(self.config['training']['final_div_factor']),
                anneal_strategy='cos'
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        elif self.config['training']['lr_scheduler'] == 'trapezoidal':
            if total_steps is None:
                raise ValueError("total_steps must be specified in config for trapezoidal scheduler")
            scheduler = TrapezoidalLR(
                optimizer,
                warmup_steps=self.config['training']['warmup_steps'],
                decay_steps=self.config['training']['decay_steps'],
                total_steps=total_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        else:
            raise ValueError(f"Unknown scheduler: {self.config['training']['lr_scheduler']}")
