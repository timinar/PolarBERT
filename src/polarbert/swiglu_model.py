import torch
import torch.nn as nn
import torch.nn.functional as F
from polarbert.base_model import SimpleTransformer

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['model']['num_heads']
        self.dim = config['model']['embedding_dim']
        self.head_dim = self.dim // self.n_heads
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
        
        # Flash attention (non-causal)
        output = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attn_mask, is_causal=False
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
        self.feed_forward = SwiGLU(config)
        self.layer_norm1 = nn.LayerNorm(config['model']['embedding_dim'])
        self.layer_norm2 = nn.LayerNorm(config['model']['embedding_dim'])

    def forward(self, x, padding_mask):
        # Attention block
        attn_output = self.attention(self.layer_norm1(x), padding_mask)
        x = x + attn_output
        
        # SwiGLU feed-forward block
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + ff_output
        
        return x

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config['model']['embedding_dim']
        hidden_size = config['model']['hidden_size']
        
        self.w1 = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.w2 = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, embedding_dim, bias=False)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SwiGLUTransformer(SimpleTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['model']['num_layers'])
        ])

    def forward(self, x):
        embeddings, padding_mask, mask = self.embedding(x)
        
        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask)
        
        cls_embed = embeddings[:, 0, :]
        charge = self.charge_prediction(cls_embed)
        logits = self.unembedding(embeddings[:, 1:, :])
        
        return logits, mask, charge, padding_mask[:, 1:]

