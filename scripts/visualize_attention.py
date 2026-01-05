#!/usr/bin/env python3
"""
Visualize attention patterns from a trained PolarBERT model.

Usage:
    python scripts/visualize_attention.py --checkpoint <path> --config <path> [--num_samples 5]
"""

import torch
import torch.nn as nn
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Patch Attention to capture attention weights
class AttentionWithCapture(nn.Module):
    """Wrapper that captures attention weights during forward pass."""

    def __init__(self, original_attention):
        super().__init__()
        self.original = original_attention
        self.captured_weights = None

    def forward(self, x, padding_mask, rope_cos=None, rope_sin=None):
        # Replicate the forward pass but capture attention weights
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.original.wq(x), self.original.wk(x), self.original.wv(x)

        xq = xq.view(bsz, seqlen, self.original.n_heads, self.original.head_dim)
        xk = xk.view(bsz, seqlen, self.original.n_heads, self.original.head_dim)
        xv = xv.view(bsz, seqlen, self.original.n_heads, self.original.head_dim)

        # RoPE
        if self.original.use_rope and rope_cos is not None and rope_sin is not None:
            from polarbert.flash_model import apply_rotary_emb
            xq = apply_rotary_emb(xq, rope_cos, rope_sin)
            xk = apply_rotary_emb(xk, rope_cos, rope_sin)

        # QK Norm
        if self.original.use_qk_norm:
            xq = self.original.q_norm(xq)
            xk = self.original.k_norm(xk)

        # Transpose for attention
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Compute attention scores manually
        scale = 1.0 / xk.size(-1) if (self.original.is_mup_enabled or self.original.is_completep_enabled) else 1.0 / xk.size(-1)**0.5

        attn_scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale

        # Apply mask
        attn_mask = padding_mask.logical_not().unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        self.captured_weights = attn_weights.detach().cpu()

        # Apply attention
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.original.wo(output)

        return output


def patch_model_for_capture(model):
    """Replace attention modules with capturing versions."""
    for block in model.transformer_blocks:
        block.attention = AttentionWithCapture(block.attention)
    return model


def get_attention_weights(model):
    """Extract captured attention weights from all layers."""
    weights = []
    for block in model.transformer_blocks:
        if hasattr(block.attention, 'captured_weights') and block.attention.captured_weights is not None:
            weights.append(block.attention.captured_weights)
    return weights


def compute_attention_stats(weights, padding_mask):
    """Compute statistics about attention patterns."""
    stats = {}

    # weights: list of (bsz, n_heads, seqlen, seqlen) tensors
    n_layers = len(weights)
    bsz, n_heads, seqlen, _ = weights[0].shape

    # Get valid length per sample
    valid_lens = (~padding_mask).sum(dim=1).cpu().numpy()

    for layer_idx, w in enumerate(weights):
        layer_stats = {}

        # Average attention to CLS (position 0) across all queries
        # w[:, :, :, 0] is attention FROM all positions TO cls
        cls_attention = w[:, :, :, 0].mean(dim=2)  # (bsz, n_heads)
        layer_stats['cls_attention_mean'] = cls_attention.mean().item()
        layer_stats['cls_attention_per_head'] = cls_attention.mean(dim=0).numpy()

        # Attention entropy per head (higher = more uniform)
        # Only compute for valid positions
        entropy_per_head = []
        for head in range(n_heads):
            head_entropy = []
            for b in range(bsz):
                valid_len = valid_lens[b]
                attn = w[b, head, :valid_len, :valid_len]
                # Compute entropy for each query position
                eps = 1e-10
                ent = -(attn * torch.log(attn + eps)).sum(dim=-1).mean()
                head_entropy.append(ent.item())
            entropy_per_head.append(np.mean(head_entropy))
        layer_stats['entropy_per_head'] = np.array(entropy_per_head)

        # Position bias: average attention received by each position
        pos_attention = w.mean(dim=(0, 1, 2)).numpy()  # (seqlen,)
        layer_stats['position_attention'] = pos_attention

        stats[f'layer_{layer_idx}'] = layer_stats

    return stats


def plot_attention_matrices(weights, sample_idx, save_path, max_seq=64):
    """Plot attention matrices for all layers and heads."""
    n_layers = len(weights)
    n_heads = weights[0].shape[1]
    seqlen = min(weights[0].shape[2], max_seq)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2*n_heads, 2*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_heads == 1:
        axes = axes.reshape(-1, 1)

    for layer_idx, w in enumerate(weights):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            attn = w[sample_idx, head_idx, :seqlen, :seqlen].numpy()
            im = ax.imshow(attn, cmap='viridis', vmin=0, vmax=min(1, attn.max()*1.2))
            ax.set_title(f'L{layer_idx}H{head_idx}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer_idx == 0:
                ax.set_xlabel(f'H{head_idx}', fontsize=8)
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx}', fontsize=8)

    plt.suptitle(f'Attention Patterns (Sample {sample_idx})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention matrices to {save_path}")


def plot_attention_stats(stats, save_path):
    """Plot summary statistics."""
    n_layers = len(stats)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. CLS attention per layer/head
    ax = axes[0, 0]
    cls_data = np.array([stats[f'layer_{i}']['cls_attention_per_head'] for i in range(n_layers)])
    im = ax.imshow(cls_data, cmap='Reds', aspect='auto')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Attention TO CLS (position 0)')
    plt.colorbar(im, ax=ax)

    # 2. Entropy per layer/head
    ax = axes[0, 1]
    entropy_data = np.array([stats[f'layer_{i}']['entropy_per_head'] for i in range(n_layers)])
    im = ax.imshow(entropy_data, cmap='Blues', aspect='auto')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Attention Entropy (higher=more uniform)')
    plt.colorbar(im, ax=ax)

    # 3. Position bias (first layer)
    ax = axes[1, 0]
    pos_attn = stats['layer_0']['position_attention'][:32]  # First 32 positions
    ax.bar(range(len(pos_attn)), pos_attn)
    ax.set_xlabel('Position')
    ax.set_ylabel('Avg Attention Received')
    ax.set_title('Position Bias (Layer 0, first 32 pos)')
    ax.axhline(y=1/len(pos_attn), color='r', linestyle='--', label='uniform')

    # 4. Position bias (last layer)
    ax = axes[1, 1]
    pos_attn = stats[f'layer_{n_layers-1}']['position_attention'][:32]
    ax.bar(range(len(pos_attn)), pos_attn)
    ax.set_xlabel('Position')
    ax.set_ylabel('Avg Attention Received')
    ax.set_title(f'Position Bias (Layer {n_layers-1}, first 32 pos)')
    ax.axhline(y=1/len(pos_attn), color='r', linestyle='--', label='uniform')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention stats to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='attention_viz', help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create model
    from polarbert.finetuning import SimpleTransformerCls
    model = SimpleTransformerCls(config)

    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('pretrained_model.', ''): v for k, v in checkpoint['state_dict'].items()
                      if k.startswith('pretrained_model.')}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cuda()

    # Patch for attention capture
    model = patch_model_for_capture(model)

    # Load some data - use small batch for visualization
    config['training']['per_device_batch_size'] = 32
    from polarbert.utils.data import get_dataloaders
    _, val_loader = get_dataloaders(config, dataset_type='kaggle')

    # Get a batch - format is [[{features, dom_id}, lengths], [labels, ...]]
    batch = next(iter(val_loader))
    x_dict, lengths = batch[0][0], batch[0][1]
    x = {k: v.cuda() for k, v in x_dict.items()}
    lengths = lengths.cuda()

    # Forward pass to capture attention
    with torch.no_grad():
        _ = model((x, lengths))

    # Get captured weights
    weights = get_attention_weights(model)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute padding mask
    max_seq = x['features'].shape[1]
    padding_mask = torch.arange(max_seq, device=lengths.device)[None, :] >= lengths[:, None]

    # Plot individual samples
    for i in range(min(args.num_samples, len(lengths))):
        plot_attention_matrices(weights, i, output_dir / f'attention_sample_{i}.png')

    # Compute and plot stats
    stats = compute_attention_stats(weights, padding_mask.cpu())
    plot_attention_stats(stats, output_dir / 'attention_stats.png')

    # Print summary
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*60)

    n_layers = len(weights)
    for layer_idx in range(n_layers):
        layer_stats = stats[f'layer_{layer_idx}']
        print(f"\nLayer {layer_idx}:")
        print(f"  CLS attention (mean): {layer_stats['cls_attention_mean']:.4f}")
        print(f"  CLS attention by head: {layer_stats['cls_attention_per_head'].round(3)}")
        print(f"  Entropy by head: {layer_stats['entropy_per_head'].round(3)}")

        # Flag potential sink behavior
        if layer_stats['cls_attention_mean'] > 0.15:
            print(f"  ⚠️  HIGH CLS attention - potential sink behavior")

        low_entropy_heads = np.where(layer_stats['entropy_per_head'] < 1.5)[0]
        if len(low_entropy_heads) > 0:
            print(f"  ⚠️  Low entropy heads: {low_entropy_heads} - concentrated attention")

    print("\n" + "="*60)
    print(f"Visualizations saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
