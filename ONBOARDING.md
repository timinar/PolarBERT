# PolarBERT Onboarding Guide

## What is PolarBERT?

PolarBERT is a foundation model for the IceCube neutrino telescope. It is pretrained on masked DOM (Digital Optical Module) prediction — given a sequence of sensor activations from a neutrino event, the model learns to predict masked DOM identities and charge values. The pretrained encoder is then finetuned for downstream tasks like directional reconstruction (predicting the neutrino's incoming direction as a 3D unit vector).

## Setup

```bash
conda activate polarbert
pip install -e .
```

The package installs as `polarbert` and all imports use this namespace (e.g., `from polarbert.finetuning import DirectionalHead`).

## Model Architecture

PolarBERT uses a Flash Attention transformer with the following structure:

1. **Embedding layer** (`src/polarbert/embedding.py`): Converts raw event data into embeddings
   - DOM positions → position embedding (Linear, 3→128)
   - Hit features (time, charge, aux) → feature embedding (Linear, 3→128)
   - Concatenated to form 256-dim token embeddings
   - A learnable CLS token is prepended to the sequence

2. **Transformer encoder** (`src/polarbert/flash_model.py`): 8 layers, 256-dim, 8 heads
   - Pre-norm architecture (LayerNorm → Attention → residual, LayerNorm → FFN → residual)
   - SDPA-based attention with efficient attention backend
   - Optional features: RoPE, QK Norm (configurable via config)

3. **Task heads** (`src/polarbert/finetuning.py`):
   - `SimpleTransformerCls`: Encoder-only backbone that returns the CLS token embedding
   - `DirectionalHead`: CLS → Linear(256, 1024) → GELU → Linear(1024, 3) → L2 normalize
   - `EnergyRegressionHead`: Same architecture but outputs scalar energy

The default architecture parameters are:
| Parameter | Value |
|-----------|-------|
| `embedding_dim` | 256 |
| `dom_embed_dim` | 128 |
| `num_heads` | 8 |
| `num_layers` | 8 |
| `hidden_size` (FFN) | 1024 |

## Loading a Checkpoint

### Pretraining checkpoint (`.pth`)

```python
import torch
from polarbert.flash_model import FlashTransformer

config = {
    'model': {
        'embedding_dim': 256, 'dom_embed_dim': 128,
        'num_heads': 8, 'num_layers': 8, 'hidden_size': 1024,
        'lambda_charge': 1.0, 'activation': 'gelu',
        'use_dom_positions': True, 'use_final_layer_norm': True,
    },
    'training': {
        'mask_prob': 0.25, 'lr_scheduler': 'constant',
        'initial_lr': 1e-4, 'weight_decay': 0.01,
    },
    'data': {
        'sensor_geometry_path': '/path/to/sensor_geometry.csv',
    },
}

model = FlashTransformer(config)
state = torch.load('checkpoint.pth', map_location='cpu', weights_only=True)
if 'state_dict' in state:
    state = state['state_dict']
model.load_state_dict(state, strict=False)
model.eval()
```

### Finetuned checkpoint (backbone + head)

```python
from polarbert.finetuning import DirectionalHead, SimpleTransformerCls

# Config must include 'directional' and 'pretrained' sections
config['model']['directional'] = {'hidden_size': 1024}

backbone = SimpleTransformerCls(config)
model = DirectionalHead(config, backbone)

state = torch.load('finetuned_checkpoint.pth', map_location='cpu', weights_only=True)
if 'state_dict' in state:
    state = state['state_dict']
model.load_state_dict(state, strict=True)
model.eval()
```

## Data Format

PolarBERT uses memory-mapped `.npy` arrays for efficient data loading. Each dataset directory contains:

- `x.npy` — structured array with fields: `dom_id` (uint16), `time` (float16), `charge` (float16), `aux` (float16)
- `l.npy` — sequence lengths per event
- `y.npy` — targets (azimuth, zenith for direction)
- `c.npy` — auxiliary info (total charge)

To create datasets from raw Kaggle data, see `scripts/create_memmapped_dataset_kaggle.py`.

## Running Inference

Here's a minimal example for getting predictions from a finetuned model:

```python
import torch
import numpy as np
from polarbert.finetuning import DirectionalHead, SimpleTransformerCls
from polarbert.icecube_dataset import IceCubeDataset
from polarbert.utils.data import default_transform

# Load model (see "Loading a Checkpoint" above)
# ...

# Load data
dataset = IceCubeDataset(
    data_dir='/path/to/memmapped_data',
    num_events=1000,
    transform=default_transform,
)

# Run inference
model.eval()
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(dataset, batch_size=256):
        inp, targets = batch
        predictions = model(inp)  # (batch_size, 3) unit vectors
```

For a complete inference example with per-event evaluation, see `scripts/run_inference_for_filtering.py`.

## Extracting Attention Maps

For interpretability work, `scripts/visualize_attention.py` provides utilities to capture attention weights:

```python
from scripts.visualize_attention import patch_model_for_capture, get_attention_weights

# Patch the model to capture attention weights during forward pass
model = patch_model_for_capture(model)

# Run a forward pass
with torch.no_grad():
    output = model(inp)

# Get captured weights: list of (batch, heads, seq, seq) tensors, one per layer
weights = get_attention_weights(model)

# weights[layer_idx] has shape (batch_size, num_heads, seq_len, seq_len)
# The CLS token is at position 0
cls_attention = weights[0][0, :, 0, :]  # Head attention from CLS, layer 0
```

Run the full visualization script:
```bash
python scripts/visualize_attention.py --checkpoint /path/to/model.pth --config /path/to/config.yaml
```

## Key Files Reference

| File | Description |
|------|-------------|
| `src/polarbert/flash_model.py` | Flash Attention transformer (pretraining model) |
| `src/polarbert/finetuning.py` | Finetuning heads (DirectionalHead, EnergyRegressionHead) |
| `src/polarbert/embedding.py` | IceCube event embedding layer |
| `src/polarbert/base_model.py` | Base transformer and optimizer configuration |
| `src/polarbert/pretraining.py` | Pretraining entry point and model registry |
| `src/polarbert/loss_functions.py` | Angular distance loss functions |
| `src/polarbert/icecube_dataset.py` | Memory-mapped dataset loader |
| `scripts/visualize_attention.py` | Attention weight extraction and visualization |
| `scripts/run_inference_for_filtering.py` | Full inference pipeline example |
| `scripts/create_memmapped_dataset_kaggle.py` | Dataset creation from raw Kaggle data |
| `configs/polarbert.example.yaml` | Example pretraining config |
| `configs/finetuning.example.yaml` | Example finetuning config |
