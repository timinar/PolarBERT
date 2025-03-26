# PolarBERT

A foundation model for the IceCube neutrino telescope, implementing masked modeling and transfer learning approaches.

## Features
- Memory-efficient data handling with memory-mapped datasets
- Multiple transformer architectures (Standard, Flash Attention, SwiGLU)
- Two-stage training: pretraining and finetuning
- Distributed training support with SLURM integration
- Automated checkpoint management and experiment tracking

## Installation
```bash
pip install -e .
```

## Preparing the data
We use a very efficient memory-mapped dataset. This allows us to load the data very quickly and to use it in a memory-efficient way.
The downside is that we subsample the long sequences to a fixed sequence length on the preprocessing step.


1) Download the data from the kaggle [IceCube competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/data)
ans save it to `<kagle data path>`.\
It is convenient to use the kaggle API to download the data (see the details [here](https://github.com/Kaggle/kaggle-api#api-credentials)):
```bash
kaggle competitions download -c icecube-neutrinos-in-deep-ice
```  
2) Adjust the paths in the `configs/prepare_datasets.yaml` file. 

3) Run the preprocessing script:
```bash
python scripts/prepare_memmaped_data.py --config_path configs/prepare_datasets.yaml
```

## Configuration
Example configuration files are provided in `configs/*.example.yaml`. To use them:

1. Copy the example config to create your actual config:
```bash
cp configs/polarbert.example.yaml configs/polarbert.yaml
cp configs/finetuning.example.yaml configs/finetuning.yaml
```

2. Update the paths and parameters in your config files:
- Set data directories
- Adjust model parameters if needed
- Configure training settings
- Set pretrained model path for finetuning

Notice that the dataloader has been tested with `num_workers: 1`. We recommend using this setting to avoid potential issues.

Note: Actual config files with paths are excluded from git to avoid sharing system-specific paths.

# Training
## Pretraining
Pretrain the model on masked DOM prediction and charge regression:

```bash
# Local development
python -m polarbert.pretraining \
    --config configs/polarbert.yaml \
    --model_type flash \  # Options: base, flash, swiglu
    --name my_experiment

# In SLURM job script
srun python -m polarbert.pretraining \
    --config configs/polarbert.yaml \
    --model_type flash \
    --job_id "${SLURM_JOB_ID}"
```

Available model architectures:

base: Standard Transformer
flash: Flash Attention Transformer (recommended)
swiglu: SwiGLU Activation Transformer
Finetuning
Finetune a pretrained model on directional prediction:

Update checkpoint path in configs/finetuning.yaml:
```yaml
pretrained:
  checkpoint_path: '/path/to/your/checkpoint.pth'
  model_type: 'flash'  # same as pretraining
  freeze_backbone: false  # whether to freeze pretrained weights
```
Start finetuning:
```bash
# Local development
python -m polarbert.finetuning \
    --config configs/finetuning.yaml \
    --name my_finetuning

# In SLURM job script
srun python -m polarbert.finetuning \
    --config configs/finetuning.yaml \
    --job_id "${SLURM_JOB_ID}"
```



Models and Checkpoints
Checkpoints are saved under:

Pretraining: checkpoints/<model_name>/
Finetuning: checkpoints/finetuned_<name>/
Each training run saves:

Best model based on validation loss
Last model state
Final model state