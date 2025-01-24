# PolarBERT
Foundation model for IceCube neutrino telescope.

# Installation
```bash
pip install -e .
```

# Preparing the data
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