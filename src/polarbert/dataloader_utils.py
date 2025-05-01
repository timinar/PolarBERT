import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Any, Optional, Callable

from polarbert.config import PolarBertConfig


def target_transform_prometheus(y, c):
    y = np.vstack([y['initial_state_azimuth'].astype(np.float32), y['initial_state_zenith'].astype(np.float32)]).T
    return y, c.astype(np.float32)


def target_transform_kaggle(y, c):
    return y.astype(np.float32), c.astype(np.float32)

def default_target_transform(y, c):
    return None, c.astype(np.float32)

def default_transform(x, l):
    # Ensure a writable copy is returned
    return x.astype(np.float32).copy(), l.astype(np.int32).copy()

def get_dataloaders(
        config: PolarBertConfig,
        dataset_type: str,
        transform=default_transform,
        target_transform=default_target_transform,
        override_batch_size: Optional[int]=None,
    ) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation dataloaders using a PolarBertConfig object."""

    if dataset_type == 'prometheus':
        from polarbert.prometheus_dataset import IceCubeDataset
        data_dir=config.data.prometheus_dir
    elif dataset_type == 'kaggle':
        from polarbert.icecube_dataset import IceCubeDataset
        data_dir=config.data.train_dir
    else:
        assert False, f"Unknown dataset type: {dataset_type}"
    
    full_dataset = IceCubeDataset(
        data_dir=data_dir,
        batch_size=override_batch_size if override_batch_size is not None else config.data.max_per_device_batch_size,
        transform=transform,
        target_transform=target_transform
    )
    train_events = config.data.train_events
    val_events = config.data.val_events

    if dataset_type == 'prometheus':
        if val_events is None:
            raise ValueError("Number of validation events must be specified for the Prometheus dataset")
        val_dataset = full_dataset.slice(0, val_events)
        train_dataset = full_dataset.slice(val_events, val_events + train_events) if train_events else full_dataset.slice(val_events, None)
    elif dataset_type == 'kaggle':
        # Training dataset
        train_dataset = full_dataset.slice(0, train_events)
        # Validation dataset with optional subsampling
        full_val_dataset = IceCubeDataset(
            data_dir=config.data.val_dir, 
            batch_size=override_batch_size if override_batch_size is not None else config.data.max_per_device_batch_size,
            transform=transform,
            target_transform=target_transform
        )
        val_dataset = full_val_dataset.slice(0, val_events)
    else:
        assert False
    
    loader_kwargs = {
        'batch_size': None,
        'num_workers': config.data.num_workers,
        'pin_memory': config.data.pin_memory,
        'persistent_workers': config.data.persistent_workers
    }
    
    return (
        DataLoader(train_dataset, **loader_kwargs),
        DataLoader(val_dataset, **loader_kwargs)
    )