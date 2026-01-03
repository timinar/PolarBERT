import numpy as np
from typing import Any, Callable, Optional
from torch.utils.data import DataLoader

def default_transform(x: dict[str, np.ndarray], l: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = {
        'features': x['features'].astype(np.float32),
        'dom_id': x['dom_id'].astype(np.int64),
    }
    return x, l.astype(np.int64)

def add_random_time_offset(std: float) -> Callable:
    def _add_random_time_offset(x: dict[str, np.ndarray], l: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        x, l = default_transform(x, l)
        x['features'] = x['features'].copy()
        time_offset = np.random.normal(0, std, (x['features'].shape[0], 1))
        x['features'][:,:,0] += time_offset
        return x, l
    return _add_random_time_offset

def default_target_transform(y: np.ndarray, c: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray]:
    return None, c.astype(np.float32)

def get_dataloaders(
        config: dict[str, Any],
        dataset_type: str,
        transform=default_transform,
        target_transform=default_target_transform,
        override_batch_size: Optional[int]=None,
    ) -> tuple[DataLoader, DataLoader]:

    if dataset_type == 'prometheus':
        from polarbert.prometheus_dataset import IceCubeDataset
    elif dataset_type == 'kaggle':
        from polarbert.icecube_dataset import IceCubeDataset
    else:
        assert False, f"Unknown dataset type: {dataset_type}"

    # Optional shuffle seed for reproducible data ordering
    shuffle_seed = config['data'].get('shuffle_seed', None)

    full_dataset = IceCubeDataset(
        data_dir=config['data']['train_dir'],
        batch_size=override_batch_size if override_batch_size is not None else config['training']['per_device_batch_size'],
        transform=transform,
        target_transform=target_transform,
        shuffle_seed=shuffle_seed
    )
    train_events = config['data'].get('train_events', None)
    val_events = config['data'].get('val_events', None)

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
            data_dir=config['data']['val_dir'], 
            batch_size=override_batch_size if override_batch_size is not None else config['training']['per_device_batch_size'],
            transform=transform,
            target_transform=target_transform
        )
        val_dataset = full_val_dataset.slice(0, val_events)
    else:
        assert False
    
    loader_kwargs = {
        'batch_size': None,
        'num_workers': config['data']['num_workers'],
        'pin_memory': config['data']['pin_memory'],
        'persistent_workers': config['data']['persistent_workers']
    }
    
    return (
        DataLoader(train_dataset, **loader_kwargs),
        DataLoader(val_dataset, **loader_kwargs)
    )