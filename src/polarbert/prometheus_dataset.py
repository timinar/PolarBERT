import numpy as np
import torch
from torch.utils.data import IterableDataset
import json
import copy
import os

class IceCubeDataset(IterableDataset):
    """
    dataset = IceCubeDataset(
        '/groups/pheno/inar/icecube_kaggle/memmaped_1M_127', 
        batch_size=2048,
        transform=lambda x: x.astype(np.float32), 
        target_transform=lambda x: x.astype(np.float32)
    )
    
    each batch contrains (x, l), (y, c)
    x - features, l -  event length (len(selected_idx))
    y - labels, c - charge

    features x contrain (see prepare_memmaped_data.py):
        0 - time ((time[selected_idx] - 1e4) / 3e4)
        1 - charge (np.log10(charge[selected_idx]) / 3.0)
        2 - auxiliary (auxiliary[selected_idx] - 0.5)
        3 - dom_id (sensor_id[selected_idx] + 1)
        all padding values are 0
    """
    def __init__(self, data_dir: str, batch_size: int, start=0, end=None, transform=None, target_transform=None):
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform
        
        for filename in ['x.npy', 'l.npy', 'c.npy', 'memmap_properties.json']:
            if not os.path.isfile(os.path.join(data_dir, filename)):
                raise FileNotFoundError(f'{filename} not found in {data_dir}')
        
        self.has_labels = os.path.isfile(os.path.join(data_dir, 'y.npy'))
        
        with open(os.path.join(data_dir, 'memmap_properties.json'), 'r') as f:
            memmap_props = json.load(f)
        
        self.x = np.memmap(
            os.path.join(data_dir, 'x.npy'), mode='r',
            shape=tuple(memmap_props['x']['shape']),
            dtype=memmap_props['x']['dtype'],
        )
        self.l = np.memmap(
            os.path.join(data_dir, 'l.npy'), mode='r',
            shape=tuple(memmap_props['l']['shape']),
            dtype=memmap_props['l']['dtype'],
        )
        self.c = np.memmap(
            os.path.join(data_dir, 'c.npy'), mode='r',
            shape=tuple(memmap_props['c']['shape']),
            dtype=memmap_props['c']['dtype'],
        )
        
        if self.has_labels:
            self.y = np.memmap(
                os.path.join(data_dir, 'y.npy'), mode='r',
                shape=tuple(memmap_props['y']['shape']),
                dtype=[tuple(dt) for dt in memmap_props['y']['dtype']],
            )
            self.labels = [field_name for field_name, _ in memmap_props['y']['dtype']]
        else:
            self.y = None
            self.labels = None
            
        if end is None:
            end = self.x.shape[0]
        assert(end > start)
        self.start = start
        self.end = end
        self.SEQ_LENGTH = self.x.shape[1]
        self.N_FEATURES = self.x.shape[2]

    def __len__(self):
        return (self.end - self.start) // self.batch_size - 1

    def __iter__(self):
        def generator():
            Nevents = self.x.shape[0]
            rand_int = np.random.randint(0, self.batch_size)
            batch_start_indices = np.arange(Nevents)[
                self.start + rand_int : self.end - self.batch_size + 1 : self.batch_size]
            np.random.shuffle(batch_start_indices)
            for idx in batch_start_indices:
                assert(idx >= self.start)
                assert(idx + self.batch_size <= self.end)
                x = self.x[idx:idx+self.batch_size,:,:]
                l = self.l[idx:idx+self.batch_size]
                
                if self.transform:
                    x, l = self.transform(x, l)
                
                if self.has_labels:
                    y = self.y[idx:idx+self.batch_size]  # remove second dimension
                    c = self.c[idx:idx+self.batch_size]
                    if self.target_transform:
                        y, c = self.target_transform(y, c)
                    yield (x, l), (y, c)
                else:
                    yield (x, l), None
        return generator()
    
    def slice(self, start, end):
        if end is None or end > self.x.shape[0]:
            end = self.x.shape[0]
        assert(end > start)
        assert(start >= 0)
        slc = copy.copy(self) # Shallow copy
        slc.start = start
        slc.end = end
        return slc


from torch.utils.data import DataLoader

def train_validation_loaders(dataset, train_ratio=0.8, pin_memory=False, persistent_workers=True):
    """
    usage example:

    train_dataloader, val_dataloader = train_validation_loaders(
    dataset,
    train_ratio = TRAIN_RATIO,
    pin_memory=True,
    )
    """

    #val_size = len(dataset) - train_size
    total_batches = len(dataset)
    train_batches = int(train_ratio * total_batches)
    val_batches = total_batches - train_batches
    train_size = train_batches * dataset.batch_size
    val_size = val_batches * dataset.batch_size
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataset = dataset.slice(0, train_size)
    val_dataset = dataset.slice(train_size, train_size+val_size)
    # Note: the split is not random anymore
    train_dataloader = DataLoader(
        train_dataset, batch_size=None, num_workers=1,
        pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_dataloader = DataLoader(
        val_dataset, batch_size=None, num_workers=1,
        pin_memory=pin_memory, persistent_workers=persistent_workers)
    return train_dataloader, val_dataloader