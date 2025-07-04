import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from torch.utils.data import IterableDataset
import json
import copy
import os
from typing import Dict

from polarbert.dataset_utils import _safe_dtype, _safe_memmap, _validate_memory_mapping

class IceCubeDataset(IterableDataset):
    """
    Prometheus dataset for memory-mapped structured arrays.
    
    Returns batches where x is a dictionary with 'features' and 'dom_id' keys:
    - x['features']: (batch_size, seq_length, 3) array of [time, charge, aux]
    - x['dom_id']: (batch_size, seq_length) array of DOM IDs (uint16)
    
    Example:
        dataset = IceCubeDataset(
            '/path/to/memmapped_data', 
            batch_size=2048,
            transform=None, 
            target_transform=lambda y, c: (y.astype(np.float32), c.astype(np.float32))
        )
        
        for (x, l), (y, c) in dataset:
            features = x['features']  # Shape: (batch_size, seq_length, 3)
            dom_ids = x['dom_id']     # Shape: (batch_size, seq_length)
    """
    def __init__(self, data_dir: str, batch_size: int, start=0, end=None, transform=None, *, target_transform):
        # target_transform is now a mandatory keyword-only argument
        
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform
        
        for filename in ['x.npy', 'l.npy', 'c.npy', 'memmap_properties.json']:
            if not os.path.isfile(os.path.join(data_dir, filename)):
                raise FileNotFoundError(f'{filename} not found in {data_dir}')
        
        self.has_labels = os.path.isfile(os.path.join(data_dir, 'y.npy'))
        
        try:
            with open(os.path.join(data_dir, 'memmap_properties.json'), 'r') as f:
                memmap_props = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load memmap properties: {e}")
        
        self.x = _safe_memmap(
            os.path.join(data_dir, 'x.npy'),
            shape=tuple(memmap_props['x']['shape']),
            dtype=_safe_dtype(memmap_props['x']['dtype']),
            name='x'
        )
        
        # Validate the memory-mapped structured array format
        _validate_memory_mapping(self.x)
        
        self.l = _safe_memmap(
            os.path.join(data_dir, 'l.npy'),
            shape=tuple(memmap_props['l']['shape']),
            dtype=_safe_dtype(memmap_props['l']['dtype']),
            name='l'
        )
        
        self.c = _safe_memmap(
            os.path.join(data_dir, 'c.npy'),
            shape=tuple(memmap_props['c']['shape']),
            dtype=_safe_dtype(memmap_props['c']['dtype']),
            name='c'
        )
        
        if self.has_labels:
            self.y = _safe_memmap(
                os.path.join(data_dir, 'y.npy'),
                shape=tuple(memmap_props['y']['shape']),
                dtype=_safe_dtype(memmap_props['y']['dtype']),
                name='y'
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

    def __len__(self):
        return (self.end - self.start) // self.batch_size - 1
    
    @property
    def num_events(self):
        """Return the number of events (not batches) in this dataset."""
        return self.end - self.start
    
    @staticmethod
    def _unpack_features(x: np.ndarray) -> Dict[str, np.ndarray]:
        # Note: Field validation is performed once during initialization by _validate_memory_mapping()
        return {
            'features': structured_to_unstructured(x[['time', 'charge', 'aux']], dtype=np.float16, casting='safe'),
            'dom_id': x['dom_id'],
        }
    
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
                x = self._unpack_features(self.x[idx:idx+self.batch_size,:])
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
        # Convert relative indices to absolute indices
        abs_start = self.start + start
        
        if end is None:
            abs_end = self.end  # Use current slice's end
        else:
            abs_end = self.start + end
        
        # Ensure we don't go beyond the current slice or original dataset
        if abs_end > self.end:
            abs_end = self.end
        if abs_end > self.x.shape[0]:
            abs_end = self.x.shape[0]
            
        assert(abs_end > abs_start)
        assert(abs_start >= 0)
        
        slc = copy.copy(self) # Shallow copy
        slc.start = abs_start
        slc.end = abs_end
        return slc
