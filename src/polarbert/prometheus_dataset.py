import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from torch.utils.data import IterableDataset
import json
import copy
import os
from typing import Dict

def _safe_dtype(dtype_spec):
    """
    Safely create a numpy dtype from JSON-serialized dtype specification.
    
    Handles both simple dtypes (strings) and structured dtypes (lists of [name, type] pairs).
    """
    if isinstance(dtype_spec, str):
        # Simple dtype like 'float32', 'int32', etc.
        return np.dtype(dtype_spec)
    elif isinstance(dtype_spec, list):
        # Check if it's JSON format (list of lists) and non-empty
        if len(dtype_spec) > 0 and isinstance(dtype_spec[0], list):
            # JSON format: [["name", "type"], ...] - convert to tuples
            return np.dtype([tuple(field) for field in dtype_spec])
        else:
            # Already in correct format, empty list, or other valid formats
            return np.dtype(dtype_spec)
    else:
        # Fallback - let numpy handle it
        return np.dtype(dtype_spec)

def _safe_memmap(file_path: str, shape: tuple, dtype, mode: str = 'r', name: str = 'memmap') -> np.memmap:
    """
    Safely create a memory-mapped array with error handling.
    
    Args:
        file_path: Path to the .npy file
        shape: Shape tuple for the memmap
        dtype: Data type for the memmap
        mode: File access mode (default: 'r')
        name: Name for error reporting
        
    Returns:
        np.memmap: The created memory-mapped array
        
    Raises:
        ValueError: If memmap creation fails
    """
    try:
        return np.memmap(file_path, mode=mode, shape=shape, dtype=dtype)
    except (ValueError, TypeError, OSError) as e:
        raise ValueError(f"Failed to create {name} memmap: {e}")

def _validate_memory_mapping(x_memmap: np.memmap) -> None:
    """
    Validate that the memory-mapped structured array has the expected format.
    
    Args:
        x_memmap: Memory-mapped structured array containing features and DOM IDs
        
    Raises:
        ValueError: If required fields are missing or have incorrect types
        TypeError: If the array is not a structured array
    """
    # Check if it's a structured array
    if not hasattr(x_memmap.dtype, 'names') or x_memmap.dtype.names is None:
        raise TypeError(f"Expected structured array, got dtype: {x_memmap.dtype}")
    
    # Define expected fields and their types
    expected_fields = {
        'time': np.float16,
        'charge': np.float16, 
        'aux': np.float16,
        'dom_id': np.uint16
    }
    
    # Check for missing fields
    missing_fields = [field for field in expected_fields.keys() if field not in x_memmap.dtype.names]
    if missing_fields:
        available_fields = list(x_memmap.dtype.names)
        raise ValueError(
            f"Missing required fields in structured array: {missing_fields}. "
            f"Available fields: {available_fields}"
        )
    
    # Check field types
    type_mismatches = []
    for field_name, expected_type in expected_fields.items():
        actual_type = x_memmap.dtype.fields[field_name][0]
        if actual_type != expected_type:
            type_mismatches.append(f"'{field_name}': expected {expected_type}, got {actual_type}")
    
    if type_mismatches:
        raise ValueError(f"Field type mismatches in structured array: {'; '.join(type_mismatches)}")
    
    # Check for unexpected extra fields (warning, not error)
    extra_fields = [field for field in x_memmap.dtype.names if field not in expected_fields]
    if extra_fields:
        import logging
        logging.warning(f"Unexpected extra fields in structured array: {extra_fields}")

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
        if end is None or end > self.x.shape[0]:
            end = self.x.shape[0]
        assert(end > start)
        assert(start >= 0)
        slc = copy.copy(self) # Shallow copy
        slc.start = start
        slc.end = end
        return slc
