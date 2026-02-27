import numpy as np
import json
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

def _validate_kaggle_targets(y_memmap: np.memmap) -> None:
    """
    Validate that the Kaggle target array has the expected format.
    
    Args:
        y_memmap: Memory-mapped array containing Kaggle targets (non-structured)
        
    Raises:
        ValueError: If the target array has incorrect shape or type
    """
    # For Kaggle, targets should be a simple 2D array with neutrino direction (azimuth, zenith)
    if len(y_memmap.shape) != 2:
        raise ValueError(f"Expected 2D target array for Kaggle dataset, got shape: {y_memmap.shape}")
    
    if y_memmap.shape[1] != 2:
        raise ValueError(f"Expected 2 target columns (azimuth, zenith) for Kaggle dataset, got: {y_memmap.shape[1]}")
    
    # Note: We don't validate dtype since angles can be accurately represented 
    # in various floating-point formats (float16, float32, etc.) and this is 
    # a user choice when generating the memory-mapped files 