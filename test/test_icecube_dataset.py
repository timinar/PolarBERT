#!/usr/bin/env python3
"""
Unit tests for icecube_dataset.py (Kaggle dataset)

Tests the IceCubeDataset class and its Kaggle-specific functionality:
- Optional target_transform parameter (unlike Prometheus)
- Simple 2D target arrays (azimuth, zenith) instead of structured arrays
- Kaggle-specific target validation
- Shared utility functions from dataset_utils
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src directory to path for importing
import sys
sys.path.insert(0, 'src')

from polarbert.icecube_dataset import IceCubeDataset
from polarbert.dataset_utils import (
    _safe_dtype, 
    _safe_memmap, 
    _validate_memory_mapping,
    _validate_kaggle_targets
)


class TestValidateKaggleTargets:
    """Test the Kaggle-specific target validation function"""
    
    def test_valid_kaggle_targets(self):
        """Test validation of correct Kaggle target format"""
        # Create valid 2D target array (azimuth, zenith)
        targets = np.random.random((100, 2)).astype(np.float16) * 2 * np.pi
        
        # Should not raise any exception
        _validate_kaggle_targets(targets)
    
    def test_valid_kaggle_targets_float32(self):
        """Test that float32 targets are also accepted"""
        targets = np.random.random((50, 2)).astype(np.float32) * 2 * np.pi
        
        # Should not raise any exception (dtype validation removed)
        _validate_kaggle_targets(targets)
    
    def test_wrong_dimensions(self):
        """Test rejection of non-2D target arrays"""
        # 1D array
        targets_1d = np.random.random(100)
        with pytest.raises(ValueError, match="Expected 2D target array"):
            _validate_kaggle_targets(targets_1d)
        
        # 3D array
        targets_3d = np.random.random((10, 2, 3))
        with pytest.raises(ValueError, match="Expected 2D target array"):
            _validate_kaggle_targets(targets_3d)
    
    def test_wrong_number_of_columns(self):
        """Test rejection of arrays with wrong number of target columns"""
        # Only 1 column (missing zenith)
        targets_1col = np.random.random((100, 1))
        with pytest.raises(ValueError, match="Expected 2 target columns"):
            _validate_kaggle_targets(targets_1col)
        
        # Too many columns
        targets_3col = np.random.random((100, 3))
        with pytest.raises(ValueError, match="Expected 2 target columns"):
            _validate_kaggle_targets(targets_3col)
    
    def test_various_float_types_accepted(self):
        """Test that various floating-point types are accepted"""
        for dtype in [np.float16, np.float32, np.float64]:
            targets = np.random.random((10, 2)).astype(dtype)
            # Should not raise exception - dtype validation was removed
            _validate_kaggle_targets(targets)


class TestIceCubeDatasetAPI:
    """Test the IceCubeDataset API for Kaggle dataset"""
    
    def test_optional_target_transform(self):
        """Test that target_transform is optional for Kaggle dataset"""
        # Should work without target_transform
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(FileNotFoundError):  # Expected due to fake path
                IceCubeDataset('/fake/path', batch_size=32)
    
    def test_target_transform_as_keyword(self):
        """Test that target_transform works as keyword argument"""
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(FileNotFoundError):  # Expected due to fake path
                IceCubeDataset(
                    '/fake/path',
                    batch_size=32,
                    target_transform=lambda y, c: (y.astype(np.float32), c.astype(np.float32))
                )
    
    def test_target_transform_as_positional(self):
        """Test that target_transform works as positional argument"""
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(FileNotFoundError):  # Expected due to fake path
                IceCubeDataset(
                    '/fake/path', 
                    32, 
                    0, 
                    None, 
                    None, 
                    lambda y, c: (y, c)
                )
    
    def test_missing_required_files(self):
        """Test error handling for missing required files"""
        with patch('os.path.isfile') as mock_isfile:
            # Simulate missing l.npy file
            mock_isfile.side_effect = lambda path: 'l.npy' not in path
            
            with pytest.raises(FileNotFoundError, match="l.npy not found"):
                IceCubeDataset('/fake/path', batch_size=32)
    
    def test_invalid_memmap_properties_json(self):
        """Test error handling for invalid memmap properties"""
        with patch('os.path.isfile', return_value=True):
            # Mock invalid JSON
            with patch('builtins.open', mock_open(read_data='{"broken": json')):
                with pytest.raises(ValueError, match="Failed to load memmap properties"):
                    IceCubeDataset('/fake/path', batch_size=32)
    
    @patch('polarbert.dataset_utils._safe_memmap')
    @patch('polarbert.dataset_utils._validate_memory_mapping')
    def test_memmap_creation_error_handling(self, mock_validate, mock_safe_memmap):
        """Test error handling during memmap creation"""
        # Setup mocks
        mock_validate.return_value = None  
        mock_safe_memmap.side_effect = ValueError("Failed to create x memmap: test error")
        
        mock_props = {
            'x': {'shape': [100, 127], 'dtype': [['time', '<f2'], ['charge', '<f2'], ['aux', '<f2'], ['dom_id', '<u2']]},
            'l': {'shape': [100], 'dtype': 'int32'},
            'c': {'shape': [100], 'dtype': 'float32'}
        }
        
        with patch('os.path.isfile', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_props))):
                with pytest.raises(ValueError, match="Failed to create x memmap"):
                    IceCubeDataset('/fake/path', batch_size=32)


class TestIceCubeDatasetFunctionality:
    """Test IceCubeDataset functionality with mock data"""
    
    def create_mock_kaggle_dataset_files(self, tmp_dir):
        """Helper to create mock Kaggle dataset files"""
        # Create structured array data for features
        dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16),
            ('aux', np.float16),
            ('dom_id', np.uint16)
        ])
        
        # Create test data
        n_events, seq_length = 20, 127
        x_data = np.zeros((n_events, seq_length), dtype=dtype)
        
        # Fill with realistic test data
        for i in range(n_events):
            event_length = np.random.randint(60, seq_length)
            for j in range(event_length):
                x_data[i, j]['time'] = (j * 15 + np.random.random() * 10 - 5) / 100  # Normalized time
                x_data[i, j]['charge'] = np.random.exponential(1.5)  # Realistic charge distribution
                x_data[i, j]['aux'] = np.random.choice([-0.5, 0.5])  # Auxiliary signal
                x_data[i, j]['dom_id'] = np.uint16(np.random.randint(1, 5161))  # DOM ID 1-5160
        
        l_data = np.array([np.sum(x_data[i]['dom_id'] > 0) for i in range(n_events)], dtype=np.int32)
        c_data = np.random.exponential(2.0, n_events).astype(np.float32)  # Charge values
        y_data = np.random.random((n_events, 2)).astype(np.float16) * 2 * np.pi  # Azimuth, zenith angles
        
        # Save files using proper memmap format
        x_memmap = np.memmap(tmp_dir / 'x.npy', dtype=dtype, mode='w+', shape=x_data.shape)
        x_memmap[:] = x_data[:]
        del x_memmap
        
        l_memmap = np.memmap(tmp_dir / 'l.npy', dtype=l_data.dtype, mode='w+', shape=l_data.shape)
        l_memmap[:] = l_data[:]
        del l_memmap
        
        c_memmap = np.memmap(tmp_dir / 'c.npy', dtype=c_data.dtype, mode='w+', shape=c_data.shape)
        c_memmap[:] = c_data[:]
        del c_memmap
        
        y_memmap = np.memmap(tmp_dir / 'y.npy', dtype=y_data.dtype, mode='w+', shape=y_data.shape)
        y_memmap[:] = y_data[:]
        del y_memmap
        
        # Create memmap properties
        props = {
            'x': {'shape': list(x_data.shape), 'dtype': dtype.descr},
            'l': {'shape': list(l_data.shape), 'dtype': str(l_data.dtype)},
            'c': {'shape': list(c_data.shape), 'dtype': str(c_data.dtype)},
            'y': {'shape': list(y_data.shape), 'dtype': str(y_data.dtype)}
        }
        
        with open(tmp_dir / 'memmap_properties.json', 'w') as f:
            json.dump(props, f)
        
        return x_data, l_data, c_data, y_data
    
    def test_dataset_initialization_without_target_transform(self):
        """Test dataset initialization without target_transform"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            x_data, l_data, c_data, y_data = self.create_mock_kaggle_dataset_files(tmp_path)
            
            # Create dataset without target_transform
            dataset = IceCubeDataset(
                str(tmp_path),
                batch_size=4
            )
            
            # Test basic properties
            assert dataset.batch_size == 4
            assert dataset.SEQ_LENGTH == 127
            assert dataset.has_labels == True
            assert len(dataset) == 4  # (20 events // 4 batch_size) - 1
    
    def test_dataset_initialization_with_target_transform(self):
        """Test dataset initialization with target_transform"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            x_data, l_data, c_data, y_data = self.create_mock_kaggle_dataset_files(tmp_path)
            
            # Create dataset with target_transform
            def kaggle_target_transform(y, c):
                return y.astype(np.float32), c.astype(np.float32)
            
            dataset = IceCubeDataset(
                str(tmp_path),
                batch_size=3,
                target_transform=kaggle_target_transform
            )
            
            # Test basic properties
            assert dataset.batch_size == 3
            assert dataset.target_transform is not None
    
    def test_unpack_features_functionality(self):
        """Test the _unpack_features static method"""
        # Create test structured array
        dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16),
            ('aux', np.float16),
            ('dom_id', np.uint16)
        ])
        
        test_data = np.zeros((8, 15), dtype=dtype)
        test_data['time'] = np.random.random((8, 15)).astype(np.float16)
        test_data['charge'] = np.random.random((8, 15)).astype(np.float16)
        test_data['aux'] = np.random.choice([-0.5, 0.5], size=(8, 15)).astype(np.float16)
        test_data['dom_id'] = np.random.randint(1, 5161, size=(8, 15)).astype(np.uint16)
        
        result = IceCubeDataset._unpack_features(test_data)
        
        # Check return structure
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'dom_id' in result
        
        # Check shapes
        assert result['features'].shape == (8, 15, 3)  # time, charge, aux
        assert result['dom_id'].shape == (8, 15)
        
        # Check data consistency
        np.testing.assert_array_equal(result['features'][:, :, 0], test_data['time'])
        np.testing.assert_array_equal(result['features'][:, :, 1], test_data['charge'])
        np.testing.assert_array_equal(result['features'][:, :, 2], test_data['aux'])
        np.testing.assert_array_equal(result['dom_id'], test_data['dom_id'])
    
    def test_dataset_slicing(self):
        """Test dataset slicing functionality"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            x_data, l_data, c_data, y_data = self.create_mock_kaggle_dataset_files(tmp_path)
            
            dataset = IceCubeDataset(str(tmp_path), batch_size=3)
            
            # Test slicing
            sliced_dataset = dataset.slice(5, 15)
            assert sliced_dataset.start == 5
            assert sliced_dataset.end == 15
            
            # Test slice with None end
            full_slice = dataset.slice(0, None)
            assert full_slice.end == 20  # Total number of events
            
            # Test slice bounds checking
            bounded_slice = dataset.slice(18, 25)  # End beyond dataset size
            assert bounded_slice.end == 20  # Should be clamped to dataset size


class TestKaggleDatasetIteration:
    """Test dataset iteration and batch generation"""
    
    def create_small_test_dataset(self, tmp_dir, n_events=12):
        """Create a small test dataset for iteration testing"""
        dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16),
            ('aux', np.float16),
            ('dom_id', np.uint16)
        ])
        
        x_data = np.zeros((n_events, 127), dtype=dtype)
        
        # Create deterministic test data for reproducible tests
        np.random.seed(42)
        for i in range(n_events):
            for j in range(min(100, 127)):  # Fill first 100 positions
                x_data[i, j]['time'] = (i * 10 + j) / 1000
                x_data[i, j]['charge'] = 1.0 + i * 0.1 + j * 0.01
                x_data[i, j]['aux'] = 0.5 if (i + j) % 2 == 0 else -0.5
                x_data[i, j]['dom_id'] = np.uint16(1 + (i * 127 + j) % 5160)
        
        l_data = np.full(n_events, 100, dtype=np.int32)  # All events have 100 active positions
        c_data = np.arange(n_events, dtype=np.float32) + 1.0
        y_data = np.column_stack([
            np.linspace(0, 2*np.pi, n_events),  # Azimuth
            np.linspace(0, np.pi, n_events)     # Zenith
        ]).astype(np.float16)
        
        # Save using memmap format
        for name, data in [('x', x_data), ('l', l_data), ('c', c_data), ('y', y_data)]:
            memmap_file = np.memmap(tmp_dir / f'{name}.npy', dtype=data.dtype, mode='w+', shape=data.shape)
            memmap_file[:] = data[:]
            del memmap_file
        
        props = {
            'x': {'shape': list(x_data.shape), 'dtype': dtype.descr},
            'l': {'shape': list(l_data.shape), 'dtype': str(l_data.dtype)},
            'c': {'shape': list(c_data.shape), 'dtype': str(c_data.dtype)},
            'y': {'shape': list(y_data.shape), 'dtype': str(y_data.dtype)}
        }
        
        with open(tmp_dir / 'memmap_properties.json', 'w') as f:
            json.dump(props, f)
        
        return x_data, l_data, c_data, y_data
        
    def test_iteration_without_transform(self):
        """Test dataset iteration without transforms"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            x_data, l_data, c_data, y_data = self.create_small_test_dataset(tmp_path)
            
            dataset = IceCubeDataset(str(tmp_path), batch_size=3, start=0, end=9)
            
            batches = list(dataset)
            assert len(batches) == 2  # (9 events // 3 batch_size) - 1
            
            for (x, l), (y, c) in batches:
                # Check types and shapes
                assert isinstance(x, dict)
                assert 'features' in x and 'dom_id' in x
                assert x['features'].shape == (3, 127, 3)
                assert x['dom_id'].shape == (3, 127)
                assert l.shape == (3,)
                assert y.shape == (3, 2)  # Kaggle targets are 2D (azimuth, zenith)
                assert c.shape == (3,)
                
                # Check data types
                assert x['features'].dtype == np.float16
                assert x['dom_id'].dtype == np.uint16
                assert l.dtype == np.int32
                assert y.dtype == np.float16
                assert c.dtype == np.float32
    
    def test_iteration_with_transforms(self):
        """Test dataset iteration with transforms"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            x_data, l_data, c_data, y_data = self.create_small_test_dataset(tmp_path)
            
            def transform(x, l):
                return {
                    'features': x['features'].astype(np.float32),
                    'dom_id': x['dom_id'].astype(np.int64),
                }, l.astype(np.int64)
            
            def target_transform(y, c):
                return y.astype(np.float32), c.astype(np.float32)
            
            dataset = IceCubeDataset(
                str(tmp_path), 
                batch_size=4, 
                start=0, 
                end=8,
                transform=transform,
                target_transform=target_transform
            )
            
            batches = list(dataset)
            assert len(batches) == 1  # (8 events // 4 batch_size) - 1
            
            for (x, l), (y, c) in batches:
                # Check that transforms were applied
                assert x['features'].dtype == np.float32
                assert x['dom_id'].dtype == np.int64
                assert l.dtype == np.int64
                assert y.dtype == np.float32
                assert c.dtype == np.float32
    
    def test_no_labels_dataset(self):
        """Test dataset without labels (y.npy file missing)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create dataset without y.npy file
            dtype = np.dtype([
                ('time', np.float16),
                ('charge', np.float16),
                ('aux', np.float16),
                ('dom_id', np.uint16)
            ])
            
            n_events = 8
            x_data = np.zeros((n_events, 127), dtype=dtype)
            l_data = np.full(n_events, 50, dtype=np.int32)
            c_data = np.ones(n_events, dtype=np.float32)
            
            # Save only x, l, c (no y)
            for name, data in [('x', x_data), ('l', l_data), ('c', c_data)]:
                memmap_file = np.memmap(tmp_path / f'{name}.npy', dtype=data.dtype, mode='w+', shape=data.shape)
                memmap_file[:] = data[:]
                del memmap_file
            
            props = {
                'x': {'shape': list(x_data.shape), 'dtype': dtype.descr},
                'l': {'shape': list(l_data.shape), 'dtype': str(l_data.dtype)},
                'c': {'shape': list(c_data.shape), 'dtype': str(c_data.dtype)}
            }
            
            with open(tmp_path / 'memmap_properties.json', 'w') as f:
                json.dump(props, f)
            
            dataset = IceCubeDataset(str(tmp_path), batch_size=2)
            assert dataset.has_labels == False
            
            batches = list(dataset)
            for (x, l), targets in batches:
                assert targets is None  # No labels available


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_realistic_kaggle_data_flow(self):
        """Test complete data flow with realistic Kaggle-like data"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create realistic test data mimicking actual Kaggle dataset
            dtype = np.dtype([
                ('time', np.float16),
                ('charge', np.float16),
                ('aux', np.float16),
                ('dom_id', np.uint16)
            ])
            
            n_events, seq_length = 50, 127
            x_data = np.zeros((n_events, seq_length), dtype=dtype)
            
            # Create realistic event structures
            np.random.seed(123)  # For reproducible tests
            for i in range(n_events):
                # Realistic event length (most events don't fill full sequence)
                event_length = np.random.randint(30, 120)
                
                # Simulate time-ordered hits
                base_time = np.random.random() * 1000
                hit_times = np.sort(np.random.random(event_length) * 500) + base_time
                
                for j in range(event_length):
                    x_data[i, j]['time'] = (hit_times[j] - 1000) / 3000  # Normalized time
                    x_data[i, j]['charge'] = np.random.lognormal(0, 1)  # Log-normal charge distribution
                    x_data[i, j]['aux'] = 0.5 if np.random.random() > 0.8 else -0.5  # Mostly signal hits
                    x_data[i, j]['dom_id'] = np.uint16(np.random.randint(1, 5161))  # Valid DOM range
            
            # Realistic sequence lengths and charges
            l_data = np.array([np.sum(x_data[i]['dom_id'] > 0) for i in range(n_events)], dtype=np.int32)
            c_data = np.array([np.sum(x_data[i]['charge']) for i in range(n_events)], dtype=np.float32)
            
            # Realistic angular targets (isotropic distribution would be more complex, but uniform for testing)
            y_data = np.column_stack([
                np.random.random(n_events) * 2 * np.pi,  # Azimuth: 0 to 2π
                np.random.random(n_events) * np.pi       # Zenith: 0 to π
            ]).astype(np.float16)
            
            # Save using proper memmap format
            for name, data in [('x', x_data), ('l', l_data), ('c', c_data), ('y', y_data)]:
                memmap_file = np.memmap(tmp_path / f'{name}.npy', dtype=data.dtype, mode='w+', shape=data.shape)
                memmap_file[:] = data[:]
                del memmap_file
            
            props = {
                'x': {'shape': list(x_data.shape), 'dtype': dtype.descr},
                'l': {'shape': list(l_data.shape), 'dtype': str(l_data.dtype)},
                'c': {'shape': list(c_data.shape), 'dtype': str(c_data.dtype)},
                'y': {'shape': list(y_data.shape), 'dtype': str(y_data.dtype)}
            }
            
            with open(tmp_path / 'memmap_properties.json', 'w') as f:
                json.dump(props, f)
            
            # Test complete pipeline
            dataset = IceCubeDataset(
                str(tmp_path),
                batch_size=8,
                transform=lambda x, l: ({
                    'features': x['features'].astype(np.float32),
                    'dom_id': x['dom_id'].astype(np.int64)
                }, l.astype(np.int64)),
                target_transform=lambda y, c: (y.astype(np.float32), c.astype(np.float32))
            )
            
            # Validate complete data flow
            batch_count = 0
            total_events_processed = 0
            
            for (x, l), (y, c) in dataset:
                batch_count += 1
                batch_size = x['features'].shape[0]
                total_events_processed += batch_size
                
                # Validate batch structure
                assert isinstance(x, dict)
                assert x['features'].shape == (batch_size, 127, 3)
                assert x['dom_id'].shape == (batch_size, 127)
                assert l.shape == (batch_size,)
                assert y.shape == (batch_size, 2)  # Kaggle: azimuth, zenith
                assert c.shape == (batch_size,)
                
                # Validate data types after transforms
                assert x['features'].dtype == np.float32
                assert x['dom_id'].dtype == np.int64
                assert l.dtype == np.int64
                assert y.dtype == np.float32
                assert c.dtype == np.float32
                
                # Check data ranges
                assert np.all(x['dom_id'][x['dom_id'] > 0] <= 5160)
                assert np.all(x['dom_id'] >= 0)
                assert np.all(y[:, 0] >= 0) and np.all(y[:, 0] <= 2*np.pi)  # Azimuth
                assert np.all(y[:, 1] >= 0) and np.all(y[:, 1] <= np.pi)    # Zenith
                
                # Validate that auxiliary values are reasonable
                aux_values = x['features'][:, :, 2]
                unique_aux = np.unique(aux_values)
                # Should mostly be -0.5 and 0.5, with 0.0 for padding positions
                expected_aux_values = [-0.5, 0.0, 0.5]
                assert np.all(np.isin(unique_aux, expected_aux_values))
            
            assert batch_count > 0
            assert total_events_processed <= n_events
            
            print(f"✓ Processed {batch_count} batches with {total_events_processed} total events")


# Additional utility test that applies to both datasets
class TestSharedUtilities:
    """Test shared utility functions from dataset_utils"""
    
    def test_safe_dtype_with_kaggle_format(self):
        """Test _safe_dtype with Kaggle-specific dtype formats"""
        # Test structured dtype for Kaggle features
        kaggle_dtype_spec = [["time", "<f2"], ["charge", "<f2"], ["aux", "<f2"], ["dom_id", "<u2"]]
        result = _safe_dtype(kaggle_dtype_spec)
        expected = np.dtype([("time", "<f2"), ("charge", "<f2"), ("aux", "<f2"), ("dom_id", "<u2")])
        assert result == expected
        
        # Test simple dtype for Kaggle targets
        target_dtype_spec = "float16"
        result = _safe_dtype(target_dtype_spec)
        expected = np.dtype("float16")
        assert result == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 