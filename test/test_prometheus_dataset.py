#!/usr/bin/env python3
"""
Unit tests for prometheus_dataset.py

Tests the IceCubeDataset class and its helper functions to ensure:
- Proper API signature enforcement
- Field validation and consistency
- Error handling for memmap creation
- Structured array processing
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

from polarbert.prometheus_dataset import (
    IceCubeDataset, 
    _safe_dtype, 
    _safe_memmap, 
    _validate_memory_mapping
)


class TestSafeDtype:
    """Test the _safe_dtype helper function"""
    
    def test_simple_dtype_string(self):
        """Test simple dtype strings"""
        result = _safe_dtype('float32')
        expected = np.dtype('float32')
        assert result == expected
    
    def test_simple_dtype_int(self):
        """Test simple integer dtype"""
        result = _safe_dtype('int64')
        expected = np.dtype('int64')
        assert result == expected
    
    def test_structured_dtype_json_format(self):
        """Test structured dtype in JSON format (list of lists)"""
        json_dtype = [["time", "<f2"], ["charge", "<f2"], ["dom_id", "<u2"]]
        result = _safe_dtype(json_dtype)
        expected = np.dtype([("time", "<f2"), ("charge", "<f2"), ("dom_id", "<u2")])
        assert result == expected
    
    def test_structured_dtype_tuple_format(self):
        """Test structured dtype in tuple format"""
        tuple_dtype = [("time", np.float16), ("charge", np.float16)]
        result = _safe_dtype(tuple_dtype)
        expected = np.dtype(tuple_dtype)
        assert result == expected
    
    def test_empty_list(self):
        """Test empty list handling"""
        result = _safe_dtype([])
        expected = np.dtype([])
        assert result == expected
    
    def test_fallback_handling(self):
        """Test fallback for other types"""
        # This should work with numpy's built-in dtype handling
        result = _safe_dtype(np.float32)
        expected = np.dtype(np.float32)
        assert result == expected


class TestSafeMemmap:
    """Test the _safe_memmap helper function"""
    
    def test_successful_memmap_creation(self):
        """Test successful memmap creation"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            # Create a test array and save it properly for memmap usage
            test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            
            # Create a proper memmap file (not using np.save which adds headers)
            temp_memmap = np.memmap(tmp.name, dtype=test_data.dtype, mode='w+', shape=test_data.shape)
            temp_memmap[:] = test_data[:]
            del temp_memmap  # Close the memmap
            
            try:
                # Test _safe_memmap
                result = _safe_memmap(
                    tmp.name,
                    shape=test_data.shape,
                    dtype=test_data.dtype,
                    name='test'
                )
                
                assert isinstance(result, np.memmap)
                assert result.shape == test_data.shape
                assert result.dtype == test_data.dtype
                np.testing.assert_array_equal(result, test_data)
                
            finally:
                os.unlink(tmp.name)
    
    def test_invalid_file_path(self):
        """Test error handling for invalid file paths"""
        with pytest.raises(ValueError, match="Failed to create test memmap"):
            _safe_memmap(
                '/nonexistent/path.npy',
                shape=(10, 10),
                dtype=np.float32,
                name='test'
            )
    
    def test_invalid_shape(self):
        """Test error handling for invalid shapes"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            # Create a small test file using memmap
            test_data = np.array([1, 2, 3], dtype=np.int32)
            temp_memmap = np.memmap(tmp.name, dtype=test_data.dtype, mode='w+', shape=test_data.shape)
            temp_memmap[:] = test_data[:]
            del temp_memmap  # Close the memmap
            
            try:
                # Try to create memmap with wrong shape
                with pytest.raises(ValueError, match="Failed to create shape_test memmap"):
                    _safe_memmap(
                        tmp.name,
                        shape=(100, 100),  # Much larger than actual data
                        dtype=np.float32,
                        name='shape_test'
                    )
            finally:
                os.unlink(tmp.name)
    
    def test_custom_mode(self):
        """Test memmap creation with custom mode"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            test_data = np.array([1, 2, 3], dtype=np.int32)
            # Create proper memmap file
            temp_memmap = np.memmap(tmp.name, dtype=test_data.dtype, mode='w+', shape=test_data.shape)
            temp_memmap[:] = test_data[:]
            del temp_memmap  # Close the memmap
            
            try:
                result = _safe_memmap(
                    tmp.name,
                    shape=test_data.shape,
                    dtype=test_data.dtype,
                    mode='r',  # Explicit read-only mode
                    name='readonly_test'
                )
                
                assert isinstance(result, np.memmap)
                assert result.mode == 'r'
                
            finally:
                os.unlink(tmp.name)


class TestValidateMemoryMapping:
    """Test the _validate_memory_mapping function"""
    
    def test_valid_structured_array(self):
        """Test validation of correct structured array"""
        correct_dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16), 
            ('aux', np.float16),
            ('dom_id', np.uint16)
        ])
        
        mock_array = np.zeros(10, dtype=correct_dtype)
        
        # Should not raise any exception
        _validate_memory_mapping(mock_array)
    
    def test_non_structured_array(self):
        """Test rejection of non-structured arrays"""
        regular_array = np.zeros((10, 4), dtype=np.float32)
        
        with pytest.raises(TypeError, match="Expected structured array"):
            _validate_memory_mapping(regular_array)
    
    def test_missing_required_fields(self):
        """Test rejection of arrays missing required fields"""
        incomplete_dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16),
            # Missing 'aux' and 'dom_id'
        ])
        
        mock_array = np.zeros(10, dtype=incomplete_dtype)
        
        with pytest.raises(ValueError, match="Missing required fields"):
            _validate_memory_mapping(mock_array)
    
    def test_wrong_field_types(self):
        """Test rejection of arrays with wrong field types"""
        wrong_dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float32),  # Should be float16
            ('aux', np.float16),
            ('dom_id', np.int32)     # Should be uint16
        ])
        
        mock_array = np.zeros(10, dtype=wrong_dtype)
        
        with pytest.raises(ValueError, match="Field type mismatches"):
            _validate_memory_mapping(mock_array)
    
    def test_extra_fields_warning(self):
        """Test that extra fields don't cause errors (warning is logged but not tested)"""
        extra_fields_dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16), 
            ('aux', np.float16),
            ('dom_id', np.uint16),
            ('extra_field', np.int8)  # Extra field
        ])
        
        mock_array = np.zeros(10, dtype=extra_fields_dtype)
        
        # Should not raise exception even with extra fields
        # Note: We're not testing the warning itself, just that it doesn't break
        _validate_memory_mapping(mock_array)


class TestIceCubeDatasetAPI:
    """Test the IceCubeDataset API and initialization"""
    
    def test_mandatory_target_transform_keyword_only(self):
        """Test that target_transform must be provided as keyword argument"""
        # Should work with keyword argument
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(FileNotFoundError):  # Expected due to fake path
                IceCubeDataset(
                    '/fake/path',
                    batch_size=32,
                    target_transform=lambda y, c: (y, c)
                )
    
    def test_missing_target_transform_raises_error(self):
        """Test that missing target_transform raises TypeError"""
        with pytest.raises(TypeError, match="target_transform"):
            IceCubeDataset('/fake/path', batch_size=32)
    
    def test_positional_target_transform_raises_error(self):
        """Test that positional target_transform raises TypeError"""
        with pytest.raises(TypeError):
            # Try to pass target_transform as positional argument
            IceCubeDataset('/fake/path', 32, 0, None, None, lambda y, c: (y, c))
    
    def test_missing_required_files(self):
        """Test error handling for missing required files"""
        with patch('os.path.isfile') as mock_isfile:
            # Simulate missing x.npy file
            mock_isfile.side_effect = lambda path: 'x.npy' not in path
            
            with pytest.raises(FileNotFoundError, match="x.npy not found"):
                IceCubeDataset(
                    '/fake/path',
                    batch_size=32,
                    target_transform=lambda y, c: (y, c)
                )
    
    def test_invalid_memmap_properties_json(self):
        """Test error handling for invalid memmap properties"""
        with patch('os.path.isfile', return_value=True):
            # Mock invalid JSON
            with patch('builtins.open', mock_open(read_data='invalid json')):
                with pytest.raises(ValueError, match="Failed to load memmap properties"):
                    IceCubeDataset(
                        '/fake/path',
                        batch_size=32,
                        target_transform=lambda y, c: (y, c)
                    )
    
    @patch('polarbert.prometheus_dataset._safe_memmap')
    @patch('polarbert.prometheus_dataset._validate_memory_mapping')
    def test_memmap_creation_error_handling(self, mock_validate, mock_safe_memmap):
        """Test error handling during memmap creation"""
        # Setup mocks
        mock_validate.return_value = None
        mock_safe_memmap.side_effect = ValueError("Memmap creation failed")
        
        mock_props = {
            'x': {'shape': [100, 127], 'dtype': 'float32'},
            'l': {'shape': [100], 'dtype': 'int32'},
            'c': {'shape': [100], 'dtype': 'float32'}
        }
        
        with patch('os.path.isfile', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_props))):
                with pytest.raises(ValueError, match="Memmap creation failed"):
                    IceCubeDataset(
                        '/fake/path',
                        batch_size=32,
                        target_transform=lambda y, c: (y, c)
                    )


class TestIceCubeDatasetFunctionality:
    """Test IceCubeDataset functionality with mock data"""
    
    def create_mock_dataset_files(self, tmp_dir):
        """Helper to create mock dataset files"""
        # Create structured array data
        dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16),
            ('aux', np.float16),
            ('dom_id', np.uint16)
        ])
        
        # Create test data
        n_events, seq_length = 10, 127
        x_data = np.zeros((n_events, seq_length), dtype=dtype)
        
        # Fill with test data
        for i in range(n_events):
            for j in range(seq_length):
                x_data[i, j]['time'] = i * 100 + j
                x_data[i, j]['charge'] = np.random.random() * 10
                x_data[i, j]['aux'] = np.random.choice([0, 1])
                x_data[i, j]['dom_id'] = np.uint16(np.random.randint(1, 5160 + 1))  # Valid DOM ID range 1-5160
        
        l_data = np.random.randint(50, seq_length, size=n_events).astype(np.int32)
        c_data = np.random.random(n_events).astype(np.float32)
        
        # Save data files
        np.save(tmp_dir / 'x.npy', x_data)
        np.save(tmp_dir / 'l.npy', l_data)
        np.save(tmp_dir / 'c.npy', c_data)
        
        # Create memmap properties
        props = {
            'x': {'shape': list(x_data.shape), 'dtype': dtype.descr},
            'l': {'shape': list(l_data.shape), 'dtype': str(l_data.dtype)},
            'c': {'shape': list(c_data.shape), 'dtype': str(c_data.dtype)}
        }
        
        with open(tmp_dir / 'memmap_properties.json', 'w') as f:
            json.dump(props, f)
        
        return x_data, l_data, c_data
    
    def test_dataset_initialization_and_basic_functionality(self):
        """Test dataset initialization with real files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            x_data, l_data, c_data = self.create_mock_dataset_files(tmp_path)
            
            # Create dataset
            dataset = IceCubeDataset(
                str(tmp_path),
                batch_size=3,
                target_transform=lambda y, c: (y, c) if y is not None else None
            )
            
            # Test basic properties
            assert dataset.batch_size == 3
            assert dataset.SEQ_LENGTH == 127
            assert len(dataset) == 2  # (10 events // 3 batch_size) - 1
    
    def test_unpack_features_functionality(self):
        """Test the _unpack_features static method"""
        # Create test structured array
        dtype = np.dtype([
            ('time', np.float16),
            ('charge', np.float16),
            ('aux', np.float16),
            ('dom_id', np.uint16)
        ])
        
        test_data = np.zeros((5, 10), dtype=dtype)
        test_data['time'] = np.random.random((5, 10)).astype(np.float16)
        test_data['charge'] = np.random.random((5, 10)).astype(np.float16)
        test_data['aux'] = np.random.choice([0, 1], size=(5, 10)).astype(np.float16)
        test_data['dom_id'] = np.random.randint(1, 5160 + 1, size=(5, 10)).astype(np.uint16)
        
        result = IceCubeDataset._unpack_features(test_data)
        
        # Check return structure
        assert isinstance(result, dict)
        assert 'features' in result
        assert 'dom_id' in result
        
        # Check shapes
        assert result['features'].shape == (5, 10, 3)  # time, charge, aux
        assert result['dom_id'].shape == (5, 10)
        
        # Check data consistency
        np.testing.assert_array_equal(result['features'][:, :, 0], test_data['time'])
        np.testing.assert_array_equal(result['features'][:, :, 1], test_data['charge'])
        np.testing.assert_array_equal(result['features'][:, :, 2], test_data['aux'])
        np.testing.assert_array_equal(result['dom_id'], test_data['dom_id'])


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow from files to batch iteration"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create comprehensive test data
            dtype = np.dtype([
                ('time', np.float16),
                ('charge', np.float16),
                ('aux', np.float16),
                ('dom_id', np.uint16)
            ])
            
            n_events, seq_length = 6, 127
            x_data = np.zeros((n_events, seq_length), dtype=dtype)
            
            # Create realistic test data
            for i in range(n_events):
                event_length = np.random.randint(80, seq_length)
                for j in range(event_length):
                    x_data[i, j]['time'] = j * 10 + np.random.random()
                    x_data[i, j]['charge'] = np.random.exponential(2.0)
                    x_data[i, j]['aux'] = np.random.choice([0, 1])
                    x_data[i, j]['dom_id'] = np.uint16(np.random.randint(1, 5160 + 1))
            
            l_data = np.array([np.sum(x_data[i]['dom_id'] > 0) for i in range(n_events)], dtype=np.int32)
            c_data = np.random.random(n_events).astype(np.float32)
            
            # Save files using proper memmap format (not np.save which adds headers)
            x_memmap = np.memmap(tmp_path / 'x.npy', dtype=dtype, mode='w+', shape=x_data.shape)
            x_memmap[:] = x_data[:]
            del x_memmap
            
            l_memmap = np.memmap(tmp_path / 'l.npy', dtype=l_data.dtype, mode='w+', shape=l_data.shape)
            l_memmap[:] = l_data[:]
            del l_memmap
            
            c_memmap = np.memmap(tmp_path / 'c.npy', dtype=c_data.dtype, mode='w+', shape=c_data.shape)
            c_memmap[:] = c_data[:]
            del c_memmap
            
            props = {
                'x': {'shape': list(x_data.shape), 'dtype': dtype.descr},
                'l': {'shape': list(l_data.shape), 'dtype': str(l_data.dtype)},
                'c': {'shape': list(c_data.shape), 'dtype': str(c_data.dtype)}
            }
            
            with open(tmp_path / 'memmap_properties.json', 'w') as f:
                json.dump(props, f)
            
            # Test dataset creation and iteration
            dataset = IceCubeDataset(
                str(tmp_path),
                batch_size=2,
                target_transform=lambda y, c: (y, c) if y is not None else None
            )
            
            # Test iteration
            batches = list(dataset)
            assert len(batches) > 0
            
            # Test batch structure
            for (x, l), target in batches:
                assert isinstance(x, dict)
                assert 'features' in x and 'dom_id' in x
                assert x['features'].shape[0] == 2  # batch_size
                assert x['features'].shape[1] == 127  # seq_length
                assert x['features'].shape[2] == 3  # time, charge, aux
                assert x['dom_id'].shape == (2, 127)
                assert l.shape == (2,)
                
                # Verify DOM ID ranges
                active_dom_ids = x['dom_id'][x['dom_id'] > 0]
                if len(active_dom_ids) > 0:
                    assert np.all(active_dom_ids >= 1)
                    assert np.all(active_dom_ids <= 5160)


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 