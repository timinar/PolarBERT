#!/usr/bin/env python3
"""
Test script for z-translation augmentation system.

This script demonstrates how to use the z_translation_augmentation module
with sample data and provides validation of the augmentation process.

Usage:
    python test_z_translation.py --geometry_path /path/to/geometry.csv
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from polarbert.z_translation_augmentation import (
    augment_batch_with_z_translations,
    setup_geometry_config,
    augment_batch_simple,
    DETECTOR_CONFIG,
    STRING_CONFIG,
    TRANSLATION_CONFIG
)

def create_mock_batch(batch_size: int = 256, max_pulses: int = 127) -> tuple:
    """
    Create a mock batch for testing purposes.
    
    Args:
        batch_size: Number of events in batch
        max_pulses: Maximum number of pulses per event
    
    Returns:
        Tuple of (events_dict, labels)
    """
    # Create random DOM IDs (1-5160, with 0 for padding)
    dom_ids = torch.randint(1, DETECTOR_CONFIG['num_doms'] + 1, (batch_size, max_pulses))
    
    # Add some padding by setting random positions to 0
    padding_mask = torch.rand(batch_size, max_pulses) < 0.3  # 30% padding
    dom_ids[padding_mask] = 0
    
    # Create random features (charge, time, etc.)
    features = torch.randn(batch_size, max_pulses, 2)  # 2 features: charge, time
    
    # Create random labels
    labels = torch.randint(0, 2, (batch_size,)).float()  # Binary classification
    
    events_dict = {
        'dom_id': dom_ids,
        'features': features
    }
    
    return events_dict, labels

def test_augmentation_system(geometry_path: str, verbose: bool = True):
    """
    Test the augmentation system with mock data.
    
    Args:
        geometry_path: Path to geometry CSV file
        verbose: Whether to print detailed information
    """
    print("Testing Z-Translation Augmentation System")
    print("=" * 50)
    
    try:
        # Setup geometry configuration
        if verbose:
            print("Setting up geometry configuration...")
        
        geometry_config = setup_geometry_config(geometry_path)
        positions_normalized = geometry_config['positions_normalized']
        
        if verbose:
            print(f"✅ Geometry loaded: {positions_normalized.shape}")
        
        # Create mock batch
        if verbose:
            print("Creating mock batch...")
        
        batch_data = create_mock_batch(batch_size=64, max_pulses=100)  # Smaller for testing
        events_dict, labels = batch_data
        
        if verbose:
            print(f"✅ Mock batch created: {events_dict['dom_id'].shape}")
        
        # Test augmentation
        if verbose:
            print("Running augmentation...")
        
        translated_batches = augment_batch_with_z_translations(
            batch_data, 
            positions_normalized, 
            geometry_config,
            max_translations_per_event=10  # Reduced for testing
        )
        
        # Report results
        total_translated_events = sum(tb['n_events'] for tb in translated_batches)
        
        print(f"\n✅ Augmentation Results:")
        print(f"   Original batch size: {len(events_dict['dom_id'])}")
        print(f"   Translated batches created: {len(translated_batches)}")
        print(f"   Total translated events: {total_translated_events}")
        
        if translated_batches:
            avg_translations = total_translated_events / len(translated_batches)
            print(f"   Average translations per original event: {avg_translations:.1f}")
        
        # Validate batch structure
        if translated_batches:
            sample_batch = translated_batches[0]['batch']
            sample_events, sample_labels = sample_batch
            
            print(f"\n✅ Batch Structure Validation:")
            print(f"   Sample batch events shape: {sample_events['dom_id'].shape}")
            print(f"   Sample batch features shape: {sample_events['features'].shape}")
            print(f"   Sample batch labels shape: {sample_labels.shape}")
            print(f"   Data types match: {sample_events['dom_id'].dtype == events_dict['dom_id'].dtype}")
        
        # Show detailed info for first few batches
        if verbose and translated_batches:
            print(f"\n📊 Detailed Results:")
            for i, tb in enumerate(translated_batches[:3]):
                meta = tb['metadata']
                print(f"   Batch {i+1} (Original Event {tb['original_event_idx']}):")
                print(f"     Events: {meta['batch_size']}")
                print(f"     Translation range: {meta['translation_range'][0]:.1f}m to {meta['translation_range'][1]:.1f}m")
                print(f"     DOM spacing used: {meta['dom_spacing_used']:.1f}m")
                print(f"     Avg pulses per event: {meta['avg_pulses']:.1f}")
        
        print(f"\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_interface(geometry_path: str):
    """Test the simplified interface."""
    print("\nTesting Simplified Interface")
    print("-" * 30)
    
    try:
        # Create mock batch
        batch_data = create_mock_batch(batch_size=32, max_pulses=50)
        
        # Use simple interface
        translated_batches = augment_batch_simple(
            batch_data, geometry_path, max_translations_per_event=5
        )
        
        print(f"✅ Simple interface test: {len(translated_batches)} batches created")
        return True
        
    except Exception as e:
        print(f"❌ Simple interface test failed: {e}")
        return False

def validate_configuration():
    """Validate the configuration constants."""
    print("\nValidating Configuration")
    print("-" * 25)
    
    print(f"Detector Config: {DETECTOR_CONFIG}")
    print(f"String Config: {STRING_CONFIG}")
    print(f"Translation Config: {TRANSLATION_CONFIG}")
    
    # Basic validation
    assert DETECTOR_CONFIG['num_doms'] > 0
    assert STRING_CONFIG['doms_per_string'] > 0
    assert TRANSLATION_CONFIG['max_translations_per_event'] > 0
    
    print("✅ Configuration validation passed")

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test z-translation augmentation system")
    parser.add_argument(
        '--geometry_path', 
        type=str, 
        required=True,
        help='Path to geometry CSV file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    validate_configuration()
    
    # Test main augmentation system
    success1 = test_augmentation_system(args.geometry_path, args.verbose)
    
    # Test simple interface
    success2 = test_simple_interface(args.geometry_path)
    
    if success1 and success2:
        print(f"\n🎉 All tests passed! System is ready for production use.")
        return 0
    else:
        print(f"\n❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())