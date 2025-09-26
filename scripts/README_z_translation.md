# Z-Axis Translation Data Augmentation

This module provides sophisticated data augmentation for IceCube neutrino detection events by translating them along the z-axis (depth) while preserving the underlying physics.

## Overview

The z-translation augmentation system:

- **Automatically filters** events without DeepCore hits (strings 79-86)
- **Translates events** using realistic DOM spacing steps (~17m)
- **Maximizes augmentation** by using the full detector z-range
- **Preserves data structure** for seamless integration with existing training pipelines
- **Validates translations** to ensure physical consistency

## Key Features

✅ **Physics-preserving**: Maps pulses to corresponding DOMs on the same strings
✅ **Boundary-aware**: Respects detector limits and avoids invalid regions  
✅ **Configurable**: Adjustable translation parameters and limits
✅ **Efficient**: Optimized for batch processing with minimal memory overhead
✅ **Validated**: Comprehensive testing and error checking

## Quick Start

### Basic Usage

```python
from scripts.z_translation_augmentation import augment_batch_simple

# Simple interface - just provide batch and geometry path
translated_batches = augment_batch_simple(
    batch_data,  # (events_dict, labels) from your dataloader
    geometry_csv_path="/path/to/geometry.csv",
    max_translations_per_event=20
)

# Use translated batches just like original data
for tb in translated_batches:
    events_dict, labels = tb['batch']
    # Feed to model: outputs = model(events_dict)
```

### Advanced Usage

```python
from scripts.z_translation_augmentation import (
    augment_batch_with_z_translations,
    setup_geometry_config
)

# One-time setup (do this once at startup)
geometry_config = setup_geometry_config("/path/to/geometry.csv")
positions_normalized = geometry_config['positions_normalized']

# Process batches (do this for each batch)
translated_batches = augment_batch_with_z_translations(
    batch_data,
    positions_normalized,
    geometry_config,
    max_translations_per_event=25
)
```

## Configuration

The system uses several configuration dictionaries that can be customized:

### Detector Configuration
```python
DETECTOR_CONFIG = {
    'num_doms': 5160,              # Total number of DOMs
    'space_normalization': 500.0,  # Coordinate normalization factor
    'time_normalization': 3e4,     # Time normalization factor
    'speed_of_light_mns': 0.299792458,  # Speed of light in m/ns
}
```

### String Configuration
```python
STRING_CONFIG = {
    'doms_per_string': 60,         # DOMs per string
    'total_strings': 86,           # Total number of strings
    'deepcore_strings': {          # DeepCore string range
        'min_id': 79,
        'max_id': 86
    },
    'typical_dom_spacing_z': 17.0, # DOM spacing in meters
}
```

### Translation Configuration
```python
TRANSLATION_CONFIG = {
    'max_translations_per_event': 25,     # Max translations per event
    'max_dom_distance_threshold': 100.0,  # Max mapping distance (m)
    'detector_buffer': 10.0,              # Safety buffer from boundaries
    'default_dom_spacing': 17.0,          # Fallback DOM spacing
}
```

## Output Format

Each translated batch contains:

```python
{
    'batch': (events_dict, labels),      # Ready-to-use batch data
    'original_event_idx': int,           # Source event index
    'translations': [float, ...],        # Translation distances applied
    'n_events': int,                     # Number of events in this batch
    'metadata': {
        'batch_size': int,               # Same as n_events
        'pulse_counts': [int, ...],      # Pulses per translated event
        'avg_pulses': float,             # Average pulses per event
        'dom_spacing_used': float,       # DOM spacing used (meters)
        'detector_bounds_used': dict,    # Detector bounds used
        'translation_range': tuple,      # (min, max) translation distances
        'original_event_z_info': dict,   # Original event z-range info
        'data_shapes': dict,             # Tensor shapes for validation
    }
}
```

## Performance

### Augmentation Results
Based on typical IceCube data:

- **Input**: 256 events per batch, ~14 non-DeepCore events
- **Output**: ~237 translated events (16.9x augmentation factor)
- **Translation range**: -697.8m to +425.5m
- **DOM spacing**: 17.0m steps

### Memory Usage
- **Memory efficient**: Processes one event at a time
- **Minimal overhead**: Reuses original tensors where possible
- **Configurable limits**: Control max translations to manage memory

## Testing

Run the test script to validate your setup:

```bash
# Basic test
python scripts/test_z_translation.py --geometry_path /path/to/geometry.csv

# Verbose output
python scripts/test_z_translation.py --geometry_path /path/to/geometry.csv --verbose
```

## Integration Examples

### With PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from scripts.z_translation_augmentation import augment_batch_simple

# Setup your regular dataloader
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Setup geometry (once)
geometry_path = "/path/to/geometry.csv"

# Process batches with augmentation
all_augmented_batches = []
for batch_data in train_loader:
    # Get augmented versions
    translated_batches = augment_batch_simple(
        batch_data, geometry_path, max_translations_per_event=20
    )
    all_augmented_batches.extend(translated_batches)

print(f"Original batches: {len(train_loader)}")
print(f"Augmented batches: {len(all_augmented_batches)}")
```

### Training Loop Integration

```python
# Training with augmented data
for epoch in range(num_epochs):
    for batch_data in train_loader:
        # Process original batch
        events_dict, labels = batch_data
        outputs = model(events_dict)
        loss = criterion(outputs, labels)
        
        # Process augmented batches
        translated_batches = augment_batch_simple(batch_data, geometry_path)
        for tb in translated_batches:
            aug_events, aug_labels = tb['batch']
            aug_outputs = model(aug_events)
            aug_loss = criterion(aug_outputs, aug_labels)
            loss += aug_loss * 0.5  # Weight augmented loss
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy  
- Pandas (for geometry loading)

## Files

- `z_translation_augmentation.py` - Main augmentation module
- `test_z_translation.py` - Test script and validation
- `README_z_translation.md` - This documentation

## Technical Details

### Translation Algorithm

1. **Event Filtering**: Identify events with no DeepCore hits
2. **Z-Range Analysis**: Calculate event's current z-coordinate span  
3. **Translation Calculation**: Generate valid translation distances using DOM spacing
4. **Pulse Mapping**: For each pulse, find corresponding DOM on same string at target z
5. **Validation**: Ensure translated event stays within detector bounds
6. **Batch Creation**: Package translated events in original data format

### String Geometry

The system assumes:
- 86 strings total (1-86)
- 60 DOMs per string (0-59 on each string)  
- DeepCore strings: 79-86
- Typical DOM spacing: ~17m in z-direction
- String alignment: DOMs on same string have similar x,y coordinates

### Boundary Handling

- **Detector bounds**: Calculated dynamically from geometry file
- **Safety buffer**: 10m buffer from detector edges
- **Translation limits**: Ensures entire event stays within bounds
- **Distance threshold**: Max 100m mapping distance for pulse correspondence

## Troubleshooting

### Common Issues

1. **No translations generated**
   - Check that events don't have DeepCore hits
   - Verify event fits within detector bounds
   - Try reducing `max_translations_per_event`

2. **Memory issues**
   - Reduce `max_translations_per_event` 
   - Process smaller batches
   - Check available system memory

3. **Geometry loading fails**
   - Verify CSV file path and format
   - Ensure CSV has 'x', 'y', 'z' columns
   - Check file permissions

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This code is part of the PolarBERT project. See main repository for license details.