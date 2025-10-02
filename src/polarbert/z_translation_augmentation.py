#!/usr/bin/env python3
"""
Z-Axis Translation Data Augmentation for IceCube Events

This script provides functionality to augment training data by translating events
along the z-axis (depth) while preserving the physics and avoiding DeepCore regions.

Key features:
- Automatically filters events without DeepCore hits
- Translates events using DOM spacing steps for realistic augmentation
- Maximizes data augmentation by using full detector range
- Preserves original data structure for seamless integration

Usage:
    from scripts.z_translation_augmentation import augment_batch_with_z_translations
    
    augmented_batches = augment_batch_with_z_translations(
        batch_data, positions, geometry_config
    )

Author: Generated from PolarBERT geometry analysis
Date: September 2025
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# IceCube Detector Constants
DETECTOR_CONFIG = {
    'num_doms': 5160,
    'space_normalization': 500.0,  # meters, used to normalize coordinates
    'time_normalization': 3e4,     # nanoseconds, used to normalize time
    'speed_of_light_mns': 0.299792458,  # m/ns
}

# String and DOM Configuration
STRING_CONFIG = {
    'doms_per_string': 60,
    'total_strings': 86,
    'deepcore_strings': {
        'min_id': 79,
        'max_id': 86
    },
    'typical_dom_spacing_z': 17.0,  # meters, typical spacing between DOMs on same string
}

# Translation Parameters
TRANSLATION_CONFIG = {
    'max_translations_per_event': 25,
    'max_dom_distance_threshold': 100.0,  # meters, max distance to consider for pulse mapping
    'detector_buffer': 10.0,  # meters, safety buffer from detector boundaries
    'default_dom_spacing': 17.0,  # meters, fallback if spacing calculation fails
}

# Detector Boundaries (will be calculated dynamically but these are typical values)
DETECTOR_BOUNDS = {
    'non_deepcore_z_range': (-512.8, 524.6),  # meters, typical range for non-DeepCore DOMs
    'deepcore_z_range': (-505.4, 191.0),      # meters, typical DeepCore range
}

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_dom_to_string_mapping(num_doms: Optional[int] = None, doms_per_string: Optional[int] = None) -> Dict[int, int]:
    """
    Create mapping from DOM ID to string ID.
    
    Args:
        num_doms: Total number of DOMs (default from config)
        doms_per_string: DOMs per string (default from config)
    
    Returns:
        Dictionary mapping dom_id (0-based) to string_id (1-based)
    """
    if num_doms is None:
        num_doms = DETECTOR_CONFIG['num_doms']
    if doms_per_string is None:
        doms_per_string = STRING_CONFIG['doms_per_string']
    
    return {
        dom_id: (dom_id // doms_per_string) + 1
        for dom_id in range(num_doms)
    }

def setup_detector_geometry(geometry_csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and setup detector geometry from CSV file.
    
    Args:
        geometry_csv_path: Path to geometry CSV file with x,y,z columns
    
    Returns:
        Tuple of (positions_normalized, positions_meters)
        - positions_normalized: Tensor of shape [num_doms+1, 3] (index 0 is padding)
        - positions_meters: Tensor of shape [num_doms, 3] (actual coordinates in meters)
    """
    import pandas as pd
    
    geometry = pd.read_csv(geometry_csv_path)
    num_doms = DETECTOR_CONFIG['num_doms']
    space_norm = DETECTOR_CONFIG['space_normalization']
    
    # Create normalized positions tensor (index 0 is padding)
    positions_normalized = torch.zeros(num_doms + 1, 3, dtype=torch.float32)
    positions_meters = torch.from_numpy(geometry[['x', 'y', 'z']].values.astype(np.float32))
    
    # Store normalized positions (skip index 0 for padding)
    positions_normalized[1:num_doms + 1] = positions_meters / space_norm
    
    return positions_normalized, positions_meters

def calculate_dom_z_spacing(positions_meters: torch.Tensor, 
                          dom_to_string_map: Dict[int, int]) -> float:
    """
    Calculate typical DOM spacing in z-direction by sampling several strings.
    
    Args:
        positions_meters: Tensor of DOM positions in meters
        dom_to_string_map: Mapping from DOM ID to string ID
    
    Returns:
        Median DOM spacing in meters
    """
    sample_strings = [1, 10, 20, 30, 40, 50, 60, 70]  # Sample different string types
    spacings = []
    
    for string_id in sample_strings:
        if STRING_CONFIG['deepcore_strings']['min_id'] <= string_id <= STRING_CONFIG['deepcore_strings']['max_id']:
            continue  # Skip DeepCore strings
            
        # Find DOMs on this string
        string_doms = []
        for dom_id in range(len(positions_meters)):
            if dom_to_string_map.get(dom_id) == string_id:
                string_doms.append(dom_id)
        
        if len(string_doms) < 2:
            continue
            
        # Get z-coordinates and calculate spacings
        z_coords = [positions_meters[dom_id, 2].item() for dom_id in string_doms]
        z_coords = sorted(z_coords)
        string_spacings = [z_coords[i+1] - z_coords[i] for i in range(len(z_coords)-1)]
        spacings.extend(string_spacings)
    
    if spacings:
        median_spacing = float(np.median(spacings))
        logger.info(f"Calculated DOM spacing: {median_spacing:.1f}m from {len(spacings)} measurements")
        return median_spacing
    else:
        logger.warning(f"Could not calculate DOM spacing, using default: {TRANSLATION_CONFIG['default_dom_spacing']}m")
        return TRANSLATION_CONFIG['default_dom_spacing']

def calculate_detector_z_bounds(positions_meters: torch.Tensor, 
                              dom_to_string_map: Dict[int, int]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate z-coordinate bounds for detector regions.
    
    Args:
        positions_meters: Tensor of DOM positions in meters
        dom_to_string_map: Mapping from DOM ID to string ID
    
    Returns:
        Dictionary with 'non_deepcore' and 'deepcore' z-ranges
    """
    deepcore_z = []
    non_deepcore_z = []
    
    deepcore_min = STRING_CONFIG['deepcore_strings']['min_id']
    deepcore_max = STRING_CONFIG['deepcore_strings']['max_id']
    
    for dom_id in range(len(positions_meters)):
        string_id = dom_to_string_map.get(dom_id)
        z_coord = positions_meters[dom_id, 2].item()
        
        if deepcore_min <= string_id <= deepcore_max:
            deepcore_z.append(z_coord)
        else:
            non_deepcore_z.append(z_coord)
    
    bounds = {
        'non_deepcore': (min(non_deepcore_z), max(non_deepcore_z)),
        'deepcore': (min(deepcore_z), max(deepcore_z))
    }
    
    logger.info(f"Detector bounds - Non-DeepCore: {bounds['non_deepcore'][0]:.1f} to {bounds['non_deepcore'][1]:.1f}m")
    logger.info(f"Detector bounds - DeepCore: {bounds['deepcore'][0]:.1f} to {bounds['deepcore'][1]:.1f}m")
    
    return bounds

def filter_non_deepcore_events(events_dict: Dict[str, torch.Tensor], 
                             dom_to_string_map: Dict[int, int]) -> List[int]:
    """
    Filter events that have no pulses in DeepCore strings.
    
    Args:
        events_dict: Dictionary containing 'dom_id' and 'features' tensors
        dom_to_string_map: Mapping from DOM ID to string ID
    
    Returns:
        List of event indices that have no DeepCore hits
    """
    non_deepcore_events = []
    batch_size = events_dict['dom_id'].shape[0]
    
    deepcore_min = STRING_CONFIG['deepcore_strings']['min_id']
    deepcore_max = STRING_CONFIG['deepcore_strings']['max_id']
    
    for event_idx in range(batch_size):
        dom_ids = events_dict['dom_id'][event_idx]
        valid_dom_ids = dom_ids[dom_ids > 0]  # Remove padding
        
        has_deepcore = False
        for dom_id in valid_dom_ids:
            # Convert to 0-based indexing for string mapping
            string_id = dom_to_string_map.get(int(dom_id.item()) - 1, 0)
            if deepcore_min <= string_id <= deepcore_max:
                has_deepcore = True
                break
        
        if not has_deepcore:
            non_deepcore_events.append(event_idx)
    
    return non_deepcore_events

def get_event_z_range(event_data: Dict[str, torch.Tensor], 
                     positions_normalized: torch.Tensor) -> Optional[Dict[str, float]]:
    """
    Analyze the z-coordinate range of a specific event.
    
    Args:
        event_data: Dictionary with 'dom_id' and 'features' for single event
        positions_normalized: Normalized position tensor
    
    Returns:
        Dictionary with z_min, z_max, z_span, z_center in meters, or None if no valid DOMs
    """
    valid_mask = event_data['dom_id'] > 0
    valid_dom_ids = event_data['dom_id'][valid_mask]
    
    if len(valid_dom_ids) == 0:
        return None
    
    # Convert to meters
    event_positions = positions_normalized[valid_dom_ids] * DETECTOR_CONFIG['space_normalization']
    z_coords = event_positions[:, 2]
    
    return {
        'z_min': z_coords.min().item(),
        'z_max': z_coords.max().item(),
        'z_span': (z_coords.max() - z_coords.min()).item(),
        'z_center': z_coords.mean().item()
    }

def calculate_possible_translations(event_z_info: Dict[str, float], 
                                  detector_bounds: Dict[str, Tuple[float, float]], 
                                  dom_spacing: float,
                                  max_translations: Optional[int] = None) -> List[float]:
    """
    Calculate all possible translation distances using DOM spacing steps.
    
    Args:
        event_z_info: Dictionary with event z-range information
        detector_bounds: Dictionary with detector z-bounds
        dom_spacing: DOM spacing in meters
        max_translations: Maximum number of translations to generate
    
    Returns:
        List of translation distances in meters
    """
    if max_translations is None:
        max_translations = TRANSLATION_CONFIG['max_translations_per_event']
    
    assert max_translations is not None  # Type checker hint
    
    event_z_min = event_z_info['z_min']
    event_z_max = event_z_info['z_max']
    
    detector_min, detector_max = detector_bounds['non_deepcore']
    buffer = TRANSLATION_CONFIG['detector_buffer']
    
    # Apply safety buffer
    safe_detector_min = detector_min + buffer
    safe_detector_max = detector_max - buffer
    
    # Calculate translation limits
    max_downward_translation = safe_detector_min - event_z_min
    max_upward_translation = safe_detector_max - event_z_max
    
    translations = []
    
    # Downward translations (negative)
    current_translation = -dom_spacing
    while current_translation >= max_downward_translation and len(translations) < max_translations // 2:
        translations.append(current_translation)
        current_translation -= dom_spacing
    
    # Upward translations (positive)
    current_translation = dom_spacing
    while current_translation <= max_upward_translation and len(translations) < max_translations:
        translations.append(current_translation)
        current_translation += dom_spacing
    
    return sorted(translations)

def translate_event_z_axis(event_data: Dict[str, torch.Tensor], 
                         positions_normalized: torch.Tensor,
                         dom_to_string_map: Dict[int, int], 
                         translation_z_meters: float) -> Optional[Dict[str, Any]]:
    """
    Translate an event along the z-axis by finding corresponding DOMs.
    
    Args:
        event_data: Dictionary with 'dom_id' and 'features' for single event
        positions_normalized: Normalized position tensor
        dom_to_string_map: Mapping from DOM ID to string ID
        translation_z_meters: Translation distance in meters
    
    Returns:
        Dictionary with translated event data, or None if translation fails
    """
    original_dom_ids = event_data['dom_id']
    original_features = event_data['features']
    
    # Remove padding
    valid_mask = original_dom_ids > 0
    valid_dom_ids = original_dom_ids[valid_mask]
    valid_features = original_features[valid_mask]
    
    if len(valid_dom_ids) == 0:
        return None
    
    # Get original positions in meters
    space_norm = DETECTOR_CONFIG['space_normalization']
    original_positions = positions_normalized[valid_dom_ids] * space_norm
    
    # Calculate target z-coordinates
    target_z_coords = original_positions[:, 2] + translation_z_meters
    
    # Find closest DOMs at target z-coordinates
    new_dom_ids = []
    new_features = []
    
    deepcore_min = STRING_CONFIG['deepcore_strings']['min_id']
    deepcore_max = STRING_CONFIG['deepcore_strings']['max_id']
    max_distance = TRANSLATION_CONFIG['max_dom_distance_threshold']
    
    for i, (original_dom_id, target_z) in enumerate(zip(valid_dom_ids, target_z_coords)):
        # Get string ID of original DOM (convert to 0-based)
        original_string_id = dom_to_string_map.get(int(original_dom_id.item()) - 1, 0)
        
        # Skip DeepCore strings
        if deepcore_min <= original_string_id <= deepcore_max:
            continue
            
        # Find all DOMs on the same string
        string_dom_candidates = []
        for dom_id in range(1, len(positions_normalized)):  # Skip index 0 (padding)
            candidate_string_id = dom_to_string_map.get(dom_id - 1, 0)
            if candidate_string_id == original_string_id:
                string_dom_candidates.append(dom_id)
        
        if not string_dom_candidates:
            continue
        
        # Find closest DOM on this string to target z-coordinate
        candidate_positions = positions_normalized[string_dom_candidates] * space_norm
        z_distances = torch.abs(candidate_positions[:, 2] - target_z)
        closest_idx = torch.argmin(z_distances)
        closest_dom_id = string_dom_candidates[closest_idx]
        
        # Check if reasonably close
        if z_distances[closest_idx].item() > max_distance:
            continue
            
        new_dom_ids.append(closest_dom_id)
        new_features.append(valid_features[i])
    
    if len(new_dom_ids) == 0:
        return None
    
    # Create new event data with same structure as input
    new_dom_ids = torch.tensor(new_dom_ids, dtype=original_dom_ids.dtype)
    new_features = torch.stack(new_features)
    
    # Pad to same length as original
    original_length = len(original_dom_ids)
    if len(new_dom_ids) < original_length:
        padding_length = original_length - len(new_dom_ids)
        dom_padding = torch.zeros(padding_length, dtype=new_dom_ids.dtype)
        feature_padding = torch.zeros(padding_length, new_features.shape[1], dtype=new_features.dtype)
        
        new_dom_ids = torch.cat([new_dom_ids, dom_padding])
        new_features = torch.cat([new_features, feature_padding])
    elif len(new_dom_ids) > original_length:
        new_dom_ids = new_dom_ids[:original_length]
        new_features = new_features[:original_length]
    
    return {
        'dom_id': new_dom_ids,
        'features': new_features,
        'translation_z': translation_z_meters,
        'original_pulses': len(valid_dom_ids),
        'translated_pulses': len([x for x in new_dom_ids if x > 0])
    }

# =============================================================================
# MAIN AUGMENTATION FUNCTION
# =============================================================================

def augment_batch_with_z_translations(batch_data: Tuple[Dict[str, torch.Tensor], torch.Tensor],
                                     positions_normalized: torch.Tensor,
                                     geometry_config: Dict[str, Any],
                                     max_translations_per_event: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Main function to augment a batch with z-axis translations.
    
    Args:
        batch_data: Tuple of (events_dict, labels) from dataloader
        positions_normalized: Normalized DOM positions tensor [num_doms+1, 3]
        geometry_config: Dictionary containing:
            - 'positions_meters': DOM positions in meters [num_doms, 3]
            - 'dom_to_string_map': Optional DOM-to-string mapping
        max_translations_per_event: Maximum translations per event
    
    Returns:
        List of dictionaries, each containing:
        - 'batch': Translated batch data (events_dict, labels)
        - 'original_event_idx': Index of original event
        - 'translations': List of translation distances applied
        - 'metadata': Additional information and statistics
    """
    if max_translations_per_event is None:
        max_translations_per_event = TRANSLATION_CONFIG['max_translations_per_event']
    
    events_dict, labels = batch_data
    positions_meters = geometry_config['positions_meters']
    
    # Create or use provided DOM-to-string mapping
    if 'dom_to_string_map' in geometry_config:
        dom_to_string_map = geometry_config['dom_to_string_map']
    else:
        dom_to_string_map = create_dom_to_string_mapping()
    
    # Calculate DOM spacing and detector bounds
    dom_spacing = calculate_dom_z_spacing(positions_meters, dom_to_string_map)
    detector_bounds = calculate_detector_z_bounds(positions_meters, dom_to_string_map)
    
    # Find events without DeepCore hits
    non_deepcore_indices = filter_non_deepcore_events(events_dict, dom_to_string_map)
    
    logger.info(f"Found {len(non_deepcore_indices)} non-DeepCore events out of {len(events_dict['dom_id'])} total events")
    
    translated_batches = []
    total_translations = 0
    
    for event_idx in non_deepcore_indices:
        event_data = {
            'dom_id': events_dict['dom_id'][event_idx],
            'features': events_dict['features'][event_idx]
        }
        
        # Get event z-range
        event_z_info = get_event_z_range(event_data, positions_normalized)
        if not event_z_info:
            continue
        
        # Calculate possible translations
        translations = calculate_possible_translations(
            event_z_info, detector_bounds, dom_spacing, max_translations_per_event
        )
        
        if not translations:
            logger.debug(f"No translations possible for event {event_idx}")
            continue
        
        # Create translated events
        translated_events = []
        translated_labels = []
        successful_translations = []
        
        # Add the original event as the first sample (translation = 0.0)
        translated_events.append({
            'dom_id': event_data['dom_id'].unsqueeze(0),
            'features': event_data['features'].unsqueeze(0)
        })
        translated_labels.append(labels[event_idx].unsqueeze(0))
        successful_translations.append(0.0)
        
        for translation_z in translations:
            translated_event = translate_event_z_axis(
                event_data, positions_normalized, dom_to_string_map, translation_z
            )
            
            if translated_event and translated_event['translated_pulses'] > 0:
                translated_events.append({
                    'dom_id': translated_event['dom_id'].unsqueeze(0),
                    'features': translated_event['features'].unsqueeze(0)
                })
                translated_labels.append(labels[event_idx].unsqueeze(0))
                successful_translations.append(translation_z)
        
        if translated_events:
            # Combine all translations into a single batch
            batch_dom_ids = torch.cat([te['dom_id'] for te in translated_events], dim=0)
            batch_features = torch.cat([te['features'] for te in translated_events], dim=0)
            batch_labels = torch.cat(translated_labels, dim=0)
            
            translated_batch = (
                {'dom_id': batch_dom_ids, 'features': batch_features},
                batch_labels
            )
            
            # Calculate metadata
            n_events = len(translated_events)
            pulse_counts = [(batch_dom_ids[i] > 0).sum().item() for i in range(n_events)]
            
            batch_info = {
                'batch': translated_batch,
                'original_event_idx': event_idx,
                'translations': successful_translations,
                'n_events': n_events,
                'metadata': {
                    'batch_size': n_events,
                    'pulse_counts': pulse_counts,
                    'avg_pulses': np.mean(pulse_counts),
                    'dom_spacing_used': dom_spacing,
                    'detector_bounds_used': detector_bounds,
                    'translation_range': (min(successful_translations), max(successful_translations)),
                    'original_event_z_info': event_z_info,
                    'data_shapes': {
                        'dom_id': batch_dom_ids.shape,
                        'features': batch_features.shape,
                        'labels': batch_labels.shape
                    }
                }
            }
            
            translated_batches.append(batch_info)
            total_translations += n_events
            
            logger.debug(f"Event {event_idx}: {len(successful_translations)}/{len(translations)} translations successful")
    
    logger.info(f"Successfully created {len(translated_batches)} translated batches with {total_translations} total events")
    logger.info(f"Augmentation factor: {total_translations / len(non_deepcore_indices):.1f}x")
    
    return translated_batches

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def setup_geometry_config(geometry_csv_path: str) -> Dict[str, Any]:
    """
    Convenience function to setup geometry configuration from CSV file.
    
    Args:
        geometry_csv_path: Path to geometry CSV file
    
    Returns:
        Dictionary with geometry configuration for augment_batch_with_z_translations
    """
    positions_normalized, positions_meters = setup_detector_geometry(geometry_csv_path)
    dom_to_string_map = create_dom_to_string_mapping()
    
    return {
        'positions_normalized': positions_normalized,
        'positions_meters': positions_meters,
        'dom_to_string_map': dom_to_string_map
    }

def augment_batch_simple(batch_data: Tuple[Dict[str, torch.Tensor], torch.Tensor],
                        geometry_csv_path: str,
                        max_translations_per_event: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Simplified interface for batch augmentation.
    
    Args:
        batch_data: Tuple of (events_dict, labels) from dataloader
        geometry_csv_path: Path to geometry CSV file
        max_translations_per_event: Maximum translations per event
    
    Returns:
        List of translated batches
    """
    # Setup geometry
    positions_normalized, positions_meters = setup_detector_geometry(geometry_csv_path)
    geometry_config = {
        'positions_meters': positions_meters,
        'dom_to_string_map': create_dom_to_string_mapping()
    }
    
    # Perform augmentation
    return augment_batch_with_z_translations(
        batch_data, positions_normalized, geometry_config, max_translations_per_event
    )

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the z-translation augmentation system.
    """
    print("Z-Axis Translation Augmentation System")
    print("=" * 50)
    
    # Example configuration
    geometry_csv_path = "/path/to/geometry.csv"  # Replace with actual path
    
    # Setup geometry (do this once)
    try:
        positions_normalized, positions_meters = setup_detector_geometry(geometry_csv_path)
        geometry_config = {
            'positions_meters': positions_meters,
            'dom_to_string_map': create_dom_to_string_mapping()
        }
        print("✅ Geometry setup complete")
    except Exception as e:
        print(f"❌ Geometry setup failed: {e}")
        exit(1)
    
    # Example batch (replace with actual batch from dataloader)
    # batch_data = (events_dict, labels)
    
    # Perform augmentation
    # translated_batches = augment_batch_with_z_translations(
    #     batch_data, positions_normalized, geometry_config,
    #     max_translations_per_event=20
    # )
    
    # print(f"Generated {len(translated_batches)} augmented batches")
    # print(f"Total augmented events: {sum(tb['n_events'] for tb in translated_batches)}")
    
    print("Ready for use!")