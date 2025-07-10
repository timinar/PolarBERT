# scripts/prepare_memmaped_orca.py

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm
import os
from pathlib import Path
import json
import argparse
import re # For extracting file numbers
from typing import Optional, Dict, List, Tuple
from scipy.spatial import cKDTree

# Default configuration values
DEFAULT_SEQ_LENGTH = 127
DEFAULT_N_FEATURES = 4  # time, charge, auxiliary, dom_id (1-indexed)

# --- SensorMapping Class (Integrated) ---
class SensorMapping:
    """
    Handles mapping of pulse coordinates to sensor IDs using a geometry file.
    Uses a KD-tree for efficient nearest neighbor search.
    """
    def __init__(self, geometry_file: str):
        print(f"SensorMapping: Loading geometry from {geometry_file}")
        try:
            self.sensor_geometry = pd.read_csv(geometry_file)
        except Exception as e:
            raise FileNotFoundError(f"Could not read geometry file: {geometry_file}. Error: {e}")

        if not {'x', 'y', 'z'}.issubset(self.sensor_geometry.columns):
            raise ValueError("Geometry CSV must contain 'x', 'y', 'z' columns.")

        self.geometry_coords = self.sensor_geometry[['x', 'y', 'z']].values
        
        if 'sensor_id' in self.sensor_geometry.columns:
            self.sensor_ids_map = self.sensor_geometry['sensor_id'].values
            print("SensorMapping: Using 'sensor_id' column from geometry file for mapping.")
        else:
            self.sensor_ids_map = self.sensor_geometry.index.values
            print("SensorMapping: Using row index from geometry file as 'sensor_id' for mapping.")

        self.kdtree = None
        self.z_offset = 0.0
        print(f"SensorMapping: Loaded {len(self.geometry_coords)} sensor positions.")
        
    def calculate_z_offset(self, data_file_path: str) -> float:
        print(f"SensorMapping: Calculating Z-offset using pulse data from: {data_file_path}")
        try:
            sample_data = pq.read_table(data_file_path, columns=['sensor_pos_z'])
            data_z = sample_data['sensor_pos_z'].to_numpy()
            if len(data_z) == 0:
                print("SensorMapping: Warning - No 'sensor_pos_z' data in the sample file for Z-offset calibration.")
                return 0.0
            
            z_prom_min, z_prom_max = np.min(data_z), np.max(data_z)
            z_prom_center = (z_prom_max + z_prom_min) / 2
            
            geom_z_min, geom_z_max = np.min(self.geometry_coords[:,2]), np.max(self.geometry_coords[:,2])
            z_geom_center = (geom_z_max + geom_z_min) / 2
            
            calculated_offset = z_prom_center - z_geom_center
            print(f"SensorMapping: Prometheus Z center: {z_prom_center:.2f}, Geometry Z center: {z_geom_center:.2f}, Calculated Z-offset: {calculated_offset:.2f}")
            return calculated_offset
        except Exception as e:
            print(f"SensorMapping: Warning - Error during Z-offset calculation: {e}. Using Z-offset of 0.")
            return 0.0
    
    def initialize_mapping(self, pulse_data_file_path: Optional[str] = None) -> None:
        if pulse_data_file_path is not None:
            self.z_offset = self.calculate_z_offset(pulse_data_file_path)
        else:
            self.z_offset = 0.0
            print("SensorMapping: No pulse data file provided for Z-offset calibration. Using Z-offset of 0.")
            
        adjusted_geom_coords = self.geometry_coords.copy()
        adjusted_geom_coords[:, 2] -= self.z_offset

        self.kdtree = cKDTree(adjusted_geom_coords)
        print(f"SensorMapping: Initialized KD-tree with {len(adjusted_geom_coords)} sensors. Applied Z-offset: {self.z_offset:.2f} meters.")
    
    def coords_to_sensor_ids(self, coords: np.ndarray) -> np.ndarray:
        if self.kdtree is None:
            print("SensorMapping: Warning - KD-tree not initialized. Initializing now without Z-offset calibration.")
            self.initialize_mapping(pulse_data_file_path=None)
            
        distances, indices_in_geometry = self.kdtree.query(coords, k=1)
        mapped_sensor_ids = self.sensor_ids_map[indices_in_geometry]
        return mapped_sensor_ids

# --- Metadata Handling Class ---
class OrcaMetadata:
    """
    Handles finding ORCA files and creating combined metadata across multiple files.
    """
    def __init__(self, base_path: Path, file_range: Tuple[int, int]):
        self.base_path = base_path
        self.truth_dir = base_path / 'mc_truth'
        self.pulse_dir = base_path / 'pulses'
        self.file_range = file_range
        
        if not self.truth_dir.is_dir() or not self.pulse_dir.is_dir():
            raise FileNotFoundError(f"Required directories 'mc_truth' or 'pulses' not found in {base_path}")

        self.truth_files, self.pulse_files = self._find_files()
        
        if not self.truth_files or not self.pulse_files:
             raise FileNotFoundError(f"No matching ORCA files found in range {file_range} within {base_path}")
        
        print(f"Found {len(self.pulse_files)} pulse files and {len(self.truth_files)} truth files in range {file_range}.")
        print(f"First pulse file: {self.pulse_files[0].name}")
        print(f"Last pulse file: {self.pulse_files[-1].name}")

    def _extract_file_num(self, path: Path) -> Optional[int]:
        """Extracts the number from filenames like 'pulses_806.parquet'."""
        match = re.search(r'_(\d+)\.parquet$', path.name)
        return int(match.group(1)) if match else None

    def _find_files(self) -> Tuple[List[Path], List[Path]]:
        """Finds and sorts truth and pulse files within the specified number range."""
        start_num, end_num = self.file_range
        
        all_truth = sorted(self.truth_dir.glob('mc_truth_*.parquet'))
        all_pulses = sorted(self.pulse_dir.glob('pulses_*.parquet'))
        
        valid_truth_files = []
        valid_pulse_files = []
        found_nums = set()

        for f in all_pulses:
            num = self._extract_file_num(f)
            if num is not None and start_num <= num <= end_num:
                 valid_pulse_files.append(f)
                 found_nums.add(num)
        
        for f in all_truth:
             num = self._extract_file_num(f)
             if num is not None and num in found_nums:
                  valid_truth_files.append(f)

        # Ensure we have matching pairs
        pulse_nums = {self._extract_file_num(f) for f in valid_pulse_files}
        truth_nums = {self._extract_file_num(f) for f in valid_truth_files}
        
        if pulse_nums != truth_nums:
             print("Warning: Mismatch between found pulse file numbers and truth file numbers.")
             print(f"Pulse file numbers: {sorted(list(pulse_nums))}")
             print(f"Truth file numbers: {sorted(list(truth_nums))}")
             # Filter to keep only pairs with matching numbers
             common_nums = pulse_nums.intersection(truth_nums)
             valid_pulse_files = [f for f in valid_pulse_files if self._extract_file_num(f) in common_nums]
             valid_truth_files = [f for f in valid_truth_files if self._extract_file_num(f) in common_nums]
             print(f"Processing only the {len(common_nums)} matching file pairs.")

        # Sort based on number to ensure correct order
        valid_pulse_files.sort(key=self._extract_file_num)
        valid_truth_files.sort(key=self._extract_file_num)

        return valid_truth_files, valid_pulse_files

    def create_metadata_table(self, labels_to_save: List[tuple[str, np.dtype]]) -> pa.Table:
        """
        Iterates through file pairs, matches events, and builds a combined metadata table.
        """
        all_meta_batches = []
        total_matched_events = 0

        for truth_file, pulse_file in tqdm(zip(self.truth_files, self.pulse_files), total=len(self.truth_files), desc="Generating Metadata"):
            batch_id = self._extract_file_num(pulse_file) # Use file number as batch_id
            if batch_id is None:
                print(f"Warning: Could not extract batch_id from {pulse_file.name}. Skipping file.")
                continue

            try:
                truth_table = pq.read_table(truth_file)
                pulse_table = pq.read_table(pulse_file)
            except Exception as e:
                print(f"Warning: Error reading {truth_file} or {pulse_file}: {e}. Skipping pair.")
                continue

            # --- Event matching logic (same as create_orca_metadata before) ---
            event_counts_in_pulses = pc.value_counts(pulse_table['event_no'])
            unique_event_nos_in_pulses = event_counts_in_pulses.field('values').to_numpy()
            counts_per_event_pulse = event_counts_in_pulses.field('counts').to_numpy()

            cumulative_counts = np.cumsum(counts_per_event_pulse)
            first_pulse_indices = np.concatenate([[0], cumulative_counts[:-1]])
            last_pulse_indices = cumulative_counts - 1

            pulse_meta_lookup = {
                event_no: {'first_pulse_index': first_idx, 'last_pulse_index': last_idx}
                for event_no, first_idx, last_idx in zip(unique_event_nos_in_pulses, first_pulse_indices, last_pulse_indices)
            }

            batch_meta_columns = {'batch_id': [], 'event_id': [], 'first_pulse_index': [], 'last_pulse_index': []}
            for label_name, _ in labels_to_save:
                batch_meta_columns[label_name] = []

            truth_event_nos = truth_table.column('event_no').to_numpy()
            truth_label_data = {name: truth_table.column(name).to_numpy() for name, _ in labels_to_save}

            matched_in_batch = 0
            for i in range(len(truth_event_nos)):
                event_no = truth_event_nos[i]
                if event_no in pulse_meta_lookup:
                    batch_meta_columns['batch_id'].append(batch_id) # Add batch_id
                    batch_meta_columns['event_id'].append(event_no)
                    batch_meta_columns['first_pulse_index'].append(pulse_meta_lookup[event_no]['first_pulse_index'])
                    batch_meta_columns['last_pulse_index'].append(pulse_meta_lookup[event_no]['last_pulse_index'])
                    for label_name, _ in labels_to_save:
                        batch_meta_columns[label_name].append(truth_label_data[label_name][i])
                    matched_in_batch += 1
            
            if matched_in_batch > 0:
                 # Convert batch dict to PyArrow Table
                 pa_arrays = []
                 pa_schema_fields = []
                 
                 pa_arrays.append(pa.array(batch_meta_columns['batch_id'], type=pa.int64())) # Assuming batch_id fits in int64
                 pa_schema_fields.append(pa.field('batch_id', pa.int64()))
                 pa_arrays.append(pa.array(batch_meta_columns['event_id'], type=pa.int64()))
                 pa_schema_fields.append(pa.field('event_id', pa.int64()))
                 pa_arrays.append(pa.array(batch_meta_columns['first_pulse_index'], type=pa.int64()))
                 pa_schema_fields.append(pa.field('first_pulse_index', pa.int64()))
                 pa_arrays.append(pa.array(batch_meta_columns['last_pulse_index'], type=pa.int64()))
                 pa_schema_fields.append(pa.field('last_pulse_index', pa.int64()))

                 for label_name, numpy_dtype in labels_to_save:
                     pa_type = pa.from_numpy_dtype(numpy_dtype)
                     pa_arrays.append(pa.array(batch_meta_columns[label_name], type=pa_type))
                     pa_schema_fields.append(pa.field(label_name, pa_type))

                 all_meta_batches.append(pa.Table.from_arrays(pa_arrays, schema=pa.schema(pa_schema_fields)))
                 total_matched_events += matched_in_batch

        if not all_meta_batches:
             print("Error: No matched events found across all file pairs.")
             return None

        print(f"Concatenating metadata from {len(all_meta_batches)} batches. Total matched events: {total_matched_events}")
        combined_metadata_table = pa.concat_tables(all_meta_batches)
        return combined_metadata_table

# --- event_to_seq function remains the same ---
def event_to_seq(features: pa.Table, sensor_mapping: SensorMapping, seq_length: int, n_features: int, dtype_np=np.float32):
    """
    Converts a PyArrow Table of pulse features for a single event into a fixed-length numpy sequence.
    """
    coords = np.stack([
        features.column('sensor_pos_x').to_numpy(),
        features.column('sensor_pos_y').to_numpy(),
        features.column('sensor_pos_z').to_numpy(),
    ], axis=1)

    sensor_id_0indexed = sensor_mapping.coords_to_sensor_ids(coords)

    N_pulses = sensor_id_0indexed.shape[0]
    charge = features.column('charge').to_numpy()
    time = features.column('t').to_numpy()
    auxiliary = 1 - features.column('is_signal').to_numpy()

    selected_idx = np.arange(N_pulses)

    if N_pulses > seq_length:
        non_aux_indices = np.where(auxiliary == 0)[0]
        aux_indices = np.where(auxiliary == 1)[0]

        if len(non_aux_indices) >= seq_length:
            selected_idx = np.random.choice(non_aux_indices, size=seq_length, replace=False)
        else:
            num_needed_from_aux = seq_length - len(non_aux_indices)
            if len(aux_indices) >= num_needed_from_aux:
                chosen_aux_indices = np.random.choice(aux_indices, size=num_needed_from_aux, replace=False)
            else:
                chosen_aux_indices = aux_indices
            selected_idx = np.concatenate((non_aux_indices, chosen_aux_indices))
            if len(selected_idx) > seq_length:
                selected_idx = selected_idx[:seq_length]
            elif len(selected_idx) < seq_length and N_pulses >= seq_length:
                 selected_idx = np.random.choice(np.arange(N_pulses), size=seq_length, replace=False)
    
    if N_pulses < seq_length:
        selected_idx = np.arange(N_pulses)

    selected_idx = np.sort(selected_idx)

    T_evt = np.zeros((seq_length, n_features), dtype=dtype_np)
    current_event_len = len(selected_idx)

    if current_event_len > 0:
        T_evt[:current_event_len, 0] = (time[selected_idx] - 1e4) / 3e4
        T_evt[:current_event_len, 1] = np.log10(np.maximum(charge[selected_idx], 1e-5)) / 3.0
        T_evt[:current_event_len, 2] = auxiliary[selected_idx] - 0.5
        T_evt[:current_event_len, 3] = sensor_id_0indexed[selected_idx] + 1

    total_charge_for_event = charge.sum() if N_pulses > 0 else 0.0
    return T_evt, current_event_len, total_charge_for_event

# --- Main Processing Function (Adapted from original process_batches) ---
def process_orca_files_to_memmap(
    output_dir_str: str,
    base_data_path: Path, # Path to the main ORCA data folder
    metadata_table: pa.Table, # Combined metadata for all batches
    sensor_mapping: SensorMapping,
    labels_to_save: List[tuple[str, np.dtype]],
    max_events: int,
    seq_length: int,
    n_features: int,
    output_dtype_str: str = 'float16'
):
    """
    Processes events defined in the combined metadata table, loading pulse data
    batch by batch (file by file) and saving to memory-mapped files.
    """
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dtype_np = np.dtype(output_dtype_str)

    num_events_in_meta = len(metadata_table)
    num_events_to_process = min(num_events_in_meta, max_events)

    if num_events_to_process == 0:
        print("No events to process based on metadata and max_events limit.")
        return

    print(f"Will process {num_events_to_process} events (limited by max_events or available metadata).")
    
    # Slice the metadata table if max_events is less than total available
    if num_events_to_process < num_events_in_meta:
        metadata_table_sliced = metadata_table.slice(0, num_events_to_process)
    else:
        metadata_table_sliced = metadata_table
    
    # Define memmap properties based on the number of events to process
    memmap_props = {
        'x': {'shape': (num_events_to_process, seq_length, n_features), 'dtype': output_dtype_np.name},
        'l': {'shape': (num_events_to_process,), 'dtype': 'int32'},
        'c': {'shape': (num_events_to_process,), 'dtype': 'float32'}
    }
    structured_fields_y = [(name, dt) for name, dt in labels_to_save]
    structured_array_dtype_y = np.dtype(structured_fields_y)
    memmap_props['y'] = {
        'shape': (num_events_to_process,),
        'dtype': structured_array_dtype_y.descr
    }

    # Create memmap files
    print(f"Creating memory-mapped files in {output_dir} for {num_events_to_process} events...")
    x_memmap = np.memmap(output_dir / 'x.npy', mode='w+', dtype=output_dtype_np, shape=memmap_props['x']['shape'])
    l_memmap = np.memmap(output_dir / 'l.npy', mode='w+', dtype='int32', shape=memmap_props['l']['shape'])
    c_memmap = np.memmap(output_dir / 'c.npy', mode='w+', dtype='float32', shape=memmap_props['c']['shape'])
    y_memmap = np.memmap(output_dir / 'y.npy', mode='w+', dtype=structured_array_dtype_y, shape=memmap_props['y']['shape'])

    # Group metadata by batch_id (which corresponds to file number)
    # Convert sliced metadata to pandas for easier grouping
    meta_df = metadata_table_sliced.to_pandas()
    
    processed_event_count = 0
    
    # Iterate through each batch (file) represented in the metadata
    for batch_id, group_df in tqdm(meta_df.groupby('batch_id'), desc="Processing Batches"):
        
        pulse_file_path = base_data_path / 'pulses' / f'pulses_{batch_id}.parquet'
        if not pulse_file_path.exists():
            print(f"Warning: Pulse file not found for batch_id {batch_id}: {pulse_file_path}. Skipping batch.")
            continue
            
        try:
            # Load pulse data only for the current batch_id
            pulse_data_table = pq.read_table(pulse_file_path)
        except Exception as e:
            print(f"Warning: Error reading pulse file {pulse_file_path}: {e}. Skipping batch.")
            continue

        # Process each event within this batch
        for idx_in_memmap, (meta_row_index, meta_row) in enumerate(group_df.iterrows()):
            # meta_row_index is the original index from meta_df before grouping
            # We need the index within the overall memmap files (0 to num_events_to_process - 1)
            # This assumes the meta_df was created by slicing the full metadata table from the start.
            current_memmap_idx = processed_event_count # Use a running counter for the memmap index
            
            if current_memmap_idx >= num_events_to_process:
                 # This should not happen if slicing was done correctly, but as a safeguard
                 print("Warning: Exceeded max_events limit during processing. Stopping.")
                 break 

            first_pulse_idx = int(meta_row['first_pulse_index'])
            last_pulse_idx = int(meta_row['last_pulse_index'])
            num_pulses_for_event = (last_pulse_idx - first_pulse_idx + 1)

            if num_pulses_for_event <= 0:
                print(f"Warning: Event {meta_row['event_id']} (batch {batch_id}) has non-positive pulse count ({num_pulses_for_event}). Skipping.")
                # Fill memmap with zeros for this event index
                x_memmap[current_memmap_idx, :, :] = 0
                l_memmap[current_memmap_idx] = 0
                c_memmap[current_memmap_idx] = 0.0
                for label_name, _ in labels_to_save:
                     y_memmap[current_memmap_idx][label_name] = 0
                processed_event_count += 1
                continue

            # Slice the pulse data table loaded for this batch
            event_pulse_features_table = pulse_data_table.slice(offset=first_pulse_idx, length=num_pulses_for_event)
            
            x_evt, l_evt, c_evt = event_to_seq(
                event_pulse_features_table, sensor_mapping, seq_length, n_features, dtype_np=output_dtype_np
            )
            
            # Write to the correct index in the memmap files
            x_memmap[current_memmap_idx, :, :] = x_evt
            l_memmap[current_memmap_idx] = l_evt
            c_memmap[current_memmap_idx] = c_evt
            
            for label_name, numpy_dtype in labels_to_save:
                y_memmap[current_memmap_idx][label_name] = numpy_dtype(meta_row[label_name])
                
            processed_event_count += 1 # Increment the counter for the next memmap index

        if processed_event_count >= num_events_to_process:
             break # Stop processing batches if max_events reached

    # Flush memmaps
    del x_memmap, l_memmap, c_memmap, y_memmap

    # Save final properties
    with open(output_dir / 'memmap_properties.json', 'w') as f:
        # Ensure properties reflect the actual number processed
        final_props = memmap_props.copy()
        final_props['x']['shape'] = (processed_event_count, seq_length, n_features)
        final_props['l']['shape'] = (processed_event_count,)
        final_props['c']['shape'] = (processed_event_count,)
        final_props['y']['shape'] = (processed_event_count,)
        json.dump(final_props, f, indent=4)

    print(f"Successfully processed and saved {processed_event_count} events to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare ORCA data from multiple Parquet file pairs into memory-mapped format.")
    # --- Updated Arguments ---
    parser.add_argument('--base_path', type=str, required=True, help="Path to the base directory containing 'mc_truth' and 'pulses' subdirectories.")
    parser.add_argument('--file_range', type=int, nargs=2, required=True, help="Range of file numbers to process (e.g., 806 821).")
    parser.add_argument('--geometry_file', type=str, required=True, help="Path to the ORCA sensor geometry CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the memory-mapped files.")
    parser.add_argument('--max_events', type=int, default=1000000, help="Maximum number of events to process across all files.")
    parser.add_argument('--seq_length', type=int, default=DEFAULT_SEQ_LENGTH, help="Fixed sequence length for an event.")
    parser.add_argument('--dtype', type=str, default='float16', help="NumPy dtype for features (e.g., 'float16', 'float32').")
    parser.add_argument('--calibrate_sensor_mapping', action='store_true', help="Run sensor_mapping.initialize_mapping with Z-offset calibration using the first pulse file found.")
    # --- Removed Arguments ---
    # parser.add_argument('--truth_file', ...)
    # parser.add_argument('--pulse_file', ...)

    args = parser.parse_args()

    # Validate file range
    if args.file_range[0] > args.file_range[1]:
        raise ValueError(f"Invalid file range: Start ({args.file_range[0]}) cannot be greater than end ({args.file_range[1]}).")
    file_range_tuple = tuple(args.file_range)

    # Define the labels to extract from the ORCA truth file.
    labels_to_save_orca: List[tuple[str, np.dtype]] = [
        ('initial_state_azimuth', np.float32),
        ('initial_state_zenith', np.float32),
        ('initial_state_energy', np.float32),
        # ('interaction_type', np.int8), # Example
        # ('initial_state_type', np.int8), # Example
    ]
    print(f"Expecting the following labels in truth data: {[name for name, _ in labels_to_save_orca]}")

    print("Initializing SensorMapping...")
    sensor_mapper = SensorMapping(args.geometry_file)
    
    # Initialize KD-tree, optionally with Z-offset calibration using the *first* pulse file
    orca_metadata_handler = OrcaMetadata(Path(args.base_path), file_range_tuple)
    
    pulse_file_for_calibration = orca_metadata_handler.pulse_files[0] if args.calibrate_sensor_mapping and orca_metadata_handler.pulse_files else None
    try:
        sensor_mapper.initialize_mapping(pulse_data_file_path=pulse_file_for_calibration)
    except Exception as e:
        print(f"Error during SensorMapping initialization: {e}. Ensure geometry and pulse files are correct.")
        exit(1)

    print("Creating combined metadata from ORCA files...")
    combined_orca_metadata_table = orca_metadata_handler.create_metadata_table(labels_to_save_orca)

    if combined_orca_metadata_table is not None and len(combined_orca_metadata_table) > 0:
        process_orca_files_to_memmap( # Use the multi-file processing function
            output_dir_str=args.output_dir,
            base_data_path=Path(args.base_path), # Pass base path for loading pulse files
            metadata_table=combined_orca_metadata_table,
            sensor_mapping=sensor_mapper,
            labels_to_save=labels_to_save_orca,
            max_events=args.max_events,
            seq_length=args.seq_length,
            n_features=DEFAULT_N_FEATURES,
            output_dtype_str=args.dtype
        )
    else:
        print("No metadata created or no matched events found. Exiting.")
