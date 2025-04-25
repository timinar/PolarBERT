import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from pathlib import Path
import json
from typing import Optional, Dict, List
from prometheus.sensor_mapping import SensorMapping
import re

BATCH_SIZE = 25_000  # Adjusted to match events per parquet file
SEQ_LENGTH = 127
N_FEATURES = 4  # time, charge, aux, dom_id

class DataSource:
    PULSES = 'pulses'
    PULSES_NO_NOISE = 'pulses_no_noise'
    MERGED_PHOTONS = 'merged_photons'

    @staticmethod
    def get_file_range(source: str) -> tuple[int, int]:
        # v1
        # ranges = {
        #     DataSource.PULSES: (344, 359),
        #     DataSource.PULSES_NO_NOISE: (240, 359),
        #     DataSource.MERGED_PHOTONS: (0, 239)
        # }
        # v2
        ranges = {
            DataSource.PULSES: (329, 344),
            DataSource.PULSES_NO_NOISE: (225, 344),
            DataSource.MERGED_PHOTONS: (0, 224)
        }
        return ranges.get(source)

class PrometheusMetadata:
    def __init__(self, base_path: Path, data_source: str, file_range: tuple[int, int] = None):
        """
        Initialize metadata processor
        
        Args:
            base_path: Base path to prometheus data
            data_source: One of 'pulses', 'pulses_no_noise', 'merged_photons'
            file_range: Optional tuple of (start_file_num, end_file_num) inclusive
        """
        self.base_path = Path(base_path)
        self.data_source = data_source
        
        if file_range is None:
            file_range = DataSource.get_file_range(data_source)
            if file_range is None:
                raise ValueError(f"Invalid data source: {data_source}")
        
        # Get all files and filter by number range
        all_data_files = sorted(self.base_path.glob(f'{data_source}/{data_source}_*.parquet'))
        all_truth_files = sorted(self.base_path.glob('mc_truth/mc_truth_*.parquet'))
        
        start_num, end_num = file_range
        def extract_num(path):
            return int(re.search(r'\d+', path.name).group())
        
        self.data_files = [
            f for f in all_data_files 
            if start_num <= extract_num(f) <= end_num
        ]
        # Match truth files to data files
        data_nums = {extract_num(f) for f in self.data_files}
        self.truth_files = [
            f for f in all_truth_files 
            if extract_num(f) in data_nums
        ]
        
        print(f"Processing {len(self.data_files)} {data_source} files")
        print(f"First file: {self.data_files[0].name}")
        print(f"Last file: {self.data_files[-1].name}")
        
    def create_metadata(self, labels: List[tuple[str, np.dtype]]) -> pa.Table:
        """Create a metadata table similar to Kaggle's format"""
        all_meta = []
        batch_id = 1
        
        for truth_file, data_file in tqdm(zip(self.truth_files, self.data_files), 
                                         desc="Processing files"):
            # Read truth and data
            truth = pq.read_table(truth_file)
            data = pq.read_table(data_file)
            
            # Group data by event_no using pyarrow
            event_counts = pc.value_counts(data['event_no'])
            # Convert StructArray to arrays we can work with
            unique_events = event_counts.field('values').to_numpy()
            counts = event_counts.field('counts').to_numpy()
            
            # Calculate cumulative sums for data indices
            cumsum = np.cumsum(counts)
            first_indices = np.concatenate([[0], cumsum[:-1]])
            
            # Create metadata entries
            meta_batch = {
                'batch_id': [],
                'event_id': [],
                'first_pulse_index': [],
                'last_pulse_index': []
            }
            # Only keep label names for arrow reading
            label_names = [lbl[0] for lbl in labels]
            for name in label_names:
                meta_batch[name] = []
            
            # Create index lookup for truth data
            truth_event_nos = truth.column('event_no').to_numpy()
            truth_cols = {name: truth.column(name).to_numpy() for name in label_names}
            truth_lookup = {}
            for i, evt in enumerate(truth_event_nos):
                truth_lookup[evt] = {}
                for name in label_names:
                    truth_lookup[evt][name] = truth_cols[name][i]
            
            for event_no, first_idx, last_idx in zip(unique_events, first_indices, cumsum):
                if event_no in truth_lookup:
                    meta_batch['batch_id'].append(batch_id)
                    meta_batch['event_id'].append(event_no)
                    meta_batch['first_pulse_index'].append(first_idx)
                    meta_batch['last_pulse_index'].append(last_idx - 1)
                    for name in label_names:
                        meta_batch[name].append(truth_lookup[event_no][name])
            
            all_meta.append(pa.Table.from_pydict(meta_batch))
            batch_id += 1
            
        return pa.concat_tables(all_meta)

def event_to_seq(features: pa.Table, sensor_mapping: SensorMapping, dtype=np.float32):
    coords = np.stack([
        features.column('sensor_pos_x').to_numpy(),
        features.column('sensor_pos_y').to_numpy(),
        features.column('sensor_pos_z').to_numpy(),
    ], axis=1)
    
    sensor_id = sensor_mapping.coords_to_sensor_ids(coords)
    N_pulses = sensor_id.shape[0]
    charge = features.column('charge').to_numpy()
    time = features.column('t').to_numpy()
    auxiliary = 1 - features.column('is_signal').to_numpy()  # inverse of is_signal
    
    if N_pulses > SEQ_LENGTH:
        # Sample and select pulses
        naux_idx = np.where(auxiliary == 0)[0]
        aux_idx = np.where(auxiliary == 1)[0]
        if len(naux_idx) < SEQ_LENGTH:
            max_length_possible = min(SEQ_LENGTH, N_pulses)
            num_to_sample = max_length_possible - len(naux_idx)
            aux_idx_sample = np.random.choice(aux_idx, size=num_to_sample, replace=False)
            selected_idx = np.concatenate((naux_idx, aux_idx_sample))
        else:
            selected_idx = np.random.choice(naux_idx, size=SEQ_LENGTH, replace=False)
        selected_idx = np.sort(selected_idx)
    else:
        selected_idx = range(N_pulses)
    
    T_evt = np.zeros((SEQ_LENGTH, N_FEATURES), dtype=dtype)
    # Scale time similarly to original data
    T_evt[:len(selected_idx), 0] = (time[selected_idx] - 1e4) / 3e4
    T_evt[:len(selected_idx), 1] = np.log10(charge[selected_idx]) / 3.0
    T_evt[:len(selected_idx), 2] = auxiliary[selected_idx] - 0.5
    T_evt[:len(selected_idx), 3] = sensor_id[selected_idx] + 1
    
    event_length = len(selected_idx)
    total_charge = charge.sum()
    
    return T_evt, event_length, total_charge

def process_prometheus_batch(
    batch_meta: Dict, 
    data_file: Path,
    batch_start_idx: int,
    x_slice: np.memmap, 
    y_slice: Optional[np.memmap],
    l_slice: np.memmap, 
    c_slice: np.memmap,
    sensor_mapping: SensorMapping,
    labels: List[tuple[str, np.dtype]],
    dtype: np.dtype=np.float32
) -> None:
    data = pq.read_table(data_file)
    for i in tqdm(range(len(batch_meta['event_id']))):
        if y_slice is not None:
            # Store each label in y, converting to the specified dtype
            for name, dt in labels:
                y_slice[batch_start_idx + i][name] = dt(batch_meta[name][i])
        
        features = data.slice(
            offset=batch_meta['first_pulse_index'][i],
            length=batch_meta['last_pulse_index'][i] - batch_meta['first_pulse_index'][i] + 1
        )
        x, l, c = event_to_seq(features, sensor_mapping, dtype=dtype)
        x_slice[batch_start_idx + i, :, :] = x
        l_slice[batch_start_idx + i] = l
        c_slice[batch_start_idx + i] = c

def process_batches(
    output_dir: str,
    base_path: Path,
    metadata_table: pa.Table,
    sensor_mapping: SensorMapping,
    *, labels: List[tuple[str, np.dtype]],
    include_truth: bool=True,
    dtype: np.dtype=np.float32
) -> None:
    N_events = len(metadata_table)
    print(f"Processing {N_events} events")
    
    # Setup memmap files
    memmap_props = {
        'x': {'shape': (N_events, SEQ_LENGTH, N_FEATURES), 'dtype': np.dtype(dtype).name},
        'l': {'shape': (N_events,), 'dtype': np.dtype(np.int32).name},
        'c': {'shape': (N_events,), 'dtype': np.dtype(np.float32).name}
    }
    
    if include_truth:
        structured_fields = [(name, dt) for (name, dt) in labels]
        structured_array_dtype = np.dtype(structured_fields)
        memmap_props['y'] = {
            'shape': (N_events,),
            'dtype': structured_array_dtype.descr
        }
    
    # Create memory mapped files
    x = np.memmap(os.path.join(output_dir, 'x.npy'), mode='w+', **memmap_props['x'])
    l = np.memmap(os.path.join(output_dir, 'l.npy'), mode='w+', **memmap_props['l'])
    c = np.memmap(os.path.join(output_dir, 'c.npy'), mode='w+', **memmap_props['c'])
    if include_truth:
        y = np.memmap(
            os.path.join(output_dir, 'y.npy'),
            mode='w+',
            shape=memmap_props['y']['shape'],
            dtype=structured_array_dtype
        )
    else:
        y = None
    
    # Convert metadata to dictionary for easier processing
    meta_dict = metadata_table.to_pydict()
    
    # Create a mapping from batch_id to actual data file paths
    data_files = sorted(base_path.glob(f'{data_source}/{data_source}_*.parquet'))
    data_file_lookup = {
        i+1: f for i, f in enumerate(data_files)
    }
    
    # Group events by batch_id
    unique_batch_ids = sorted(set(meta_dict['batch_id']))
    current_idx = 0
    
    for batch_id in tqdm(unique_batch_ids, desc="Processing batches"):
        # Get indices for current batch
        batch_mask = [i for i, bid in enumerate(meta_dict['batch_id']) if bid == batch_id]
        
        # Create batch metadata
        batch_meta = {
            k: [meta_dict[k][i] for i in batch_mask] for k in ['event_id','first_pulse_index','last_pulse_index']
        }
        for name, _ in labels:
            batch_meta[name] = [meta_dict[name][i] for i in batch_mask]
        
        # Get the actual data file path
        data_file = data_file_lookup[batch_id]
        
        # Process batch
        process_prometheus_batch(
            batch_meta,
            data_file,  # Use the actual file path instead of constructing it
            current_idx,
            x, y, l, c,
            sensor_mapping,
            labels=labels,
            dtype=dtype
        )
        
        current_idx += len(batch_mask)
    
    # Save memmap properties
    with open(os.path.join(output_dir, 'memmap_properties.json'), 'w+') as f:
        json.dump(memmap_props, f, indent=4)

if __name__ == '__main__':
    # Use home directory instead of hardcoded paths
    HOME = Path.home()
    BASE_PATH = HOME / 'prometheus_data2'
    
    # Process each data source
    for data_source in [DataSource.PULSES, DataSource.PULSES_NO_NOISE, DataSource.MERGED_PHOTONS]:
        OUTPUT_DIR = HOME / f'prometheus_data_updated/memmaped_{data_source}'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize sensor mapping
        sensor_mapping = SensorMapping(HOME / 'icecube_kaggle/sensor_geometry.csv')
        
        # Use first file of the source for z-offset calibration
        first_file = next(BASE_PATH.glob(f'{data_source}/{data_source}_*.parquet'))
        sensor_mapping.initialize_mapping(first_file)
        
        # Create metadata for the data source
        labels_to_save = [
            ('initial_state_azimuth', np.float16),
            ('initial_state_zenith', np.float16),
            ('initial_state_energy', np.float32),
            ('interaction', np.int8),
            ('initial_state_type', np.int8)
        ]
        metadata = PrometheusMetadata(BASE_PATH, data_source)
        meta_table = metadata.create_metadata(labels_to_save)
        
        # Save metadata
        output_name = f'{data_source}_meta_{metadata.data_files[0].stem[-3:]}_{metadata.data_files[-1].stem[-3:]}.parquet'
        pq.write_table(meta_table, os.path.join(OUTPUT_DIR, output_name))
        
        # Process batches
        process_batches(
            OUTPUT_DIR,
            BASE_PATH,
            meta_table,
            sensor_mapping,
            labels=labels_to_save,
            include_truth=True,
            dtype=np.float16
        )