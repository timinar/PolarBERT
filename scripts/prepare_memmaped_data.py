import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
# import pyarrow.compute as pc
from tqdm import tqdm
import os
import json
import yaml
import argparse
from typing import Optional, Dict

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def event_to_seq(features: pa.Table, seq_length: int, n_features: int, dtype=np.float32) -> tuple:
    sensor_id = features.column('sensor_id').to_numpy()
    N_pulses = sensor_id.shape[0]
    charge = features.column('charge').to_numpy()
    time = features.column('time').to_numpy()
    auxiliary = features.column('auxiliary').to_numpy()
    
    if N_pulses > seq_length:
        naux_idx = np.where(auxiliary == 0)[0]
        aux_idx = np.where(auxiliary == 1)[0]
        if len(naux_idx) < seq_length:
            max_length_possible = min(seq_length, N_pulses)
            num_to_sample = max_length_possible - len(naux_idx)
            aux_idx_sample = np.random.choice(aux_idx, size=num_to_sample, replace=False)
            selected_idx = np.concatenate((naux_idx, aux_idx_sample))
        else:
            selected_idx = np.random.choice(naux_idx, size=seq_length, replace=False)
        selected_idx = np.sort(selected_idx)
    else:
        selected_idx = range(N_pulses)
    
    T_evt = np.zeros((seq_length, n_features), dtype=dtype)
    T_evt[:len(selected_idx), 0] = (time[selected_idx] - 1e4) / 3e4
    T_evt[:len(selected_idx), 1] = np.log10(charge[selected_idx]) / 3.0
    T_evt[:len(selected_idx), 2] = auxiliary[selected_idx] - 0.5 # aux = True is BAD, so 0.5 is bad
    T_evt[:len(selected_idx), 3] = sensor_id[selected_idx] + 1  # +1 is needed since we use 0 for padding
    
    event_length = len(selected_idx)
    total_charge = charge.sum()
    
    return T_evt, event_length, total_charge

def process_kaggle_batch(
    batch_meta: Dict, train_features: pa.Table,
    x_slice: np.memmap, y_slice: Optional[np.memmap],
    l_slice: np.memmap, c_slice: np.memmap,
    seq_length: int, n_features: int, batch_size: int,
    dtype: np.dtype=np.float32
) -> None:
    assert(len(batch_meta['batch_id']) == batch_size)
    for i in tqdm(range(batch_size)):
        if y_slice is not None:
            y_slice[i, 0] = batch_meta['azimuth'][i]
            y_slice[i, 1] = batch_meta['zenith'][i]
        features = train_features.slice(
            offset=batch_meta['first_pulse_index'][i],
            length=batch_meta['last_pulse_index'][i] - batch_meta['first_pulse_index'][i] + 1
        )
        x, l, c = event_to_seq(features, seq_length, n_features, dtype=dtype)
        x_slice[i, :, :] = x
        l_slice[i] = l
        c_slice[i] = c

def process_batches(
    config: Dict,
    metadata: pq.ParquetFile,
    dataset_type: str
) -> None:
    output_dir = os.path.join(config['paths']['output_dir'], config[dataset_type]['directory_name'])
    data_dir = config['paths']['data_dir']
    batch_size = config['data']['batch_size']
    seq_length = config['data']['seq_length']
    n_features = config['data']['n_features']
    start_at_batch = config[dataset_type]['start_at_batch']
    stop_at_batch = config[dataset_type]['stop_at_batch']
    include_truth = config[dataset_type]['include_truth']
    dtype = np.dtype(config[dataset_type]['dtype'])

    N_batches = stop_at_batch - start_at_batch + 1
    N_events = N_batches * batch_size
    
    memmap_props = {
        'x': {'shape': (N_events, seq_length, n_features), 'dtype': np.dtype(dtype).name},
        'l': {'shape': (N_events,), 'dtype': np.dtype(np.int32).name},
        'c': {'shape': (N_events,), 'dtype': np.dtype(np.float32).name}
    }
    
    x = np.memmap(os.path.join(output_dir, 'x.npy'), mode='w+', **memmap_props['x'])
    l = np.memmap(os.path.join(output_dir, 'l.npy'), mode='w+', **memmap_props['l'])
    c = np.memmap(os.path.join(output_dir, 'c.npy'), mode='w+', **memmap_props['c'])
    
    if include_truth:
        memmap_props['y'] = {'shape': (N_events, 2), 'dtype': np.dtype(dtype).name}
        y = np.memmap(os.path.join(output_dir, 'y.npy'), mode='w+', **memmap_props['y'])
    
    for (i, batch_meta) in enumerate(metadata.iter_batches(batch_size=batch_size)):
        batch_id = batch_meta[0][0].as_py()
        if batch_id < start_at_batch:
            continue
        if stop_at_batch is not None and batch_id > stop_at_batch:
            break
        
        batch_meta_dict = batch_meta.to_pydict()
        assert(batch_meta_dict['batch_id'][0] == batch_id)
        assert(batch_meta_dict['batch_id'][-1] == batch_id)
        assert(len(batch_meta_dict['batch_id']) == batch_size)
        
        train_features = pq.read_table(os.path.join(data_dir, 'train', f'batch_{batch_id}.parquet'))

        adjusted_i = i - (start_at_batch - 1)
        batch_slice = slice(adjusted_i*batch_size, (adjusted_i+1)*batch_size)

        assert(batch_slice.stop <= N_events)
        process_kaggle_batch(
            batch_meta_dict, train_features,
            x[batch_slice, :, :], 
            y[batch_slice, :] if include_truth else None,
            l[batch_slice],
            c[batch_slice],
            seq_length, n_features, batch_size,
            dtype=dtype
        )
    
    with open(os.path.join(output_dir, 'memmap_properties.json'), 'w+') as f:
        json.dump(memmap_props, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process kaggle IceCube data using configuration from a YAML file.')
    parser.add_argument('--config_path', type=str, help='Path to the configuration YAML file', default='../configs/prepare_datasets.yaml')
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    for dataset_type in ['pretraining', 'validation', 'finetuning']:
        os.makedirs(os.path.join(config['paths']['output_dir'], config[dataset_type]['directory_name']), exist_ok=True)
        metadata = pq.ParquetFile(os.path.join(config['paths']['data_dir'], "train_meta.parquet"))
        print(f"Processing {dataset_type} dataset")
        process_batches(config, metadata, dataset_type)