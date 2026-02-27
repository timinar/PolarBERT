import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pyarrow import csv
from tqdm import tqdm
import os
import json
import yaml
import argparse
from typing import Optional, Dict

FEATURES_DTYPE = np.dtype([('time', np.float16), ('charge', np.float16), ('aux', np.float16), ('dom_id', np.uint16)])


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sensor_geometry(sensor_geometry_path: str) -> np.ndarray:
    geometry_table = csv.read_csv(sensor_geometry_path)
    dom_pos = np.vstack(
        [geometry_table.column(col).to_numpy() for col in ['x', 'y', 'z']]
    ).T.copy()
    return dom_pos


def event_to_seq(features, seq_length: int, n_doms: int):
    sensor_id = features.column('sensor_id').to_numpy()
    N_pulses = sensor_id.shape[0]
    charge = features.column('charge').to_numpy()
    time = features.column('time').to_numpy()
    auxiliary = features.column('auxiliary').to_numpy()

    assert np.all((sensor_id >= 0) & (sensor_id < n_doms)), f"Invalid sensor IDs found: {np.unique(sensor_id[~((sensor_id >= 0) & (sensor_id < n_doms))])}"

    if N_pulses > seq_length:
        # Sample and select pulses as per your current logic
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

    T_evt = np.zeros(seq_length, dtype=FEATURES_DTYPE)
    T_evt['time'  ][:len(selected_idx)] = (time[selected_idx] - 1e4) / 3e4
    T_evt['charge'][:len(selected_idx)] = np.log10(charge[selected_idx]) / 3.0
    T_evt['aux'   ][:len(selected_idx)] = auxiliary[selected_idx] - 0.5
    T_evt['dom_id'][:len(selected_idx)] = sensor_id[selected_idx] + 1

    event_length = len(selected_idx)
    total_charge = charge.sum()

    return T_evt, event_length, total_charge


def process_kaggle_batch(
    batch_meta: Dict, train_features: pa.Table,
    x_slice: np.memmap, y_slice: Optional[np.memmap],
    l_slice: np.memmap, c_slice: np.memmap,
    batch_size: int, seq_length: int, n_doms: int
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
        x, l, c = event_to_seq(features, seq_length, n_doms)
        x_slice[i, :] = x
        l_slice[i] = l
        c_slice[i] = c


def process_batches(
    config: dict, split_name: str, output_dir: str
) -> None:
    paths = config['paths']
    data = config['data']
    split = config['splits'][split_name]

    data_dir = paths['data_dir']
    batch_size = data['batch_size']
    seq_length = data['seq_length']
    n_doms = data['n_doms']

    start_at_batch = split['start_at_batch']
    stop_at_batch = split['stop_at_batch']
    include_truth = split['include_truth']
    target_dtype = np.dtype(split['target_dtype'])

    train_or_test = 'train' if include_truth else 'test'
    metadata = pq.ParquetFile(os.path.join(data_dir, f'{train_or_test}_meta.parquet'))

    N_batches = stop_at_batch - start_at_batch + 1
    N_events = N_batches * batch_size

    memmap_props = {
        'x': {'shape': (N_events, seq_length), 'dtype': FEATURES_DTYPE.descr},
        'l': {'shape': (N_events,), 'dtype': np.dtype(np.int32).name},
        'c': {'shape': (N_events,), 'dtype': np.dtype(np.float32).name}
    }

    x = np.memmap(os.path.join(output_dir, 'x.npy'), mode='w+', **memmap_props['x'])
    l = np.memmap(os.path.join(output_dir, 'l.npy'), mode='w+', **memmap_props['l'])
    c = np.memmap(os.path.join(output_dir, 'c.npy'), mode='w+', **memmap_props['c'])

    if include_truth:
        memmap_props['y'] = {'shape': (N_events, 2), 'dtype': target_dtype.name}
        y = np.memmap(os.path.join(output_dir, 'y.npy'), mode='w+', **memmap_props['y'])

    for (i, batch_meta) in enumerate(metadata.iter_batches(batch_size=batch_size)):
        batch_id = batch_meta[0][0].as_py()
        if batch_id < start_at_batch:
            continue
        if stop_at_batch is not None and batch_id > stop_at_batch:
            break

        batch_meta_dict = batch_meta.to_pydict()
        assert(batch_meta_dict['batch_id'][ 0] == batch_id)
        assert(batch_meta_dict['batch_id'][-1] == batch_id)
        assert(len(batch_meta_dict['batch_id']) == batch_size)

        train_features = pq.read_table(os.path.join(data_dir, train_or_test, f'batch_{batch_id}.parquet'))

        adjusted_i = i - (start_at_batch - 1)
        batch_slice = slice(adjusted_i*batch_size, (adjusted_i+1)*batch_size)

        assert(batch_slice.stop <= N_events)
        process_kaggle_batch(
            batch_meta_dict, train_features,
            x[batch_slice, :],
            y[batch_slice, :] if include_truth else None,
            l[batch_slice],
            c[batch_slice],
            batch_size, seq_length, n_doms
        )

    with open(os.path.join(output_dir, 'memmap_properties.json'), 'w+') as f:
        json.dump(memmap_props, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create memory-mapped datasets from IceCube Kaggle data')
    parser.add_argument('--config', type=str, required=True, help='Path to prepare_datasets.yaml')
    parser.add_argument('--split', type=str, default=None, help='Process only this split (default: all)')
    args = parser.parse_args()

    config = load_config(args.config)
    splits = [args.split] if args.split else list(config['splits'].keys())

    for split_name in splits:
        print(f"Processing {split_name} split...")
        output_dir = os.path.join(config['paths']['output_dir'], split_name)
        os.makedirs(output_dir, exist_ok=True)
        process_batches(config, split_name, output_dir)
