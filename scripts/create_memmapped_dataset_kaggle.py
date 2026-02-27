import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pyarrow import csv
from tqdm import tqdm
import os
import json
from typing import Optional, Dict

#DATA_DIR = '/groups/pheno/inar/icecube_kaggle'
DATA_DIR = '/groups/hep/jlt/icecube_kaggle'
TRAIN_OR_TEST = 'train'
BATCH_SIZE = 200_000
N_DOMS = 5160
SEQ_LENGTH = 127
FEATURES_DTYPE = np.dtype([('time', np.float16), ('charge', np.float16), ('aux', np.float16), ('dom_id', np.uint16)])

METADATA = pq.ParquetFile(os.path.join(DATA_DIR, f'{TRAIN_OR_TEST}_meta.parquet'))

geometry_table = csv.read_csv(os.path.join(DATA_DIR, 'sensor_geometry.csv'))
DOM_POS = np.vstack(
    [geometry_table.column(col).to_numpy() for col in ['x', 'y', 'z']]
).T.copy()
del geometry_table

def event_to_seq(features):
    sensor_id = features.column('sensor_id').to_numpy()
    N_pulses = sensor_id.shape[0]
    charge = features.column('charge').to_numpy()
    time = features.column('time').to_numpy()
    auxiliary = features.column('auxiliary').to_numpy()

    assert np.all((sensor_id >= 0) & (sensor_id < N_DOMS)), f"Invalid sensor IDs found: {np.unique(sensor_id[~((sensor_id >= 0) & (sensor_id < N_DOMS))])}"
    
    if N_pulses > SEQ_LENGTH:
        # Sample and select pulses as per your current logic
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
    
    T_evt = np.zeros(SEQ_LENGTH, dtype=FEATURES_DTYPE)
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
    l_slice: np.memmap, c_slice: np.memmap
) -> None:
    assert(len(batch_meta['batch_id']) == BATCH_SIZE)
    for i in tqdm(range(BATCH_SIZE)):
        if y_slice is not None:
            y_slice[i, 0] = batch_meta['azimuth'][i]
            y_slice[i, 1] = batch_meta['zenith'][i]
        features = train_features.slice(
            offset=batch_meta['first_pulse_index'][i],
            length=batch_meta['last_pulse_index'][i] - batch_meta['first_pulse_index'][i] + 1
        )
        x, l, c = event_to_seq(features)
        x_slice[i, :] = x
        l_slice[i] = l
        c_slice[i] = c

def process_batches(
    output_dir: str,
    metadata: pq.ParquetFile,
    start_at_batch: int, stop_at_batch: int,
    *, include_truth: bool=True, target_dtype: np.dtype=np.float32
) -> None:
    N_batches = stop_at_batch - start_at_batch + 1
    N_events = N_batches * BATCH_SIZE
    
    memmap_props = {
        'x': {'shape': (N_events, SEQ_LENGTH), 'dtype': FEATURES_DTYPE.descr},
        'l': {'shape': (N_events,), 'dtype': np.dtype(np.int32).name},
        'c': {'shape': (N_events,), 'dtype': np.dtype(np.float32).name}
    }
    
    x = np.memmap(os.path.join(output_dir, 'x.npy'), mode='w+', **memmap_props['x'])
    l = np.memmap(os.path.join(output_dir, 'l.npy'), mode='w+', **memmap_props['l'])
    c = np.memmap(os.path.join(output_dir, 'c.npy'), mode='w+', **memmap_props['c'])
    
    if include_truth:
        memmap_props['y'] = {'shape': (N_events, 2), 'dtype': np.dtype(target_dtype).name}
        y = np.memmap(os.path.join(output_dir, 'y.npy'), mode='w+', **memmap_props['y'])
    
    for (i, batch_meta) in enumerate(metadata.iter_batches(batch_size=BATCH_SIZE)):
        batch_id = batch_meta[0][0].as_py()
        if batch_id < start_at_batch:
            continue
        if stop_at_batch is not None and batch_id > stop_at_batch:
            break
        
        batch_meta_dict = batch_meta.to_pydict()
        assert(batch_meta_dict['batch_id'][ 0] == batch_id)
        assert(batch_meta_dict['batch_id'][-1] == batch_id)
        assert(len(batch_meta_dict['batch_id']) == BATCH_SIZE)
        
        train_features = pq.read_table(os.path.join(DATA_DIR, TRAIN_OR_TEST, f'batch_{batch_id}.parquet'))

        adjusted_i = i - (start_at_batch - 1)
        batch_slice = slice(adjusted_i*BATCH_SIZE, (adjusted_i+1)*BATCH_SIZE)

        assert(batch_slice.stop <= N_events)
        process_kaggle_batch(
            batch_meta_dict, train_features,
            x[batch_slice, :], 
            y[batch_slice, :] if include_truth else None,
            l[batch_slice],
            c[batch_slice]
        )
    
    with open(os.path.join(output_dir, 'memmap_properties.json'), 'w+') as f:
        json.dump(memmap_props, f, indent=4)


if __name__ == '__main__':
    OUTPUT_DIR = '/groups/hep/jlt/icecube_kaggle-v2/memmapped_train_130M_127'
    #OUTPUT_DIR = '/groups/hep/jlt/icecube_kaggle-v2/memmapped_eval_1.2M_127'
    #OUTPUT_DIR = '/groups/hep/jlt/icecube_kaggle-v2/memmapped_test_0.6M_127'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_batches(
        OUTPUT_DIR, METADATA,
        start_at_batch=1, stop_at_batch=650,
        # start_at_batch=651, stop_at_batch=656,
        # start_at_batch=657, stop_at_batch=659,
        include_truth=(TRAIN_OR_TEST == 'train'),
        target_dtype=np.float16
    )