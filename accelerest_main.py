import argparse
import os
import json
import subprocess
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, SequentialSampler

from accelerest.datasets.sleep_dataset import AccelerometryDataset
from accelerest.input import (
    SUPPORTED_FILE_TYPES,
    find_input_files,
    load_accelerometry,
    output_stem,
)

def parse_args():
    parser = argparse.ArgumentParser()
    # IO parameters
    parser.add_argument('--data_file_dir', type=str, required = True,
                        help='Path to folder with raw accelerometry data files to run AcceleRest on.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to folder to save accelerest outputs in.'
                             'Defaults to <data_file_dir>/accelerest_outputs')
    parser.add_argument('--file_type', type=str, default='auto', choices=SUPPORTED_FILE_TYPES,
                        help='Input type. auto finds h5, cwa, cwa.gz, gt3x, gt3x.gz, csv, and csv.gz.')

    parser.add_argument('--save_preprocessed', action='store_true',
                        help='Whether to save the preprocessed data.')
    parser.add_argument('--no_detect_nonwear', action='store_true',
                        help='Do not mark non-wear periods using actipy (enabled by default).')
    
    # Select which predictions to return
    parser.add_argument('--lstm_sleepstages', action='store_true',
                        help='Return sleep stages predicted with an LSTM-C head.')
    parser.add_argument('--linear_sleepstages', action='store_true',
                        help='Return sleep stages predicted with a linear head.')
    parser.add_argument('--linear_respevents', action='store_true',
                        help='Return respiratory events predicted with a linear head.')

    # Processing parameters
    parser.add_argument('--context_window_shift', type=int, default=1,
                        help='The number of 30 sec patches to shift consecutive windows.')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Batch size for inference.')

    # Select intermediate outputs to save
    parser.add_argument('--get_embeddings', action='store_true',
                        help='Return model embeddings.')
    parser.add_argument('--window_wise_predictions', action='store_true',
                        help='Return predictions for each context window.')
    args = parser.parse_args()

    # Dynamically set default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_file_dir, "accelerest_outputs")

    return args

def outputs_exist(output_dir: str, args):
    prefixes = []
    if args.lstm_sleepstages:
        prefixes.append('lstm_sleepstages')
    if args.linear_sleepstages:
        prefixes.append('linear_sleepstages')
    if args.linear_respevents:
        prefixes.append('linear_respevents')

    prefix_preds_exist = []
    for prefix in prefixes:
        soft_preds_exists = os.path.isfile(
            os.path.join(output_dir, f'{prefix}_soft_preds.npy')
        )

        if args.window_wise_predictions:
            window_wise_preds_exists = os.path.isfile(
                os.path.join(output_dir, f"{prefix}_window_wise_logits.dat")
            )
            window_wise_preds_meta_exists = os.path.isfile(
                os.path.join(output_dir, f"{prefix}_window_wise_logits_meta.npy")
            )
            prefix_preds_exist.append(
                soft_preds_exists and
                window_wise_preds_exists and
                window_wise_preds_meta_exists
            )
        
        else:
            prefix_preds_exist.append(soft_preds_exists)

    return all(prefix_preds_exist)

def eval(args, device):
    model = torch.hub.load(
        'NielsRLorenzen/AcceleRest',
        'accelerest_multihead',
        linear_sleepstage = args.linear_sleepstages,
        lstm_sleepstage = args.lstm_sleepstages,
        linear_respevent = args.linear_respevents,
        trust_repo='check',
        force_reload = True,
    )

    model.to(device)
    model.eval()

    # Get files
    print(f'Searching for {args.file_type} input files')
    files = find_input_files(args.data_file_dir, args.file_type)

    print(f'Processing {len(files)} files.')

    # Make overall output dir
    os.makedirs(args.output_dir, exist_ok = True)

    stems = [output_stem(file) for file in files]
    duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
    for i, file in enumerate(files):
        # Make individual output dirs if they don't exist
        individual_output_dir = os.path.join(
            args.output_dir,
            output_stem(file) if output_stem(file) not in duplicate_stems else os.path.basename(file),
        )
        if os.path.exists(individual_output_dir):
            # Check if all output files exist
            if outputs_exist(individual_output_dir, args):
                continue
        
        else:
            os.makedirs(individual_output_dir)

        eval_single(file, model, device, individual_output_dir, args)


def init_storage(output_dir, prefix, num_windows, num_patches, window_size, num_classes, args):
    storage = {
        "sum_logits": torch.zeros((num_patches, num_classes), dtype=torch.float32),
        "position_count": torch.zeros((num_patches, 1), dtype=torch.float32),
        "memmap": None,
        "shape": (num_windows, window_size, num_classes),
    }

    if args.window_wise_predictions:
        storage["memmap"] = np.memmap(
            os.path.join(output_dir, f"{prefix}_window_wise_logits.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(num_windows, window_size, num_classes),
        )

    return storage

def store_outputs(storage: dict, y_hat: torch.Tensor, batch_start_patch: int):
    batch_size, window_size, num_classes = y_hat.shape
 
    # Fill slice of memmap corresponding to batch (optional)
    if storage["memmap"] is not None:
        storage["memmap"][batch_start_patch // storage["step_patches"]: batch_start_patch // storage["step_patches"] + batch_size, :, :] = y_hat.numpy()
 
    for i in range(batch_size):
        window_start = batch_start_patch + i * storage["step_patches"]
        # Mask for valid (non-nan) logits
        valid_mask = ~torch.isnan(y_hat[i])
        # Fill in predictions summing patch logits across context windows
        # Only add valid logits
        storage["sum_logits"][window_start: window_start + window_size][valid_mask] += y_hat[i][valid_mask]
        # Store number of context windows for each patch (Used to avg. later)
        # Increment position_count only for valid positions
        storage["position_count"][window_start: window_start + window_size][valid_mask.any(dim=-1, keepdim=True)] += 1
 
def finalize_head_storage(storage, output_dir, prefix):
    # Save memmap and meta data for reading
    if storage["memmap"] is not None:
        storage["memmap"].flush()
        meta = {
            "shape": list(storage["shape"]),
            "dtype": "float32",
        }
        np.save(
            os.path.join(output_dir, f"{prefix}_window_wise_logits_meta.npy"),
            meta,
        )

    # Save the soft predictions based on cross-context window average logits
    avg_logits = storage["sum_logits"].div(
        storage["position_count"].clamp_min(1)
    )
    soft_preds = avg_logits.softmax(dim=-1)
    # With context_window_shift > 1 some epochs are deliberately not visited.
    # Preserve that fact instead of silently returning a uniform probability.
    soft_preds[storage["position_count"].squeeze(-1) == 0] = torch.nan
    np.save(
        os.path.join(output_dir, f"{prefix}_soft_preds.npy"),
        soft_preds.numpy(),
    )

def eval_single(file, model, device, output_dir, args):
    try:
        data_array, timestamps, processing_info = load_accelerometry(
            file, args.file_type, detect_nonwear=not args.no_detect_nonwear,
        )
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f"[WARNING] Cannot use {file}: {error}")
        return

    with open(os.path.join(output_dir, "input_processing.json"), "w") as handle:
        json.dump(processing_info, handle, indent=2, default=str)
    if args.save_preprocessed and timestamps is not None:
        preprocessed = pd.DataFrame(data_array.T, columns=["x", "y", "z"], index=timestamps)
        preprocessed.index.name = "time"
        preprocessed.to_csv(os.path.join(output_dir, "preprocessed_30hz.csv"))
    try:
        subject_dataset = AccelerometryDataset(
            accelerometry=data_array,
            patch_size_samples=model.patch_size,
            context_window_patches=model.max_seq_len,
            step_patches=args.context_window_shift,
        )
    except RuntimeError as e:
        if "Number of samples" in str(e):
            print(f"[WARNING] Caught error: {str(e)} for file {file}, skipping...")
            return
        raise

    loader = DataLoader(
        subject_dataset,
        batch_size=args.max_batch_size,
        sampler=SequentialSampler(subject_dataset),
        shuffle=False,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )
    
    num_windows = len(subject_dataset)
    window_size = model.max_seq_len
    num_patches = (num_windows - 1) * args.context_window_shift + window_size
    if timestamps is not None:
        patch_starts = timestamps[np.arange(num_patches) * model.patch_size]
        pd.DataFrame({"epoch_start": patch_starts}).to_csv(
            os.path.join(output_dir, "epoch_start_times.csv"), index=False,
        )
    
    print(f'Processing file: {os.path.basename(file)}')
    print(f'Number of windows: {num_windows}')

    storage = {}
    
    with torch.no_grad():
        for batch_idx, x in enumerate(loader): 
            batch_start_idx = batch_idx * args.max_batch_size

            x = x.to(device)
            outputs = model(x)

            for name, logits in outputs.items():
                if name not in storage.keys():
                    # Initialize output storage for each prediction head
                    _, _, num_classes = logits.shape
                    storage[name] = init_storage(
                        output_dir, name, num_windows, num_patches, window_size, num_classes, args,
                    )   
                    storage[name]["step_patches"] = args.context_window_shift
                logits = logits.cpu()
                store_outputs(storage[name], logits, batch_start_idx * args.context_window_shift)

    for name in storage.keys():
        finalize_head_storage(storage[name], output_dir, name)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    eval(args, device)

if __name__ == '__main__':
    args = parse_args()
    main(args)
