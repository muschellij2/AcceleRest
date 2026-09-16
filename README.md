# AcceleRest

## Getting started
### Step 1: Clone this repo
Change your working directory to where you want to clone the repo:
```
cd /path/to/repo/
```
and paste the following command:
```
git clone https://github.com/NielsRLorenzen/AcceleRest.git
```

### Step 2: Input files and preprocessing

AcceleRest now automatically accepts one recording per file in any of these formats:

| Input | Handling |
| --- | --- |
| `.cwa`, `.cwa.gz` | Decoded by ActiPy's Axivity reader, then low-pass filtered, gravity-calibrated, non-wear flagged, and resampled to 30 Hz. |
| `.gt3x`, `.gt3x.gz` | Decoded by ActiPy's ActiGraph reader using the same processing pipeline. Compressed files are passed directly to ActiPy; do not unpack them first. |
| `.csv`, `.csv.gz` | Must contain a timestamp (`time`, `timestamp`, `datetime`, or `date`) and `x`, `y`, `z` columns (common `acc_x/y/z`, `acceleration_x/y/z`, and `accelerometer_x/y/z` aliases also work). Gzip-compressed CSV is read directly by pandas and is processed with ActiPy at 30 Hz. |
| `.h5` | Backward-compatible model-ready input: `data/accelerometry` must be a `(3, n_samples)` array already at 30 Hz. |

CSV axes must be in **g**, not raw device counts, mg, or m/s². AcceleRest rejects CSV/raw data sampled below 30 Hz (upsampling cannot restore the signal the model needs), missing/duplicate timestamps, all-missing data, implausible gravity scale, missing axes, and recordings shorter than the model context. The full context is 256 30-second patches: **2 hours 8 minutes** of usable 30-Hz data. ActiPy writes calibration diagnostics to each output's `input_processing.json`; a calibration diagnostic of `CalibOK: 0` can occur for short recordings with too few stationary windows, even when the axes are already device-calibrated.

The command line defaults to `--file_type auto`, so it scans all formats above. Use `--file_type gt3x.gz`, for example, to restrict a batch. `--no_detect_nonwear` retains data instead of using ActiPy's non-wear flagger. Use `--save_preprocessed` to save `preprocessed_30hz.csv` alongside outputs for timestamped inputs.

After processing, the following code should result in the number of samples being printed:

```
with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
    data = f['data/accelerometry']
    n_samples = data.shape[1]
    print(n_samples)
```

### Step 3: Running AcceleRest
Use this commandline prompt and edit the paths appropriately to your directory structure: 
```
python /path/to/repo/AcceleRest/accelerest_main.py --data_file_dir /path/to/data/dir/ --output_dir /path/to/output/dir/ --lstm_sleepstages --linear_sleepstages --linear_respevents --context_window_shift 1 --max_batch_size 16 --window_wise_predictions
```
The --data_file_dir should be a path to a folder with supported raw, CSV, or appropriately formatted .h5 files (see above).
The output_dir will have a subdirectory for each input file with the same name containing the outputs for that file, depending on which flags were used.

These flags determine what prediction heads are used:
```
--lstm_sleepstages # For lightweight LSTM sequence-model sleep stage predictions.
--linear_sleepstages # For Linear patch-wise (30s epochs) sleep stage predictions.
--linear_respevents # For Linear patch-wise respiratory event predictions.
```
You must specify at least one prediction head. If more are selected they are all run on the output of the same AcceleRest encoder backbone to save compute.

The other options control the following behaviors:
```
 --context_window_shift # Number of patches (30s epochs) to shift the model context window between consecutive forward-passes. Prediction are averaged across overlapping context windows.
 --max_batch_size # Number of context windows to include in a single forward-pass. Higher is faster but more memory intensive.
 --window_wise_predictions # Include to return a memmap file with an array of predictions per context window in addition to the cross-window averaged predictions.
 ```
### Step 4: Analysing Outputs
For each specified prediction head the following files are saved:
```
/path/to/output/dir/{prefix}_soft_preds.npy
```
Where the {prefix} is the corresponding prediction head flag. This file contains a [n_patches X n_classes] array of the patch-wise probabilities of each class.

For raw and CSV inputs, `epoch_start_times.csv` maps every output row to its 30-second epoch start time. `input_processing.json` records the detected format, sample rates, and ActiPy diagnostics.

When `--context_window_shift` is greater than 1, epochs not reached by any context window are retained as rows with `NaN` probabilities rather than being mistaken for uniform predictions.

For sleep stages the class indeces correspond to:
deep: 0, light: 1, rem: 2, wake: 3

```
# To get the probabilitis for each stage:
soft_preds = np.load(os.path.join(output_dir/original_filename, "{prefix}_soft_preds.npy"), allow_pickle=True).item()
soft_preds[:, 0] # for patch-wise deep sleep probabilities.
soft_preds[:, 2] # for patch-wise REM sleep probabilities.

# To get the hard predictions for each patch
hard_preds = np.argmax(soft_preds, axis = 1)
hard_preds[0] # The predicted sleep stage for the first patch.

```

For respiratory events the index label map is:
no_event: 0, event: 1

if the window_wise_prediction flag was used:
```
/path/to/output/dir/{prefix}_window_wise_logits.dat
/path/to/output/dir/{prefix}_window_wise_logits_meta.npy
```
The {prefix}_window_wise_logits.dat file contains the [n_windows X n_classes X window_lenght] array of context window-wise soft predictions.
The {prefix}_window_wise_logits_meta.npy contains the shape and data type of the array in the .dat file and is used when loading the array.

```
# How to load memmap output
meta = np.load(os.path.join(output_dir/original_filename, "{prefix}_window_wise_logits_meta.npy"), allow_pickle=True).item()
mm = np.memmap(
  os.path.join(output_dir/original_filename, "{prefix}_window_wise_logits.dat"),
  mode="r", dtype=meta["dtype"], shape=tuple(meta["shape"])
)
```
