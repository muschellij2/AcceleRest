"""Input readers and validation for AcceleRest inference.

The model consumes gravity-calibrated, 30 Hz acceleration in ``(3, samples)``
order.  Raw Axivity and ActiGraph files are decoded by actipy; CSV is handled
as a timestamped accelerometer table and passed through actipy's *processing*
API, rather than pretending that it is a device file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import actipy


RAW_SUFFIXES = (".cwa", ".cwa.gz", ".gt3x", ".gt3x.gz")
SUPPORTED_FILE_TYPES = ("auto", "h5", "cwa", "cwa.gz", "gt3x", "gt3x.gz", "csv", "csv.gz")
_TIME_NAMES = ("time", "timestamp", "datetime", "date_time", "date")
_AXIS_NAMES = {
    "x": ("x", "acc_x", "acceleration_x", "accelerometer_x"),
    "y": ("y", "acc_y", "acceleration_y", "accelerometer_y"),
    "z": ("z", "acc_z", "acceleration_z", "accelerometer_z"),
}


def infer_file_type(path: str | Path) -> str:
    """Return the supported type implied by *path*, or raise a useful error."""
    name = Path(path).name.lower()
    for suffix in (".cwa.gz", ".gt3x.gz", ".csv.gz", ".gt3x", ".cwa", ".h5", ".csv"):
        if name.endswith(suffix):
            return suffix[1:]
    raise ValueError(
        f"Unsupported input {path!s}. Expected one of: "
        ".h5, .cwa, .cwa.gz, .gt3x, .gt3x.gz, or .csv."
    )


def output_stem(path: str | Path) -> str:
    """Remove a supported compound suffix without collapsing distinct inputs."""
    name = Path(path).name
    suffix = "." + infer_file_type(path)
    return name[: -len(suffix)]


def find_input_files(directory: str | Path, file_type: str = "auto") -> list[str]:
    """Find only supported files, deterministically, in one directory."""
    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unknown file type {file_type!r}; choose from {SUPPORTED_FILE_TYPES}.")
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    files = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            detected = infer_file_type(path)
        except ValueError:
            continue
        if file_type == "auto" or detected == file_type:
            files.append(str(path))
    return sorted(files)


def _column_matches(columns: pd.Index, candidates: tuple[str, ...]) -> list[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    return [lookup[name] for name in candidates if name in lookup]


def _column_lookup(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    matches = _column_matches(columns, candidates)
    return matches[0] if matches else None


def _read_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, compression="infer")
    time_matches = _column_matches(data.columns, _TIME_NAMES)
    axis_matches = {axis: _column_matches(data.columns, names) for axis, names in _AXIS_NAMES.items()}
    if len(time_matches) > 1:
        raise ValueError(f"CSV has multiple recognised timestamp columns: {time_matches}.")
    ambiguous_axes = {axis: matches for axis, matches in axis_matches.items() if len(matches) > 1}
    if ambiguous_axes:
        details = "; ".join(f"{axis}: {matches}" for axis, matches in ambiguous_axes.items())
        raise ValueError(
            f"CSV has ambiguous acceleration columns ({details}). Keep exactly one recognised "
            "column for each of x, y, and z."
        )
    time_col = time_matches[0] if time_matches else None
    axes = {axis: matches[0] if matches else None for axis, matches in axis_matches.items()}
    if time_col is None or any(column is None for column in axes.values()):
        raise ValueError(
            "CSV input must contain a timestamp column (time, timestamp, datetime, "
            "or date) and x, y, z columns. Recognised axis aliases are acc_x/y/z, "
            "acceleration_x/y/z, and accelerometer_x/y/z."
        )
    # ``mixed`` is needed when CSV writers omit fractional seconds on exact
    # boundaries but retain them for the other samples in the same column.
    time = pd.to_datetime(data[time_col], errors="coerce", utc=True, format="mixed")
    if time.isna().any():
        raise ValueError(f"CSV has {time.isna().sum()} unparseable timestamps in {time_col!r}.")
    out = data[[axes["x"], axes["y"], axes["z"]]].copy()
    out.columns = ["x", "y", "z"]
    out = out.apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(time, name="time")
    if out.index.has_duplicates:
        raise ValueError("CSV timestamps must be unique; duplicate timestamps cannot be resampled safely.")
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out


def _sample_rate(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        raise ValueError("Input has fewer than three timestamps; sampling rate cannot be determined.")
    seconds = np.diff(index.asi8) / 1e9
    if np.any(seconds <= 0):
        raise ValueError("Timestamps must be strictly increasing.")
    rate = 1.0 / float(np.median(seconds))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Could not determine a positive sampling rate from timestamps.")
    return rate


def _process_frame(data: pd.DataFrame, sample_rate: float, detect_nonwear: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    # A lower input rate cannot contain the 0--15 Hz signal AcceleRest expects.
    if sample_rate < 30:
        raise ValueError(
            f"Input sampling rate is {sample_rate:.3g} Hz. AcceleRest requires raw "
            "data sampled at least at 30 Hz; upsampling lower-rate data is not valid."
        )
    # Actipy's lowpass requires a cutoff below Nyquist.  This preserves the
    # model's 15-Hz input bandwidth for common 30-Hz and higher recordings.
    lowpass_hz = min(15.0, sample_rate / 2.0 * 0.999)
    # Actipy 3.8 uses NumPy structured arrays internally and therefore cannot
    # process timezone-aware DatetimeIndex values.  Work in naive UTC and put
    # the timezone back afterwards so CSV timestamps retain their meaning.
    timezone = data.index.tz
    process_data = data[["x", "y", "z"]].copy()
    if timezone is not None:
        process_data.index = process_data.index.tz_convert("UTC").tz_localize(None)
    process_data.index.name = "time"
    processed, info = actipy.process(
        process_data,
        sample_rate=sample_rate,
        lowpass_hz=lowpass_hz,
        calibrate_gravity=True,
        detect_nonwear=detect_nonwear,
        resample_hz=30,
        verbose=False,
    )
    if timezone is not None:
        processed.index = processed.index.tz_localize("UTC").tz_convert(timezone)
        processed.index.name = "time"
    return processed, info


def load_accelerometry(path: str | Path, file_type: str = "auto", detect_nonwear: bool = True) -> tuple[np.ndarray, pd.DatetimeIndex | None, dict[str, Any]]:
    """Load, validate and standardise one recording for AcceleRest.

    Returns the model array, its 30-Hz timestamps when available, and processing
    metadata.  HDF5 is retained for backwards compatibility because it already
    stores the model-ready ``data/accelerometry`` array.
    """
    path = str(path)
    detected = infer_file_type(path)
    if file_type != "auto" and detected != file_type:
        raise ValueError(f"{path} is {detected!r}, not the requested {file_type!r} format.")

    if detected == "h5":
        with h5py.File(path, "r", rdcc_nbytes=1024**3) as handle:
            if "data/accelerometry" not in handle:
                raise ValueError("HDF5 input is missing required dataset 'data/accelerometry'.")
            array = np.asarray(handle["data/accelerometry"], dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 3:
            raise ValueError("HDF5 data/accelerometry must have shape (3, n_samples).")
        return array, None, {"input_type": "h5", "processed_by": "caller"}

    if detected in ("csv", "csv.gz"):
        raw = _read_csv(path)
        sample_rate = _sample_rate(raw.index)
        frame, info = _process_frame(raw, sample_rate, detect_nonwear)
        info = {"input_type": detected, "input_sample_rate_hz": sample_rate, **info}
    else:
        # Actipy natively decompresses .cwa.gz and .gt3x.gz before selecting its
        # Axivity or ActiGraph reader, so do not manually unpack these files.
        raw, raw_info = actipy.read_device(
            path, lowpass_hz=None, calibrate_gravity=False, detect_nonwear=False,
            resample_hz=None, verbose=False,
        )
        sample_rate = float(raw_info["SampleRate"])
        frame, info = _process_frame(raw, sample_rate, detect_nonwear)
        info = {"input_type": detected, "input_sample_rate_hz": sample_rate, **raw_info, **info}

    if frame.empty:
        raise ValueError("No acceleration samples remain after reading and preprocessing.")
    array = frame[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True).T
    if array.shape[0] != 3 or array.shape[1] == 0:
        raise ValueError("Preprocessing did not produce a non-empty 3-axis acceleration signal.")
    finite = np.isfinite(array).all(axis=0)
    if not finite.any():
        raise ValueError("All samples are missing after preprocessing (for example, all were non-wear).")
    median_magnitude = float(np.median(np.linalg.norm(array[:, finite], axis=0)))
    if not 0.25 <= median_magnitude <= 2.0:
        raise ValueError(
            f"Median acceleration magnitude is {median_magnitude:.3g}; input appears not to be in g. "
            "Convert CSV axes to gravitational units before inference."
        )
    output_rate = _sample_rate(frame.index)
    if not np.isclose(output_rate, 30, rtol=1e-3):
        raise ValueError(f"Actipy did not produce the required 30-Hz signal (got {output_rate:.6g} Hz).")
    info["output_sample_rate_hz"] = output_rate
    info["median_acceleration_magnitude_g"] = median_magnitude
    return array, frame.index, info
