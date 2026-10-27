"""Input readers and validation for AcceleRest inference.

The model consumes gravity-calibrated, 30 Hz acceleration in ``(3, samples)``
order. Raw device files are read and processed by ``actipy.read_device``, so
any format actipy reads (Axivity ``.cwa``, ActiGraph ``.gt3x``, GENEActiv and
Matrix ``.bin``, optionally ``.gz`` or ``.zip`` compressed) is supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import actipy


# Device formats read by actipy.read_device, which also decompresses these.
ACTIPY_FILE_TYPES = ("cwa", "gt3x", "bin")
COMPRESSION_SUFFIXES = (".gz", ".zip")
SUPPORTED_FILE_TYPES = ("auto", "h5", *ACTIPY_FILE_TYPES)


def _split_suffix(path: str | Path) -> tuple[str, str]:
    """Return ``(file_type, full_suffix)``, e.g. ``("cwa", ".cwa.gz")``."""
    name = Path(path).name.lower()
    compression = next((s for s in COMPRESSION_SUFFIXES if name.endswith(s)), "")
    base = name[: len(name) - len(compression)]
    for file_type in ("h5", *ACTIPY_FILE_TYPES):
        if base.endswith("." + file_type) and not (file_type == "h5" and compression):
            return file_type, "." + file_type + compression
    raise ValueError(
        f"Unsupported input {path!s}. Expected .h5, or a .cwa, .gt3x or .bin "
        "file (optionally .gz or .zip compressed)."
    )


def infer_file_type(path: str | Path) -> str:
    """Return the format of *path* (h5, cwa, gt3x or bin), or raise."""
    return _split_suffix(path)[0]


def output_stem(path: str | Path) -> str:
    """Remove a supported compound suffix without collapsing distinct inputs."""
    name = Path(path).name
    return name[: -len(_split_suffix(path)[1])]


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


def load_accelerometry(path: str | Path, file_type: str = "auto", detect_nonwear: bool = True) -> tuple[np.ndarray, pd.DatetimeIndex | None, dict[str, Any]]:
    """Load and standardise one recording for AcceleRest.

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

    # actipy skips the lowpass filter when 15 Hz is not below Nyquist (30-Hz devices).
    frame, info = actipy.read_device(
        path,
        lowpass_hz=15,
        calibrate_gravity=True,
        detect_nonwear=detect_nonwear,
        resample_hz=30,
        verbose=False,
    )
    info = {"input_type": detected, **info}

    # Upsampling cannot restore the signal content the model expects.
    if float(info.get("SampleRate", 0)) < 30:
        raise ValueError(
            f"Device sampling rate is {info.get('SampleRate')} Hz; AcceleRest "
            "requires data sampled at 30 Hz or higher."
        )
    if frame.empty:
        raise ValueError("No acceleration samples remain after reading and preprocessing.")
    array = frame[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True).T
    if not np.isfinite(array).any():
        raise ValueError("All samples are missing after preprocessing (for example, all were non-wear).")
    return array, frame.index, info
