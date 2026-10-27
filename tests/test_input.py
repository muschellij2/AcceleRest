import h5py
import numpy as np
import pandas as pd
import pytest

from accelerest.input import find_input_files, infer_file_type, load_accelerometry, output_stem


def _fake_frame(n=90):
    index = pd.date_range("2025-01-01", periods=n, freq="33333333ns", name="time")
    return pd.DataFrame({"x": 0.0, "y": 0.0, "z": 1.0}, index=index)


@pytest.mark.parametrize(
    "name, file_type, stem",
    [
        ("a.cwa", "cwa", "a"),
        ("a.CWA.gz", "cwa", "a"),
        ("a.gt3x", "gt3x", "a"),
        ("a.gt3x.zip", "gt3x", "a"),
        ("a.b.bin.gz", "bin", "a.b"),
        ("a.h5", "h5", "a"),
    ],
)
def test_suffixes(name, file_type, stem):
    assert infer_file_type(name) == file_type
    assert output_stem(name) == stem


@pytest.mark.parametrize("name", ["a.csv", "a.h5.gz", "a.txt", "a.gz"])
def test_unsupported_suffixes_are_rejected(name):
    with pytest.raises(ValueError, match="Unsupported input"):
        infer_file_type(name)


def test_h5_validation_and_file_discovery(tmp_path):
    with h5py.File(tmp_path / "ready.h5", "w") as handle:
        handle.create_dataset("data/accelerometry", data=np.ones((3, 20), dtype=np.float32))
    (tmp_path / "raw.gt3x.gz").write_bytes(b"not decoded in this test")
    (tmp_path / "ignore.txt").write_text("ignore")

    array, times, info = load_accelerometry(tmp_path / "ready.h5")
    assert array.shape == (3, 20)
    assert times is None
    assert info["input_type"] == "h5"
    assert find_input_files(tmp_path) == [str(tmp_path / "raw.gt3x.gz"), str(tmp_path / "ready.h5")]
    assert find_input_files(tmp_path, "gt3x") == [str(tmp_path / "raw.gt3x.gz")]


@pytest.mark.parametrize("suffix", [".cwa", ".cwa.gz", ".gt3x", ".gt3x.gz", ".bin", ".bin.gz"])
def test_device_files_are_passed_directly_to_actipy(tmp_path, monkeypatch, suffix):
    # The file need not be a real recording: this verifies that AcceleRest hands
    # the path, compressed or not, to a single actipy.read_device call.
    filename = tmp_path / f"device{suffix}"
    filename.write_bytes(b"placeholder")
    frame = _fake_frame()
    seen = {}

    def fake_read(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return frame, {"SampleRate": 30.0, "CalibOK": 1}

    monkeypatch.setattr("accelerest.input.actipy.read_device", fake_read)
    array, times, info = load_accelerometry(filename, detect_nonwear=False)

    assert seen["path"] == str(filename)
    assert seen["kwargs"]["resample_hz"] == 30
    assert seen["kwargs"]["calibrate_gravity"] is True
    assert seen["kwargs"]["detect_nonwear"] is False
    assert array.shape == (3, 90)
    assert times.equals(frame.index)
    assert info["input_type"] == suffix.split(".")[1]


def test_device_below_30hz_is_rejected(tmp_path, monkeypatch):
    filename = tmp_path / "slow.bin"
    filename.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "accelerest.input.actipy.read_device", lambda path, **kwargs: (_fake_frame(), {"SampleRate": 10.0})
    )
    with pytest.raises(ValueError, match="30 Hz or higher"):
        load_accelerometry(filename)


def test_requested_type_must_match(tmp_path):
    (tmp_path / "x.cwa").write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="not the requested"):
        load_accelerometry(tmp_path / "x.cwa", file_type="gt3x")
