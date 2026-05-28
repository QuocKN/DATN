#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert IQ data in a .mat file to spectrogram PNG images."""

from __future__ import annotations

import os
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np

from bin_iq_to_spectrogram import compute_spectrogram, save_spectrogram_image


# ========================
# CONFIG
# ========================
INPUT_MAT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/DRFF_R2/evn/outdoor_environment.mat"
OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/DRFF_R2/evn/spectrograms"

# Set to a variable name inside .mat if you know it, else None to auto-detect
MAT_IQ_KEY: str | None = None

SAMPLE_RATE = 100_000_000  # Hz
STFT_POINT = 1024
DURATION_TIME = 0.03  # seconds per spectrogram
CHUNK_SIZE = 1_000_000  # IQ samples per chunk
OUTPUT_PREFIX = "spectrogram"
WAVEFORM_PREFIX = "waveform"

# Set to None to read full signal, or number of seconds to limit processing
MAX_DURATION_SECONDS: int | None = None
SAVE_WAVEFORM = True


def save_waveform_image(
    iq_data: np.ndarray,
    sample_rate: int,
    output_path: str,
    source_name: str,
    chunk_index: int,
) -> None:
    """Save 3-panel waveform image: I, Q, and |IQ|."""
    time_axis = np.arange(iq_data.size) / sample_rate
    i_values = iq_data.real
    q_values = iq_data.imag
    iq_magnitude = np.abs(iq_data)

    figure, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=120, sharex=True)

    axes[0].plot(time_axis, i_values, linewidth=0.8, color="tab:blue")
    axes[0].set_ylabel("I")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, q_values, linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("Q")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_axis, iq_magnitude, linewidth=0.8, color="tab:green")
    axes[2].set_ylabel("|IQ|")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.25)

    figure.suptitle(f"{source_name} | chunk={chunk_index} | samples={iq_data.size}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _to_complex_iq(data: Any) -> np.ndarray | None:
    """Try converting a MATLAB variable to 1D complex IQ array."""
    arr = np.asarray(data)
    if arr.size == 0:
        return None

    arr = np.squeeze(arr)

    # Already complex -> flatten to 1D
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False).reshape(-1)

    # Numeric real arrays
    if not np.issubdtype(arr.dtype, np.number):
        return None

    # Shape (..., 2) interpreted as [I, Q]
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        reshaped = arr.reshape(-1, 2).astype(np.float32, copy=False)
        return reshaped[:, 0] + 1j * reshaped[:, 1]

    # Shape (2, N) interpreted as [I; Q]
    if arr.ndim == 2 and arr.shape[0] == 2:
        reshaped = arr.astype(np.float32, copy=False)
        return reshaped[0, :] + 1j * reshaped[1, :]

    return None


def _is_hdf5_mat(mat_path: str) -> bool:
    """Check whether .mat file is MATLAB v7.3 (HDF5-based)."""
    with open(mat_path, "rb") as handle:
        signature = handle.read(8)
    return signature == b"\x89HDF\r\n\x1a\n"


def _find_iq_pair_from_dict(data_dict: dict[str, Any]) -> tuple[str, np.ndarray] | None:
    """Find paired keys like RF0_I + RF0_Q and combine them."""
    keys = list(data_dict.keys())
    for key_i in keys:
        if key_i.startswith("__"):
            continue
        key_u = key_i.upper()
        if not key_u.endswith("_I"):
            continue

        key_q = key_i[:-2] + "_Q"
        if key_q not in data_dict:
            continue

        i_arr = np.squeeze(np.asarray(data_dict[key_i]))
        q_arr = np.squeeze(np.asarray(data_dict[key_q]))
        if i_arr.size == 0 or q_arr.size == 0:
            continue
        if i_arr.shape != q_arr.shape:
            continue
        if not np.issubdtype(i_arr.dtype, np.number) or not np.issubdtype(q_arr.dtype, np.number):
            continue

        i_flat = i_arr.astype(np.float32, copy=False).reshape(-1)
        q_flat = q_arr.astype(np.float32, copy=False).reshape(-1)
        if i_flat.size < 1024:
            continue
        return key_i[:-2], i_flat + 1j * q_flat
    return None


def _find_iq_pair_names(keys: list[str], preferred_base: str | None = None) -> tuple[str, str, str]:
    """
    Find paired key names like RF0_I + RF0_Q without loading full arrays.
    Returns: (base_key, key_i, key_q)
    """
    if preferred_base:
        key_i = f"{preferred_base}_I"
        key_q = f"{preferred_base}_Q"
        if key_i in keys and key_q in keys:
            return preferred_base, key_i, key_q

    for key_i in keys:
        key_u = key_i.upper()
        if not key_u.endswith("_I"):
            continue
        key_q = key_i[:-2] + "_Q"
        if key_q in keys:
            return key_i[:-2], key_i, key_q

    raise ValueError("Cannot find paired *_I and *_Q variables in MAT file.")


def _pick_iq_key_from_dict(data_dict: dict[str, Any]) -> tuple[str, np.ndarray]:
    """Pick the most likely IQ variable from loaded MATLAB dict."""
    candidates: list[tuple[int, str, np.ndarray]] = []

    pair = _find_iq_pair_from_dict(data_dict)
    if pair is not None:
        return pair

    for key, value in data_dict.items():
        if key.startswith("__"):
            continue

        iq = _to_complex_iq(value)
        if iq is None:
            continue

        if iq.size < 1024:
            continue

        score = 0
        key_l = key.lower()
        if "iq" in key_l:
            score += 100
        if key_l.endswith("_i") or key_l.endswith("_q"):
            score -= 50
        if "rf" in key_l:
            score += 40
        if "signal" in key_l:
            score += 20
        if "data" in key_l:
            score += 10
        if np.iscomplexobj(iq):
            score += 30
        score += min(iq.size // 1000, 50)
        candidates.append((score, key, iq))

    if not candidates:
        raise ValueError("No IQ-like numeric variable found in MAT file.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_key, best_iq = candidates[0]
    return best_key, best_iq


def _load_iq_with_scipy(mat_path: str, mat_iq_key: str | None) -> tuple[str, np.ndarray]:
    import scipy.io as sio

    data_dict = sio.loadmat(mat_path)

    if mat_iq_key is not None:
        key_i = f"{mat_iq_key}_I"
        key_q = f"{mat_iq_key}_Q"
        if key_i in data_dict and key_q in data_dict:
            i_flat = np.squeeze(np.asarray(data_dict[key_i])).astype(np.float32, copy=False).reshape(-1)
            q_flat = np.squeeze(np.asarray(data_dict[key_q])).astype(np.float32, copy=False).reshape(-1)
            if i_flat.shape != q_flat.shape:
                raise ValueError(f"'{key_i}' and '{key_q}' have different shapes.")
            return mat_iq_key, i_flat + 1j * q_flat

        if mat_iq_key not in data_dict:
            raise KeyError(f"Key '{mat_iq_key}' not found in MAT file.")
        iq = _to_complex_iq(data_dict[mat_iq_key])
        if iq is None:
            raise ValueError(f"Key '{mat_iq_key}' is not in a supported IQ format.")
        return mat_iq_key, iq

    return _pick_iq_key_from_dict(data_dict)


def _load_iq_with_h5py(mat_path: str, mat_iq_key: str | None) -> tuple[str, np.ndarray]:
    import h5py

    with h5py.File(mat_path, "r") as handle:
        datasets: dict[str, Any] = {}

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[()]

        handle.visititems(visitor)

    # Flatten dataset names to final segment for easier key matching
    flat: dict[str, Any] = {}
    for full_name, value in datasets.items():
        short_name = full_name.split("/")[-1]
        if short_name not in flat:
            flat[short_name] = value
        flat[full_name] = value

    if mat_iq_key is not None:
        key_i = f"{mat_iq_key}_I"
        key_q = f"{mat_iq_key}_Q"
        if key_i in flat and key_q in flat:
            i_flat = np.squeeze(np.asarray(flat[key_i])).astype(np.float32, copy=False).reshape(-1)
            q_flat = np.squeeze(np.asarray(flat[key_q])).astype(np.float32, copy=False).reshape(-1)
            if i_flat.shape != q_flat.shape:
                raise ValueError(f"'{key_i}' and '{key_q}' have different shapes.")
            return mat_iq_key, i_flat + 1j * q_flat

        if mat_iq_key not in flat:
            raise KeyError(f"Key '{mat_iq_key}' not found in MAT file datasets.")
        iq = _to_complex_iq(flat[mat_iq_key])
        if iq is None:
            raise ValueError(f"Key '{mat_iq_key}' is not in a supported IQ format.")
        return mat_iq_key, iq

    return _pick_iq_key_from_dict(flat)


def iter_iq_chunks_from_mat_h5(
    mat_path: str,
    chunk_size: int,
    mat_iq_key: str | None = None,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield IQ chunks from MATLAB v7.3 file by reading I/Q datasets per chunk."""
    import h5py

    with h5py.File(mat_path, "r") as handle:
        key_names = list(handle.keys())
        base_key, key_i, key_q = _find_iq_pair_names(key_names, mat_iq_key)

        ds_i = handle[key_i]
        ds_q = handle[key_q]

        total_samples = int(np.prod(ds_i.shape))
        if int(np.prod(ds_q.shape)) != total_samples:
            raise ValueError(f"'{key_i}' and '{key_q}' do not have matching shapes.")

        if max_iq_samples is not None:
            total_samples = min(total_samples, max_iq_samples)

        print(f"Using MAT variable pair: {base_key} ({key_i}, {key_q})")
        print(f"Total IQ samples: {total_samples}")

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)

            i_part = np.asarray(ds_i[start:end]).reshape(-1).astype(np.float32, copy=False)
            q_part = np.asarray(ds_q[start:end]).reshape(-1).astype(np.float32, copy=False)

            yield i_part + 1j * q_part


def load_iq_from_mat(mat_path: str, mat_iq_key: str | None = None) -> tuple[str, np.ndarray]:
    """Load IQ from MAT (supports scipy MAT and HDF5 MAT v7.3)."""
    scipy_error = None

    try:
        return _load_iq_with_scipy(mat_path, mat_iq_key)
    except Exception as e:
        scipy_error = e

    try:
        return _load_iq_with_h5py(mat_path, mat_iq_key)
    except Exception as h5_error:
        raise RuntimeError(
            "Failed to load MAT with both scipy and h5py. "
            f"scipy error: {scipy_error}; h5py error: {h5_error}"
        )


def convert_mat_to_spectrograms(
    mat_path: str,
    output_dir: str,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
    chunk_size: int,
    prefix: str,
    max_duration_seconds: int | None,
    mat_iq_key: str | None,
) -> int:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    os.makedirs(output_dir, exist_ok=True)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)
        print(f"Reading first {max_duration_seconds}s ({max_iq_samples} IQ samples)...")

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] chunk_size={chunk_size} is smaller than required {min_samples_needed}. "
            f"Using {min_samples_needed} instead."
        )
        chunk_size = min_samples_needed

    saved_count = 0
    waveform_dir = os.path.join(output_dir, "waveforms")
    if SAVE_WAVEFORM:
        os.makedirs(waveform_dir, exist_ok=True)

    if _is_hdf5_mat(mat_path):
        chunk_iter = iter_iq_chunks_from_mat_h5(
            mat_path=mat_path,
            chunk_size=chunk_size,
            mat_iq_key=mat_iq_key,
            max_iq_samples=max_iq_samples,
        )
    else:
        # Fallback for non-HDF5 MAT: load once then iterate in-memory slices.
        key_name, iq_data = load_iq_from_mat(mat_path, mat_iq_key)
        if max_iq_samples is not None:
            iq_data = iq_data[:max_iq_samples]
        print(f"Using MAT variable: {key_name}")
        print(f"Total IQ samples: {iq_data.size}")

        def _in_memory_iter() -> Generator[np.ndarray, None, None]:
            for start in range(0, iq_data.size, chunk_size):
                yield iq_data[start : start + chunk_size]

        chunk_iter = _in_memory_iter()

    for index, chunk in enumerate(chunk_iter, start=1):
        if chunk.size < min_samples_needed:
            print(f"[SKIP] Chunk {index}: only {chunk.size} samples, need {min_samples_needed}")
            continue

        try:
            frequencies, times, spectrum = compute_spectrogram(
                chunk,
                sample_rate=sample_rate,
                stft_point=stft_point,
                duration_time=duration_time,
            )
            output_path = os.path.join(output_dir, f"{prefix}_{index:06d}.png")
            save_spectrogram_image(
                frequencies=frequencies,
                times=times,
                spectrum=spectrum,
                output_path=output_path,
            )
            if SAVE_WAVEFORM:
                waveform_path = os.path.join(waveform_dir, f"{WAVEFORM_PREFIX}_{index:06d}.png")
                save_waveform_image(
                    iq_data=chunk[: int(sample_rate * duration_time)],
                    sample_rate=sample_rate,
                    output_path=waveform_path,
                    source_name=os.path.basename(mat_path),
                    chunk_index=index,
                )
            saved_count += 1
            print(f"[OK] Saved {output_path}")
        except Exception as e:
            print(f"[ERROR] Chunk {index}: {e}")

    print(f"\nDone. Total spectrograms saved: {saved_count}")
    print(f"Output directory: {output_dir}")
    return saved_count


def main() -> None:
    convert_mat_to_spectrograms(
        mat_path=INPUT_MAT_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=CHUNK_SIZE,
        prefix=OUTPUT_PREFIX,
        max_duration_seconds=MAX_DURATION_SECONDS,
        mat_iq_key=MAT_IQ_KEY,
    )


if __name__ == "__main__":
    main()
