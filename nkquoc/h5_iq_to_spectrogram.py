#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert HDF5 int16 interleaved IQ datasets (I,Q,I,Q,...) to spectrogram PNGs.

Each HDF5 dataset is assumed to be a 1D int16 array with layout:
I_0, Q_0, I_1, Q_1, ...
"""

from __future__ import annotations

import os
from typing import Generator, Iterable

import h5py
import numpy as np

from bin_iq_to_spectrogram import compute_spectrogram, save_spectrogram_image


# ========================
# CONFIG
# ========================
INPUT_H5_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/Wifi_Blue/2_4ghz_bluetooth.h5"
OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Wifi_Blue/2_4ghz_bluetooth_spectrograms"

SAMPLE_RATE = 30_000_000  # Hz (1 second = 30,000,000 IQ samples)
STFT_POINT = 1024
DURATION_TIME = 0.1  # seconds per spectrogram
CHUNK_SIZE = 1_000_000  # IQ samples per chunk
OUTPUT_PREFIX = "spectrogram"

# Limit how many datasets to process (None = all)
MAX_DATASETS: int | None = None

# Optionally restrict to a list of dataset names (None = all)
DATASET_NAMES: list[str] | None = None

# Set to None to read full dataset, or number of seconds to limit per dataset
MAX_DURATION_SECONDS: int | None = None

# Normalize int16 to [-1, 1]
NORMALIZE = False


def _sorted_dataset_names(names: Iterable[str]) -> list[str]:
    def _key(x: str) -> tuple[int, str]:
        return (0, x) if x.isdigit() else (1, x)

    numeric = sorted([n for n in names if n.isdigit()], key=lambda s: int(s))
    non_numeric = sorted([n for n in names if not n.isdigit()])
    return numeric + non_numeric


def iter_iq_chunks_from_h5_dataset(
    dataset: h5py.Dataset,
    chunk_size: int,
    normalize: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield complex IQ chunks from an HDF5 dataset storing interleaved int16 IQ.
    """
    total_int16 = int(np.prod(dataset.shape))
    total_iq = total_int16 // 2

    if total_int16 % 2 != 0:
        total_iq = (total_int16 - 1) // 2

    if max_iq_samples is not None:
        total_iq = min(total_iq, max_iq_samples)

    for start_iq in range(0, total_iq, chunk_size):
        end_iq = min(start_iq + chunk_size, total_iq)
        start = start_iq * 2
        end = end_iq * 2

        raw = np.asarray(dataset[start:end], dtype=np.int16)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        if raw.size == 0:
            break

        iq_pairs = raw.reshape(-1, 2)
        i_values = iq_pairs[:, 0].astype(np.float32)
        q_values = iq_pairs[:, 1].astype(np.float32)

        if normalize:
            i_values /= 32768.0
            q_values /= 32768.0

        yield i_values + 1j * q_values


def convert_h5_to_spectrograms(
    h5_path: str,
    output_dir: str,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
    chunk_size: int,
    prefix: str,
    dataset_names: list[str] | None,
    max_datasets: int | None,
    max_duration_seconds: int | None,
    normalize: bool,
) -> int:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    os.makedirs(output_dir, exist_ok=True)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)
        print(f"Reading first {max_duration_seconds}s ({max_iq_samples} IQ samples) per dataset...")

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] chunk_size={chunk_size} is smaller than required {min_samples_needed}. "
            f"Using {min_samples_needed} instead."
        )
        chunk_size = min_samples_needed

    total_saved = 0

    with h5py.File(h5_path, "r") as handle:
        available = _sorted_dataset_names(list(handle.keys()))

        if dataset_names is not None:
            targets = [name for name in dataset_names if name in handle]
            missing = [name for name in dataset_names if name not in handle]
            if missing:
                print(f"[WARN] Missing datasets: {missing}")
        else:
            targets = available

        if max_datasets is not None:
            targets = targets[:max_datasets]

        for dataset_name in targets:
            dataset = handle[dataset_name]
            if not isinstance(dataset, h5py.Dataset):
                continue
            if dataset.dtype != np.int16:
                print(f"[SKIP] {dataset_name}: expected int16, got {dataset.dtype}")
                continue

            print(f"Processing dataset: {dataset_name} shape={dataset.shape}")
            dataset_prefix = f"{prefix}_{dataset_name}"

            for index, iq_chunk in enumerate(
                iter_iq_chunks_from_h5_dataset(
                    dataset=dataset,
                    chunk_size=chunk_size,
                    normalize=normalize,
                    max_iq_samples=max_iq_samples,
                ),
                start=1,
            ):
                if iq_chunk.size < min_samples_needed:
                    print(
                        f"[SKIP] {dataset_name} chunk {index}: "
                        f"only {iq_chunk.size} samples, need {min_samples_needed}"
                    )
                    continue

                try:
                    frequencies, times, spectrum = compute_spectrogram(
                        iq_chunk,
                        sample_rate=sample_rate,
                        stft_point=stft_point,
                        duration_time=duration_time,
                    )

                    output_path = os.path.join(output_dir, f"{dataset_prefix}_{index:06d}.png")
                    save_spectrogram_image(
                        frequencies=frequencies,
                        times=times,
                        spectrum=spectrum,
                        output_path=output_path,
                    )
                    total_saved += 1
                except Exception as exc:
                    print(f"[ERROR] {dataset_name} chunk {index}: {exc}")

    print(f"\nDone. Total spectrograms saved: {total_saved}")
    print(f"Output directory: {output_dir}")
    return total_saved


def main() -> None:
    convert_h5_to_spectrograms(
        h5_path=INPUT_H5_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=CHUNK_SIZE,
        prefix=OUTPUT_PREFIX,
        dataset_names=DATASET_NAMES,
        max_datasets=MAX_DATASETS,
        max_duration_seconds=MAX_DURATION_SECONDS,
        normalize=NORMALIZE,
    )


if __name__ == "__main__":
    main()
