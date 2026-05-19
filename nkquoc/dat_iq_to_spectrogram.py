#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert binary RF IQ data stored in a .dat file into spectrogram PNG images.

This variant is designed for .dat files whose payload is float32 interleaved IQ:
I_0, Q_0, I_1, Q_1, ... (little-endian float32).
"""

from __future__ import annotations

import os
from typing import Generator

import numpy as np

from bin_iq_to_spectrogram import compute_spectrogram, save_spectrogram_image


# ========================
# CONFIG
# ========================
INPUT_DAT_PATH = "E:\\DATN_DATA\\RF\\DroneDetect\\MA1_0000_00.dat"
OUTPUT_DIR = "E:\\DATN_DATA\\RF\\DroneDetect\\spectrograms"
SAMPLE_RATE = 28_000_000  # Hz (adjust as needed)
STFT_POINT = 2048
DURATION_TIME = 0.1  # seconds per spectrogram
CHUNK_SIZE = 1_000_000  # IQ samples per chunk (will auto-increase if needed)
OUTPUT_PREFIX = "spectrogram_MA1_0000_03"

# Set to None to read entire file, or set to N seconds to read only first N seconds
MAX_DURATION_SECONDS = 2

# Supported values: "float32_iq" or "int16_iq"
DAT_FORMAT = "float32_iq"

# Set to True only when DAT_FORMAT == "int16_iq"
NORMALIZE_INT16 = False


def iter_iq_chunks_from_dat(
    dat_path: str,
    chunk_size: int,
    dat_format: str = "float32_iq",
    normalize_int16: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield complex IQ chunks from a .dat file.

    float32_iq layout: [I(float32), Q(float32), ...]
    int16_iq layout:   [I(int16),   Q(int16),   ...]
    """
    if dat_format not in {"float32_iq", "int16_iq"}:
        raise ValueError("dat_format must be 'float32_iq' or 'int16_iq'")

    if dat_format == "float32_iq":
        dtype = np.dtype("<f4")
        bytes_per_scalar = 4
    else:
        dtype = np.dtype("<i2")
        bytes_per_scalar = 2

    total_read = 0
    with open(dat_path, "rb") as f:
        while True:
            if max_iq_samples is not None and total_read >= max_iq_samples:
                break

            bytes_to_read = 2 * chunk_size * bytes_per_scalar
            raw_bytes = f.read(bytes_to_read)
            if not raw_bytes:
                break

            data = np.frombuffer(raw_bytes, dtype=dtype)
            if data.size % 2 != 0:
                data = data[:-1]
            if data.size == 0:
                break

            iq_pairs = data.reshape(-1, 2)
            iq_count = iq_pairs.shape[0]

            if dat_format == "float32_iq":
                i_values = iq_pairs[:, 0].astype(np.float32, copy=False)
                q_values = iq_pairs[:, 1].astype(np.float32, copy=False)
            else:
                i_values = iq_pairs[:, 0].astype(np.float32)
                q_values = iq_pairs[:, 1].astype(np.float32)
                if normalize_int16:
                    i_values /= 32768.0
                    q_values /= 32768.0

            iq_data = i_values + 1j * q_values
            total_read += iq_count
            yield iq_data


def convert_dat_to_spectrograms(
    dat_path: str,
    output_dir: str,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
    chunk_size: int,
    prefix: str,
    dat_format: str,
    normalize_int16: bool,
    max_duration_seconds: int | None,
) -> int:
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"DAT file not found: {dat_path}")

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
    for index, iq_chunk in enumerate(
        iter_iq_chunks_from_dat(
            dat_path=dat_path,
            chunk_size=chunk_size,
            dat_format=dat_format,
            normalize_int16=normalize_int16,
            max_iq_samples=max_iq_samples,
        ),
        start=1,
    ):
        if iq_chunk.size < min_samples_needed:
            print(f"[SKIP] Chunk {index}: only {iq_chunk.size} samples, need {min_samples_needed}")
            continue

        try:
            frequencies, times, spectrum = compute_spectrogram(
                iq_chunk,
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
            saved_count += 1
            print(f"[OK] Saved {output_path}")
        except Exception as e:
            print(f"[ERROR] Chunk {index}: {e}")

    print(f"\nDone. Total spectrograms saved: {saved_count}")
    print(f"Output directory: {output_dir}")
    return saved_count


def main() -> None:
    convert_dat_to_spectrograms(
        dat_path=INPUT_DAT_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=CHUNK_SIZE,
        prefix=OUTPUT_PREFIX,
        dat_format=DAT_FORMAT,
        normalize_int16=NORMALIZE_INT16,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )


if __name__ == "__main__":
    main()
