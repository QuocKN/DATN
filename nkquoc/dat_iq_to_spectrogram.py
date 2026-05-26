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

import matplotlib.pyplot as plt
import numpy as np

from bin_iq_to_spectrogram import compute_spectrogram, save_spectrogram_image


# ========================
# CONFIG
# ========================
INPUT_DAT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/CDRF/Rowan_Outdoor/mavic2/DJI_Mavic2_2422.5_10_2412.5_30_50_hovering_14.0.dat"
OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/CDRF/Rowan_Outdoor/mavic2/spectrograms"
SAMPLE_RATE = 60_000_000  # Hz (adjust as needed)
STFT_POINT = 1024
DURATION_TIME = 0.05  # seconds per spectrogram
CHUNK_SIZE = 1_000_000  # IQ samples per chunk (will auto-increase if needed)
OUTPUT_PREFIX = "spectrogram"
WAVEFORM_PREFIX = "waveform"

# Set to None to read entire file, or set to N seconds to read only first N seconds
MAX_DURATION_SECONDS = 20

# Supported values: "float32_iq" or "int16_iq"
DAT_FORMAT = "float32_iq"

# Set to True only when DAT_FORMAT == "int16_iq"
NORMALIZE_INT16 = False
SAVE_WAVEFORM = True


def save_waveform_image(
    iq_data: np.ndarray,
    sample_rate: int,
    output_path: str,
    source_name: str,
    chunk_index: int,
    title: str | None = None,
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

    if title:
        figure.suptitle(title)
    else:
        figure.suptitle(f"{source_name} | chunk={chunk_index} | samples={iq_data.size}")

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


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

    waveform_dir = os.path.join(output_dir, "waveforms")
    if SAVE_WAVEFORM:
        os.makedirs(waveform_dir, exist_ok=True)

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
            if SAVE_WAVEFORM:
                waveform_path = os.path.join(waveform_dir, f"{WAVEFORM_PREFIX}_{index:06d}.png")
                save_waveform_image(
                    iq_data=iq_chunk[: int(sample_rate * duration_time)],
                    sample_rate=sample_rate,
                    output_path=waveform_path,
                    source_name=os.path.basename(dat_path),
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
