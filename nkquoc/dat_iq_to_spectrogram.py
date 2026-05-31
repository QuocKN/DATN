#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert RF IQ data stored in a .dat file into spectrogram PNG images.

Supported DAT payload formats:
- float32_iq: interleaved little-endian float32 [I, Q, I, Q, ...]
- int16_iq:   interleaved little-endian int16   [I, Q, I, Q, ...]
"""

from __future__ import annotations

import os
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import iq_spectrogram_core as spectrogram_core
    from .bin_spectrogram_converter import ProcessingConfig, preprocess_iq_chunk
    from .iq_spectrogram_core import compute_spectrogram, save_spectrogram_image, save_waveform_image
except ImportError:
    import iq_spectrogram_core as spectrogram_core
    from bin_spectrogram_converter import ProcessingConfig, preprocess_iq_chunk
    from iq_spectrogram_core import compute_spectrogram, save_spectrogram_image, save_waveform_image


# ========================
# CONFIG
# ========================
INPUT_DAT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/DroneDetect/MA1_0000_02/MA1_0000_02.dat"
OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/DroneDetect/MA1_0000_02/spectrograms"
SAMPLE_RATE = 60_000_000
STFT_POINT = 1024
DURATION_TIME = 0.05
CHUNK_SIZE = 4096
OUTPUT_PREFIX = "spectrogram_MA1_0000_02"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform_MA1_0000_02"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True
MAX_DURATION_SECONDS = 5

DAT_FORMAT = "float32_iq"  # "float32_iq" | "int16_iq"
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

ENABLE_DESPIKE = False
DESPIKE_PERCENTILE = 99.5
ENABLE_REPAIR_CLIPPED = False
ENABLE_IMPULSE_BLANKER = False
BLANKER_MEDIAN_KERNEL = 129
BLANKER_THRESHOLD_SIGMA = 6.0
BLANKER_MAX_SPIKE_WIDTH = 10
ENABLE_PEAK_THINNING = False
THIN_PERCENTILE = 99.2
THIN_TARGET_WIDTH = 2
THIN_APPLY_MAX_RUN = 48
ENABLE_SATURATION_THINNING = False
SATURATION_LEVEL = 1850.0
SAT_TARGET_WIDTH = 1
SAT_MAX_RUN = 256

ENABLE_SPEC_COLUMN_DENOISE = True
SPEC_COLUMN_QUANTILE = 60.0
ENABLE_SPEC_DB_CLIP = False
SPEC_CLIP_DB_MIN = -80.0
SPEC_CLIP_DB_MAX = 15.0


def apply_spectrogram_config() -> None:
    spectrogram_core.IMAGE_SIZE = IMAGE_SIZE
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = ENABLE_SPEC_COLUMN_DENOISE
    spectrogram_core.SPEC_COLUMN_QUANTILE = SPEC_COLUMN_QUANTILE
    spectrogram_core.ENABLE_SPEC_DB_CLIP = ENABLE_SPEC_DB_CLIP
    spectrogram_core.SPEC_CLIP_DB_MIN = SPEC_CLIP_DB_MIN
    spectrogram_core.SPEC_CLIP_DB_MAX = SPEC_CLIP_DB_MAX


def build_processing_config() -> ProcessingConfig:
    return ProcessingConfig(
        save_waveform=SAVE_WAVEFORM,
        waveform_prefix=WAVEFORM_PREFIX,
        waveform_max_points=WAVEFORM_MAX_POINTS,
        enable_despike=ENABLE_DESPIKE,
        despike_percentile=DESPIKE_PERCENTILE,
        enable_repair_clipped=ENABLE_REPAIR_CLIPPED,
        enable_impulse_blanker=ENABLE_IMPULSE_BLANKER,
        blanker_median_kernel=BLANKER_MEDIAN_KERNEL,
        blanker_threshold_sigma=BLANKER_THRESHOLD_SIGMA,
        blanker_max_spike_width=BLANKER_MAX_SPIKE_WIDTH,
        enable_peak_thinning=ENABLE_PEAK_THINNING,
        thin_percentile=THIN_PERCENTILE,
        thin_target_width=THIN_TARGET_WIDTH,
        thin_apply_max_run=THIN_APPLY_MAX_RUN,
        enable_saturation_thinning=ENABLE_SATURATION_THINNING,
        saturation_level=SATURATION_LEVEL,
        sat_target_width=SAT_TARGET_WIDTH,
        sat_max_run=SAT_MAX_RUN,
    )


def iter_iq_chunks_from_dat(
    dat_path: str,
    chunk_size: int,
    dat_format: str = "float32_iq",
    normalize_int16: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield complex IQ chunks from a DAT file."""
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

            samples_to_read = chunk_size
            if max_iq_samples is not None:
                samples_to_read = min(samples_to_read, max_iq_samples - total_read)
                if samples_to_read <= 0:
                    break

            bytes_to_read = 2 * samples_to_read * bytes_per_scalar
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

            total_read += iq_count
            yield i_values + 1j * q_values


def convert_dat_to_spectrograms(
    dat_path: str,
    output_dir: str,
    sample_rate: int = 28_000_000,
    stft_point: int = 2048,
    duration_time: float = 0.05,
    chunk_size: int = 4096,
    prefix: str = "spectrogram",
    dat_format: str = "float32_iq",
    normalize_int16: bool = False,
    max_duration_seconds: int | None = None,
    processing: ProcessingConfig | None = None,
) -> int:
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"DAT file not found: {dat_path}")

    processing = processing or ProcessingConfig()
    os.makedirs(output_dir, exist_ok=True)
    waveform_dir = os.path.join(output_dir, "waveforms")
    if processing.save_waveform:
        os.makedirs(waveform_dir, exist_ok=True)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)
        print(f"Reading first {max_duration_seconds}s ({max_iq_samples} IQ samples)...")

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] chunk_size={chunk_size} is smaller than the required minimum "
            f"{min_samples_needed} samples (max(STFT_POINT, SAMPLE_RATE * DURATION_TIME)). "
            f"Using {min_samples_needed} instead."
        )
        chunk_size = min_samples_needed

    waveform_dir = os.path.join(output_dir, "waveforms")
    if SAVE_WAVEFORM:
        os.makedirs(waveform_dir, exist_ok=True)

    saved_count = 0
    chunks = iter_iq_chunks_from_dat(
        dat_path=dat_path,
        chunk_size=chunk_size,
        dat_format=dat_format,
        normalize_int16=normalize_int16,
        max_iq_samples=max_iq_samples,
    )
    for index, iq_chunk in enumerate(chunks, start=1):
        if iq_chunk.size < min_samples_needed:
            print(f"[SKIP] Chunk {index}: only {iq_chunk.size} samples, need {min_samples_needed}")
            continue

        iq_chunk = preprocess_iq_chunk(iq_chunk, processing)

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

            if processing.save_waveform:
                waveform_path = os.path.join(
                    waveform_dir, f"{processing.waveform_prefix}_{index:06d}.png"
                )
                waveform_title = (
                    f"{os.path.basename(dat_path)} | chunk={index} | "
                    f"samples={iq_chunk.size}"
                )
                save_waveform_image(
                    iq=iq_chunk,
                    sample_rate=sample_rate,
                    output_path=waveform_path,
                    title=waveform_title,
                    max_points=processing.waveform_max_points,
                )

            saved_count += 1
            print(f"[OK] Saved {output_path}")
        except Exception as exc:
            print(f"[ERROR] Chunk {index}: {exc}")
            continue

    print(f"\nDone. Total spectrograms saved: {saved_count}")
    print(f"Output directory: {output_dir}")
    return saved_count


def main() -> None:
    apply_spectrogram_config()
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
        processing=build_processing_config(),
    )


if __name__ == "__main__":
    main()
