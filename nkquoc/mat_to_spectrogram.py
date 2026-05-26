#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert RF IQ data in MAT file to spectrogram PNG images."""

from __future__ import annotations

import os

try:
    from . import iq_spectrogram_core as spectrogram_core
    from .bin_spectrogram_converter import ProcessingConfig, preprocess_iq_chunk
    from .iq_spectrogram_core import compute_spectrogram, save_spectrogram_image, save_waveform_image
    from .mat_iq_reader import iter_iq_chunks_from_mat
except ImportError:
    import iq_spectrogram_core as spectrogram_core
    from bin_spectrogram_converter import ProcessingConfig, preprocess_iq_chunk
    from iq_spectrogram_core import compute_spectrogram, save_spectrogram_image, save_waveform_image
    from mat_iq_reader import iter_iq_chunks_from_mat


# ========================
# CONFIG
# ========================
INPUT_MAT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/DRFF_R2/mavic3C_1_Ascend_c17_u1_d2.mat"
OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/DRFF_R2/spectrograms"

# Set to base key if needed (e.g., "RF0" -> RF0_I/RF0_Q), else None for auto-detect
MAT_IQ_KEY: str | None = None

SAMPLE_RATE = 100_000_000
STFT_POINT = 1024
DURATION_TIME = 0.05
CHUNK_SIZE = 1_000_000
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True
MAX_DURATION_SECONDS: int | None = None

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


def convert_mat_to_spectrograms(
    mat_path: str,
    output_dir: str,
    sample_rate: int = 20_000_000,
    stft_point: int = 1024,
    duration_time: float = 0.05,
    chunk_size: int = 1_000_000,
    prefix: str = "spectrogram",
    mat_iq_key: str | None = None,
    max_duration_seconds: int | None = None,
    processing: ProcessingConfig | None = None,
) -> int:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

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

    saved_count = 0
    chunks = iter_iq_chunks_from_mat(
        mat_path=mat_path,
        chunk_size=chunk_size,
        mat_iq_key=mat_iq_key,
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
                    f"{os.path.basename(mat_path)} | chunk={index} | "
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
    convert_mat_to_spectrograms(
        mat_path=INPUT_MAT_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=CHUNK_SIZE,
        prefix=OUTPUT_PREFIX,
        mat_iq_key=MAT_IQ_KEY,
        max_duration_seconds=MAX_DURATION_SECONDS,
        processing=build_processing_config(),
    )


if __name__ == "__main__":
    main()
