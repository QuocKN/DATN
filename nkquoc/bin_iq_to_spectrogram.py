#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert binary RF IQ data (int16 interleaved I,Q,I,Q,...) into spectrogram PNG images.

This file is intentionally kept as a small entrypoint. The reusable pieces live in:
- bin_iq_reader.py
- iq_spectrogram_core.py
- iq_preprocessing.py
- bin_spectrogram_converter.py
"""

from __future__ import annotations

try:
    from . import iq_spectrogram_core as spectrogram_core
    from .bin_iq_reader import iter_iq_chunks_from_bin
    from .bin_spectrogram_converter import ProcessingConfig, convert_bin_to_spectrograms as _convert_bin
    from .iq_preprocessing import (
        blank_impulsive_spikes,
        check_iq_amplitude,
        despike_iq,
        repair_clipped_iq,
        thin_high_amplitude_runs,
        thin_saturation_runs,
    )
    from .iq_spectrogram_core import (
        STFT,
        compute_spectrogram,
        downsample_for_plot,
        save_spectrogram_image,
        save_waveform_image,
    )
except ImportError:
    import iq_spectrogram_core as spectrogram_core
    from bin_iq_reader import iter_iq_chunks_from_bin
    from bin_spectrogram_converter import ProcessingConfig, convert_bin_to_spectrograms as _convert_bin
    from iq_preprocessing import (
        blank_impulsive_spikes,
        check_iq_amplitude,
        despike_iq,
        repair_clipped_iq,
        thin_high_amplitude_runs,
        thin_saturation_runs,
    )
    from iq_spectrogram_core import (
        STFT,
        compute_spectrogram,
        downsample_for_plot,
        save_spectrogram_image,
        save_waveform_image,
    )


# ========================
# CONFIG
# ========================
INPUT_BIN_PATH = "e:\\DATN_DATA\\RF\\Tu_thu\\2toan.bin"
OUTPUT_DIR = "e:\\DATN_DATA\\RF\\Tu_thu\\2toan_spectrograms_refactor"
# INPUT_BIN_PATH = "E:\\DATN_DATA\\RF\\1toan.bin"
# OUTPUT_DIR = "E:\\DATN_DATA\\RF\\1toan_spectrograms"

SAMPLE_RATE = 28_000_000
STFT_POINT = 1024
DURATION_TIME = 0.05
CHUNK_SIZE = 4096
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True
MAX_DURATION_SECONDS = 500
NORMALIZE = False

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


def convert_bin_to_spectrograms(
    bin_path: str,
    output_dir: str,
    sample_rate: int = 28_000_000,
    stft_point: int = 2048,
    duration_time: float = 0.05,
    chunk_size: int = 4096,
    prefix: str = "spectrogram",
    normalize: bool = False,
    max_duration_seconds: int | None = None,
) -> int:
    apply_spectrogram_config()
    return _convert_bin(
        bin_path=bin_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        stft_point=stft_point,
        duration_time=duration_time,
        chunk_size=chunk_size,
        prefix=prefix,
        normalize=normalize,
        max_duration_seconds=max_duration_seconds,
        processing=build_processing_config(),
    )


def main() -> None:
    convert_bin_to_spectrograms(
        bin_path=INPUT_BIN_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=CHUNK_SIZE,
        prefix=OUTPUT_PREFIX,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )


if __name__ == "__main__":
    main()
