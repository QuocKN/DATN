#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert RF IQ data in MAT file to spectrogram PNG images."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nkquoc.base import iq_spectrogram_core as spectrogram_core
from nkquoc.mat_spectrogram.converter import convert_mat_to_spectrograms

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
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True
MAX_DURATION_SECONDS: int | None = None

ENABLE_SPEC_COLUMN_DENOISE = True
SPEC_COLUMN_QUANTILE = 60.0


def apply_spectrogram_config() -> None:
    spectrogram_core.IMAGE_SIZE = IMAGE_SIZE
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = ENABLE_SPEC_COLUMN_DENOISE
    spectrogram_core.SPEC_COLUMN_QUANTILE = SPEC_COLUMN_QUANTILE


def main() -> None:
    apply_spectrogram_config()
    chunk_size = max(STFT_POINT, int(SAMPLE_RATE * DURATION_TIME))
    convert_mat_to_spectrograms(
        mat_path=INPUT_MAT_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=chunk_size,
        prefix=OUTPUT_PREFIX,
        mat_iq_key=MAT_IQ_KEY,
        max_duration_seconds=MAX_DURATION_SECONDS,
        save_waveform=SAVE_WAVEFORM,
        waveform_prefix=WAVEFORM_PREFIX,
        waveform_max_points=WAVEFORM_MAX_POINTS,
    )


if __name__ == "__main__":
    main()
