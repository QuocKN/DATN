#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert binary RF IQ data (int16 interleaved I,Q,I,Q,...) into spectrogram PNG images.

This file is intentionally kept as a small entrypoint. The reusable pieces live in:
- nkquoc/iq_spectrogram_core.py
- nkquoc/bin_spectrogram/converter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nkquoc.base import iq_spectrogram_core as spectrogram_core
from nkquoc.bin_spectrogram.converter import convert_bin_to_spectrograms

# ========================
# CONFIG
# ========================
INPUT_BIN_PATH = r"G:\DATN_DATA\RF\Tu_thu\signal1.bin"
OUTPUT_DIR = r"G:\DATN_DATA\RF\Tu_thu\signal1_spectrograms_v1"
# INPUT_BIN_PATH = "E:\\DATN_DATA\\RF\\1toan.bin"
# OUTPUT_DIR = "E:\\DATN_DATA\\RF\\1toan_spectrograms"

SAMPLE_RATE = 60_000_000
STFT_POINT = 1024
DURATION_TIME = 0.05
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True
MAX_DURATION_SECONDS = 500
NORMALIZE = False

ENABLE_SPEC_COLUMN_DENOISE = False
SPEC_COLUMN_QUANTILE =40.0


def apply_spectrogram_config() -> None:
    spectrogram_core.IMAGE_SIZE = IMAGE_SIZE
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = ENABLE_SPEC_COLUMN_DENOISE
    spectrogram_core.SPEC_COLUMN_QUANTILE = SPEC_COLUMN_QUANTILE


def main() -> None:
    apply_spectrogram_config()
    chunk_size = max(STFT_POINT, int(SAMPLE_RATE * DURATION_TIME))
    convert_bin_to_spectrograms(
        bin_path=INPUT_BIN_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=chunk_size,
        prefix=OUTPUT_PREFIX,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
        save_waveform=SAVE_WAVEFORM,
        waveform_prefix=WAVEFORM_PREFIX,
        waveform_max_points=WAVEFORM_MAX_POINTS,
    )


if __name__ == "__main__":
    main()
