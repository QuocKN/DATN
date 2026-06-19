#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert RF IQ data stored in a .dat file into spectrogram PNG images.

Supported DAT payload formats:
- float32_iq: interleaved little-endian float32 [I, Q, I, Q, ...]
- int16_iq:   interleaved little-endian int16   [I, Q, I, Q, ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nkquoc.base import iq_spectrogram_core as spectrogram_core
from nkquoc.dat_spectrogram.converter import convert_dat_to_spectrograms

# ========================
# CONFIG
# ========================
INPUT_DAT_PATH = r"g:\DATN_DATA\RF\BladeRF\drone_indoor.dat"
OUTPUT_DIR = r"G:\DATN_DATA\RF\BladeRF\drone_indoor_spectrograms"
SAMPLE_RATE = 40_000_000
STFT_POINT = 2048
DURATION_TIME = 0.05
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True
MAX_DURATION_SECONDS = 5

DAT_FORMAT = "float32_iq"  # "float32_iq" | "int16_iq"
NORMALIZE_INT16 = False

ENABLE_SPEC_COLUMN_DENOISE = True
SPEC_COLUMN_QUANTILE = 10.0


def apply_spectrogram_config() -> None:
    spectrogram_core.IMAGE_SIZE = IMAGE_SIZE
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = ENABLE_SPEC_COLUMN_DENOISE
    spectrogram_core.SPEC_COLUMN_QUANTILE = SPEC_COLUMN_QUANTILE


def main() -> None:
    apply_spectrogram_config()
    chunk_size = max(STFT_POINT, int(SAMPLE_RATE * DURATION_TIME))
    convert_dat_to_spectrograms(
        dat_path=INPUT_DAT_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=chunk_size,
        prefix=OUTPUT_PREFIX,
        dat_format=DAT_FORMAT,
        normalize_int16=NORMALIZE_INT16,
        max_duration_seconds=MAX_DURATION_SECONDS,
        save_waveform=SAVE_WAVEFORM,
        waveform_prefix=WAVEFORM_PREFIX,
        waveform_max_points=WAVEFORM_MAX_POINTS,
    )


if __name__ == "__main__":
    main()
