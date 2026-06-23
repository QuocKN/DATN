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
INPUT_DAT_PATH = r"F:\Data_22_6"
OUTPUT_DIR = r"F:\Data_22_6_spectrogram_0_05"
SAMPLE_RATE = 40_000_000
STFT_POINT = 1024
DURATION_TIME = 0.05
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = False
MAX_DURATION_SECONDS = 20


DAT_FORMAT = "float32_iq"  # "float32_iq" | "int16_iq"
NORMALIZE_INT16 = False

ENABLE_SPEC_COLUMN_DENOISE = False
SPEC_COLUMN_QUANTILE = 10.0


def apply_spectrogram_config() -> None:
    spectrogram_core.IMAGE_SIZE = IMAGE_SIZE
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = ENABLE_SPEC_COLUMN_DENOISE
    spectrogram_core.SPEC_COLUMN_QUANTILE = SPEC_COLUMN_QUANTILE


def main() -> None:
    apply_spectrogram_config()
    chunk_size = max(STFT_POINT, int(SAMPLE_RATE * DURATION_TIME))

    input_path = Path(INPUT_DAT_PATH)
    output_root = Path(OUTPUT_DIR)
    dat_files = sorted(input_path.glob("*.dat")) if input_path.is_dir() else [input_path]

    if not dat_files:
        raise FileNotFoundError(f"No DAT files found: {INPUT_DAT_PATH}")

    total_saved = 0
    for dat_file in dat_files:
        output_dir = output_root / f"{dat_file.stem}_spectrograms" if input_path.is_dir() else output_root
        print(f"\n=== Converting {dat_file} -> {output_dir} ===")
        total_saved += convert_dat_to_spectrograms(
            dat_path=str(dat_file),
            output_dir=str(output_dir),
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

    print(f"\nAll DAT files converted. Total spectrograms saved: {total_saved}")


if __name__ == "__main__":
    main()
