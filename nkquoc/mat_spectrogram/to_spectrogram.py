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
INPUT_MAT_PATH = r"f:\Data_22_6_spectrogram_0_05\MavicRC1"
OUTPUT_DIR = r"f:\Data_22_6_spectrogram_0_05\MavicRC1\spectrograms"

# Set to base key if needed (e.g., "RF0" -> RF0_I/RF0_Q), else None for auto-detect
MAT_IQ_KEY: str | None = None

SAMPLE_RATE = 100_000_000
STFT_POINT = 1024
DURATION_TIME = 0.05
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = False
MAX_DURATION_SECONDS = 20

ENABLE_SPEC_COLUMN_DENOISE = False
SPEC_COLUMN_QUANTILE = 10.0
ENABLE_SPEC_ROW_DENOISE = False
SPEC_ROW_QUANTILE = 50.0
ENABLE_SPEC_DC_MASK = True
SPEC_DC_MASK_BINS = 1
ENABLE_SPEC_FIXED_DB_RANGE = False
SPEC_DB_VMIN = -57.0
SPEC_DB_VMAX = 16.0
ENABLE_SPEC_DB_CLIP = False
SPECTROGRAM_CMAP = "jet"
SAVE_GRAYSCALE = False
REMOVE_DC = True


def apply_spectrogram_config() -> None:
    spectrogram_core.IMAGE_SIZE = IMAGE_SIZE
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = ENABLE_SPEC_COLUMN_DENOISE
    spectrogram_core.SPEC_COLUMN_QUANTILE = SPEC_COLUMN_QUANTILE
    spectrogram_core.ENABLE_SPEC_ROW_DENOISE = ENABLE_SPEC_ROW_DENOISE
    spectrogram_core.SPEC_ROW_QUANTILE = SPEC_ROW_QUANTILE
    spectrogram_core.ENABLE_SPEC_DC_MASK = ENABLE_SPEC_DC_MASK
    spectrogram_core.SPEC_DC_MASK_BINS = SPEC_DC_MASK_BINS
    spectrogram_core.ENABLE_SPEC_FIXED_DB_RANGE = ENABLE_SPEC_FIXED_DB_RANGE
    spectrogram_core.SPEC_DB_VMIN = SPEC_DB_VMIN
    spectrogram_core.SPEC_DB_VMAX = SPEC_DB_VMAX
    spectrogram_core.ENABLE_SPEC_DB_CLIP = ENABLE_SPEC_DB_CLIP
    spectrogram_core.SPECTROGRAM_CMAP = SPECTROGRAM_CMAP
    spectrogram_core.SAVE_GRAYSCALE = SAVE_GRAYSCALE


def main() -> None:
    apply_spectrogram_config()
    chunk_size = max(STFT_POINT, int(SAMPLE_RATE * DURATION_TIME))

    input_path = Path(INPUT_MAT_PATH)
    output_root = Path(OUTPUT_DIR)
    mat_files = sorted(input_path.glob("*.mat")) if input_path.is_dir() else [input_path]

    if not mat_files:
        raise FileNotFoundError(f"No MAT files found: {INPUT_MAT_PATH}")

    total_saved = 0
    for mat_file in mat_files:
        output_dir = output_root / f"{mat_file.stem}_spectrograms" if input_path.is_dir() else output_root
        print(f"\n=== Converting {mat_file} -> {output_dir} ===")
        total_saved += convert_mat_to_spectrograms(
            mat_path=str(mat_file),
            output_dir=str(output_dir),
            sample_rate=SAMPLE_RATE,
            stft_point=STFT_POINT,
            duration_time=DURATION_TIME,
            chunk_size=chunk_size,
            prefix=OUTPUT_PREFIX,
            mat_iq_key=MAT_IQ_KEY,
            remove_dc=REMOVE_DC,
            max_duration_seconds=MAX_DURATION_SECONDS,
            save_waveform=SAVE_WAVEFORM,
            waveform_prefix=WAVEFORM_PREFIX,
            waveform_max_points=WAVEFORM_MAX_POINTS,
        )

    print(f"\nAll MAT files converted. Total spectrograms saved: {total_saved}")


if __name__ == "__main__":
    main()
