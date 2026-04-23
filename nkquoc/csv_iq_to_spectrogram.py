#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read two CSV files:
- High/I CSV: in-phase samples
- Low/Q CSV: quadrature samples

Then convert the I/Q stream into spectrogram images and save them into a folder.

Assumptions:
- Both CSV files contain numeric values in the same order and same length.
- The CSV may have one long row or multiple rows; all numeric tokens are collected.
- Output spectrograms are saved as PNG files.
"""

import csv
import os
import random
import re
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from graphic.RawDataProcessor import STFT


# ========================
# CONFIG
# ========================
DATASET_ROOT = r"C:\Users\DiepHM\Documents\data\DroneRF"
OUTPUT_ROOT = r"C:\Users\DiepHM\Documents\data\DroneRF\DroneRF_spectrogram"

SAMPLE_RATE = 40000000
STFT_POINT = 2048
DURATION_TIME = 0.05
CHUNK_SIZE = 4096
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224

TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42


def iter_csv_numbers(csv_path: str) -> Generator[float, None, None]:
    """Yield numeric values from a CSV file without loading everything into RAM."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            for token in row:
                token = token.strip()
                if token:
                    yield float(token)


def iter_iq_chunks(
    high_csv_path: str,
    low_csv_path: str,
    chunk_size: int,
) -> Generator[np.ndarray, None, None]:
    """Yield complex IQ chunks from two CSV streams."""
    high_iter = iter_csv_numbers(high_csv_path)
    low_iter = iter_csv_numbers(low_csv_path)

    while True:
        high_chunk = []
        low_chunk = []

        try:
            for _ in range(chunk_size):
                high_chunk.append(next(high_iter))
                low_chunk.append(next(low_iter))
        except StopIteration:
            break

        if len(high_chunk) != len(low_chunk) or not high_chunk:
            break

        i_values = np.asarray(high_chunk, dtype=np.float32)
        q_values = np.asarray(low_chunk, dtype=np.float32)
        yield i_values + 1j * q_values


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute two-sided STFT spectrogram using RawDataProcessor.STFT."""
    slice_point = int(sample_rate * duration_time)
    if slice_point <= 0:
        raise ValueError("sample_rate * duration_time must be positive")

    segment = data[:slice_point]
    if segment.size < stft_point:
        raise ValueError(
            f"Chunk too small for STFT: got {segment.size} samples, need at least {stft_point}."
        )

    frequencies, times, spectrum = STFT(
        segment,
        onside=False,
        stft_point=stft_point,
        fs=sample_rate,
        duration_time=duration_time,
    )
    frequencies = np.fft.fftshift(frequencies)
    spectrum = np.fft.fftshift(spectrum, axes=0)
    return frequencies, times, spectrum


def save_spectrogram_image(
    frequencies: np.ndarray,
    times: np.ndarray,
    spectrum: np.ndarray,
    output_path: str,
    title: str | None = None,
) -> None:
    """Save a spectrogram image without axes or labels."""
    magnitude_db = 10 * np.log10(np.abs(spectrum) + 1e-12)
    extent = (times.min(), times.max(), frequencies.min(), frequencies.max())

    dpi = 100
    figure = plt.figure(figsize=(IMAGE_SIZE / dpi, IMAGE_SIZE / dpi), dpi=dpi)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.imshow(magnitude_db, extent=extent, aspect="auto", origin="lower", cmap="jet")
    axes.axis("off")
    if title:
        axes.set_title(title)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)

    # Enforce exact 224x224 output to match model input size.
    with Image.open(output_path) as image:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        image.save(output_path)


def convert_csv_pair_to_spectrograms(
    high_csv_path: str,
    low_csv_path: str,
    output_dir: str,
    sample_rate: int = 40_000_000,
    stft_point: int = 2048,
    duration_time: float = 0.05,
    chunk_size: int = 4096,
    prefix: str = "spectrogram",
) -> int:
    """Convert paired I/Q CSV files into a folder of spectrogram PNG images."""
    if not os.path.exists(high_csv_path):
        raise FileNotFoundError(f"High/I CSV not found: {high_csv_path}")
    if not os.path.exists(low_csv_path):
        raise FileNotFoundError(f"Low/Q CSV not found: {low_csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] chunk_size={chunk_size} is smaller than the required minimum "
            f"{min_samples_needed} samples (max(STFT_POINT, SAMPLE_RATE * DURATION_TIME)). "
            f"Using {min_samples_needed} instead."
        )
        chunk_size = min_samples_needed

    saved_count = 0
    for index, iq_chunk in enumerate(iter_iq_chunks(high_csv_path, low_csv_path, chunk_size), start=1):
        if iq_chunk.size < min_samples_needed:
            continue

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
        print(f"Saved {output_path}")

    print(f"Done. Total spectrograms saved for pair: {saved_count}")
    return saved_count


def parse_role_and_key(csv_name: str):
    base = os.path.splitext(csv_name)[0]
    # Example: 10100H_0 or 10100L_0
    match = re.match(r"^(.*?)([HhLl])_(\d+)$", base)
    if not match:
        return None, None

    prefix, role, index = match.groups()
    role = role.upper()
    pair_key = f"{prefix}_{index}"
    return role, pair_key


def find_all_hl_pairs(dataset_root: str):
    pairs = {}

    for root, _, files in os.walk(dataset_root):
        for file_name in files:
            if not file_name.lower().endswith(".csv"):
                continue

            role, pair_key = parse_role_and_key(file_name)
            if role is None:
                continue

            csv_path = os.path.join(root, file_name)
            rel_parent = os.path.relpath(root, dataset_root)
            rel_parts = rel_parent.split(os.sep)
            class_name = rel_parts[0] if rel_parts else "unknown"

            if pair_key not in pairs:
                pairs[pair_key] = {
                    "H": None,
                    "L": None,
                    "class_name": class_name,
                    "pair_key": pair_key,
                }

            pairs[pair_key][role] = csv_path

    complete = []
    missing = []
    for pair in pairs.values():
        if pair["H"] and pair["L"]:
            complete.append(pair)
        else:
            missing.append(pair)

    return complete, missing


def split_pairs(complete_pairs):
    rng = random.Random(RANDOM_SEED)
    by_class = {}
    for pair in complete_pairs:
        by_class.setdefault(pair["class_name"], []).append(pair)

    train_pairs = []
    valid_pairs = []
    test_pairs = []

    for class_pairs in by_class.values():
        rng.shuffle(class_pairs)
        n = len(class_pairs)

        n_train = int(round(n * TRAIN_RATIO))
        n_valid = int(round(n * VALID_RATIO))

        # Keep split sizes valid and leave remaining to test.
        n_train = min(n_train, n)
        n_valid = min(n_valid, n - n_train)
        n_test = n - n_train - n_valid

        train_pairs.extend(class_pairs[:n_train])
        valid_pairs.extend(class_pairs[n_train:n_train + n_valid])
        test_pairs.extend(class_pairs[n_train + n_valid:n_train + n_valid + n_test])

    return train_pairs, valid_pairs, test_pairs


def process_split(split_name: str, split_pairs_list, processed_pairs: int, total_pairs: int):
    split_root = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(split_root, exist_ok=True)

    total_images = 0
    current_processed = processed_pairs
    for i, pair in enumerate(split_pairs_list, start=1):
        class_dir = os.path.join(split_root, pair["class_name"], pair["pair_key"])
        current_processed += 1
        overall_pct = (current_processed / total_pairs) * 100 if total_pairs > 0 else 100.0
        print(
            f"[{split_name}] {i}/{len(split_pairs_list)} -> {pair['pair_key']} "
            f"| overall {current_processed}/{total_pairs} ({overall_pct:.2f}%)"
        )
        saved = convert_csv_pair_to_spectrograms(
            high_csv_path=pair["H"],
            low_csv_path=pair["L"],
            output_dir=class_dir,
            sample_rate=SAMPLE_RATE,
            stft_point=STFT_POINT,
            duration_time=DURATION_TIME,
            chunk_size=CHUNK_SIZE,
            prefix=OUTPUT_PREFIX,
        )
        total_images += saved

    print(f"[{split_name}] pairs={len(split_pairs_list)}, images={total_images}")
    return total_images, current_processed


def main() -> None:
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    ratio_sum = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.0")

    complete_pairs, missing_pairs = find_all_hl_pairs(DATASET_ROOT)
    print(f"Found complete H/L pairs: {len(complete_pairs)}")
    print(f"Found incomplete pairs: {len(missing_pairs)}")
    if missing_pairs:
        print("Incomplete pair keys (missing H or L):")
        for pair in missing_pairs[:20]:
            print(f"- {pair['pair_key']} (H={pair['H'] is not None}, L={pair['L'] is not None})")

    if not complete_pairs:
        print("No complete H/L pairs found. Nothing to process.")
        return

    train_pairs, valid_pairs, test_pairs = split_pairs(complete_pairs)
    print(f"Split -> train={len(train_pairs)}, valid={len(valid_pairs)}, test={len(test_pairs)}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    total_pairs = len(train_pairs) + len(valid_pairs) + len(test_pairs)
    processed_pairs = 0

    total_train, processed_pairs = process_split("train", train_pairs, processed_pairs, total_pairs)
    total_valid, processed_pairs = process_split("valid", valid_pairs, processed_pairs, total_pairs)
    total_test, processed_pairs = process_split("test", test_pairs, processed_pairs, total_pairs)

    print("All done.")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Total images -> train={total_train}, valid={total_valid}, test={total_test}")


if __name__ == "__main__":
    main()
