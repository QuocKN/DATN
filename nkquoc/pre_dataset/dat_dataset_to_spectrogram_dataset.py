#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a spectrogram image dataset from a directory tree of .dat RF recordings.

Expected input layout:
    DATASET_ROOT/
        CLEAN/
            DRONE_FOLDER/
                *.dat
        WIFI/
            DRONE_FOLDER/
                *.dat
        BLUE/
            DRONE_FOLDER/
                *.dat
        BOTH/
            DRONE_FOLDER/
                *.dat

The script splits by recording file first, then converts each split into
spectrogram PNG images.

Output layout:
    OUTPUT_ROOT/
    train/CLEAN/<label>/*.png
    train/WIFI/<label>/*.png
    train/BLUE/<label>/*.png
    train/BOTH/<label>/*.png
    valid/...
    test/...

Drone folder information is kept both in folder name (<label>) and filename.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import random
import re
from typing import Generator

import numpy as np
from tqdm import tqdm
from nkquoc.base.iq_spectrogram_core import compute_spectrogram, save_spectrogram_image


# ========================
# CONFIG
# ========================
DATASET_ROOT = r"C:\Users\DiepHM\Documents\data"
SOURCE_FOLDERS = ("CLEAN", "WIFI", "BLUE", "BOTH")
OUTPUT_ROOT = r"C:\Users\DiepHM\Documents\data\DroneDetect_spectrogram_dataset"

SAMPLE_RATE = 60_000_000
STFT_POINT = 2048

# Each recording is 2 seconds long. Use 100 ms windows with 50% overlap.
WINDOW_SECONDS = 0.1
OVERLAP_RATIO = 0.5

MAX_DURATION_SECONDS = 2

# Supported values: "float32_iq" or "int16_iq"
DAT_FORMAT = "float32_iq"
NORMALIZE_INT16 = False

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

SKIP_EXISTING_IMAGES = True

# Parallel conversion settings
ENABLE_PARALLEL = True
PARALLEL_BACKEND = "process"  # "process" or "thread"
NUM_WORKERS = 5  # 0 = auto (cpu_count - 1)


def safe_name(text: str) -> str:
    """Convert a path segment into a filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def derive_label(source_root: str, file_path: str) -> str:
    """Derive the class label from the directory containing a .dat file."""
    parent_dir = os.path.dirname(file_path)
    relative_dir = os.path.relpath(parent_dir, source_root)
    if relative_dir in {".", ""}:
        return "unknown"

    parts = [part for part in relative_dir.split(os.sep) if part not in {".", ""}]
    return safe_name("__".join(parts))


def discover_dat_files(dataset_root: str, source_folders: tuple[str, ...]):
    """Collect all .dat recordings from the available source folders."""
    records = []

    for source_folder in source_folders:
        source_root = os.path.join(dataset_root, source_folder)
        if not os.path.isdir(source_root):
            continue

        for root, _, files in os.walk(source_root):
            files = sorted(files)
            for file_name in files:
                if not file_name.lower().endswith(".dat"):
                    continue

                dat_path = os.path.join(root, file_name)
                label = derive_label(source_root, dat_path)
                records.append(
                    {
                        "source_group": source_folder,
                        "label": label,
                        "dat_path": dat_path,
                        "file_stem": os.path.splitext(file_name)[0],
                    }
                )

    return sorted(records, key=lambda item: (item["label"], item["source_group"], item["dat_path"]))


def split_counts_for_group(
    n: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Split a small group while keeping valid/test non-empty whenever possible."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 1, 1, 1

    valid = max(1, int(round(n * valid_ratio)))
    test = max(1, int(round(n * test_ratio)))
    train = n - valid - test

    if train < 1:
        train = 1
        excess = train + valid + test - n

        while excess > 0 and valid > 1:
            valid -= 1
            excess -= 1

        while excess > 0 and test > 1:
            test -= 1
            excess -= 1

        train = n - valid - test

    if train < 1:
        # Final fallback for degenerate rounding cases.
        train = max(1, n - 2)
        valid = 1 if n >= 2 else 0
        test = n - train - valid
        if test < 0:
            test = 0

    return train, valid, test


def split_records(records):
    """Split recording-level records by class label before spectrogram generation."""
    rng = random.Random(RANDOM_SEED)
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    train_records = []
    valid_records = []
    test_records = []

    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        n_train, n_valid, n_test = split_counts_for_group(
            len(items),
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
        )

        train_records.extend(items[:n_train])
        valid_records.extend(items[n_train:n_train + n_valid])
        test_records.extend(items[n_train + n_valid:n_train + n_valid + n_test])

    rng.shuffle(train_records)
    rng.shuffle(valid_records)
    rng.shuffle(test_records)
    return train_records, valid_records, test_records


def iter_iq_chunks_from_dat(
    dat_path: str,
    window_samples: int,
    hop_samples: int,
    dat_format: str = "float32_iq",
    normalize_int16: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield overlapping complex IQ windows from a .dat file."""
    if dat_format not in {"float32_iq", "int16_iq"}:
        raise ValueError("dat_format must be 'float32_iq' or 'int16_iq'")

    if dat_format == "float32_iq":
        dtype = np.dtype("<f4")
        bytes_per_scalar = 4
    else:
        dtype = np.dtype("<i2")
        bytes_per_scalar = 2

    hop_samples = max(1, hop_samples)
    total_read = 0
    buffer = np.empty(0, dtype=np.complex64)

    with open(dat_path, "rb") as f:
        while True:
            if max_iq_samples is not None and total_read >= max_iq_samples:
                break

            while buffer.size < window_samples:
                remaining_needed = window_samples - buffer.size
                if max_iq_samples is not None:
                    remaining_needed = min(remaining_needed, max_iq_samples - total_read)
                if remaining_needed <= 0:
                    break

                raw_bytes = f.read(2 * remaining_needed * bytes_per_scalar)
                if not raw_bytes:
                    break

                data = np.frombuffer(raw_bytes, dtype=dtype)
                if data.size % 2 != 0:
                    data = data[:-1]
                if data.size == 0:
                    break

                iq_pairs = data.reshape(-1, 2)
                i_values = iq_pairs[:, 0].astype(np.float32, copy=False)
                q_values = iq_pairs[:, 1].astype(np.float32, copy=False)

                if dat_format == "int16_iq" and normalize_int16:
                    i_values /= 32768.0
                    q_values /= 32768.0

                iq_chunk = i_values + 1j * q_values
                total_read += iq_chunk.size
                if buffer.size == 0:
                    buffer = iq_chunk
                else:
                    buffer = np.concatenate((buffer, iq_chunk))

            if buffer.size < window_samples:
                break

            yield buffer[:window_samples]
            buffer = buffer[hop_samples:]


def convert_dat_to_spectrograms(
    dat_path: str,
    output_dir: str,
    sample_rate: int,
    stft_point: int,
    window_seconds: float,
    overlap_ratio: float,
    dat_format: str,
    normalize_int16: bool,
    max_duration_seconds: int | None,
    prefix: str,
    source_group: str,
    label: str,
    show_window_progress: bool = True,
) -> int:
    """Convert a single .dat recording into multiple spectrogram images."""
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"DAT file not found: {dat_path}")

    os.makedirs(output_dir, exist_ok=True)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)

    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError("window_seconds must be positive")

    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    hop_samples = int(round(window_samples * (1.0 - overlap_ratio)))
    hop_samples = max(1, hop_samples)

    if window_samples < stft_point:
        raise ValueError(
            f"window size {window_samples} is smaller than STFT point {stft_point}."
        )

    estimated_windows = None
    if max_iq_samples is not None and max_iq_samples >= window_samples:
        estimated_windows = ((max_iq_samples - window_samples) // hop_samples) + 1

    safe_prefix = safe_name(prefix)
    safe_source = safe_name(source_group)
    safe_label = safe_name(label)
    base_name = safe_name(os.path.splitext(os.path.basename(dat_path))[0])

    saved_count = 0
    chunk_iter = iter_iq_chunks_from_dat(
        dat_path=dat_path,
        window_samples=window_samples,
        hop_samples=hop_samples,
        dat_format=dat_format,
        normalize_int16=normalize_int16,
        max_iq_samples=max_iq_samples,
    )
    if show_window_progress:
        chunk_iter = tqdm(
            chunk_iter,
            total=estimated_windows,
            desc=f"windows {base_name}",
            unit="win",
            dynamic_ncols=True,
            leave=False,
        )

    for index, iq_chunk in enumerate(chunk_iter, start=1):
        if iq_chunk.size < window_samples:
            continue

        output_path = os.path.join(
            output_dir,
            f"{safe_prefix}__{safe_source}__{safe_label}__{base_name}__w{index:05d}.png",
        )

        if SKIP_EXISTING_IMAGES and os.path.exists(output_path):
            continue

        frequencies, times, spectrum = compute_spectrogram(
            iq_chunk,
            sample_rate=sample_rate,
            stft_point=stft_point,
            duration_time=window_seconds,
        )

        save_spectrogram_image(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            output_path=output_path,
        )
        saved_count += 1

    return saved_count


def resolve_num_workers() -> int:
    if NUM_WORKERS and NUM_WORKERS > 0:
        return NUM_WORKERS
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def process_one_record(split_name: str, total: int, index: int, record: dict) -> int:
    print(
        f"[{split_name}] {index}/{total} -> {record['source_group']}/{record['label']} "
        f"| {os.path.basename(record['dat_path'])}"
    )
    split_root = os.path.join(OUTPUT_ROOT, split_name)
    source_dir = os.path.join(split_root, safe_name(record["source_group"]))
    label_dir = os.path.join(source_dir, safe_name(record["label"]))

    return convert_dat_to_spectrograms(
        dat_path=record["dat_path"],
        output_dir=label_dir,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        window_seconds=WINDOW_SECONDS,
        overlap_ratio=OVERLAP_RATIO,
        dat_format=DAT_FORMAT,
        normalize_int16=NORMALIZE_INT16,
        max_duration_seconds=MAX_DURATION_SECONDS,
        prefix="spectrogram",
        source_group=record["source_group"],
        label=record["label"],
        show_window_progress=True,
    )


def process_split(split_name: str, records: list[dict]) -> int:
    split_root = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(split_root, exist_ok=True)

    if not records:
        print(f"[{split_name}] recordings=0, images=0")
        return 0

    total_saved = 0

    use_parallel = ENABLE_PARALLEL and len(records) > 1
    if use_parallel:
        worker_count = resolve_num_workers()
        backend = PARALLEL_BACKEND.strip().lower()
        executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

        print(f"[{split_name}] parallel={backend}, workers={worker_count}")
        with executor_cls(max_workers=worker_count) as executor:
            futures = {}
            total = len(records)
            completed = 0
            for index, record in enumerate(records, start=1):
                future = executor.submit(process_one_record, split_name, total, index, record)
                futures[future] = record

            for future in as_completed(futures):
                record = futures[future]
                try:
                    total_saved += future.result()
                except Exception as exc:
                    print(
                        f"[ERROR] {split_name} -> {record['source_group']}/{record['label']} "
                        f"| {os.path.basename(record['dat_path'])}: {exc}"
                    )
                completed += 1
                print(f"[{split_name}] completed {completed}/{total} files")
    else:
        total = len(records)
        completed = 0
        for index, record in enumerate(records, start=1):
            print(
                f"[{split_name}] {index}/{total} -> {record['source_group']}/{record['label']} "
                f"| {os.path.basename(record['dat_path'])}"
            )
            source_dir = os.path.join(split_root, safe_name(record["source_group"]))
            label_dir = os.path.join(source_dir, safe_name(record["label"]))
            total_saved += convert_dat_to_spectrograms(
                dat_path=record["dat_path"],
                output_dir=label_dir,
                sample_rate=SAMPLE_RATE,
                stft_point=STFT_POINT,
                window_seconds=WINDOW_SECONDS,
                overlap_ratio=OVERLAP_RATIO,
                dat_format=DAT_FORMAT,
                normalize_int16=NORMALIZE_INT16,
                max_duration_seconds=MAX_DURATION_SECONDS,
                prefix="spectrogram",
                source_group=record["source_group"],
                label=record["label"],
                show_window_progress=True,
            )
            completed += 1
            print(f"[{split_name}] completed {completed}/{total} files")

    print(f"[{split_name}] recordings={len(records)}, images={total_saved}")
    return total_saved


def main() -> None:
    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    records = discover_dat_files(DATASET_ROOT, SOURCE_FOLDERS)
    if not records:
        print("No .dat files found. Nothing to process.")
        return

    train_records, valid_records, test_records = split_records(records)

    print(f"Found recordings: {len(records)}")
    print(
        f"Split -> train={len(train_records)}, valid={len(valid_records)}, test={len(test_records)}"
    )
    print(f"Output root: {OUTPUT_ROOT}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total_train = process_split("train", train_records)
    total_valid = process_split("valid", valid_records)
    total_test = process_split("test", test_records)

    print("All done.")
    print(f"Total images -> train={total_train}, valid={total_valid}, test={total_test}")


if __name__ == "__main__":
    main()
