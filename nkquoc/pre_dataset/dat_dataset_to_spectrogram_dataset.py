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

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Callable, Generator

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nkquoc.base.iq_spectrogram_core import compute_spectrogram, save_spectrogram_image


# ========================
# CONFIG
# ========================
DATASET_ROOT = r"F:\DroneDetect_V2"
SOURCE_FOLDERS = ("CLEAN", "WIFI", "BLUE", "BOTH")
OUTPUT_ROOT = r"E:\DATN_Data_21_6\dataset_40_0_1\drone"

SAMPLE_RATE = 40_000_000
STFT_POINT = 1024

# Each recording is limited to 5 seconds. Use 50 ms windows.
WINDOW_SECONDS = 0.1
OVERLAP_RATIO = 0.05

MAX_DURATION_SECONDS = 5

ENABLE_IMAGE_SCALING = True
SCALE_TOP_RATIO = 1 / 6
SCALE_BOTTOM_RATIO = 5 / 6

# Supported values: "float32_iq" or "int16_iq"
DAT_FORMAT = "float32_iq"
NORMALIZE_INT16 = False

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

SKIP_EXISTING_IMAGES = True
ENABLE_PROGRESS_LOG = True
PROGRESS_LOG_PATH = ""  # Empty = OUTPUT_ROOT/_spectrogram_progress.json
PROGRESS_LOCK_STALE_SECONDS = 60
CHECKPOINT_EVERY_WINDOWS = 1
RESUME_FROM_EXISTING_IMAGES = True

# Parallel conversion settings
ENABLE_PARALLEL = True
PARALLEL_BACKEND = "process"  # "process" or "thread"
NUM_WORKERS = 4  # 0 = auto (cpu_count - 1)


def safe_name(text: str) -> str:
    """Convert a path segment into a filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def now_text() -> str:
    """Return a compact local timestamp for progress logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalized_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def get_progress_log_path() -> str:
    if PROGRESS_LOG_PATH:
        return normalized_path(PROGRESS_LOG_PATH)
    return os.path.join(normalized_path(OUTPUT_ROOT), "_spectrogram_progress.json")


def progress_config_snapshot() -> dict:
    return {
        "dataset_root": normalized_path(DATASET_ROOT),
        "source_folders": list(SOURCE_FOLDERS),
        "output_root": normalized_path(OUTPUT_ROOT),
        "sample_rate": SAMPLE_RATE,
        "stft_point": STFT_POINT,
        "window_seconds": WINDOW_SECONDS,
        "overlap_ratio": OVERLAP_RATIO,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "dat_format": DAT_FORMAT,
        "normalize_int16": NORMALIZE_INT16,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "enable_image_scaling": ENABLE_IMAGE_SCALING,
        "scale_top_ratio": SCALE_TOP_RATIO,
        "scale_bottom_ratio": SCALE_BOTTOM_RATIO,
        "skip_existing_images": SKIP_EXISTING_IMAGES,
    }


def progress_config_signature() -> str:
    encoded = json.dumps(
        progress_config_snapshot(),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def new_progress_state() -> dict:
    config = progress_config_snapshot()
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return {
        "version": 1,
        "config_signature": hashlib.sha1(encoded).hexdigest(),
        "config": config,
        "created_at": now_text(),
        "updated_at": now_text(),
        "records": {},
    }


@contextmanager
def progress_file_lock() -> Generator[None, None, None]:
    """Coordinate progress JSON updates across parallel worker processes."""
    progress_path = get_progress_log_path()
    progress_dir = os.path.dirname(progress_path)
    os.makedirs(progress_dir, exist_ok=True)

    lock_path = f"{progress_path}.lock"
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                json.dump({"pid": os.getpid(), "created_at": now_text()}, lock_file)
            break
        except FileExistsError:
            try:
                lock_age = time.time() - os.path.getmtime(lock_path)
                if lock_age > PROGRESS_LOCK_STALE_SECONDS:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            time.sleep(0.1)

    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def read_progress_state_unlocked() -> dict:
    progress_path = get_progress_log_path()
    if not os.path.exists(progress_path):
        return new_progress_state()

    try:
        with open(progress_path, "r", encoding="utf-8") as progress_file:
            state = json.load(progress_file)
    except json.JSONDecodeError:
        backup_path = f"{progress_path}.corrupt.{int(time.time())}"
        os.replace(progress_path, backup_path)
        print(f"[progress] Corrupt progress log moved to: {backup_path}")
        return new_progress_state()

    if state.get("config_signature") != progress_config_signature():
        print("[progress] Config changed; current progress log will be ignored.")
        return new_progress_state()

    state.setdefault("records", {})
    return state


def write_progress_state_unlocked(state: dict) -> None:
    progress_path = get_progress_log_path()
    progress_dir = os.path.dirname(progress_path)
    os.makedirs(progress_dir, exist_ok=True)

    state["updated_at"] = now_text()
    temp_path = f"{progress_path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as progress_file:
        json.dump(state, progress_file, indent=2, sort_keys=True)
        progress_file.write("\n")

    os.replace(temp_path, progress_path)


def progress_record_key(split_name: str, record: dict) -> str:
    payload = {
        "split": split_name,
        "source_group": record["source_group"],
        "label": record["label"],
        "dat_path": normalized_path(record["dat_path"]),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def upsert_progress_entry(state: dict, split_name: str, record: dict) -> dict:
    key = progress_record_key(split_name, record)
    entry = state.setdefault("records", {}).setdefault(key, {})
    entry.update(
        {
            "split": split_name,
            "source_group": record["source_group"],
            "label": record["label"],
            "dat_path": normalized_path(record["dat_path"]),
            "file_stem": record.get("file_stem", ""),
        }
    )
    return entry


def load_progress_records_snapshot() -> dict:
    if not ENABLE_PROGRESS_LOG:
        return {}

    with progress_file_lock():
        state = read_progress_state_unlocked()
        return dict(state.get("records", {}))


def begin_record_progress(split_name: str, record: dict) -> int | None:
    if not ENABLE_PROGRESS_LOG:
        return 1

    with progress_file_lock():
        state = read_progress_state_unlocked()
        entry = upsert_progress_entry(state, split_name, record)
        if entry.get("status") == "done":
            return None

        try:
            next_window_index = int(entry.get("next_window_index", 1))
        except (TypeError, ValueError):
            next_window_index = 1

        next_window_index = max(1, next_window_index)
        entry.setdefault("started_at", now_text())
        entry["status"] = "running"
        entry["next_window_index"] = next_window_index
        entry["updated_at"] = now_text()
        write_progress_state_unlocked(state)

    return next_window_index


def set_record_next_window(split_name: str, record: dict, next_window_index: int) -> None:
    if not ENABLE_PROGRESS_LOG:
        return

    with progress_file_lock():
        state = read_progress_state_unlocked()
        entry = upsert_progress_entry(state, split_name, record)
        if entry.get("status") == "done":
            return

        entry["status"] = "running"
        entry["next_window_index"] = max(1, int(next_window_index))
        entry["updated_at"] = now_text()
        write_progress_state_unlocked(state)


def checkpoint_record_progress(
    split_name: str,
    record: dict,
    next_window_index: int,
    saved_delta: int,
) -> None:
    if not ENABLE_PROGRESS_LOG:
        return

    with progress_file_lock():
        state = read_progress_state_unlocked()
        entry = upsert_progress_entry(state, split_name, record)
        if entry.get("status") == "done":
            return

        try:
            current_next_window = int(entry.get("next_window_index", 1))
        except (TypeError, ValueError):
            current_next_window = 1

        entry["status"] = "running"
        entry["next_window_index"] = max(current_next_window, int(next_window_index))
        entry["saved_images"] = int(entry.get("saved_images", 0)) + int(saved_delta)
        entry["updated_at"] = now_text()
        write_progress_state_unlocked(state)


def finish_record_progress(split_name: str, record: dict, saved_count: int) -> None:
    if not ENABLE_PROGRESS_LOG:
        return

    with progress_file_lock():
        state = read_progress_state_unlocked()
        entry = upsert_progress_entry(state, split_name, record)
        entry["status"] = "done"
        entry["last_run_new_images"] = int(saved_count)
        entry["completed_at"] = now_text()
        entry["updated_at"] = now_text()
        write_progress_state_unlocked(state)


def infer_next_window_index_from_existing_images(
    output_dir: str,
    prefix: str,
    source_group: str,
    label: str,
    dat_path: str,
) -> int:
    """Find the first missing output window from already written PNG files."""
    if not (RESUME_FROM_EXISTING_IMAGES and SKIP_EXISTING_IMAGES):
        return 1
    if not os.path.isdir(output_dir):
        return 1

    safe_prefix = safe_name(prefix)
    safe_source = safe_name(source_group)
    safe_label = safe_name(label)
    base_name = safe_name(os.path.splitext(os.path.basename(dat_path))[0])
    expected_prefix = f"{safe_prefix}__{safe_source}__{safe_label}__{base_name}__w"
    expected_suffix = ".png"

    existing_indices = set()
    for file_name in os.listdir(output_dir):
        if not file_name.startswith(expected_prefix) or not file_name.endswith(expected_suffix):
            continue

        index_text = file_name[len(expected_prefix):-len(expected_suffix)]
        if index_text.isdigit():
            existing_indices.add(int(index_text))

    next_window_index = 1
    while next_window_index in existing_indices:
        next_window_index += 1

    return next_window_index


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


def maybe_scale_spectrogram_image(output_path: str) -> None:
    """Apply the same in-place crop/resize used by nkquoc.scale_image."""
    if not ENABLE_IMAGE_SCALING:
        return

    from nkquoc.scale_image import scale_image

    scale_image(Path(output_path), SCALE_TOP_RATIO, SCALE_BOTTOM_RATIO)


def save_spectrogram_image_atomic(
    frequencies: np.ndarray,
    times: np.ndarray,
    spectrum: np.ndarray,
    output_path: str,
) -> None:
    """Write a PNG through a temp file so interrupted runs do not leave partial output."""
    temp_output_path = f"{output_path}.{os.getpid()}.tmp.png"
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    try:
        save_spectrogram_image(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            output_path=temp_output_path,
        )
        maybe_scale_spectrogram_image(temp_output_path)
        os.replace(temp_output_path, output_path)
    except Exception:
        try:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except OSError:
            pass
        raise


def iter_iq_chunks_from_dat(
    dat_path: str,
    window_samples: int,
    hop_samples: int,
    dat_format: str = "float32_iq",
    normalize_int16: bool = False,
    max_iq_samples: int | None = None,
    start_window_index: int = 1,
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
    start_window_index = max(1, int(start_window_index))
    start_iq_sample = (start_window_index - 1) * hop_samples
    if max_iq_samples is not None and start_iq_sample >= max_iq_samples:
        return

    total_read = start_iq_sample
    buffer = np.empty(0, dtype=np.complex64)

    with open(dat_path, "rb") as f:
        if start_iq_sample > 0:
            f.seek(2 * start_iq_sample * bytes_per_scalar, os.SEEK_SET)

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
    start_window_index: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
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

    start_window_index = max(1, int(start_window_index))
    if estimated_windows is not None:
        estimated_windows = max(0, estimated_windows - start_window_index + 1)

    safe_prefix = safe_name(prefix)
    safe_source = safe_name(source_group)
    safe_label = safe_name(label)
    base_name = safe_name(os.path.splitext(os.path.basename(dat_path))[0])

    saved_count = 0
    pending_saved_delta = 0
    windows_since_checkpoint = 0
    last_next_window_index = start_window_index

    def mark_window_processed(next_window_index: int, saved_delta: int) -> None:
        nonlocal pending_saved_delta, windows_since_checkpoint, last_next_window_index
        pending_saved_delta += int(saved_delta)
        windows_since_checkpoint += 1
        last_next_window_index = next_window_index

        checkpoint_every = max(1, int(CHECKPOINT_EVERY_WINDOWS))
        if progress_callback is None or windows_since_checkpoint < checkpoint_every:
            return

        progress_callback(last_next_window_index, pending_saved_delta)
        pending_saved_delta = 0
        windows_since_checkpoint = 0

    chunk_iter = iter_iq_chunks_from_dat(
        dat_path=dat_path,
        window_samples=window_samples,
        hop_samples=hop_samples,
        dat_format=dat_format,
        normalize_int16=normalize_int16,
        max_iq_samples=max_iq_samples,
        start_window_index=start_window_index,
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

    for index, iq_chunk in enumerate(chunk_iter, start=start_window_index):
        if iq_chunk.size < window_samples:
            break

        output_path = os.path.join(
            output_dir,
            f"{safe_prefix}__{safe_source}__{safe_label}__{base_name}__w{index:05d}.png",
        )

        if SKIP_EXISTING_IMAGES and os.path.exists(output_path):
            mark_window_processed(index + 1, saved_delta=0)
            continue

        frequencies, times, spectrum = compute_spectrogram(
            iq_chunk,
            sample_rate=sample_rate,
            stft_point=stft_point,
            duration_time=window_seconds,
        )

        save_spectrogram_image_atomic(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            output_path=output_path,
        )
        saved_count += 1
        mark_window_processed(index + 1, saved_delta=1)

    if progress_callback is not None and windows_since_checkpoint > 0:
        progress_callback(last_next_window_index, pending_saved_delta)

    return saved_count


def resolve_num_workers() -> int:
    if NUM_WORKERS and NUM_WORKERS > 0:
        return NUM_WORKERS
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def process_one_record(split_name: str, total: int, index: int, record: dict) -> int:
    split_root = os.path.join(OUTPUT_ROOT, split_name)
    source_dir = os.path.join(split_root, safe_name(record["source_group"]))
    label_dir = os.path.join(source_dir, safe_name(record["label"]))

    start_window_index = begin_record_progress(split_name, record)
    if start_window_index is None:
        print(
            f"[{split_name}] {index}/{total} -> {record['source_group']}/{record['label']} "
            f"| {os.path.basename(record['dat_path'])} | already done"
        )
        return 0

    existing_next_window = infer_next_window_index_from_existing_images(
        output_dir=label_dir,
        prefix="spectrogram",
        source_group=record["source_group"],
        label=record["label"],
        dat_path=record["dat_path"],
    )
    if start_window_index <= 1:
        start_window_index = existing_next_window
    else:
        start_window_index = min(start_window_index, existing_next_window)

    set_record_next_window(split_name, record, start_window_index)

    resume_text = ""
    if start_window_index > 1:
        resume_text = f" | resume window {start_window_index}"

    print(
        f"[{split_name}] {index}/{total} -> {record['source_group']}/{record['label']} "
        f"| {os.path.basename(record['dat_path'])}{resume_text}"
    )

    def checkpoint(next_window_index: int, saved_delta: int) -> None:
        checkpoint_record_progress(
            split_name=split_name,
            record=record,
            next_window_index=next_window_index,
            saved_delta=saved_delta,
        )

    saved_count = convert_dat_to_spectrograms(
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
        start_window_index=start_window_index,
        progress_callback=checkpoint,
    )

    finish_record_progress(split_name, record, saved_count)
    return saved_count


def process_split(split_name: str, records: list[dict]) -> int:
    split_root = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(split_root, exist_ok=True)

    original_count = len(records)
    if not records:
        print(f"[{split_name}] recordings=0, images=0")
        return 0

    skipped_done = 0
    if ENABLE_PROGRESS_LOG:
        progress_records = load_progress_records_snapshot()
        pending_records = []
        for record in records:
            key = progress_record_key(split_name, record)
            if progress_records.get(key, {}).get("status") == "done":
                skipped_done += 1
                continue
            pending_records.append(record)
        records = pending_records

    if skipped_done:
        print(f"[{split_name}] resume: skipped {skipped_done} completed recordings")

    if not records:
        print(f"[{split_name}] recordings={original_count}, processed=0, images=0")
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
            total_saved += process_one_record(split_name, total, index, record)
            completed += 1
            print(f"[{split_name}] completed {completed}/{total} files")

    print(
        f"[{split_name}] recordings={original_count}, processed={len(records)}, "
        f"skipped_done={skipped_done}, images={total_saved}"
    )
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
    if ENABLE_PROGRESS_LOG:
        print(f"Progress log: {get_progress_log_path()}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total_train = process_split("train", train_records)
    total_valid = process_split("valid", valid_records)
    total_test = process_split("test", test_records)

    print("All done.")
    print(f"Total images -> train={total_train}, valid={total_valid}, test={total_test}")


if __name__ == "__main__":
    main()
