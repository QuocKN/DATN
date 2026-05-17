#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert RF raw data to spectrogram PNGs with two supported inputs:
1) Kaggle extracted files: <sample>/data/0 (float32 x_iq stored as shape (2, N))
2) Local .bin files: int16 interleaved I,Q,I,Q,...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Generator

import numpy as np
from bin_iq_to_spectrogram import compute_spectrogram, save_spectrogram_image


def iter_iq_chunks_from_float32(
    path: Path,
    chunk_size: int,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Read x_iq stored as float32 tensor with shape (2, N):
    - first N values: I channel
    - next N values: Q channel
    """
    total_read = 0
    a = np.fromfile(path, dtype=np.float32)
    if a.size < 2:
        return
    n = a.size // 2
    i_all = a[:n]
    q_all = a[n : 2 * n]
    n = min(i_all.size, q_all.size)
    i_all = i_all[:n]
    q_all = q_all[:n]

    start = 0
    while start < n:
        if max_iq_samples is not None and total_read >= max_iq_samples:
            break
        end = min(start + chunk_size, n)
        i_values = i_all[start:end]
        q_values = q_all[start:end]
        if i_values.size == 0:
            break
        yield i_values + 1j * q_values
        total_read += i_values.size
        start = end


def convert_float32_to_spectrograms(
    data_path: Path,
    output_dir: Path,
    sample_name: str,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
    chunk_size: int,
    prefix: str,
    max_duration_seconds: int | None,
    image_size: int,
) -> int:
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    os.makedirs(output_dir, exist_ok=True)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))

    saved = 0
    for idx, iq_chunk in enumerate(
        iter_iq_chunks_from_float32(data_path, chunk_size, max_iq_samples),
        start=1,
    ):
        if iq_chunk.size < stft_point:
            continue
        try:
            effective_duration = duration_time
            if iq_chunk.size < min_samples_needed:
                # Fallback for short files/chunks: use available duration.
                effective_duration = iq_chunk.size / sample_rate
            f, t, s = compute_spectrogram(iq_chunk, sample_rate, stft_point, effective_duration)
            out = output_dir / f"{sample_name}_{prefix}_{idx:06d}.png"
            save_spectrogram_image(
                frequencies=f,
                times=t,
                spectrum=s,
                output_path=str(out),
            )
            saved += 1
        except Exception:
            continue
    return saved


def iter_iq_chunks_from_int16_bin(
    bin_path: Path,
    chunk_size: int,
    normalize: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    total_read = 0
    with bin_path.open("rb") as f:
        while True:
            if max_iq_samples is not None and total_read >= max_iq_samples:
                break

            raw_bytes = f.read(2 * chunk_size * 2)  # 2 channels * int16(2 bytes)
            if not raw_bytes:
                break

            int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
            if len(int16_data) % 2 != 0:
                int16_data = int16_data[:-1]
            if len(int16_data) == 0:
                break

            iq_pairs = int16_data.reshape(-1, 2)
            i_values = iq_pairs[:, 0].astype(np.float32)
            q_values = iq_pairs[:, 1].astype(np.float32)

            if normalize:
                i_values /= 32768.0
                q_values /= 32768.0

            total_read += len(iq_pairs)
            yield i_values + 1j * q_values


def convert_int16_bin_to_spectrograms(
    bin_path: Path,
    output_dir: Path,
    sample_name: str,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
    chunk_size: int,
    prefix: str,
    max_duration_seconds: int | None,
    normalize: bool,
) -> int:
    if not bin_path.exists():
        raise FileNotFoundError(f"File not found: {bin_path}")
    os.makedirs(output_dir, exist_ok=True)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] BIN chunk_size={chunk_size} < required {min_samples_needed}. "
            f"Using {min_samples_needed}."
        )
        chunk_size = min_samples_needed

    saved = 0
    for idx, iq_chunk in enumerate(
        iter_iq_chunks_from_int16_bin(
            bin_path=bin_path,
            chunk_size=chunk_size,
            normalize=normalize,
            max_iq_samples=max_iq_samples,
        ),
        start=1,
    ):
        if iq_chunk.size < stft_point:
            continue
        try:
            effective_duration = duration_time
            if iq_chunk.size < min_samples_needed:
                effective_duration = iq_chunk.size / sample_rate
            f, t, s = compute_spectrogram(iq_chunk, sample_rate, stft_point, effective_duration)
            out = output_dir / f"{sample_name}_{prefix}_{idx:06d}.png"
            save_spectrogram_image(
                frequencies=f,
                times=t,
                spectrum=s,
                output_path=str(out),
            )
            saved += 1
            if idx % 50 == 0:
                print(f"[BIN] processed chunks: {idx}, images: {saved}")
        except Exception:
            continue
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Kaggle float32 IQ and/or local int16 .bin to spectrogram.")
    parser.add_argument(
        "--mode",
        choices=["kaggle", "bin", "both"],
        default="kaggle",
        help="Input mode to process (default: kaggle).",
    )
    parser.add_argument("--root", default="/home/quocnk/Documents/NKQuoc/Data/RF/Noisy_Drone_RF_Signal_v2_kaggle")
    parser.add_argument("--output-dir", default="/home/quocnk/Documents/NKQuoc/Data/RF/Noisy_Drone_RF_Signal_v2_kaggle/spectrograms")
    parser.add_argument("--bin-path", default="e:\\DATN_DATA\\RF\\1toan.bin")
    parser.add_argument("--bin-output-dir", default="e:\\DATN_DATA\\RF\\1toan_spectrograms")
    parser.add_argument("--bin-name", default="1toan")
    parser.add_argument("--sample-rate", type=int, default=60_000_000)
    parser.add_argument("--stft-point", type=int, default=2048)
    parser.add_argument("--duration-time", type=float, default=0.03)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--prefix", default="spectrogram")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-duration-seconds", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--normalize-int16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in ("kaggle", "both"):
        root = Path(args.root).resolve()
        out = Path(args.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        data_files = sorted(root.glob("*/data/0"))
        if args.max_files is not None:
            data_files = data_files[: args.max_files]

        print(f"Root: {root}")
        print(f"Output: {out}")
        print(f"Found files: {len(data_files)}")

        ok = 0
        fail = 0
        total = 0
        for p in data_files:
            sample = p.parent.parent.name
            try:
                n = convert_float32_to_spectrograms(
                    data_path=p,
                    output_dir=out,
                    sample_name=sample,
                    sample_rate=args.sample_rate,
                    stft_point=args.stft_point,
                    duration_time=args.duration_time,
                    chunk_size=args.chunk_size,
                    prefix=args.prefix,
                    max_duration_seconds=args.max_duration_seconds,
                    image_size=args.image_size,
                )
                ok += 1
                total += n
                print(f"[OK] {sample}: {n}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {sample}: {e}")

        print("\n===== Kaggle Summary =====")
        print(f"Processed: {ok}")
        print(f"Failed: {fail}")
        print(f"Total images: {total}")

    if args.mode in ("bin", "both"):
        bin_path = Path(args.bin_path).resolve()
        bin_out = Path(args.bin_output_dir).resolve()
        bin_out.mkdir(parents=True, exist_ok=True)
        print(f"\nBIN input: {bin_path}")
        print(f"BIN output: {bin_out}")
        count = convert_int16_bin_to_spectrograms(
            bin_path=bin_path,
            output_dir=bin_out,
            sample_name=args.bin_name,
            sample_rate=args.sample_rate,
            stft_point=args.stft_point,
            duration_time=args.duration_time,
            chunk_size=args.chunk_size,
            prefix=args.prefix,
            max_duration_seconds=args.max_duration_seconds,
            normalize=args.normalize_int16,
        )
        print("\n===== BIN Summary =====")
        print(f"Total images: {count}")


if __name__ == "__main__":
    main()
