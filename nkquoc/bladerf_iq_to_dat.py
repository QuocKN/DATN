#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture bladeRF int16 interleaved IQ and save it as float32 interleaved DAT.

Output DAT layout:
    I_0, Q_0, I_1, Q_1, ... as little-endian float32 values.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np


SAMPLE_RATE = 40_000_000
RF_BANDWIDTH = 28_000_000
CENTER_FREQUENCY = 2_445_000_000
RX_GAIN = 30


def format_bladerf_value(value: float | int) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def ensure_fifo(path: Path) -> bool:
    if path.exists():
        if not stat.S_ISFIFO(path.stat().st_mode):
            raise FileExistsError(f"Path exists but is not a FIFO: {path}")
        return False
    os.mkfifo(path)
    return True


def default_bladerf_capture_path() -> Path:
    if os.name == "nt":
        return Path(tempfile.gettempdir()) / f"bladerf_iq_{os.getpid()}.bin"
    return Path(f"/tmp/bladerf_iq_{os.getpid()}.pipe")


def ensure_bladerf_capture_path(path: Path) -> bool:
    if os.name != "nt":
        return ensure_fifo(path)

    if path.exists():
        raise FileExistsError(
            f"Capture file already exists: {path}. "
            "Choose another --fifo-path or delete the old capture file first."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return True


def build_bladerf_script(args: argparse.Namespace, capture_path: Path) -> str:
    commands = [
        f"set frequency rx {format_bladerf_value(args.center_frequency)}",
        f"set samplerate rx {format_bladerf_value(args.sample_rate)}",
        f"set bandwidth rx {format_bladerf_value(args.rf_bandwidth)}",
        f"set gain rx {format_bladerf_value(args.gain)}",
        f"rx config file={capture_path} format=bin n=0",
        "rx start",
        "rx wait",
    ]
    return "; ".join(commands)


@contextmanager
def bladerf_capture_context(args: argparse.Namespace) -> Iterator[Path]:
    capture_path = Path(args.fifo_path) if args.fifo_path else default_bladerf_capture_path()
    cleanup_capture_path = ensure_bladerf_capture_path(capture_path)
    command = ["bladeRF-cli"]
    if args.device:
        command.extend(["-d", args.device])
    command.extend(["-e", build_bladerf_script(args, capture_path)])

    print("Starting bladeRF capture:")
    print(" ".join(command))

    process = subprocess.Popen(command)
    try:
        yield capture_path
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if cleanup_capture_path:
            try:
                capture_path.unlink()
            except FileNotFoundError:
                pass


def read_int16_iq_as_float32_dat_chunk(
    handle,
    scalar_remainder: np.ndarray,
    normalize: bool,
    max_bytes_per_read: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw_bytes = handle.read(max_bytes_per_read)
    if not raw_bytes:
        return np.empty(0, dtype="<f4"), scalar_remainder, 0

    scalars = np.frombuffer(raw_bytes, dtype="<i2")
    if scalar_remainder.size:
        scalars = np.concatenate((scalar_remainder, scalars))

    if scalars.size % 2 != 0:
        scalar_remainder = scalars[-1:].copy()
        scalars = scalars[:-1]
    else:
        scalar_remainder = np.empty(0, dtype=np.int16)

    if scalars.size == 0:
        return np.empty(0, dtype="<f4"), scalar_remainder, 0

    dat_chunk = scalars.astype("<f4")
    if normalize:
        dat_chunk /= 32768.0
    return dat_chunk, scalar_remainder, scalars.size // 2


def capture_or_convert_to_dat(args: argparse.Namespace) -> int:
    output_path = Path(args.output_dat)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "ab" if args.append else "wb"

    if args.input_bin:
        source_context = nullcontext(Path(args.input_bin))
        source_label = str(args.input_bin)
    else:
        source_context = bladerf_capture_context(args)
        source_label = "bladeRF"

    print(
        "IQ DAT capture: "
        f"fs={args.sample_rate:g} Hz, rf_bandwidth={args.rf_bandwidth:g} Hz, "
        f"fc={args.center_frequency:g} Hz, gain={args.gain:g}"
    )
    print(f"Input: {source_label}")
    print(f"Output DAT: {output_path} (float32 interleaved I,Q)")
    if args.normalize:
        print("Normalize: enabled, int16 values are divided by 32768")

    scalar_remainder = np.empty(0, dtype=np.int16)
    total_iq = 0
    total_bytes = 0
    last_status_time = time.monotonic()
    last_status_iq = 0
    max_iq_samples = args.max_iq_samples
    if args.max_seconds is not None:
        seconds_iq = int(round(args.max_seconds * args.sample_rate))
        max_iq_samples = seconds_iq if max_iq_samples is None else min(max_iq_samples, seconds_iq)

    with source_context as source:
        with source.open("rb") as input_handle, output_path.open(output_mode) as output_handle:
            while max_iq_samples is None or total_iq < max_iq_samples:
                dat_chunk, scalar_remainder, iq_count = read_int16_iq_as_float32_dat_chunk(
                    handle=input_handle,
                    scalar_remainder=scalar_remainder,
                    normalize=args.normalize,
                    max_bytes_per_read=args.read_bytes,
                )

                if iq_count == 0:
                    if args.input_bin and not args.follow:
                        break
                    time.sleep(args.poll_seconds)
                    continue

                if max_iq_samples is not None and total_iq + iq_count > max_iq_samples:
                    keep_iq = max_iq_samples - total_iq
                    dat_chunk = dat_chunk[: keep_iq * 2]
                    iq_count = keep_iq

                output_handle.write(dat_chunk.tobytes())
                total_iq += iq_count
                total_bytes += dat_chunk.nbytes

                now = time.monotonic()
                elapsed = max(now - last_status_time, 1e-6)
                if total_iq == iq_count or elapsed >= args.status_seconds:
                    interval_iq = total_iq - last_status_iq
                    print(
                        f"[OK] wrote={total_iq / args.sample_rate:.3f}s IQ | "
                        f"samples={total_iq:,} | dat={total_bytes / 1_000_000:.1f} MB | "
                        f"speed={interval_iq / args.sample_rate / elapsed:.2f}x"
                    )
                    last_status_time = now
                    last_status_iq = total_iq

    print(f"Done. IQ samples: {total_iq:,}. DAT bytes written: {total_bytes:,}")
    return total_iq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture bladeRF IQ and save float32 interleaved I,Q DAT."
    )
    parser.add_argument("-o", "--output-dat", required=True, help="Output .dat path")
    parser.add_argument(
        "-i",
        "--input-bin",
        default=None,
        help="Optional int16 interleaved IQ .bin input instead of live bladeRF capture",
    )
    parser.add_argument("--append", action="store_true", help="Append instead of overwrite output .dat")
    parser.add_argument("--center-frequency", type=float, default=CENTER_FREQUENCY)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--rf-bandwidth", type=int, default=RF_BANDWIDTH)
    parser.add_argument("--gain", type=float, default=RX_GAIN)
    parser.add_argument("--device", default=None, help="Optional bladeRF device selector for bladeRF-cli -d")
    parser.add_argument(
        "--fifo-path",
        default=None,
        help="Optional bladeRF capture path; FIFO on Linux, temporary .bin file on Windows",
    )
    parser.add_argument("--normalize", action="store_true", help="Scale int16 IQ to roughly [-1, 1]")
    parser.add_argument("--follow", action="store_true", help="Wait for more data at EOF with --input-bin")
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--read-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after this many seconds of IQ")
    parser.add_argument("--max-iq-samples", type=int, default=None, help="Stop after this many IQ samples")
    parser.add_argument("--status-seconds", type=float, default=1.0, help="Print status at this interval")
    return parser.parse_args()


def main() -> None:
    capture_or_convert_to_dat(parse_args())


if __name__ == "__main__":
    main()
