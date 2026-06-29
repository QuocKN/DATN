#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a growing int16 interleaved IQ .bin file into spectrogram PNGs.

Intended capture flow:
    bladeRF -> int16 IQ file/FIFO -> this script -> spectrogram samples

Binary layout:
    I_0, Q_0, I_1, Q_1, ... as signed int16 values.
"""

from __future__ import annotations

import argparse
import os
import queue
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from nkquoc.base import iq_spectrogram_core as spectrogram_core
    from nkquoc.base.iq_preprocessing import blank_impulsive_spikes, despike_iq, repair_clipped_iq
    from nkquoc.base.iq_spectrogram_core import compute_spectrogram, save_spectrogram_image
except ImportError:
    from base import iq_spectrogram_core as spectrogram_core
    from base.iq_preprocessing import blank_impulsive_spikes, despike_iq, repair_clipped_iq
    from base.iq_spectrogram_core import compute_spectrogram, save_spectrogram_image


SAMPLE_RATE = 40_000_000
RF_BANDWIDTH = 28_000_000
WINDOW_SECONDS = 0.05
HOP_SECONDS = 0.05
STFT_POINT = 1024
IMAGE_SIZE = 224
CENTER_FREQUENCY = 2_445_000_000
RX_GAIN = 30
RAM_BUFFER_SECONDS = 0.0

ENABLE_SPEC_COLUMN_DENOISE = False
SPEC_COLUMN_QUANTILE = 30.0
ENABLE_SPEC_DB_CLIP = False
SPEC_CLIP_DB_MIN = -80.0
SPEC_CLIP_DB_MAX = 15.0


def apply_spectrogram_config(args: argparse.Namespace) -> None:
    spectrogram_core.IMAGE_SIZE = args.image_size
    spectrogram_core.ENABLE_SPEC_COLUMN_DENOISE = args.column_denoise
    spectrogram_core.SPEC_COLUMN_QUANTILE = args.column_quantile
    spectrogram_core.ENABLE_SPEC_DB_CLIP = args.db_clip
    spectrogram_core.SPEC_CLIP_DB_MIN = args.db_min
    spectrogram_core.SPEC_CLIP_DB_MAX = args.db_max


def preprocess_iq_chunk(iq_chunk: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.impulse_blanker:
        iq_chunk = blank_impulsive_spikes(
            iq_chunk,
            median_kernel=args.blanker_median_kernel,
            threshold_sigma=args.blanker_threshold_sigma,
            max_spike_width=args.blanker_max_spike_width,
        )
    if args.despike:
        iq_chunk = despike_iq(iq_chunk, percentile=args.despike_percentile)
    if args.repair_clipped:
        iq_chunk = repair_clipped_iq(iq_chunk)
    return iq_chunk


def read_new_iq_samples(
    handle,
    scalar_remainder: np.ndarray,
    normalize: bool,
    max_bytes_per_read: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_bytes = handle.read(max_bytes_per_read)
    if not raw_bytes:
        return np.empty(0, dtype=np.complex64), scalar_remainder

    scalars = np.frombuffer(raw_bytes, dtype="<i2")
    if scalar_remainder.size:
        scalars = np.concatenate((scalar_remainder, scalars))

    if scalars.size % 2 != 0:
        scalar_remainder = scalars[-1:].copy()
        scalars = scalars[:-1]
    else:
        scalar_remainder = np.empty(0, dtype=np.int16)

    if scalars.size == 0:
        return np.empty(0, dtype=np.complex64), scalar_remainder

    pairs = scalars.reshape(-1, 2)
    i_values = pairs[:, 0].astype(np.float32)
    q_values = pairs[:, 1].astype(np.float32)
    if normalize:
        i_values /= 32768.0
        q_values /= 32768.0

    return (i_values + 1j * q_values).astype(np.complex64), scalar_remainder


def pop_window(buffer: deque[np.ndarray], total_samples: int, window_samples: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    remaining = window_samples

    while remaining > 0:
        head = buffer[0]
        if head.size <= remaining:
            parts.append(head)
            buffer.popleft()
            remaining -= head.size
        else:
            parts.append(head[:remaining])
            buffer[0] = head[remaining:]
            remaining = 0

    window = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if window.size != window_samples:
        raise RuntimeError(f"Expected {window_samples} samples, got {window.size}")
    return window


def drop_samples(buffer: deque[np.ndarray], samples_to_drop: int) -> None:
    remaining = samples_to_drop
    while remaining > 0 and buffer:
        head = buffer[0]
        if head.size <= remaining:
            buffer.popleft()
            remaining -= head.size
        else:
            buffer[0] = head[remaining:]
            remaining = 0


def append_samples(buffer: deque[np.ndarray], samples: np.ndarray) -> int:
    if samples.size == 0:
        return 0
    buffer.append(samples)
    return int(samples.size)


def make_output_path(output_dir: Path, prefix: str, index: int) -> Path:
    return output_dir / f"{prefix}_{index:06d}.png"


def spectrogram_to_uint8(magnitude_db: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.db_clip:
        magnitude_db = np.clip(magnitude_db, args.db_min, args.db_max)
        vmin = args.db_min
        vmax = args.db_max
    else:
        vmin = float(np.nanmin(magnitude_db))
        vmax = float(np.nanmax(magnitude_db))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(magnitude_db.shape, dtype=np.uint8)

    scaled = (magnitude_db - vmin) * (255.0 / (vmax - vmin))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def save_spectrogram_image_fast(
    spectrum: np.ndarray,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    magnitude_db = 10 * np.log10(np.abs(spectrum) + 1e-12)
    if args.column_denoise:
        col_bg = np.percentile(magnitude_db, args.column_quantile, axis=0, keepdims=True)
        magnitude_db = magnitude_db - col_bg

    image_8bit = spectrogram_to_uint8(magnitude_db, args)
    image_8bit = np.flipud(image_8bit)

    if cv2 is not None:
        resized = cv2.resize(
            image_8bit,
            (args.image_size, args.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        colored = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, colored)
        return

    image = Image.fromarray(image_8bit, mode="L").resize(
        (args.image_size, args.image_size),
        Image.Resampling.BILINEAR,
    )
    image.save(output_path)


def save_realtime_spectrogram_image(
    frequencies: np.ndarray,
    times: np.ndarray,
    spectrum: np.ndarray,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    if args.image_renderer == "fast":
        save_spectrogram_image_fast(spectrum=spectrum, output_path=output_path, args=args)
        return

    save_spectrogram_image(
        frequencies=frequencies,
        times=times,
        spectrum=spectrum,
        output_path=output_path,
    )


def format_bladerf_value(value: float | int) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def ensure_fifo(path: Path) -> bool:
    """Create a FIFO if needed. Return True when this function created it."""
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
    """Prepare bladeRF output path. Return True when this function should clean it up."""
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


def run_realtime_conversion(args: argparse.Namespace) -> int:
    apply_spectrogram_config(args)
    if not args.bladerf and args.input is None:
        raise ValueError("Provide --input/-i, or use --bladerf to capture from bladeRF-cli.")
    if args.bladerf and args.input is not None:
        raise ValueError("--input/-i is not used with --bladerf. Use --fifo-path if you need a fixed FIFO path.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(round(args.sample_rate * args.window_seconds))
    hop_samples = int(round(args.sample_rate * args.hop_seconds))
    if window_samples < args.stft_point:
        raise ValueError("window_seconds * sample_rate must be >= stft_point")
    if hop_samples <= 0:
        raise ValueError("hop_seconds must be positive")

    buffer: deque[np.ndarray] = deque()
    buffered_samples = 0
    scalar_remainder = np.empty(0, dtype=np.int16)
    saved = 0
    read_samples = 0
    start_time = time.monotonic()

    print(
        "Realtime spectrogram pipeline: "
        f"fs={args.sample_rate:g} Hz, rf_bandwidth={args.rf_bandwidth:g} Hz, "
        f"window={args.window_seconds:g}s ({window_samples:,} IQ), "
        f"hop={args.hop_seconds:g}s ({hop_samples:,} IQ), stft={args.stft_point}"
    )
    input_label = "bladeRF" if args.bladerf else ("stdin" if args.input == "-" else args.input)
    print(f"Input: {input_label}")
    print(f"Output: {output_dir}")

    if args.bladerf:
        source_context = bladerf_capture_context(args)
    elif args.input == "-":
        if args.start_at_end:
            raise ValueError("--start-at-end cannot be used with stdin")
        source_context = nullcontext(None)
    else:
        source_context = nullcontext(Path(args.input))

    with source_context as source:
        if args.bladerf:
            input_context = source.open("rb")
        elif args.input == "-":
            input_context = nullcontext(sys.stdin.buffer)
        else:
            input_context = source.open("rb")

        with input_context as handle:
            saved = read_stream_to_spectrograms(
                handle=handle,
                args=args,
                output_dir=output_dir,
                buffer=buffer,
                buffered_samples=buffered_samples,
                scalar_remainder=scalar_remainder,
                saved=saved,
                read_samples=read_samples,
                start_time=start_time,
                window_samples=window_samples,
                hop_samples=hop_samples,
            )

    print(f"Done. Saved spectrograms: {saved}")
    return saved


def read_stream_to_spectrograms(
    handle,
    args: argparse.Namespace,
    output_dir: Path,
    buffer: deque[np.ndarray],
    buffered_samples: int,
    scalar_remainder: np.ndarray,
    saved: int,
    read_samples: int,
    start_time: float,
    window_samples: int,
    hop_samples: int,
) -> int:
    if args.start_at_end:
        handle.seek(0, os.SEEK_END)

    if args.ram_buffer_seconds > 0:
        return read_threaded_stream_to_spectrograms(
            handle=handle,
            args=args,
            output_dir=output_dir,
            buffer=buffer,
            buffered_samples=buffered_samples,
            saved=saved,
            read_samples=read_samples,
            start_time=start_time,
            window_samples=window_samples,
            hop_samples=hop_samples,
        )

    while args.max_spectrograms is None or saved < args.max_spectrograms:
        iq_samples, scalar_remainder = read_new_iq_samples(
            handle=handle,
            scalar_remainder=scalar_remainder,
            normalize=args.normalize,
            max_bytes_per_read=args.read_bytes,
        )

        if iq_samples.size == 0:
            if not args.follow and not args.bladerf:
                break
            time.sleep(args.poll_seconds)
            continue

        read_samples += iq_samples.size
        buffered_samples += append_samples(buffer, iq_samples)

        while buffered_samples >= window_samples and (
            args.max_spectrograms is None or saved < args.max_spectrograms
        ):
            window = pop_window(buffer, buffered_samples, window_samples)
            buffered_samples -= window_samples

            processed = preprocess_iq_chunk(window, args)
            frequencies, times, spectrum = compute_spectrogram(
                processed,
                sample_rate=args.sample_rate,
                stft_point=args.stft_point,
                duration_time=args.window_seconds,
            )

            saved += 1
            output_path = make_output_path(output_dir, args.prefix, saved)
            save_realtime_spectrogram_image(
                frequencies=frequencies,
                times=times,
                spectrum=spectrum,
                output_path=str(output_path),
                args=args,
            )

            if hop_samples < window_samples:
                keep = window[hop_samples:]
                buffer.appendleft(keep)
                buffered_samples += keep.size
            elif hop_samples > window_samples:
                extra_drop = hop_samples - window_samples
                drop_now = min(extra_drop, buffered_samples)
                drop_samples(buffer, drop_now)
                buffered_samples -= drop_now

            elapsed = max(time.monotonic() - start_time, 1e-6)
            print(
                f"[OK] {output_path} | saved={saved} | "
                f"read={read_samples / args.sample_rate:.3f}s IQ | "
                f"speed={read_samples / args.sample_rate / elapsed:.2f}x"
            )

    return saved


def process_iq_samples(
    iq_samples: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    buffer: deque[np.ndarray],
    buffered_samples: int,
    saved: int,
    read_samples: int,
    start_time: float,
    window_samples: int,
    hop_samples: int,
) -> tuple[int, int]:
    buffered_samples += append_samples(buffer, iq_samples)

    while buffered_samples >= window_samples and (
        args.max_spectrograms is None or saved < args.max_spectrograms
    ):
        window = pop_window(buffer, buffered_samples, window_samples)
        buffered_samples -= window_samples

        processed = preprocess_iq_chunk(window, args)
        frequencies, times, spectrum = compute_spectrogram(
            processed,
            sample_rate=args.sample_rate,
            stft_point=args.stft_point,
            duration_time=args.window_seconds,
        )

        saved += 1
        output_path = make_output_path(output_dir, args.prefix, saved)
        save_realtime_spectrogram_image(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            output_path=str(output_path),
            args=args,
        )

        if hop_samples < window_samples:
            keep = window[hop_samples:]
            buffer.appendleft(keep)
            buffered_samples += keep.size
        elif hop_samples > window_samples:
            extra_drop = hop_samples - window_samples
            drop_now = min(extra_drop, buffered_samples)
            drop_samples(buffer, drop_now)
            buffered_samples -= drop_now

        elapsed = max(time.monotonic() - start_time, 1e-6)
        print(
            f"[OK] {output_path} | saved={saved} | "
            f"read={read_samples / args.sample_rate:.3f}s IQ | "
            f"speed={read_samples / args.sample_rate / elapsed:.2f}x"
        )

    return buffered_samples, saved


def estimate_ram_queue_chunks(args: argparse.Namespace) -> int:
    iq_per_read = max(1, args.read_bytes // 4)
    buffered_iq = max(1, int(round(args.ram_buffer_seconds * args.sample_rate)))
    return max(1, int(np.ceil(buffered_iq / iq_per_read)))


def iq_reader_worker(
    handle,
    args: argparse.Namespace,
    iq_queue: queue.Queue[np.ndarray | None],
    stop_event: threading.Event,
) -> None:
    scalar_remainder = np.empty(0, dtype=np.int16)
    dropped_chunks = 0
    try:
        while not stop_event.is_set():
            iq_samples, scalar_remainder = read_new_iq_samples(
                handle=handle,
                scalar_remainder=scalar_remainder,
                normalize=args.normalize,
                max_bytes_per_read=args.read_bytes,
            )

            if iq_samples.size == 0:
                if not args.follow and not args.bladerf:
                    break
                time.sleep(args.poll_seconds)
                continue

            while not stop_event.is_set():
                try:
                    iq_queue.put(iq_samples, timeout=0.05)
                    break
                except queue.Full:
                    if args.ram_buffer_overflow == "drop-newest":
                        dropped_chunks += 1
                        if dropped_chunks % 25 == 1:
                            print(f"[WARN] RAM queue full; dropped newest chunks={dropped_chunks}")
                        break
                    if args.ram_buffer_overflow == "drop-oldest":
                        try:
                            iq_queue.get_nowait()
                        except queue.Empty:
                            pass
                        dropped_chunks += 1
                        if dropped_chunks % 25 == 1:
                            print(f"[WARN] RAM queue full; dropped oldest chunks={dropped_chunks}")
                        continue
                    continue
    finally:
        iq_queue.put(None)


def read_threaded_stream_to_spectrograms(
    handle,
    args: argparse.Namespace,
    output_dir: Path,
    buffer: deque[np.ndarray],
    buffered_samples: int,
    saved: int,
    read_samples: int,
    start_time: float,
    window_samples: int,
    hop_samples: int,
) -> int:
    max_chunks = estimate_ram_queue_chunks(args)
    approx_ram_mb = args.ram_buffer_seconds * args.sample_rate * np.dtype(np.complex64).itemsize / 1_000_000
    print(
        f"RAM buffer enabled: {args.ram_buffer_seconds:g}s, "
        f"queue_chunks={max_chunks}, approx_complex64={approx_ram_mb:.1f} MB"
    )

    iq_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_chunks)
    stop_event = threading.Event()
    reader = threading.Thread(
        target=iq_reader_worker,
        args=(handle, args, iq_queue, stop_event),
        daemon=True,
    )
    reader.start()

    try:
        while args.max_spectrograms is None or saved < args.max_spectrograms:
            try:
                iq_samples = iq_queue.get(timeout=0.5)
            except queue.Empty:
                if not reader.is_alive():
                    break
                continue

            if iq_samples is None:
                break

            read_samples += iq_samples.size
            buffered_samples, saved = process_iq_samples(
                iq_samples=iq_samples,
                args=args,
                output_dir=output_dir,
                buffer=buffer,
                buffered_samples=buffered_samples,
                saved=saved,
                read_samples=read_samples,
                start_time=start_time,
                window_samples=window_samples,
                hop_samples=hop_samples,
            )
    finally:
        stop_event.set()
        reader.join(timeout=2)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create realtime spectrogram PNGs from a growing bladeRF int16 IQ .bin file."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Growing .bin file, FIFO, or '-' for stdin with int16 interleaved IQ",
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Directory for spectrogram PNGs")
    parser.add_argument("--bladerf", action="store_true", help="Capture from bladeRF-cli")
    parser.add_argument("--center-frequency", type=float, default=CENTER_FREQUENCY)
    parser.add_argument("--gain", type=float, default=RX_GAIN)
    parser.add_argument("--device", default=None, help="Optional bladeRF device selector for bladeRF-cli -d")
    parser.add_argument(
        "--fifo-path",
        default=None,
        help="Optional capture path for --bladerf; FIFO on Linux, temporary .bin file on Windows",
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--rf-bandwidth", type=int, default=RF_BANDWIDTH)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=HOP_SECONDS)
    parser.add_argument("--stft-point", type=int, default=STFT_POINT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--prefix", default="spectrogram")
    parser.add_argument(
        "--image-renderer",
        choices=["fast", "matplotlib"],
        default="matplotlib",
        help="Use direct OpenCV/Pillow PNG rendering, or the slower Matplotlib renderer.",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--follow", action="store_true", help="Wait for more samples at EOF")
    parser.add_argument("--start-at-end", action="store_true", help="Ignore existing bytes and process only new samples")
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument("--read-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--ram-buffer-seconds",
        type=float,
        default=RAM_BUFFER_SECONDS,
        help="Use a reader thread and RAM queue for this many seconds of IQ; 0 disables it.",
    )
    parser.add_argument(
        "--ram-buffer-overflow",
        choices=["block", "drop-oldest", "drop-newest"],
        default="block",
        help="When RAM queue is full: block reader, drop oldest queued IQ, or drop newest IQ.",
    )
    parser.add_argument("--max-spectrograms", type=int, default=None)

    parser.add_argument("--column-denoise", action=argparse.BooleanOptionalAction, default=ENABLE_SPEC_COLUMN_DENOISE)
    parser.add_argument("--column-quantile", type=float, default=SPEC_COLUMN_QUANTILE)
    parser.add_argument("--db-clip", action=argparse.BooleanOptionalAction, default=ENABLE_SPEC_DB_CLIP)
    parser.add_argument("--db-min", type=float, default=SPEC_CLIP_DB_MIN)
    parser.add_argument("--db-max", type=float, default=SPEC_CLIP_DB_MAX)

    parser.add_argument("--despike", action="store_true")
    parser.add_argument("--despike-percentile", type=float, default=99.5)
    parser.add_argument("--repair-clipped", action="store_true")
    parser.add_argument("--impulse-blanker", action="store_true")
    parser.add_argument("--blanker-median-kernel", type=int, default=129)
    parser.add_argument("--blanker-threshold-sigma", type=float, default=6.0)
    parser.add_argument("--blanker-max-spike-width", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    run_realtime_conversion(parse_args())


if __name__ == "__main__":
    main()
