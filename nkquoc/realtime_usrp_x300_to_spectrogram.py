#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture realtime IQ from a USRP X300/X310 and save spectrogram PNGs.

Intended capture flow:
    USRP X300 -> UHD Python API -> complex64 IQ -> spectrogram samples

Requires the UHD Python bindings:
    python -c "import uhd"
"""

from __future__ import annotations

import argparse
import struct
import time
import zlib
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .base.iq_preprocessing import blank_impulsive_spikes, despike_iq, repair_clipped_iq
except ImportError:
    from base.iq_preprocessing import blank_impulsive_spikes, despike_iq, repair_clipped_iq


SAMPLE_RATE = 60_000_000
RF_BANDWIDTH = 28_000_000
WINDOW_SECONDS = 0.05
HOP_SECONDS = 0.05
STFT_POINT = 1024
IMAGE_SIZE = 224
CENTER_FREQUENCY = 2_440_000_000
RX_GAIN = 30

ENABLE_SPEC_COLUMN_DENOISE = True
SPEC_COLUMN_QUANTILE = 30.0
ENABLE_SPEC_DB_CLIP = False
SPEC_CLIP_DB_MIN = -80.0
SPEC_CLIP_DB_MAX = 15.0


DEFAULT_DEVICE_ARGS = "type=x300"
DEFAULT_CPU_FORMAT = "fc32"
DEFAULT_WIRE_FORMAT = "sc16"
DEFAULT_SAMPLES_PER_BUFFER = 65_536

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def import_uhd() -> Any:
    try:
        import uhd
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import UHD Python bindings. Install UHD with Python support, "
            "then verify with: python -c \"import uhd\""
        ) from exc
    return uhd


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, **kwargs: Any) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, **kwargs)
        return

    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", **kwargs)
    parser.add_argument(
        "--no-" + name.lstrip("-"),
        dest=name.lstrip("-").replace("-", "_"),
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


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


def append_samples(buffer: deque, samples: np.ndarray) -> int:
    if samples.size == 0:
        return 0
    buffer.append(samples)
    return int(samples.size)


def pop_window(buffer: deque, window_samples: int) -> np.ndarray:
    parts = []
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

    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def drop_samples(buffer: deque, samples_to_drop: int) -> None:
    remaining = samples_to_drop
    while remaining > 0 and buffer:
        head = buffer[0]
        if head.size <= remaining:
            buffer.popleft()
            remaining -= head.size
        else:
            buffer[0] = head[remaining:]
            remaining = 0


def compute_spectrogram_numpy(
    data: np.ndarray,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slice_point = int(sample_rate * duration_time)
    segment = data[:slice_point]
    if segment.size < stft_point:
        raise ValueError(
            f"Chunk too small for STFT: got {segment.size} samples, need at least {stft_point}."
        )

    hop = max(1, stft_point // 2)
    frame_count = 1 + (segment.size - stft_point) // hop
    frames = np.empty((frame_count, stft_point), dtype=np.complex64)
    for idx in range(frame_count):
        start = idx * hop
        frames[idx] = segment[start : start + stft_point]

    window = np.hamming(stft_point).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1).T
    frequencies = np.fft.fftshift(np.fft.fftfreq(stft_point, d=1.0 / sample_rate))
    times = (np.arange(frame_count) * hop) / sample_rate
    return frequencies, times, spectrum


def spectrogram_to_uint8(magnitude_db: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.column_denoise:
        col_bg = np.percentile(magnitude_db, args.column_quantile, axis=0, keepdims=True)
        magnitude_db = magnitude_db - col_bg

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


def resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, size).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, size).astype(np.int64)
    return image[y_idx][:, x_idx]


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    x = gray.astype(np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.dstack((red, green, blue)).astype(np.float32).clip(0.0, 1.0) * 255.0


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("RGB PNG writer expects an HxWx3 array")

    raw_rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        PNG_SIGNATURE
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + png_chunk(b"IEND", b"")
    )


def save_spectrogram_image(spectrum: np.ndarray, output_path: Path, args: argparse.Namespace) -> None:
    magnitude_db = 10 * np.log10(np.abs(spectrum) + 1e-12)
    image_8bit = spectrogram_to_uint8(magnitude_db, args)
    image_8bit = np.flipud(image_8bit)
    image_8bit = resize_nearest(image_8bit, args.image_size)
    save_rgb_png(output_path, jet_colormap(image_8bit))


def make_output_path(output_dir: Path, prefix: str, index: int) -> Path:
    return output_dir / f"{prefix}_{index:06d}.png"


def process_iq_samples(
    iq_samples: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    buffer: deque,
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
        window = pop_window(buffer, window_samples)
        buffered_samples -= window_samples

        processed = preprocess_iq_chunk(window, args)
        _, _, spectrum = compute_spectrogram_numpy(
            processed,
            sample_rate=args.sample_rate,
            stft_point=args.stft_point,
            duration_time=args.window_seconds,
        )

        saved += 1
        output_path = make_output_path(output_dir, args.prefix, saved)
        save_spectrogram_image(spectrum=spectrum, output_path=output_path, args=args)

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


def configure_usrp(uhd: Any, args: argparse.Namespace) -> Any:
    usrp = uhd.usrp.MultiUSRP(args.device_args)

    if args.clock_source:
        usrp.set_clock_source(args.clock_source)
    if args.time_source:
        usrp.set_time_source(args.time_source)
    if args.subdev:
        usrp.set_rx_subdev_spec(args.subdev)

    channel = args.channel
    usrp.set_rx_rate(args.sample_rate, channel)
    usrp.set_rx_freq(uhd.types.TuneRequest(args.center_frequency), channel)
    usrp.set_rx_gain(args.gain, channel)

    if args.rf_bandwidth > 0:
        usrp.set_rx_bandwidth(args.rf_bandwidth, channel)
    if args.antenna:
        usrp.set_rx_antenna(args.antenna, channel)

    return usrp


def make_rx_streamer(uhd: Any, usrp: Any, args: argparse.Namespace) -> Any:
    stream_args = uhd.usrp.StreamArgs(args.cpu_format, args.wire_format)
    stream_args.channels = [args.channel]
    if args.stream_args:
        stream_args.args = args.stream_args
    return usrp.get_rx_stream(stream_args)


def start_continuous_stream(uhd: Any, rx_streamer: Any) -> None:
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)


def stop_continuous_stream(uhd: Any, rx_streamer: Any) -> None:
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)


def metadata_error_name(uhd: Any, metadata: Any) -> str:
    error_code = metadata.error_code
    for name in dir(uhd.types.RXMetadataErrorCode):
        if name.startswith("_"):
            continue
        if getattr(uhd.types.RXMetadataErrorCode, name) == error_code:
            return name
    return str(error_code)


def run_usrp_x300_to_spectrogram(args: argparse.Namespace) -> int:
    uhd = import_uhd()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(round(args.sample_rate * args.window_seconds))
    hop_samples = int(round(args.sample_rate * args.hop_seconds))
    if window_samples < args.stft_point:
        raise ValueError("window_seconds * sample_rate must be >= stft_point")
    if hop_samples <= 0:
        raise ValueError("hop_seconds must be positive")

    print(
        "USRP X300 realtime spectrogram pipeline: "
        f"device_args={args.device_args!r}, channel={args.channel}, "
        f"fs={args.sample_rate:g} Hz, rf_bandwidth={args.rf_bandwidth:g} Hz, "
        f"fc={args.center_frequency:g} Hz, gain={args.gain:g} dB, "
        f"window={args.window_seconds:g}s ({window_samples:,} IQ), "
        f"hop={args.hop_seconds:g}s ({hop_samples:,} IQ), stft={args.stft_point}"
    )
    print(f"Output: {output_dir}")

    usrp = configure_usrp(uhd, args)
    rx_streamer = make_rx_streamer(uhd, usrp, args)
    metadata = uhd.types.RXMetadata()
    recv_buffer = np.zeros((1, args.samples_per_buffer), dtype=np.complex64)

    buffer: deque[np.ndarray] = deque()
    buffered_samples = 0
    saved = 0
    read_samples = 0
    start_time = time.monotonic()
    overflow_count = 0

    start_continuous_stream(uhd, rx_streamer)
    try:
        while args.max_spectrograms is None or saved < args.max_spectrograms:
            num_rx = rx_streamer.recv(recv_buffer, metadata, args.recv_timeout)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                error_name = metadata_error_name(uhd, metadata)
                if "overflow" in error_name.lower():
                    overflow_count += 1
                    if overflow_count % 25 == 1:
                        print(f"[WARN] UHD overflow while receiving; count={overflow_count}")
                    continue
                raise RuntimeError(f"UHD RX error: {error_name}")

            if num_rx <= 0:
                continue

            iq_samples = recv_buffer[0, :num_rx].copy()
            if args.remove_dc:
                iq_samples = iq_samples - np.mean(iq_samples)

            read_samples += int(num_rx)
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
        stop_continuous_stream(uhd, rx_streamer)

    print(f"Done. Saved spectrograms: {saved}")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create realtime spectrogram PNGs from a USRP X300/X310 using UHD."
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Directory for spectrogram PNGs")
    parser.add_argument("--device-args", default=DEFAULT_DEVICE_ARGS, help="UHD device args, e.g. addr=192.168.10.2")
    parser.add_argument("--subdev", default=None, help="Optional UHD RX subdevice spec, e.g. A:0")
    parser.add_argument("--antenna", default=None, help="Optional RX antenna name, e.g. RX2 or TX/RX")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--clock-source", default=None, help="Optional clock source: internal, external, gpsdo")
    parser.add_argument("--time-source", default=None, help="Optional time source: internal, external, gpsdo")
    parser.add_argument("--center-frequency", type=float, default=CENTER_FREQUENCY)
    parser.add_argument("--gain", type=float, default=RX_GAIN)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--rf-bandwidth", type=int, default=RF_BANDWIDTH)
    parser.add_argument("--cpu-format", default=DEFAULT_CPU_FORMAT, help="UHD CPU sample format")
    parser.add_argument("--wire-format", default=DEFAULT_WIRE_FORMAT, help="UHD wire sample format")
    parser.add_argument("--stream-args", default=None, help="Optional UHD stream args string")
    parser.add_argument("--samples-per-buffer", type=int, default=DEFAULT_SAMPLES_PER_BUFFER)
    parser.add_argument("--recv-timeout", type=float, default=1.0)
    parser.add_argument("--remove-dc", action="store_true", help="Subtract chunk mean before buffering")

    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=HOP_SECONDS)
    parser.add_argument("--stft-point", type=int, default=STFT_POINT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--prefix", default="spectrogram")
    parser.add_argument("--image-renderer", choices=["fast"], default="fast", help="Numpy-only PNG renderer.")
    parser.add_argument("--normalize", action="store_true", help="Accepted for compatibility; UHD fc32 is already scaled.")
    parser.add_argument("--max-spectrograms", type=int, default=None)

    add_bool_arg(parser, "--column-denoise", default=ENABLE_SPEC_COLUMN_DENOISE)
    parser.add_argument("--column-quantile", type=float, default=SPEC_COLUMN_QUANTILE)
    add_bool_arg(parser, "--db-clip", default=ENABLE_SPEC_DB_CLIP)
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
    run_usrp_x300_to_spectrogram(parse_args())


if __name__ == "__main__":
    main()
