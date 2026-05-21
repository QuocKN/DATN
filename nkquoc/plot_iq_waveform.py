#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot IQ waveform from binary files.

Supported formats:
- int16 interleaved IQ: I0,Q0,I1,Q1,...
- float32 interleaved IQ: I0,Q0,I1,Q1,...
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

# ========================
# CONFIG
# ========================
INPUT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/2toan.bin"
# INPUT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/DroneDetect/MA1_0000_02.dat"
INPUT_DTYPE = "int16"  # "int16" or "float32"
SAMPLE_RATE = 28_000_000
START_SECONDS = 0.0
DURATION_SECONDS = 0.1
MAX_POINTS = 200_000
OUTPUT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/2toan_waveform5.png"
ENABLE_DESPIKE = True
DESPIKE_PERCENTILE = 97.5


def read_iq_segment(
    path: str,
    dtype: np.dtype,
    sample_rate: float,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if start_seconds < 0:
        raise ValueError("start_seconds must be >= 0")

    iq_count = int(sample_rate * duration_seconds)
    if iq_count <= 0:
        raise ValueError("sample_rate * duration_seconds must be >= 1 IQ sample")

    scalar_offset = int(start_seconds * sample_rate) * 2  # I and Q
    scalar_count = iq_count * 2
    byte_offset = scalar_offset * np.dtype(dtype).itemsize

    file_size = os.path.getsize(path)
    if byte_offset >= file_size:
        raise ValueError(
            f"start offset out of range: byte_offset={byte_offset}, file_size={file_size}"
        )

    with open(path, "rb") as handle:
        handle.seek(byte_offset)
        raw = np.fromfile(handle, dtype=dtype, count=scalar_count)

    if raw.size < 2:
        raise ValueError("not enough data to form IQ samples")
    if raw.size % 2 != 0:
        raw = raw[:-1]

    iq = raw.reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)


def downsample_for_plot(x: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError("max_points must be > 0")
    if x.size <= max_points:
        return x
    step = int(np.ceil(x.size / max_points))
    return x[::step]


def despike_iq(iq: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Limit extreme IQ outliers by clipping magnitude at a percentile threshold.
    Phase is preserved, only amplitude of outlier samples is reduced.
    """
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100)")

    amp = np.abs(iq)
    threshold = np.percentile(amp, percentile)
    if threshold <= 0:
        return iq

    out = iq.copy()
    mask = amp > threshold
    out[mask] = out[mask] / (amp[mask] + 1e-12) * threshold
    return out


def plot_waveform(
    iq: np.ndarray,
    sample_rate: float,
    out_path: str,
    title: str,
    max_points: int,
) -> None:
    i = downsample_for_plot(iq.real, max_points)
    q = downsample_for_plot(iq.imag, max_points)
    amp = downsample_for_plot(np.abs(iq), max_points)

    n = min(i.size, q.size, amp.size)
    i = i[:n]
    q = q[:n]
    amp = amp[:n]
    t = np.arange(n, dtype=np.float32) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, i, linewidth=0.7)
    axes[0].set_ylabel("I")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, q, linewidth=0.7, color="tab:orange")
    axes[1].set_ylabel("Q")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, amp, linewidth=0.7, color="tab:green")
    axes[2].set_ylabel("|IQ|")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if INPUT_DTYPE not in {"int16", "float32"}:
        raise ValueError("INPUT_DTYPE must be 'int16' or 'float32'")

    dtype = np.int16 if INPUT_DTYPE == "int16" else np.float32

    iq = read_iq_segment(
        path=INPUT_PATH,
        dtype=dtype,
        sample_rate=SAMPLE_RATE,
        start_seconds=START_SECONDS,
        duration_seconds=DURATION_SECONDS,
    )
    if ENABLE_DESPIKE:
        iq = despike_iq(iq, percentile=DESPIKE_PERCENTILE)

    title = (
        f"{os.path.basename(INPUT_PATH)} | dtype={INPUT_DTYPE} | "
        f"start={START_SECONDS}s | duration={DURATION_SECONDS}s"
    )
    if ENABLE_DESPIKE:
        title += f" | despike p{DESPIKE_PERCENTILE}"
    plot_waveform(
        iq=iq,
        sample_rate=SAMPLE_RATE,
        out_path=OUTPUT_PATH,
        title=title,
        max_points=MAX_POINTS,
    )
    print(f"[OK] Saved waveform: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
