from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import stft

IMAGE_SIZE = 224
ENABLE_SPEC_COLUMN_DENOISE = True
SPEC_COLUMN_QUANTILE = 60.0
ENABLE_SPEC_DB_CLIP = False
SPEC_CLIP_DB_MIN = -80.0
SPEC_CLIP_DB_MAX = 15.0


def STFT(
    data: np.ndarray,
    onside: bool = True,
    stft_point: int = 1024,
    fs: int = 100_000_000,
    duration_time: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slice_point = int(fs * duration_time)
    f, t, zxx = stft(
        data[0:slice_point],
        fs=fs,
        return_onesided=onside,
        window="hamming",
        nperseg=stft_point,
    )
    return f, t, zxx


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if ENABLE_SPEC_COLUMN_DENOISE:
        col_bg = np.percentile(magnitude_db, SPEC_COLUMN_QUANTILE, axis=0, keepdims=True)
        magnitude_db = magnitude_db - col_bg
    if ENABLE_SPEC_DB_CLIP:
        magnitude_db = np.clip(magnitude_db, SPEC_CLIP_DB_MIN, SPEC_CLIP_DB_MAX)
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

    with Image.open(output_path) as image:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        image.save(output_path)


def downsample_for_plot(x: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or x.size <= max_points:
        return x
    step = int(np.ceil(x.size / max_points))
    return x[::step]


def save_waveform_image(
    iq: np.ndarray,
    sample_rate: float,
    output_path: str,
    title: str,
    max_points: int = 20_000,
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
