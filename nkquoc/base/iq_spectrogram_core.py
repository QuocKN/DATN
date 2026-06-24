from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import stft

IMAGE_SIZE = 224
ENABLE_SPEC_COLUMN_DENOISE = False
SPEC_COLUMN_QUANTILE = 60.0
ENABLE_SPEC_ROW_DENOISE = False
SPEC_ROW_QUANTILE = 50.0
ENABLE_SPEC_DC_MASK = False
SPEC_DC_MASK_BINS = 1
ENABLE_SPEC_FIXED_DB_RANGE = True
SPEC_DB_VMIN = -57.0
SPEC_DB_VMAX = 16.0
ENABLE_SPEC_DB_CLIP = False
SPEC_CLIP_DB_MIN = -80.0
SPEC_CLIP_DB_MAX = 15.0
SPECTROGRAM_CMAP = "jet"
SAVE_GRAYSCALE = False


def STFT(
    data: np.ndarray,
    onside: bool = True,
    stft_point: int = 1024,
    fs: int = 100_000_000,
    duration_time: float = 0.1,
    detrend: str | bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slice_point = int(fs * duration_time)
    f, t, zxx = stft(
        data[0:slice_point],
        fs=fs,
        return_onesided=onside,
        window="hamming",
        nperseg=stft_point,
        detrend=detrend,
    )
    return f, t, zxx


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    stft_point: int,
    duration_time: float,
    remove_dc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slice_point = int(sample_rate * duration_time)
    if slice_point <= 0:
        raise ValueError("sample_rate * duration_time must be positive")

    segment = data[:slice_point]
    if segment.size < stft_point:
        raise ValueError(
            f"Chunk too small for STFT: got {segment.size} samples, need at least {stft_point}."
        )

    if remove_dc:
        segment = segment - np.mean(segment)

    frequencies, times, spectrum = STFT(
        segment,
        onside=False,
        stft_point=stft_point,
        fs=sample_rate,
        duration_time=duration_time,
        detrend="constant" if remove_dc else False,
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
    if ENABLE_SPEC_ROW_DENOISE:
        row_bg = np.percentile(magnitude_db, SPEC_ROW_QUANTILE, axis=1, keepdims=True)
        magnitude_db = magnitude_db - row_bg
    if ENABLE_SPEC_DC_MASK and SPEC_DC_MASK_BINS > 0:
        zero_bin = int(np.argmin(np.abs(frequencies)))
        lo = max(0, zero_bin - SPEC_DC_MASK_BINS)
        hi = min(magnitude_db.shape[0], zero_bin + SPEC_DC_MASK_BINS + 1)
        magnitude_db[lo:hi, :] = float(np.median(magnitude_db))
    if ENABLE_SPEC_DB_CLIP:
        magnitude_db = np.clip(magnitude_db, SPEC_CLIP_DB_MIN, SPEC_CLIP_DB_MAX)
    extent = (times.min(), times.max(), frequencies.min(), frequencies.max())

    dpi = 100
    figure = plt.figure(figsize=(IMAGE_SIZE / dpi, IMAGE_SIZE / dpi), dpi=dpi)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    imshow_kwargs = {}
    if ENABLE_SPEC_FIXED_DB_RANGE:
        imshow_kwargs["vmin"] = SPEC_DB_VMIN
        imshow_kwargs["vmax"] = SPEC_DB_VMAX
    axes.imshow(
        magnitude_db,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap=SPECTROGRAM_CMAP,
        **imshow_kwargs,
    )
    axes.axis("off")
    if title:
        axes.set_title(title)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)

    with Image.open(output_path) as image:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if SAVE_GRAYSCALE:
            image = image.convert("L")
        image.save(output_path)
