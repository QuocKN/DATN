#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert binary RF IQ data (int16 interleaved I,Q,I,Q,...) into spectrogram PNG images.

Reads raw int16 data from a .bin file, chunks it, and generates spectrograms.
"""

import os
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import stft


# ========================
# CONFIG
# ========================
# INPUT_BIN_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone/2toan.bin"
# OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone/spectrograms"
INPUT_BIN_PATH ="E:\\DATN_DATA\\RF\\1toan.bin"
OUTPUT_DIR = "E:\\DATN_DATA\\RF\\1toan_spectrograms"


SAMPLE_RATE = 28_000_000  # Hz (adjust as needed)
STFT_POINT = 2048
DURATION_TIME = 0.05  # seconds per spectrogram
CHUNK_SIZE = 4096  # IQ samples per chunk
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224

# Set to None to read entire file, or set to N seconds to read only first N seconds
MAX_DURATION_SECONDS = 500  # Read only first 500 seconds

# Set to True to normalize int16 to [-1, 1] range, False to keep raw values
NORMALIZE = False

def STFT(data,
         onside: bool = True,
         stft_point: int = 1024,
         fs: int = 100_000_000,
         duration_time: float = 0.1,
         ):

    slice_point = int(fs * duration_time)

    f, t, Zxx = stft(
        data[0:slice_point],
        fs=fs,
        return_onesided=onside,
        window="hamming",
        nperseg=stft_point,
    )

    return f, t, Zxx

def iter_iq_chunks_from_bin(
    bin_path: str,
    chunk_size: int,
    normalize: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield complex IQ chunks from a binary int16 interleaved file.
    
    Binary layout: I_0, Q_0, I_1, Q_1, I_2, Q_2, ...
    Each value is int16 (2 bytes).
    
    Args:
        bin_path: Path to .bin file
        chunk_size: Number of IQ samples per chunk (each chunk = 2 * chunk_size int16 values)
        normalize: If True, scale int16 to [-1, 1]
        max_iq_samples: Maximum number of IQ samples to read (None = read all)
    
    Yields:
        Complex numpy arrays of shape (chunk_size,)
    """
    total_read = 0
    with open(bin_path, "rb") as f:
        while True:
            # Check if we've hit the max limit
            if max_iq_samples is not None and total_read >= max_iq_samples:
                break
            
            # Read 2 * chunk_size int16 values (I_0, Q_0, I_1, Q_1, ...)
            bytes_to_read = 2 * chunk_size * 2  # int16 = 2 bytes
            raw_bytes = f.read(bytes_to_read)
            
            if not raw_bytes:
                break
            
            # Convert bytes to int16 array
            int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
            
            # If we got fewer samples than requested, yield only what we have
            if len(int16_data) % 2 != 0:
                # Odd number of int16 values -> drop the last one
                int16_data = int16_data[:-1]
            
            if len(int16_data) == 0:
                break
            
            # Reshape to (N, 2) where each row is [I, Q]
            iq_pairs = int16_data.reshape(-1, 2)
            iq_count = len(iq_pairs)
            
            # Extract I and Q components
            i_values = iq_pairs[:, 0].astype(np.float32)
            q_values = iq_pairs[:, 1].astype(np.float32)
            
            # Optionally normalize
            if normalize:
                i_values /= 32768.0
                q_values /= 32768.0
            
            # Combine into complex IQ data
            iq_data = i_values + 1j * q_values
            
            total_read += iq_count
            yield iq_data


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

def check_iq_amplitude(iq_data: np.ndarray, index: int | None = None) -> None:
    amp = np.abs(iq_data)
    i = iq_data.real
    q = iq_data.imag

    prefix = f"[AMP CHECK] Chunk {index}" if index is not None else "[AMP CHECK]"

    print("\n" + "=" * 60)
    print(prefix)
    print(f"I min/max       : {i.min():.2f} / {i.max():.2f}")
    print(f"Q min/max       : {q.min():.2f} / {q.max():.2f}")
    print(f"Amp min/max     : {amp.min():.2f} / {amp.max():.2f}")
    print(f"Amp mean/std    : {amp.mean():.2f} / {amp.std():.2f}")
    print("Amp percentile  :")
    print(f"  90%   = {np.percentile(amp, 90):.2f}")
    print(f"  95%   = {np.percentile(amp, 95):.2f}")
    print(f"  99%   = {np.percentile(amp, 99):.2f}")
    print(f"  99.9% = {np.percentile(amp, 99.9):.2f}")

    # Dữ liệu của bạn có vẻ là ADC 12-bit lưu trong int16
    adc_max = 2047
    adc_min = -2048

    i_clip_ratio = np.mean((i <= adc_min) | (i >= adc_max)) * 100
    q_clip_ratio = np.mean((q <= adc_min) | (q >= adc_max)) * 100

    print(f"I 12-bit clipping ratio: {i_clip_ratio:.6f}%")
    print(f"Q 12-bit clipping ratio: {q_clip_ratio:.6f}%")

    if i_clip_ratio > 0.01 or q_clip_ratio > 0.01:
        print("[WARN] Có dấu hiệu clipping/saturation theo ADC 12-bit.")

    p90 = np.percentile(amp, 90)
    p95 = np.percentile(amp, 95)
    p99 = np.percentile(amp, 99)
    p999 = np.percentile(amp, 99.9)

    if p95 > 3 * p90:
        print("[WARN] Biên độ tăng đột ngột từ p90 lên p95 -> có nhiều burst/spike mạnh.")

    if p99 > 5 * p90:
        print("[WARN] p99 lớn hơn nhiều so với p90 -> spike có thể gây sọc dọc.")

    print("=" * 60 + "\n")

def clip_iq_amplitude(iq_data: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """
    Giới hạn biên độ IQ để giảm spike mạnh gây sọc dọc.
    Không đổi pha, chỉ giảm magnitude của các mẫu quá lớn.
    """
    amp = np.abs(iq_data)
    threshold = np.percentile(amp, percentile)

    iq_clipped = iq_data.copy()
    mask = amp > threshold

    iq_clipped[mask] = iq_clipped[mask] / (amp[mask] + 1e-12) * threshold

    return iq_clipped
def limit_iq_amplitude(iq_data: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    amp = np.abs(iq_data)
    threshold = np.percentile(amp, percentile)

    iq_out = iq_data.copy()
    mask = amp > threshold

    iq_out[mask] = iq_out[mask] / (amp[mask] + 1e-12) * threshold

    return iq_out

def repair_clipped_iq(iq_data: np.ndarray,
                      adc_min: int = -2048,
                      adc_max: int = 2047) -> np.ndarray:
    """
    Thay các mẫu IQ bị clipping bằng nội suy tuyến tính.
    Chỉ dùng cho xử lý spectrogram, không khôi phục được tín hiệu gốc hoàn hảo.
    """
    i = iq_data.real.copy()
    q = iq_data.imag.copy()

    i_bad = (i <= adc_min) | (i >= adc_max)
    q_bad = (q <= adc_min) | (q >= adc_max)

    x = np.arange(len(iq_data))

    if np.any(i_bad) and np.any(~i_bad):
        i[i_bad] = np.interp(x[i_bad], x[~i_bad], i[~i_bad])

    if np.any(q_bad) and np.any(~q_bad):
        q[q_bad] = np.interp(x[q_bad], x[~q_bad], q[~q_bad])

    return i + 1j * q

def convert_bin_to_spectrograms(
    bin_path: str,
    output_dir: str,
    sample_rate: int = 28_000_000,
    stft_point: int = 2048,
    duration_time: float = 0.05,
    chunk_size: int = 4096,
    prefix: str = "spectrogram",
    normalize: bool = False,
    max_duration_seconds: int | None = None,
) -> int:
    """Convert binary IQ file into a folder of spectrogram PNG images."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Calculate max IQ samples to read if duration limit is set
    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)
        print(f"Reading first {max_duration_seconds}s ({max_iq_samples} IQ samples)...")

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] chunk_size={chunk_size} is smaller than the required minimum "
            f"{min_samples_needed} samples (max(STFT_POINT, SAMPLE_RATE * DURATION_TIME)). "
            f"Using {min_samples_needed} instead."
        )
        chunk_size = min_samples_needed

    saved_count = 0
    for index, iq_chunk in enumerate(iter_iq_chunks_from_bin(bin_path, chunk_size, normalize, max_iq_samples), start=1):
        if iq_chunk.size < min_samples_needed:
            print(f"[SKIP] Chunk {index}: only {iq_chunk.size} samples, need {min_samples_needed}")
            continue

        # check_iq_amplitude(iq_chunk, index=index)
        # Xử lý trước khi STFT
        # iq_chunk = iq_chunk - np.mean(iq_chunk)
        # iq_chunk = clip_iq_amplitude(iq_chunk, percentile=95.0)
        # iq_chunk = limit_iq_amplitude(iq_chunk, percentile=85.0)
        # iq_chunk = repair_clipped_iq(iq_chunk)
        try:
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
            print(f"[OK] Saved {output_path}")
        except Exception as e:
            print(f"[ERROR] Chunk {index}: {e}")
            continue

    print(f"\nDone. Total spectrograms saved: {saved_count}")
    print(f"Output directory: {output_dir}")
    return saved_count


def main() -> None:
    convert_bin_to_spectrograms(
        bin_path=INPUT_BIN_PATH,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        stft_point=STFT_POINT,
        duration_time=DURATION_TIME,
        chunk_size=CHUNK_SIZE,
        prefix=OUTPUT_PREFIX,
        normalize=NORMALIZE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )


if __name__ == "__main__":
    main()
