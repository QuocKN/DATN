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
INPUT_BIN_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/2toan.bin"
OUTPUT_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/spectrograms"
# INPUT_BIN_PATH ="E:\\DATN_DATA\\RF\\1toan.bin"
# OUTPUT_DIR = "E:\\DATN_DATA\\RF\\1toan_spectrograms"


SAMPLE_RATE = 28_000_000  # Hz (adjust as needed)
STFT_POINT = 1024
DURATION_TIME = 0.05  # seconds per spectrogram
CHUNK_SIZE = 4096  # IQ samples per chunk
OUTPUT_PREFIX = "spectrogram"
IMAGE_SIZE = 224
WAVEFORM_PREFIX = "waveform"
WAVEFORM_MAX_POINTS = 20_000
SAVE_WAVEFORM = True

# Set to None to read entire file, or set to N seconds to read only first N seconds
MAX_DURATION_SECONDS = 500  # Read only first 500 seconds

# Set to True to normalize int16 to [-1, 1] range, False to keep raw values
NORMALIZE = False
ENABLE_DESPIKE = True
DESPIKE_PERCENTILE = 99
ENABLE_REPAIR_CLIPPED = False
ENABLE_IMPULSE_BLANKER = False
BLANKER_MEDIAN_KERNEL = 129
BLANKER_THRESHOLD_SIGMA = 6.0
BLANKER_MAX_SPIKE_WIDTH = 100

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

def despike_iq(iq_data: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Giảm spike bằng cách giới hạn biên độ |IQ| theo percentile.
    Giữ nguyên pha, chỉ nén biên độ các mẫu outlier.
    """
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100)")

    amp = np.abs(iq_data)
    threshold = np.percentile(amp, percentile)
    if threshold <= 0:
        return iq_data

    iq_out = iq_data.copy()
    mask = amp > threshold

    iq_out[mask] = iq_out[mask] / (amp[mask] + 1e-12) * threshold

    return iq_out


def blank_impulsive_spikes(
    iq_data: np.ndarray,
    median_kernel: int = 129,
    threshold_sigma: float = 8.0,
    max_spike_width: int = 8,
) -> np.ndarray:
    """
    Khử spike hẹp theo thời gian để giảm sọc dọc spectrogram.
    Burst FHSS rộng hơn max_spike_width sẽ được giữ nguyên.
    """
    if iq_data.size < 3:
        return iq_data
    if median_kernel < 3:
        median_kernel = 3
    if median_kernel % 2 == 0:
        median_kernel += 1
    if max_spike_width < 1:
        return iq_data

    amp = np.abs(iq_data)
    kernel = np.ones(median_kernel, dtype=np.float32) / median_kernel
    baseline = np.convolve(amp, kernel, mode="same")
    resid = amp - baseline
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    sigma = 1.4826 * mad
    threshold = baseline + threshold_sigma * sigma
    candidate = amp > threshold

    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return iq_data

    spike_mask = np.zeros_like(candidate, dtype=bool)
    run_start = idx[0]
    run_prev = idx[0]
    for k in idx[1:]:
        if k == run_prev + 1:
            run_prev = k
            continue
        run_len = run_prev - run_start + 1
        if run_len <= max_spike_width:
            spike_mask[run_start : run_prev + 1] = True
        run_start = k
        run_prev = k
    run_len = run_prev - run_start + 1
    if run_len <= max_spike_width:
        spike_mask[run_start : run_prev + 1] = True

    if not np.any(spike_mask):
        return iq_data

    i = iq_data.real.copy()
    q = iq_data.imag.copy()
    x = np.arange(iq_data.size)
    good = ~spike_mask
    if np.sum(good) < 2:
        return iq_data

    i[spike_mask] = np.interp(x[spike_mask], x[good], i[good])
    q[spike_mask] = np.interp(x[spike_mask], x[good], q[good])
    return i + 1j * q

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
    waveform_dir = os.path.join(output_dir, "waveforms")
    if SAVE_WAVEFORM:
        os.makedirs(waveform_dir, exist_ok=True)

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
        if ENABLE_IMPULSE_BLANKER:
            iq_chunk = blank_impulsive_spikes(
                iq_chunk,
                median_kernel=BLANKER_MEDIAN_KERNEL,
                threshold_sigma=BLANKER_THRESHOLD_SIGMA,
                max_spike_width=BLANKER_MAX_SPIKE_WIDTH,
            )
        if ENABLE_DESPIKE:
            iq_chunk = despike_iq(iq_chunk, percentile=DESPIKE_PERCENTILE)
            #  plot_waveform(
            #     iq=iq,
            #     sample_rate=SAMPLE_RATE,
            #     out_path=OUTPUT_PATH,
            #     title=title,
            #     max_points=MAX_POINTS,
            # )
        if ENABLE_REPAIR_CLIPPED:
            iq_chunk = repair_clipped_iq(iq_chunk)
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
            if SAVE_WAVEFORM:
                waveform_path = os.path.join(
                    waveform_dir, f"{WAVEFORM_PREFIX}_{index:06d}.png"
                )
                waveform_title = (
                    f"{os.path.basename(bin_path)} | chunk={index} | "
                    f"samples={iq_chunk.size}"
                )
                save_waveform_image(
                    iq=iq_chunk,
                    sample_rate=sample_rate,
                    output_path=waveform_path,
                    title=waveform_title,
                    max_points=WAVEFORM_MAX_POINTS,
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
