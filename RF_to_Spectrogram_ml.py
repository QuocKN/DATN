import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_iq_dat_segment(file_path, Fs, start_time_ms=0.0, duration_ms=200.0):
    """Read I/Q float32 interleaved .dat segment as complex64."""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0.")
    if Fs <= 0:
        raise ValueError("Fs must be > 0.")

    start_sample = int(Fs * (start_time_ms / 1000.0))
    n_complex_samples = int(Fs * (duration_ms / 1000.0))
    if n_complex_samples <= 0:
        raise ValueError("Requested segment has zero samples.")

    count_values = n_complex_samples * 2
    offset_values = start_sample * 2

    raw = np.fromfile(
        file_path,
        dtype=np.float32,
        count=count_values,
        offset=offset_values * 4,
    )

    if raw.size < 2:
        raise ValueError(f"File {file_path} does not contain enough samples.")
    if raw.size % 2 != 0:
        raw = raw[:-1]

    return raw.view(np.complex64)


def _get_base_spec_db(
    x,
    Fs,
    fc,
    n_fft=2048,
    hop_length=None,
    window="hann",
    max_frames=None,
    eps=1e-20,
):
    """Compute PSD in dB/Hz before ML normalization."""
    _ = fc  # Kept for API compatibility.

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input signal x must be 1-D.")
    if x.size < n_fft:
        raise ValueError("Signal too short for chosen n_fft.")

    if hop_length is None:
        hop_length = n_fft // 4
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0.")

    if window == "hann":
        w = np.hanning(n_fft).astype(np.float32, copy=False)
    elif window == "rect":
        w = np.ones(n_fft, dtype=np.float32)
    else:
        raise ValueError("window must be 'hann' or 'rect'.")

    n_frames = (x.size - n_fft) // hop_length + 1
    if n_frames <= 0:
        raise ValueError("Signal too short.")

    frame_step = 1
    if max_frames is not None and max_frames > 0 and n_frames > max_frames:
        frame_step = int(np.ceil(n_frames / max_frames))

    frame_starts = np.arange(0, n_frames, frame_step, dtype=np.int64) * hop_length
    n_out_frames = frame_starts.size
    sample_stride = x.strides[0]

    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_out_frames, n_fft),
        strides=(sample_stride * hop_length * frame_step, sample_stride),
        writeable=False,
    )

    frames = frames * w[None, :]
    spectrum = np.fft.fft(frames, axis=1)
    spectrum = np.fft.fftshift(spectrum, axes=1)

    power = np.abs(spectrum) ** 2
    window_energy = float(np.sum(w * w))
    psd = power / (Fs * window_energy)
    spec_db_hz = 10.0 * np.log10(psd + eps)

    time_axis = (frame_starts / Fs) * 1000.0
    freq_axis = np.linspace(-Fs / 2, Fs / 2, n_fft, dtype=np.float64)

    return spec_db_hz.astype(np.float32, copy=False), time_axis, freq_axis


def get_spectrogram_ml(
    x,
    Fs,
    fc,
    n_fft=2048,
    hop_length=None,
    window="hann",
    max_frames=None,
    center_per_freq_bin=True,
    clip_percentiles=(1.0, 99.0),
    normalize="zscore",
    blur_kernel=3,
):
    """ML-friendly spectrogram with robust normalization."""
    spec, time_axis, freq_axis = _get_base_spec_db(
        x=x,
        Fs=Fs,
        fc=fc,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        max_frames=max_frames,
    )

    if center_per_freq_bin:
        spec = spec - np.mean(spec, axis=0, keepdims=True)

    if clip_percentiles is not None:
        low, high = clip_percentiles
        if not (0 <= low < high <= 100):
            raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100.")
        lo_val, hi_val = np.percentile(spec, [low, high])
        spec = np.clip(spec, lo_val, hi_val)

    if normalize == "zscore":
        mu = float(np.mean(spec))
        sigma = float(np.std(spec))
        if sigma > 1e-12:
            spec = (spec - mu) / sigma
        else:
            spec = spec - mu
    elif normalize == "minmax":
        min_v = float(np.min(spec))
        max_v = float(np.max(spec))
        if max_v > min_v:
            spec = (spec - min_v) / (max_v - min_v)
    elif normalize in (None, "none"):
        pass
    else:
        raise ValueError("normalize must be 'zscore', 'minmax', or None/'none'.")

    if blur_kernel is not None and blur_kernel > 1:
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd.")
        spec = cv2.GaussianBlur(spec.astype(np.float32, copy=False), (blur_kernel, blur_kernel), 0)

    return spec.astype(np.float32, copy=False), time_axis, freq_axis


def plot_spectrogram(spec, time_axis, freq_axis, save_path=None):
    plt.figure(figsize=(8, 5))

    freq_min = freq_axis[0] / 1e6
    freq_max = freq_axis[-1] / 1e6
    freq_range = freq_max - freq_min
    expand = 0.2 * freq_range

    plt.imshow(
        spec.T,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], freq_min - expand, freq_max + expand],
        cmap="jet",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz, relative)")
    plt.title("Spectrogram (ML Stable)")
    plt.colorbar(label="Normalized Power")
    plt.tight_layout()

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(save_path):
            os.remove(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved spectrogram image to: {save_path}")


def save_spectrogram_chunks_ml(
    file_path,
    Fs,
    fc,
    start_time_ms,
    end_time_ms,
    chunk_ms=5,
    n_fft=1024,
    hop_length=128,
    max_frames=8000,
    output_dir="spectrogram_chunks_ml",
):
    os.makedirs(output_dir, exist_ok=True)

    total_ms = end_time_ms - start_time_ms
    n_chunks = int(np.floor(total_ms / chunk_ms))
    print(f"Total chunks: {n_chunks}")

    for i in range(n_chunks):
        chunk_start = start_time_ms + i * chunk_ms
        x = read_iq_dat_segment(
            file_path=file_path,
            Fs=Fs,
            start_time_ms=chunk_start,
            duration_ms=chunk_ms,
        )

        if x.size < n_fft:
            print(f"Chunk {i} too short, skipped.")
            continue

        spec, t, f = get_spectrogram_ml(
            x=x,
            Fs=Fs,
            fc=fc,
            n_fft=n_fft,
            hop_length=hop_length,
            max_frames=max_frames,
        )

        img_path = os.path.join(output_dir, f"spec_{int(chunk_start)}ms.png")
        plot_spectrogram(spec, t, f, save_path=img_path)
        print(f"Saved: {img_path}")


if __name__ == "__main__":
    file_path = r"C:\Users\DiepHM\Documents\Project\RF_Processing\CLEAN\PHA_HO\PHA_0001_00.dat"
    output_dir = r"C:\Users\DiepHM\Documents\Project\RF_Processing\CLEAN\PHA_HO\spec_chunks_ml"

    Fs = 60e6
    fc = 2.4375e9
    start_time_ms = 130 
    end_time_ms = 180
    chunk_ms = 50

    save_spectrogram_chunks_ml(
        file_path=file_path,
        Fs=Fs,
        fc=fc,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        chunk_ms=chunk_ms,
        n_fft=1024,
        hop_length=128,
        max_frames=8000,
        output_dir=output_dir,
    )
