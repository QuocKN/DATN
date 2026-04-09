import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_spectrogram(
    x,
    Fs,
    fc,
    n_fft=2048,
    hop_length=None,
    window="hann",
    max_frames=None,
):
    """Compute spectrogram (power/frequency, dB) from complex IQ samples.

    Args:
        x: Complex IQ signal (1-D).
        Fs: Sampling rate (Hz).
        fc: Center frequency (Hz), kept for compatibility.
        n_fft: FFT size.
        hop_length: STFT hop size. Defaults to n_fft // 4.
        window: Window type, "hann" or "rect".
        max_frames: Optional cap on number of output time frames.
    """
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

    magnitude = np.abs(spectrum)
    spec = (magnitude * magnitude).astype(np.float32, copy=False)

    spec = 10.0 * np.log10(spec + 1e-12)  # Power in dB

    time_axis = (frame_starts / Fs) * 1000.0
    freq_axis = np.linspace(-Fs / 2, Fs / 2, n_fft, dtype=np.float64)

    spec -= np.mean(spec, axis=0, keepdims=True)
    spec = cv2.GaussianBlur(spec, (3, 3), 0)

    return spec, time_axis, freq_axis


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

def plot_spectrogram(spec, time_axis, freq_axis, save_path=None):
    plt.figure(figsize=(8, 5))

    # Mở rộng trục tần số ±20%
    freq_min = freq_axis[0] / 1e6
    freq_max = freq_axis[-1] / 1e6
    freq_range = freq_max - freq_min
    expand = 0.2 * freq_range
    plt.imshow(
        spec.T,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1],
                freq_min - expand, freq_max + expand],
        cmap='jet'
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz, relative)")
    plt.title("Spectrogram (Power/Frequency)")

    plt.colorbar(label="Power/frequency (dB/Hz)")
    plt.tight_layout()

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(save_path):
            os.remove(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved spectrogram image to: {save_path}")

    # plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\DiepHM\Documents\AI\RF_Processing\CLEAN\PHA_HO\PHA_0001_00.dat"
    save_path = r"C:\Users\DiepHM\Documents\AI\RF_Processing\CLEAN\PHA_HO\PHA_0001_00_power601.png"

    Fs = 60e6
    fc = 2.4375e9
    start_time_ms = 0
    duration_ms = 60

    x = read_iq_dat_segment(
        file_path=file_path,
        Fs=Fs,
        start_time_ms=start_time_ms,
        duration_ms=duration_ms,
    )

    print(f"Loaded IQ samples: {x.shape[0]:,}")
    print(f"Complex signal: {np.iscomplexobj(x)}")

    spec, t, f = get_spectrogram(
        x=x,
        Fs=Fs,
        fc=fc,
        n_fft=1024,
        hop_length=128,
        max_frames=8000,
    )

    plot_spectrogram(spec, t, f, save_path=save_path)
