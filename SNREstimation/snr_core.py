from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import welch

SAMPLE_RATE = 60e6
EXPECTED_BANDWIDTH = 28e6
NFFT = 409600
MAX_NFFT = 16384
SEGMENT_SECONDS = 1.0
def add_common_snr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True, help="Input IQ file")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {SAMPLE_RATE:g})",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=EXPECTED_BANDWIDTH,
        help=f"Expected signal bandwidth in Hz (default: {EXPECTED_BANDWIDTH:g})",
    )
    parser.add_argument("--nfft", type=int, default=NFFT, help=f"Analysis FFT length (default: {NFFT})")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=SEGMENT_SECONDS,
        help="Use only the first N seconds for speed; set 0 to use the whole file",
    )


def max_iq_samples(segment_seconds: float | None, sample_rate: float) -> int | None:
    if segment_seconds is None or segment_seconds <= 0:
        return None
    return int(round(segment_seconds * sample_rate))


def validate_input_file(path: str) -> Path:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return input_path


def validate_iq_samples(iq: np.ndarray) -> None:
    if iq.size == 0:
        raise ValueError("No IQ samples were read. Check the input file and sample format.")
    if not np.iscomplexobj(iq):
        raise ValueError("Expected complex IQ samples.")


def pwelch_shifted(x: np.ndarray, fs: float, nfft: int):
    use_nfft = int(min(nfft, MAX_NFFT))
    nperseg = max(256, min(int(len(x) / 10), use_nfft, 8192))
    f, pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, nfft=use_nfft, return_onesided=False)
    idx = np.argsort(f)
    f = f[idx]
    pxx = pxx[idx]
    return np.fft.fftshift(f), np.fft.fftshift(pxx)


def estimate_snr_from_psd(x: np.ndarray, fs: float, bandwidth: float, nfft: int) -> dict:
    fvec, pxx = pwelch_shifted(x, fs, nfft)
    pxx = np.maximum(np.real(pxx), 0.0)

    if fvec.size < 2:
        raise ValueError("Need at least two PSD bins to estimate SNR.")

    df = float(np.median(np.diff(fvec)))
    if df <= 0:
        raise ValueError("Invalid PSD frequency spacing.")

    max_bandwidth = fs * 0.95
    bandwidth = float(min(max(bandwidth, df), max_bandwidth))
    band_bins = max(1, int(round(bandwidth / df)))
    band_bins = min(band_bins, pxx.size)

    band_energy = np.convolve(pxx, np.ones(band_bins, dtype=np.float64), mode="valid")
    start = int(np.argmax(band_energy))
    end = start + band_bins

    f1 = float(fvec[start])
    f2 = float(fvec[end - 1] + df)

    signal_mask = np.zeros_like(pxx, dtype=bool)
    signal_mask[start:end] = True

    guard_bins = max(1, band_bins // 10)
    guard_start = max(0, start - guard_bins)
    guard_end = min(pxx.size, end + guard_bins)
    noise_mask = np.ones_like(pxx, dtype=bool)
    noise_mask[guard_start:guard_end] = False
    if not np.any(noise_mask):
        noise_mask = ~signal_mask
    if not np.any(noise_mask):
        raise ValueError("Not enough off-band bins to estimate noise. Use a smaller --bandwidth.")

    noise_density = float(np.median(pxx[noise_mask]))
    total_band_power = float(np.sum(pxx[signal_mask]) * df)
    noise_power = float(noise_density * band_bins * df)
    signal_power = max(total_band_power - noise_power, np.finfo(np.float64).tiny)
    snr_db = float(10.0 * np.log10(signal_power / max(noise_power, np.finfo(np.float64).tiny)))

    return {
        "snr_db": snr_db,
        "f1_hz": f1,
        "f2_hz": f2,
        "center_hz": 0.5 * (f1 + f2),
        "bandwidth_hz": float(band_bins * df),
        "signal_power": signal_power,
        "noise_power": noise_power,
        "noise_density": noise_density,
        "psd_bins": int(pxx.size),
        "band_bins": int(band_bins),
    }


def print_snr_result(result: dict) -> None:
    print(f"Estimated SNR: {result['snr_db']:.2f} dB")
    print(
        f"signal_band=[{result['f1_hz']:.1f}, {result['f2_hz']:.1f}] Hz, "
        f"center={result['center_hz']:.1f} Hz, bandwidth={result['bandwidth_hz']:.1f} Hz"
    )
    print(
        f"signal_power={result['signal_power']:.6g}, "
        f"noise_power={result['noise_power']:.6g}, noise_density={result['noise_density']:.6g}"
    )
    print(f"psd_bins={result['psd_bins']}, band_bins={result['band_bins']}")


def run_snr_estimation(args: argparse.Namespace, reader: Callable[[Path, int | None], np.ndarray]) -> dict:
    input_path = validate_input_file(args.input)
    max_samples = max_iq_samples(args.segment_seconds, args.sample_rate)
    iq = reader(input_path, max_samples)
    validate_iq_samples(iq)

    if max_samples is None:
        print(f"Using whole file ({len(iq)} IQ samples)")
    else:
        print(f"Using first {args.segment_seconds} s of data ({len(iq)} IQ samples)")

    result = estimate_snr_from_psd(iq, args.sample_rate, args.bandwidth, args.nfft)
    print_snr_result(result)
    return result
