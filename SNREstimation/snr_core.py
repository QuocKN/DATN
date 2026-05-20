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
    if (not np.all(np.isfinite(iq.real))) or (not np.all(np.isfinite(iq.imag))):
        raise ValueError(
            "IQ data contains NaN/Inf. Likely wrong input dtype. "
            "Try --dtype int16_iq for int16 .bin files or --dtype float32_iq for float32 files."
        )


def pwelch_shifted(x: np.ndarray, fs: float, nfft: int):
    use_nfft = int(min(nfft, MAX_NFFT))
    nperseg = max(256, min(int(len(x) / 10), use_nfft, 8192))
    f, pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, nfft=use_nfft, return_onesided=False)
    # welch(..., return_onesided=False) returns bins in FFT order; fftshift centers DC.
    return np.fft.fftshift(f), np.fft.fftshift(pxx)


def _robust_noise_density(pxx: np.ndarray) -> float:
    eps = np.finfo(np.float64).tiny
    pxx = np.maximum(np.asarray(pxx, dtype=np.float64), eps)
    pxx_db = 10.0 * np.log10(pxx)

    noise_db = float(np.median(pxx_db))
    for _ in range(3):
        keep = pxx_db <= (noise_db + 3.0)
        if not np.any(keep):
            break
        noise_db = float(np.median(pxx_db[keep]))
    return float(10.0 ** (noise_db / 10.0))


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask.astype(bool)):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _find_signal_band(
    pxx: np.ndarray,
    df: float,
    noise_density: float,
    bandwidth_hint_hz: float | None = None,
    fs: float | None = None,
) -> tuple[int, int]:
    eps = np.finfo(np.float64).tiny
    pxx = np.maximum(np.asarray(pxx, dtype=np.float64), eps)
    pxx_db = 10.0 * np.log10(pxx)
    noise_db = 10.0 * np.log10(max(noise_density, eps))

    # Light smoothing to reduce isolated spikes while preserving the occupied band.
    smooth_bins = 9
    kernel = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
    pxx_db_smooth = np.convolve(pxx_db, kernel, mode="same")

    sig_mask = pxx_db_smooth > (noise_db + 6.0)
    runs = _find_runs(sig_mask)
    if not runs:
        peak = int(np.argmax(pxx))
        half = max(1, int(round(1e6 / df)) // 2)
        return max(0, peak - half), min(pxx.size, peak + half)

    gap_bins = max(1, int(round(0.3e6 / df)))
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        if s - cur_e <= gap_bins:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Choose the run with the largest excess power above the noise floor.
    best_run = merged[0]
    best_excess = -1.0
    for s, e in merged:
        excess = float(np.sum(np.maximum(pxx[s:e] - noise_density, 0.0)))
        if excess > best_excess:
            best_excess = excess
            best_run = (s, e)

    # Use bandwidth hint only when it is clearly meaningful (not near full-band).
    if fs is not None and bandwidth_hint_hz is not None:
        bw = float(bandwidth_hint_hz)
        if (3.0 * df) <= bw <= (0.8 * fs):
            hint_bins = int(round(bw / df))
            if hint_bins > 0 and hint_bins < pxx.size:
                peak = int(np.argmax(pxx))
                start_min = max(0, peak - hint_bins + 1)
                start_max = min(peak, pxx.size - hint_bins)
                if start_max >= start_min:
                    energy = np.convolve(pxx, np.ones(hint_bins, dtype=np.float64), mode="valid")
                    local_start = start_min + int(np.argmax(energy[start_min : start_max + 1]))
                    return local_start, local_start + hint_bins
    return best_run


def estimate_snr_from_psd(x: np.ndarray, fs: float, bandwidth: float, nfft: int) -> dict:
    fvec, pxx = pwelch_shifted(x, fs, nfft)
    pxx = np.maximum(np.real(pxx), 0.0)

    if fvec.size < 2:
        raise ValueError("Need at least two PSD bins to estimate SNR.")

    df = float(np.median(np.diff(fvec)))
    if df <= 0:
        raise ValueError("Invalid PSD frequency spacing.")

    # Estimate noise first, then detect occupied signal band robustly from PSD.
    pre_noise_density = _robust_noise_density(pxx)
    start, end = _find_signal_band(
        pxx,
        df,
        pre_noise_density,
        bandwidth_hint_hz=bandwidth,
        fs=fs,
    )
    band_bins = int(end - start)
    if band_bins <= 0:
        raise ValueError("Failed to detect a valid signal band.")

    f1 = float(fvec[start])
    f2 = float(fvec[end - 1] + df)

    signal_mask = np.zeros_like(pxx, dtype=bool)
    signal_mask[start:end] = True

    guard_bins = max(1, int(round(0.5e6 / df)))
    guard_start = max(0, start - guard_bins)
    guard_end = min(pxx.size, end + guard_bins)
    noise_mask = np.ones_like(pxx, dtype=bool)
    noise_mask[guard_start:guard_end] = False
    if not np.any(noise_mask):
        noise_mask = ~signal_mask
    if not np.any(noise_mask):
        raise ValueError("Not enough off-band bins to estimate noise. Use a smaller --bandwidth.")

    noise_region = pxx[noise_mask]
    noise_density = _robust_noise_density(noise_region)
    noise_power = float(noise_density * band_bins * df)

    # Estimate signal power as excess above noise floor to avoid catastrophic cancellation
    # when requested signal bandwidth is close to full Nyquist bandwidth.
    signal_psd_excess = np.maximum(pxx[signal_mask] - noise_density, 0.0)
    signal_power = max(float(np.sum(signal_psd_excess) * df), np.finfo(np.float64).tiny)
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
