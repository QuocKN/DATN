#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from iq_readers import read_interleaved_iq
from snr_core import (
    MAX_NFFT,
    NFFT,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    _find_signal_band,
    _robust_noise_density,
    pwelch_shifted,
    validate_input_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate SNR from separate signal(noise+drone) and noise-only IQ files."
    )
    parser.add_argument("--signal-file", required=True, help="Input signal IQ file (signal+noise)")
    parser.add_argument("--noise-file", required=True, help="Input noise-only IQ file")
    parser.add_argument(
        "--dtype",
        choices=["float32_iq", "int16_iq"],
        default="float32_iq",
        help="Binary sample format: interleaved I,Q values (default: float32_iq)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize int16 IQ by 32768 before analysis. This does not change SNR, only scale.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {SAMPLE_RATE:g})",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=28e6,
        help="Bandwidth hint in Hz for signal band localization",
    )
    parser.add_argument("--nfft", type=int, default=NFFT, help=f"Analysis FFT length (default: {NFFT})")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=SEGMENT_SECONDS,
        help="Use only the first N seconds; set 0 to use whole files (up to shortest file)",
    )
    return parser.parse_args()


def _max_iq_samples(segment_seconds: float | None, sample_rate: float) -> int | None:
    if segment_seconds is None or segment_seconds <= 0:
        return None
    return int(round(segment_seconds * sample_rate))


def _validate_complex_finite(iq: np.ndarray, name: str) -> None:
    if iq.size == 0:
        raise ValueError(f"{name}: no IQ samples were read.")
    if not np.iscomplexobj(iq):
        raise ValueError(f"{name}: expected complex IQ samples.")
    if (not np.all(np.isfinite(iq.real))) or (not np.all(np.isfinite(iq.imag))):
        raise ValueError(
            f"{name}: IQ contains NaN/Inf. Check --dtype and --normalize."
        )


def main() -> None:
    args = parse_args()
    signal_path = validate_input_file(args.signal_file)
    noise_path = validate_input_file(args.noise_file)
    max_samples = _max_iq_samples(args.segment_seconds, args.sample_rate)

    sig = read_interleaved_iq(
        signal_path,
        dtype=args.dtype,
        max_iq_samples=max_samples,
        normalize=args.normalize,
    )
    nos = read_interleaved_iq(
        noise_path,
        dtype=args.dtype,
        max_iq_samples=max_samples,
        normalize=args.normalize,
    )
    _validate_complex_finite(sig, "signal-file")
    _validate_complex_finite(nos, "noise-file")

    used = min(sig.size, nos.size)
    if used < 1024:
        raise ValueError("Not enough matched samples between signal/noise files.")
    sig = sig[:used]
    nos = nos[:used]

    if max_samples is None:
        print(f"Using matched whole data: {used} IQ samples")
    else:
        print(f"Using first {args.segment_seconds} s (matched): {used} IQ samples")

    nfft = int(min(args.nfft, MAX_NFFT))
    f_sig, p_sig = pwelch_shifted(sig, args.sample_rate, nfft)
    f_nos, p_nos = pwelch_shifted(nos, args.sample_rate, nfft)

    p_sig = np.maximum(np.real(p_sig), 0.0)
    p_nos = np.maximum(np.real(p_nos), 0.0)
    if p_sig.size != p_nos.size or not np.allclose(f_sig, f_nos):
        raise ValueError("PSD grids of signal/noise files do not match.")

    df = float(np.median(np.diff(f_sig)))
    if not np.isfinite(df) or df <= 0:
        raise ValueError("Invalid PSD frequency spacing.")

    pre_noise_density = _robust_noise_density(p_sig)
    start, end = _find_signal_band(
        p_sig,
        df,
        pre_noise_density,
        bandwidth_hint_hz=args.bandwidth,
        fs=args.sample_rate,
    )
    if end <= start:
        raise ValueError("Failed to detect a valid signal band.")

    band_mask = np.zeros_like(p_sig, dtype=bool)
    band_mask[start:end] = True

    p_total = float(np.sum(p_sig[band_mask]) * df)
    p_noise = float(np.sum(p_nos[band_mask]) * df)
    p_signal = max(p_total - p_noise, np.finfo(np.float64).tiny)
    snr_db = float(10.0 * np.log10(p_signal / max(p_noise, np.finfo(np.float64).tiny)))

    f1 = float(f_sig[start])
    f2 = float(f_sig[end - 1] + df)
    print(f"Estimated SNR (noise-ref): {snr_db:.2f} dB")
    print(
        f"signal_band=[{f1:.1f}, {f2:.1f}] Hz, "
        f"center={0.5 * (f1 + f2):.1f} Hz, bandwidth={(end - start) * df:.1f} Hz"
    )
    print(
        f"total_band_power={p_total:.6g}, noise_band_power={p_noise:.6g}, "
        f"signal_band_power={p_signal:.6g}"
    )
    print(f"psd_bins={p_sig.size}, band_bins={end - start}")


if __name__ == "__main__":
    main()

