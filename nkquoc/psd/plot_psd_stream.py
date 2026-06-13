#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def read_iq_chunk(
    f,
    dtype: str,
    num_iq_samples: int,
    normalize: bool,
) -> np.ndarray:
    if dtype == "int16_iq":
        raw = np.fromfile(f, dtype=np.int16, count=num_iq_samples * 2)

        if raw.size < 2:
            return np.array([], dtype=np.complex64)

        if raw.size % 2 != 0:
            raw = raw[:-1]

        raw = raw.reshape(-1, 2)

        i = raw[:, 0].astype(np.float32)
        q = raw[:, 1].astype(np.float32)

        if normalize:
            i /= 32768.0
            q /= 32768.0

        return (i + 1j * q).astype(np.complex64)

    elif dtype == "float32_iq":
        raw = np.fromfile(f, dtype=np.float32, count=num_iq_samples * 2)

        if raw.size < 2:
            return np.array([], dtype=np.complex64)

        if raw.size % 2 != 0:
            raw = raw[:-1]

        raw = raw.reshape(-1, 2)

        i = raw[:, 0]
        q = raw[:, 1]

        return (i + 1j * q).astype(np.complex64)

    else:
        raise ValueError("dtype must be int16_iq or float32_iq")


def psd_one_chunk(
    x: np.ndarray,
    fs: float,
    nfft: int,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, len(x))

    f, pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        return_onesided=False,
        scaling="density",
    )

    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)

    return f, pxx


def average_psd_stream(
    filepath: str,
    fs: float,
    dtype: str,
    normalize: bool,
    seconds_per_chunk: float,
    num_chunks: int,
    nfft: int,
    nperseg: int,
    skip_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(filepath)
    samples_per_chunk = int(round(seconds_per_chunk * fs))
    skip_samples = int(round(skip_seconds * fs))

    if samples_per_chunk <= 0:
        raise ValueError("seconds_per_chunk must be > 0")

    bytes_per_iq = 4 if dtype == "int16_iq" else 8

    psd_sum = None
    f_ref = None
    used_chunks = 0

    with open(path, "rb") as f:
        if skip_samples > 0:
            f.seek(skip_samples * bytes_per_iq)

        for idx in range(num_chunks):
            x = read_iq_chunk(
                f=f,
                dtype=dtype,
                num_iq_samples=samples_per_chunk,
                normalize=normalize,
            )

            if len(x) < 256:
                break

            f_vec, pxx = psd_one_chunk(
                x=x,
                fs=fs,
                nfft=nfft,
                nperseg=nperseg,
            )

            if psd_sum is None:
                psd_sum = np.zeros_like(pxx, dtype=np.float64)
                f_ref = f_vec

            psd_sum += pxx
            used_chunks += 1

            print(f"{path.name}: chunk {idx + 1}/{num_chunks}, samples={len(x):,}")

    if used_chunks == 0:
        raise RuntimeError(f"No valid chunk read from {filepath}")

    psd_avg = psd_sum / used_chunks
    psd_db = 10.0 * np.log10(np.maximum(psd_avg, 1e-20))

    return f_ref, psd_db


def main():
    parser = argparse.ArgumentParser(description="Plot PSD from large IQ files using streaming chunks")

    parser.add_argument("--signal-file", required=True)
    parser.add_argument("--noise-file", default=None)

    parser.add_argument(
        "--dtype",
        choices=["int16_iq", "float32_iq"],
        default="int16_iq",
    )

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sample-rate", type=float, required=True)

    parser.add_argument("--seconds-per-chunk", type=float, default=0.02)
    parser.add_argument("--num-chunks", type=int, default=20)

    parser.add_argument("--skip-seconds", type=float, default=0.0)

    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--nperseg", type=int, default=4096)

    parser.add_argument("--save", default=None, help="Save plot to image file, e.g. psd.png")

    args = parser.parse_args()

    print("Computing signal PSD...")
    f_sig, psd_sig = average_psd_stream(
        filepath=args.signal_file,
        fs=args.sample_rate,
        dtype=args.dtype,
        normalize=args.normalize,
        seconds_per_chunk=args.seconds_per_chunk,
        num_chunks=args.num_chunks,
        nfft=args.nfft,
        nperseg=args.nperseg,
        skip_seconds=args.skip_seconds,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(f_sig / 1e6, psd_sig, label="Signal", linewidth=1.0)

    if args.noise_file:
        print("Computing noise PSD...")
        f_noise, psd_noise = average_psd_stream(
            filepath=args.noise_file,
            fs=args.sample_rate,
            dtype=args.dtype,
            normalize=args.normalize,
            seconds_per_chunk=args.seconds_per_chunk,
            num_chunks=args.num_chunks,
            nfft=args.nfft,
            nperseg=args.nperseg,
            skip_seconds=args.skip_seconds,
        )

        plt.plot(f_noise / 1e6, psd_noise, label="Noise Reference", linewidth=1.0)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("PSD using Welch, averaged over chunks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()