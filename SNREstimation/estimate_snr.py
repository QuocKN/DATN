#!/usr/bin/env python3
"""Estimate SNR from interleaved IQ .bin/.dat/.iq files.

This is a Python translation of the MATLAB flow in SNREstimation:
- locate signal band (f1,f2)
- compensate frequency offset
- downsample/resample for PSD
- compute signal and noise power and SNR

Dependencies: numpy, scipy
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from scipy.signal import welch, resample


# ========================
# CONFIG
# ========================
INPUT_FILE = r"/home/quocnk/Documents/NKQuoc/Data/DroneDetect/MAV_0000_03.dat"
SAMPLE_RATE = 60e6
EXPECTED_BANDWIDTH = 28e6
NFFT = 409600
# Performance caps to avoid huge FFTs/resamples on long recordings
# Tune these to trade accuracy vs speed
MAX_REFS = 2e6
MAX_RENFFT = 16384
MAX_NFFT = 16384
# Use only first SEGMENT_SECONDS of the recording (None = use all)
SEGMENT_SECONDS = 1.0


def read_iq_file(path: str, max_iq_samples: int | None = None) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 2 != 0:
        data = data[:-1]
    iq = data[0::2] + 1j * data[1::2]
    return iq

    # count = None if max_iq_samples is None else max_iq_samples * 2
    # data = np.fromfile(path, dtype=np.int16, count=count)
    # if data.size % 2 != 0:
    #     data = data[:-1]
    # i_values = data[0::2].astype(np.float32)
    # q_values = data[1::2].astype(np.float32)
    # iq = i_values + 1j * q_values
    # return iq


def pwelch_shifted(x: np.ndarray, fs: float, nfft: int):
    use_nfft = int(min(nfft, MAX_NFFT))
    nperseg = max(256, min(int(len(x) / 10), use_nfft, 8192))
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=nperseg, nfft=use_nfft, return_onesided=False)
    # center frequencies and PSD
    idx = np.argsort(f)
    f = f[idx]
    Pxx = Pxx[idx]
    # shift to [-fs/2, fs/2)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    return f, Pxx


def dronesOFDMFreqShiftEsti(x: np.ndarray, fs: float, bw: float, nfft: int) -> Tuple[float, float]:
    fvec, pxx = pwelch_shifted(x, fs, nfft)
    pxx_db = 10 * np.log10(np.abs(pxx) + 1e-12)
    pxx_norm = pxx_db / np.max(np.abs(pxx_db))
    pxx_norm = pxx_norm + (-np.min(pxx_norm))

    bwNfft = max(1, int(round(nfft * (bw / fs) * 0.9)))
    energy_len = len(pxx_norm) - bwNfft - 20
    if energy_len <= 0:
        # fallback: return center band
        mid = 0.0
        return mid - bw / 2, mid + bw / 2

    energy = np.zeros(energy_len)
    energy[0] = np.mean(pxx_norm[0:bwNfft] ** 2)
    for i in range(1, energy_len):
        energy[i] = energy[i - 1] - (pxx_norm[i - 1] ** 2) / bwNfft + (pxx_norm[i + bwNfft - 1] ** 2) / bwNfft

    maxIdx = int(np.argmax(energy))
    # careful with indices
    maxIdx = np.clip(maxIdx, 0, len(fvec) - bwNfft - 1)
    f1 = fvec[maxIdx]
    f2 = fvec[maxIdx + bwNfft]
    return float(f1), float(f2)


def dronesOFDMFreqCompensation(x: np.ndarray, fs: float, f: float) -> np.ndarray:
    t = np.arange(len(x)) / fs
    return x * np.exp(-1j * 2 * np.pi * f * t)


def positionFind(dataIQ: np.ndarray, fs: float, bw: float, NFFT: int):
    f1, f2 = dronesOFDMFreqShiftEsti(dataIQ, fs, bw, NFFT)
    f = (f1 + f2) / 2.0
    sig = dronesOFDMFreqCompensation(dataIQ, fs, f)
    f11 = f1 - f
    f22 = f2 - f

    refs = min(fs / 2.0, MAX_REFS)
    reNfft = int(min(NFFT // 2, MAX_RENFFT))
    if refs <= 0:
        raise ValueError("refs (resample rate) must be > 0")

    # resample sig to refs
    new_len = max(1, int(round(len(sig) * (refs / fs))))
    sigResample = resample(sig, new_len)

    # PSD on resampled
    fvec3, pxx3 = pwelch_shifted(sigResample, refs, reNfft)

    # find indices nearest to f11 and f22
    temp11 = np.abs(fvec3 - f11)
    temp22 = np.abs(fvec3 - f22)
    idx1 = int(np.argmin(temp11))
    idx2 = int(np.argmin(temp22))

    bwNoise = 1e6
    f3 = 0.75e6
    bwNoiseNfft = int(round(reNfft * (bwNoise / refs)))
    idx4 = int(idx1 - round(reNfft * (f3 / refs)))
    idx3 = int(idx4 - bwNoiseNfft)

    # clip indices
    idx1 = np.clip(idx1, 0, len(fvec3) - 1)
    idx2 = np.clip(idx2, 0, len(fvec3) - 1)
    idx3 = np.clip(idx3, 0, len(fvec3) - 1)
    idx4 = np.clip(idx4, 0, len(fvec3) - 1)

    return idx1, idx2, idx3, idx4, f1, f2


def snrEsti(dataIQ: np.ndarray, fs: float, nfft: int, f1: float, f2: float, idx1: int, idx2: int, idx3: int, idx4: int) -> float:
    f = (f1 + f2) / 2.0
    sig = dronesOFDMFreqCompensation(dataIQ, fs, f)
    refs = min(fs / 2.0, MAX_REFS)
    reNfft = max(2, int(min(nfft // 2, MAX_RENFFT)))
    new_len = max(1, int(round(len(sig) * (refs / fs))))
    sigResample = resample(sig, new_len)

    # compute FFT temp
    fft_vals = np.fft.fft(sigResample, reNfft) / float(reNfft)
    fftTemp = np.abs(np.fft.fftshift(fft_vals))

    # ensure indices in range
    L = len(fftTemp)
    i1 = np.clip(idx1, 0, L - 1)
    i2 = np.clip(idx2, 0, L - 1)
    i3 = np.clip(idx3, 0, L - 1)
    i4 = np.clip(idx4, 0, L - 1)

    if i2 <= i1:
        sigPower = np.mean(fftTemp[i1:i1+1] ** 2)
    else:
        sigPower = np.mean(fftTemp[i1:i2] ** 2)
    if i4 <= i3:
        nosPower = np.mean(fftTemp[i3:i3+1] ** 2)
    else:
        nosPower = np.mean(fftTemp[i3:i4] ** 2)

    # avoid negative or zero
    if nosPower <= 0 or sigPower <= nosPower:
        return float(-999.0)

    snr = 10.0 * np.log10((sigPower - nosPower) / nosPower)
    return float(np.real(snr))


def main():
    if not INPUT_FILE:
        raise ValueError("INPUT_FILE is empty. Set it to a valid .dat/.iq file path.")
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    # if requested, only keep a short segment to speed up processing
    max_samples = None
    if SEGMENT_SECONDS is not None and SEGMENT_SECONDS > 0:
        max_samples = int(round(SEGMENT_SECONDS * SAMPLE_RATE))

    dataIQ = read_iq_file(INPUT_FILE, max_iq_samples=max_samples)
    if max_samples is not None:
        print(f"Using first {SEGMENT_SECONDS} s of data ({len(dataIQ)} samples) to speed up processing")
    idx1, idx2, idx3, idx4, f1, f2 = positionFind(dataIQ, SAMPLE_RATE, EXPECTED_BANDWIDTH, NFFT)

    snr_value = snrEsti(dataIQ, SAMPLE_RATE, NFFT, f1, f2, idx1, idx2, idx3, idx4)

    print(f"Estimated SNR: {snr_value:.2f} dB")
    print(f"f1={f1:.1f} Hz, f2={f2:.1f} Hz")
    print(f"indices: idx1={idx1}, idx2={idx2}, idx3={idx3}, idx4={idx4}")


if __name__ == "__main__":
    main()
