from __future__ import annotations

import numpy as np


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

    adc_max = 2047
    adc_min = -2048
    i_clip_ratio = np.mean((i <= adc_min) | (i >= adc_max)) * 100
    q_clip_ratio = np.mean((q <= adc_min) | (q >= adc_max)) * 100

    print(f"I 12-bit clipping ratio: {i_clip_ratio:.6f}%")
    print(f"Q 12-bit clipping ratio: {q_clip_ratio:.6f}%")

    if i_clip_ratio > 0.01 or q_clip_ratio > 0.01:
        print("[WARN] Co dau hieu clipping/saturation theo ADC 12-bit.")

    p90 = np.percentile(amp, 90)
    p95 = np.percentile(amp, 95)
    p99 = np.percentile(amp, 99)
    if p95 > 3 * p90:
        print("[WARN] Bien do tang dot ngot tu p90 len p95 -> co nhieu burst/spike manh.")
    if p99 > 5 * p90:
        print("[WARN] p99 lon hon nhieu so voi p90 -> spike co the gay soc doc.")

    print("=" * 60 + "\n")


def despike_iq(iq_data: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Compress outlier amplitudes while preserving phase."""
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


def interpolate_masked_iq(iq_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return iq_data

    i = iq_data.real.copy()
    q = iq_data.imag.copy()
    x = np.arange(iq_data.size)
    good = ~mask
    if np.sum(good) < 2:
        return iq_data

    i[mask] = np.interp(x[mask], x[good], i[good])
    q[mask] = np.interp(x[mask], x[good], q[good])
    return i + 1j * q


def contiguous_run_mask(
    candidate: np.ndarray,
    target_width: int,
    max_run_len: int,
) -> np.ndarray:
    idx = np.flatnonzero(candidate)
    remove_mask = np.zeros_like(candidate, dtype=bool)
    if idx.size == 0:
        return remove_mask

    run_start = idx[0]
    run_prev = idx[0]
    for k in idx[1:]:
        if k == run_prev + 1:
            run_prev = k
            continue
        mark_run_edges(remove_mask, run_start, run_prev, target_width, max_run_len)
        run_start = k
        run_prev = k
    mark_run_edges(remove_mask, run_start, run_prev, target_width, max_run_len)
    return remove_mask


def mark_run_edges(
    remove_mask: np.ndarray,
    run_start: int,
    run_prev: int,
    target_width: int,
    max_run_len: int,
) -> None:
    run_len = run_prev - run_start + 1
    if target_width <= 0 and run_len <= max_run_len:
        remove_mask[run_start : run_prev + 1] = True
        return
    if target_width < run_len <= max_run_len:
        keep_start = run_start + (run_len - target_width) // 2
        keep_end = keep_start + target_width - 1
        remove_mask[run_start:keep_start] = True
        remove_mask[keep_end + 1 : run_prev + 1] = True


def blank_impulsive_spikes(
    iq_data: np.ndarray,
    median_kernel: int = 129,
    threshold_sigma: float = 8.0,
    max_spike_width: int = 8,
) -> np.ndarray:
    """Interpolate short impulse spikes while keeping wider bursts."""
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
    candidate = amp > baseline + threshold_sigma * sigma
    spike_mask = contiguous_run_mask(candidate, target_width=0, max_run_len=max_spike_width)
    return interpolate_masked_iq(iq_data, spike_mask)


def thin_high_amplitude_runs(
    iq_data: np.ndarray,
    percentile: float = 99.2,
    target_width: int = 2,
    max_run_len: int = 48,
) -> np.ndarray:
    """Thin short high-amplitude runs by interpolating their edges."""
    if iq_data.size < 3:
        return iq_data
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100)")
    if target_width < 1:
        target_width = 1
    if max_run_len < target_width:
        return iq_data

    amp = np.abs(iq_data)
    hot = amp > np.percentile(amp, percentile)
    remove_mask = contiguous_run_mask(hot, target_width=target_width, max_run_len=max_run_len)
    return interpolate_masked_iq(iq_data, remove_mask)


def thin_saturation_runs(
    iq_data: np.ndarray,
    saturation_level: float = 1850.0,
    target_width: int = 1,
    max_run_len: int = 256,
) -> np.ndarray:
    """Thin ADC-like saturated runs by interpolating their edges."""
    if iq_data.size < 3:
        return iq_data
    if target_width < 1:
        target_width = 1
    if max_run_len < target_width:
        return iq_data

    sat = (np.abs(iq_data.real) >= saturation_level) | (np.abs(iq_data.imag) >= saturation_level)
    remove_mask = contiguous_run_mask(sat, target_width=target_width, max_run_len=max_run_len)
    return interpolate_masked_iq(iq_data, remove_mask)


def repair_clipped_iq(
    iq_data: np.ndarray,
    adc_min: int = -2048,
    adc_max: int = 2047,
) -> np.ndarray:
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
