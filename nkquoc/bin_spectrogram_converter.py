from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from .bin_iq_reader import iter_iq_chunks_from_bin
    from .iq_preprocessing import (
        blank_impulsive_spikes,
        despike_iq,
        repair_clipped_iq,
        thin_high_amplitude_runs,
        thin_saturation_runs,
    )
    from .iq_spectrogram_core import compute_spectrogram, save_spectrogram_image, save_waveform_image
except ImportError:
    from bin_iq_reader import iter_iq_chunks_from_bin
    from iq_preprocessing import (
        blank_impulsive_spikes,
        despike_iq,
        repair_clipped_iq,
        thin_high_amplitude_runs,
        thin_saturation_runs,
    )
    from iq_spectrogram_core import compute_spectrogram, save_spectrogram_image, save_waveform_image


@dataclass(frozen=True)
class ProcessingConfig:
    save_waveform: bool = True
    waveform_prefix: str = "waveform"
    waveform_max_points: int = 20_000
    enable_despike: bool = False
    despike_percentile: float = 99.5
    enable_repair_clipped: bool = False
    enable_impulse_blanker: bool = False
    blanker_median_kernel: int = 129
    blanker_threshold_sigma: float = 6.0
    blanker_max_spike_width: int = 10
    enable_peak_thinning: bool = False
    thin_percentile: float = 99.2
    thin_target_width: int = 2
    thin_apply_max_run: int = 48
    enable_saturation_thinning: bool = False
    saturation_level: float = 1850.0
    sat_target_width: int = 1
    sat_max_run: int = 256


def preprocess_iq_chunk(iq_chunk, config: ProcessingConfig):
    if config.enable_saturation_thinning:
        iq_chunk = thin_saturation_runs(
            iq_chunk,
            saturation_level=config.saturation_level,
            target_width=config.sat_target_width,
            max_run_len=config.sat_max_run,
        )
    if config.enable_peak_thinning:
        iq_chunk = thin_high_amplitude_runs(
            iq_chunk,
            percentile=config.thin_percentile,
            target_width=config.thin_target_width,
            max_run_len=config.thin_apply_max_run,
        )
    if config.enable_impulse_blanker:
        iq_chunk = blank_impulsive_spikes(
            iq_chunk,
            median_kernel=config.blanker_median_kernel,
            threshold_sigma=config.blanker_threshold_sigma,
            max_spike_width=config.blanker_max_spike_width,
        )
    if config.enable_despike:
        iq_chunk = despike_iq(iq_chunk, percentile=config.despike_percentile)
    if config.enable_repair_clipped:
        iq_chunk = repair_clipped_iq(iq_chunk)
    return iq_chunk


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
    processing: ProcessingConfig | None = None,
) -> int:
    """Convert binary IQ file into a folder of spectrogram PNG images."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    processing = processing or ProcessingConfig()
    os.makedirs(output_dir, exist_ok=True)
    waveform_dir = os.path.join(output_dir, "waveforms")
    if processing.save_waveform:
        os.makedirs(waveform_dir, exist_ok=True)

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
    chunks = iter_iq_chunks_from_bin(bin_path, chunk_size, normalize, max_iq_samples)
    for index, iq_chunk in enumerate(chunks, start=1):
        if iq_chunk.size < min_samples_needed:
            print(f"[SKIP] Chunk {index}: only {iq_chunk.size} samples, need {min_samples_needed}")
            continue

        iq_chunk = preprocess_iq_chunk(iq_chunk, processing)
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
            if processing.save_waveform:
                waveform_path = os.path.join(
                    waveform_dir, f"{processing.waveform_prefix}_{index:06d}.png"
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
                    max_points=processing.waveform_max_points,
                )
            saved_count += 1
            print(f"[OK] Saved {output_path}")
        except Exception as exc:
            print(f"[ERROR] Chunk {index}: {exc}")
            continue

    print(f"\nDone. Total spectrograms saved: {saved_count}")
    print(f"Output directory: {output_dir}")
    return saved_count
