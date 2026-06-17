from __future__ import annotations

import os
from ..base.iq_spectrogram_core import compute_spectrogram, save_spectrogram_image
from ..base.iq_waveform_plotter import WaveformPlotter
from .iq_reader import iter_iq_chunks_from_bin

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
    save_waveform: bool = True,
    waveform_prefix: str = "waveform",
    waveform_max_points: int = 20_000,
    start_iq_sample: int = 0,
) -> int:
    """Convert binary IQ file into a folder of spectrogram PNG images."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    os.makedirs(output_dir, exist_ok=True)
    waveform_plotter = WaveformPlotter(
        enabled=save_waveform,
        prefix=waveform_prefix,
        max_points=waveform_max_points,
    )
    waveform_plotter.prepare_output_dir(output_dir)

    max_iq_samples = None
    if max_duration_seconds is not None:
        max_iq_samples = int(sample_rate * max_duration_seconds)
        print(f"Reading first {max_duration_seconds}s ({max_iq_samples} IQ samples)...")
    if start_iq_sample:
        print(
            f"Skipping first {start_iq_sample} IQ samples "
            f"({start_iq_sample / sample_rate:.6f}s at {sample_rate} Hz)..."
        )

    min_samples_needed = max(stft_point, int(sample_rate * duration_time))
    if chunk_size < min_samples_needed:
        print(
            f"[WARN] chunk_size={chunk_size} is smaller than the required minimum "
            f"{min_samples_needed} samples (max(STFT_POINT, SAMPLE_RATE * DURATION_TIME)). "
            f"Using {min_samples_needed} instead."
        )
        chunk_size = min_samples_needed

    saved_count = 0
    chunks = iter_iq_chunks_from_bin(
        bin_path,
        chunk_size,
        normalize,
        max_iq_samples,
        skip_iq_samples=start_iq_sample,
    )
    for index, iq_chunk in enumerate(chunks, start=1):
        if iq_chunk.size < min_samples_needed:
            print(f"[SKIP] Chunk {index}: only {iq_chunk.size} samples, need {min_samples_needed}")
            continue

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
            waveform_plotter.save(
                iq=iq_chunk,
                sample_rate=sample_rate,
                output_dir=output_dir,
                source_path=bin_path,
                index=index,
            )
            saved_count += 1
            print(f"[OK] Saved {output_path}")
        except Exception as exc:
            print(f"[ERROR] Chunk {index}: {exc}")
            continue

    print(f"\nDone. Total spectrograms saved: {saved_count}")
    print(f"Output directory: {output_dir}")
    return saved_count
