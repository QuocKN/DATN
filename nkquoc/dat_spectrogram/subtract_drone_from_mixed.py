#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate WiFi-only data by subtracting drone-only data from drone+WiFi data.

Best case:
    Use synchronized IQ .dat recordings:
        wifi ~= mixed(drone+wifi) - alpha * drone

Fallback:
    If recordings are not phase-aligned, subtract STFT power instead:
        wifi_power ~= max(power(mixed) - beta * power(drone), floor)

    If only spectrogram images are available, subtract image intensities:
        wifi_img ~= max(mixed_img - drone_weight * drone_img, 0)

Examples:
    python nkquoc/dat_spectrogram/subtract_drone_from_mixed.py \
        --mode dat \
        --drone-root /path/to/drone \
        --mixed-root /path/to/drone_wifi \
        --output-root /path/to/wifi_estimated

    python nkquoc/dat_spectrogram/subtract_drone_from_mixed.py \
        --mode stft \
        --drone-root /path/to/drone \
        --mixed-root /path/to/drone_wifi \
        --output-root /path/to/wifi_spectrograms
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable


DAT_EXTENSIONS = {".dat"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EPSILON = 1e-12


def collect_files(root: Path, extensions: set[str]) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file extension: {root}")
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)


def pair_by_relative_path(drone_root: Path, mixed_root: Path, extensions: set[str]) -> list[tuple[Path, Path, Path]]:
    drone_files = collect_files(drone_root, extensions)
    mixed_files = collect_files(mixed_root, extensions)

    if drone_root.is_file() and mixed_root.is_file():
        return [(drone_root, mixed_root, Path(mixed_root.name))]

    drone_by_rel = {path.relative_to(drone_root): path for path in drone_files}
    pairs: list[tuple[Path, Path, Path]] = []
    missing: list[Path] = []

    for mixed_path in mixed_files:
        rel_path = mixed_path.relative_to(mixed_root)
        drone_path = drone_by_rel.get(rel_path)
        if drone_path is None:
            missing.append(rel_path)
            continue
        pairs.append((drone_path, mixed_path, rel_path))

    if missing:
        print(f"[WARN] Missing {len(missing)} drone files with the same relative path.")
        for rel_path in missing[:10]:
            print(f"       {rel_path}")
        if len(missing) > 10:
            print("       ...")

    return pairs


def iq_dtype(dat_format: str) -> np.dtype:
    if dat_format == "float32_iq":
        return np.dtype("<f4")
    if dat_format == "int16_iq":
        return np.dtype("<i2")
    raise ValueError("dat_format must be 'float32_iq' or 'int16_iq'")


def read_iq_block(file_obj, iq_samples: int, dat_format: str, normalize_int16: bool) -> np.ndarray:
    dtype = iq_dtype(dat_format)
    raw = np.fromfile(file_obj, dtype=dtype, count=iq_samples * 2)
    if raw.size % 2:
        raw = raw[:-1]
    if raw.size == 0:
        return np.empty(0, dtype=np.complex64)

    pairs = raw.reshape(-1, 2)
    i_values = pairs[:, 0].astype(np.float32, copy=False)
    q_values = pairs[:, 1].astype(np.float32, copy=False)
    if dat_format == "int16_iq" and normalize_int16:
        i_values = i_values / 32768.0
        q_values = q_values / 32768.0
    return (i_values + 1j * q_values).astype(np.complex64, copy=False)


def estimate_scalar_gain(mixed: np.ndarray, drone: np.ndarray) -> complex:
    denom = np.vdot(drone, drone)
    if abs(denom) < 1e-12:
        return 1.0 + 0.0j
    return np.vdot(drone, mixed) / denom


def crop_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = min(a.shape[0], b.shape[0])
    cols = min(a.shape[1], b.shape[1])
    return a[:rows, :cols], b[:rows, :cols]


def estimate_power_scale(
    mixed_power: np.ndarray,
    drone_power: np.ndarray,
    mode: str,
    fixed_scale: float,
    mask_percentile: float,
) -> float:
    if mode == "fixed":
        return fixed_scale
    if mode == "none":
        return 1.0

    valid = drone_power > np.percentile(drone_power, mask_percentile)
    if not np.any(valid):
        return 1.0

    mixed_valid = mixed_power[valid].reshape(-1)
    drone_valid = drone_power[valid].reshape(-1)
    if mode == "median":
        ratios = mixed_valid / np.maximum(drone_valid, EPSILON)
        return max(float(np.median(ratios)), 0.0)

    denom = float(np.dot(drone_valid, drone_valid))
    if denom <= EPSILON:
        return 1.0

    scale = float(np.dot(mixed_valid, drone_valid) / denom)
    return max(scale, 0.0)


def compute_power_spectrogram(
    iq: np.ndarray,
    sample_rate: int,
    stft_point: int,
    overlap_ratio: float,
    remove_dc: bool,
    window: str,
) -> np.ndarray:
    try:
        from scipy.signal import stft
    except ImportError as exc:
        raise ImportError("STFT mode requires scipy. Install it with: pip install scipy") from exc

    if iq.size < stft_point:
        raise ValueError(f"Need at least {stft_point} IQ samples for STFT, got {iq.size}.")

    if remove_dc:
        iq = iq - np.mean(iq)

    noverlap = int(stft_point * overlap_ratio)
    noverlap = min(max(noverlap, 0), stft_point - 1)
    _, _, spectrum = stft(
        iq,
        fs=sample_rate,
        return_onesided=False,
        window=window,
        nperseg=stft_point,
        nfft=stft_point,
        noverlap=noverlap,
        detrend="constant" if remove_dc else False,
        boundary=None,
        padded=False,
    )
    spectrum = np.fft.fftshift(spectrum, axes=0)
    return (np.abs(spectrum) ** 2).astype(np.float32, copy=False)


def drone_band_suppression_db(mixed_power: np.ndarray, residual_power: np.ndarray, drone_power: np.ndarray, mask_percentile: float) -> float:
    mask = drone_power > np.percentile(drone_power, mask_percentile)
    if not np.any(mask):
        return 0.0
    before = float(np.mean(mixed_power[mask]))
    after = float(np.mean(residual_power[mask]))
    return 10.0 * np.log10((after + EPSILON) / (before + EPSILON))


def power_to_db_image(
    power: np.ndarray,
    image_size: int,
    fixed_db_range: bool,
    db_min: float,
    db_max: float,
    cmap_name: str,
    grayscale: bool,
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("STFT mode requires Pillow. Install it with: pip install Pillow") from exc

    db = 10.0 * np.log10(np.maximum(power, EPSILON))
    if fixed_db_range:
        norm = (db - db_min) / max(db_max - db_min, EPSILON)
    else:
        low, high = np.percentile(db, [1.0, 99.5])
        norm = (db - low) / max(float(high - low), EPSILON)
    norm = np.clip(norm, 0.0, 1.0)

    gray = (norm * 255.0).astype(np.uint8)
    image = Image.fromarray(gray, mode="L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    if grayscale:
        return np.asarray(image)

    try:
        from matplotlib import colormaps

        color = colormaps[cmap_name](np.asarray(image).astype(np.float32) / 255.0)
        return (color[:, :, :3] * 255.0).astype(np.uint8)
    except Exception:
        return np.asarray(image.convert("RGB"))


def save_stft_residual_image(
    drone: np.ndarray,
    mixed: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[float, float]:
    drone_power = compute_power_spectrogram(
        iq=drone,
        sample_rate=args.sample_rate,
        stft_point=args.stft_point,
        overlap_ratio=args.overlap_ratio,
        remove_dc=args.remove_dc,
        window=args.window,
    )
    mixed_power = compute_power_spectrogram(
        iq=mixed,
        sample_rate=args.sample_rate,
        stft_point=args.stft_point,
        overlap_ratio=args.overlap_ratio,
        remove_dc=args.remove_dc,
        window=args.window,
    )
    drone_power, mixed_power = crop_same_shape(drone_power, mixed_power)

    scale = estimate_power_scale(
        mixed_power=mixed_power,
        drone_power=drone_power,
        mode=args.power_scale_mode,
        fixed_scale=args.power_scale,
        mask_percentile=args.drone_mask_percentile,
    )
    residual = mixed_power - args.subtract_strength * scale * drone_power

    if args.noise_floor_ratio > 0.0:
        floor = np.percentile(mixed_power, args.noise_floor_percentile) * args.noise_floor_ratio
    else:
        floor = 0.0
    residual = np.maximum(residual, floor)

    image = power_to_db_image(
        power=residual,
        image_size=args.image_size,
        fixed_db_range=args.fixed_db_range,
        db_min=args.db_min,
        db_max=args.db_max,
        cmap_name=args.cmap,
        grayscale=args.grayscale,
    )

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("STFT mode requires Pillow. Install it with: pip install Pillow") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return scale, drone_band_suppression_db(
        mixed_power=mixed_power,
        residual_power=residual,
        drone_power=drone_power,
        mask_percentile=args.drone_mask_percentile,
    )


def write_float32_iq(file_obj, iq: np.ndarray) -> None:
    interleaved = np.empty(iq.size * 2, dtype="<f4")
    interleaved[0::2] = iq.real.astype(np.float32, copy=False)
    interleaved[1::2] = iq.imag.astype(np.float32, copy=False)
    interleaved.tofile(file_obj)


def subtract_dat_pair(
    drone_path: Path,
    mixed_path: Path,
    output_path: Path,
    dat_format: str,
    normalize_int16: bool,
    chunk_size: int,
    gain_mode: str,
    remove_dc: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0

    with drone_path.open("rb") as drone_file, mixed_path.open("rb") as mixed_file, output_path.open("wb") as output_file:
        while True:
            drone = read_iq_block(drone_file, chunk_size, dat_format, normalize_int16)
            mixed = read_iq_block(mixed_file, chunk_size, dat_format, normalize_int16)
            sample_count = min(drone.size, mixed.size)
            if sample_count == 0:
                break

            drone = drone[:sample_count]
            mixed = mixed[:sample_count]
            if remove_dc:
                drone = drone - np.mean(drone)
                mixed = mixed - np.mean(mixed)

            alpha = estimate_scalar_gain(mixed, drone) if gain_mode == "scalar" else 1.0 + 0.0j
            wifi = mixed - alpha * drone
            write_float32_iq(output_file, wifi)
            samples_written += sample_count

            if drone.size != mixed.size:
                break

    return samples_written


def subtract_image_pair(
    drone_path: Path,
    mixed_path: Path,
    output_path: Path,
    drone_weight: float,
    output_gain: float,
    normalize_output: bool,
    grayscale: bool,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Image mode requires opencv-python. Install it with: pip install opencv-python") from exc

    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    drone = cv2.imread(str(drone_path), read_flag)
    mixed = cv2.imread(str(mixed_path), read_flag)
    if drone is None:
        raise FileNotFoundError(f"Cannot read image: {drone_path}")
    if mixed is None:
        raise FileNotFoundError(f"Cannot read image: {mixed_path}")
    if drone.shape[:2] != mixed.shape[:2]:
        drone = cv2.resize(drone, (mixed.shape[1], mixed.shape[0]), interpolation=cv2.INTER_LINEAR)

    drone_f = drone.astype(np.float32) / 255.0
    mixed_f = mixed.astype(np.float32) / 255.0
    wifi = np.maximum((mixed_f - drone_weight * drone_f) * output_gain, 0.0)

    if normalize_output and float(wifi.max()) > 0.0:
        wifi = wifi / float(wifi.max())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(wifi * 255.0, 0, 255).astype(np.uint8))


def run_dat(args: argparse.Namespace) -> None:
    drone_root = Path(args.drone_root)
    mixed_root = Path(args.mixed_root)
    output_root = Path(args.output_root)
    pairs = pair_by_relative_path(drone_root, mixed_root, DAT_EXTENSIONS)
    if not pairs:
        raise FileNotFoundError("No matched .dat pairs found.")

    total_samples = 0
    for drone_path, mixed_path, rel_path in tqdm(pairs, desc="Subtracting DAT"):
        output_path = output_root / rel_path
        total_samples += subtract_dat_pair(
            drone_path=drone_path,
            mixed_path=mixed_path,
            output_path=output_path,
            dat_format=args.dat_format,
            normalize_int16=args.normalize_int16,
            chunk_size=args.chunk_size,
            gain_mode=args.gain_mode,
            remove_dc=args.remove_dc,
        )

    print(f"[OK] Created {len(pairs)} WiFi .dat files in: {output_root}")
    print(f"[OK] Total IQ samples written: {total_samples}")
    print("[NOTE] Output .dat format is float32 interleaved IQ [I, Q, I, Q, ...].")


def run_image(args: argparse.Namespace) -> None:
    drone_root = Path(args.drone_root)
    mixed_root = Path(args.mixed_root)
    output_root = Path(args.output_root)
    pairs = pair_by_relative_path(drone_root, mixed_root, IMAGE_EXTENSIONS)
    if not pairs:
        raise FileNotFoundError("No matched image pairs found.")

    for drone_path, mixed_path, rel_path in tqdm(pairs, desc="Subtracting images"):
        subtract_image_pair(
            drone_path=drone_path,
            mixed_path=mixed_path,
            output_path=output_root / rel_path,
            drone_weight=args.drone_weight,
            output_gain=args.output_gain,
            normalize_output=args.normalize_output,
            grayscale=args.grayscale,
        )

    print(f"[OK] Created {len(pairs)} WiFi spectrogram images in: {output_root}")


def run_stft(args: argparse.Namespace) -> None:
    drone_root = Path(args.drone_root)
    mixed_root = Path(args.mixed_root)
    output_root = Path(args.output_root)
    pairs = pair_by_relative_path(drone_root, mixed_root, DAT_EXTENSIONS)
    if not pairs:
        raise FileNotFoundError("No matched .dat pairs found.")

    window_samples = max(args.stft_point, int(args.sample_rate * args.duration_time))
    total_images = 0
    scales: list[float] = []
    suppressions: list[float] = []
    for drone_path, mixed_path, rel_path in tqdm(pairs, desc="Subtracting STFT power"):
        with drone_path.open("rb") as drone_file, mixed_path.open("rb") as mixed_file:
            chunk_index = 0
            while True:
                drone = read_iq_block(drone_file, window_samples, args.dat_format, args.normalize_int16)
                mixed = read_iq_block(mixed_file, window_samples, args.dat_format, args.normalize_int16)
                sample_count = min(drone.size, mixed.size)
                if sample_count < args.stft_point:
                    break

                chunk_index += 1
                output_path = output_root / rel_path.parent / f"{rel_path.stem}_wifi_residual_{chunk_index:06d}.png"
                scale, suppression_db = save_stft_residual_image(
                    drone=drone[:sample_count],
                    mixed=mixed[:sample_count],
                    output_path=output_path,
                    args=args,
                )
                scales.append(scale)
                suppressions.append(suppression_db)
                total_images += 1

                if args.max_windows and chunk_index >= args.max_windows:
                    break
                if drone.size != mixed.size:
                    break

    print(f"[OK] Created {total_images} residual WiFi spectrogram images in: {output_root}")
    if scales:
        print(f"[OK] Power scale beta mean/std: {np.mean(scales):.6g} / {np.std(scales):.6g}")
    if suppressions:
        print(
            "[OK] Drone-band residual energy mean/std: "
            f"{np.mean(suppressions):.3f} dB / {np.std(suppressions):.3f} dB "
            "(more negative = more drone removed)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate WiFi signal by subtracting drone data from drone+WiFi data.",
    )
    parser.add_argument("--mode", choices=("dat", "stft", "image"), default="stft")
    parser.add_argument("--drone-root", default="/media/quocnk/Ngocmx_disk/DroneDetect_V2/CLEAN/MP2_FY/MAV_0010_00.dat", help="Drone-only file or folder.")
    parser.add_argument("--mixed-root", default="/media/quocnk/Ngocmx_disk/DroneDetect_V2/WIFI/MP2_FY/MAV_1010_00.dat", help="Drone+WiFi file or folder.")
    parser.add_argument("--output-root", default="/media/quocnk/Ngocmx_disk/DroneDetect_V2/WIFI_ONLY", help="Output file or folder for estimated WiFi.")

    parser.add_argument("--dat-format", choices=("float32_iq", "int16_iq"), default="float32_iq")
    parser.add_argument("--normalize-int16", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2_000_000, help="IQ samples per subtraction block.")
    parser.add_argument("--gain-mode", choices=("none", "scalar"), default="scalar")
    parser.add_argument("--remove-dc", action="store_true")

    parser.add_argument("--sample-rate", type=int, default=40_000_000, help="STFT mode only.")
    parser.add_argument("--stft-point", type=int, default=1024, help="STFT mode only.")
    parser.add_argument("--duration-time", type=float, default=0.05, help="Seconds per output image in STFT mode.")
    parser.add_argument("--window", default="hann", help="STFT window, e.g. hann, hamming, blackman.")
    parser.add_argument("--overlap-ratio", type=float, default=0.50, help="STFT overlap ratio in [0, 1).")
    parser.add_argument("--max-windows", type=int, default=0, help="STFT mode only. 0 = all windows.")
    parser.add_argument("--power-scale-mode", choices=("median", "lsq", "fixed", "none"), default="fixed")
    parser.add_argument("--power-scale", type=float, default=1.0, help="Used when --power-scale-mode fixed.")
    parser.add_argument("--subtract-strength", type=float, default=1.0, help="Multiply the estimated drone power before subtraction.")
    parser.add_argument("--drone-mask-percentile", type=float, default=80.0, help="Use strongest drone-power bins for beta and suppression metrics.")
    parser.add_argument("--noise-floor-ratio", type=float, default=0.05)
    parser.add_argument("--noise-floor-percentile", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fixed-db-range", action="store_true")
    parser.add_argument("--db-min", type=float, default=-90.0)
    parser.add_argument("--db-max", type=float, default=-20.0)
    parser.add_argument("--cmap", default="jet")

    parser.add_argument("--drone-weight", type=float, default=1.0, help="Image mode only.")
    parser.add_argument("--output-gain", type=float, default=1.0, help="Image mode only.")
    parser.add_argument("--normalize-output", action="store_true", help="Image mode only.")
    parser.add_argument("--grayscale", action="store_true", help="Image mode only.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not 0 <= args.overlap_ratio < 1:
        raise ValueError("--overlap-ratio must be in [0, 1)")
    if not 0 <= args.drone_mask_percentile < 100:
        raise ValueError("--drone-mask-percentile must be in [0, 100)")
    if args.subtract_strength < 0:
        raise ValueError("--subtract-strength must be non-negative")

    if args.mode == "dat":
        run_dat(args)
    elif args.mode == "stft":
        run_stft(args)
    else:
        run_image(args)


if __name__ == "__main__":
    main()
