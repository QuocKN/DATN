#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick inspector for the beginning of a .dat file.

It reads the first bytes and prints:
- File size
- Hex dump (first N bytes)
- ASCII preview
- Reinterpretation previews as int16/uint16/int32/float32 and IQ pairs

Use this to determine whether the .dat file is raw IQ, text, or a custom format.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL.ImagePalette import raw
import numpy as np


# ========================
# CONFIG
# ========================
DAT_PATH = "/home/quocnk/Documents/NKQuoc/Data/RF/CDRF/Cage_Indoor/DJI_Mavic4_Mini/DJI_MavicMini4_20_2442_armed.dat"
HEAD_BYTES = 512           # bytes to inspect from beginning of file
HEX_PER_LINE = 16          # bytes per hex row
PREVIEW_COUNT = 16         # number of values for numeric previews


def format_hexdump(data: bytes, width: int = 16) -> str:
    lines: list[str] = []
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        lines.append(f"{offset:08X}  {hex_part:<{width * 3 - 1}}  |{ascii_part}|")
    return "\n".join(lines)


def preview_array(name: str, arr: np.ndarray, count: int = 16) -> None:
    n = min(count, arr.size)
    print(f"{name} ({arr.dtype}, n={arr.size}) first {n}:")
    print(arr[:n])
    print()


def inspect_dat_head(path: str, head_bytes: int = 512, preview_count: int = 16) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    read_len = min(head_bytes, file_size)

    with file_path.open("rb") as f:
        raw = f.read(read_len)

    print(f"File: {file_path}")
    print(f"File size: {file_size:,} bytes")
    print(f"Read: {len(raw)} bytes from start")
    print()

    print("=== HEX DUMP (HEAD) ===")
    print(format_hexdump(raw, width=HEX_PER_LINE))
    print()

    ascii_preview = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw)
    print("=== ASCII PREVIEW ===")
    print(ascii_preview)
    print()

    print("=== NUMERIC INTERPRETATION PREVIEWS ===")
    if len(raw) >= 2:
        i16 = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype="<i2")
        u16 = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype="<u2")
        preview_array("int16 little-endian", i16, preview_count)
        preview_array("uint16 little-endian", u16, preview_count)

        if i16.size >= 2:
            iq_pairs = i16[: (i16.size // 2) * 2].reshape(-1, 2)
            i_vals = iq_pairs[:, 0]
            q_vals = iq_pairs[:, 1]
            print(f"IQ pairs interpreted from int16 (pairs={iq_pairs.shape[0]}) first {min(preview_count, iq_pairs.shape[0])}:")
            print(iq_pairs[:preview_count])
            print()
            complex_iq = i_vals.astype(np.float32) + 1j * q_vals.astype(np.float32)
            preview_array("complex IQ from int16 pairs", complex_iq, preview_count)

    if len(raw) >= 4:
        raw4 = raw[: len(raw) - (len(raw) % 4)]

        i32 = np.frombuffer(raw4, dtype="<i4")
        f32 = np.frombuffer(raw4, dtype="<f4")

        preview_array("int32 little-endian", i32, preview_count)
        preview_array("float32 little-endian", f32, preview_count)

        # Preview float32 sau đoạn zero đầu file
        nz = np.flatnonzero(f32 != 0)
        if len(nz) > 0:
            start = max(0, nz[0] - 4)
            print(f"\nfloat32 first non-zero index: {nz[0]}")
            preview_array(
                f"float32 around first non-zero [{start}:{start + preview_count}]",
                f32[start:start + preview_count],
                preview_count,
            )

            # Nếu là I/Q interleaved float32
            iq = f32[0::2] + 1j * f32[1::2]
            nz_iq = np.flatnonzero(np.abs(iq) > 0)
            if len(nz_iq) > 0:
                iq_start = max(0, nz_iq[0] - 4)
                print(f"\nIQ first non-zero sample index: {nz_iq[0]}")
                preview_array(
                    f"complex IQ around first non-zero [{iq_start}:{iq_start + preview_count}]",
                    iq[iq_start:iq_start + preview_count],
                    preview_count,
                )
        else:
            print("\nfloat32: toàn bộ dữ liệu đang là 0")

    inspect_formats(raw)

def analyze_signal(iq: np.ndarray, name: str):
    print(f"=== ANALYZE: {name} ===")

    real = np.real(iq)
    imag = np.imag(iq)

    print("Mean (I, Q):", np.mean(real), np.mean(imag))
    print("Std  (I, Q):", np.std(real), np.std(imag))

    # Kiểm tra số giá trị unique
    unique_vals = np.unique(real[:5000])
    print("Unique I (first 5k):", len(unique_vals))

    # Kiểm tra step size
    diffs = np.diff(np.sort(unique_vals))
    if len(diffs) > 0:
        print("Min step in I:", np.min(diffs))

    print()


def inspect_formats(raw: bytes):
    print("=== FORMAT GUESSING ===")

    # ---- int16 IQ ----
    i16 = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype="<i2")

    if i16.size >= 2:
        iq = i16.reshape(-1, 2)
        x = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)

        analyze_signal(x, "int16 IQ (raw)")

        # normalize kiểu SC16
        x_norm_16 = x / 32768.0
        analyze_signal(x_norm_16, "int16 normalized /32768")

        # normalize kiểu SC12 (rất hay gặp)
        x_norm_12 = x / 4096.0
        analyze_signal(x_norm_12, "int16 normalized /4096 (SC12 guess)")

    # ---- float32 IQ ----
    if len(raw) >= 8:
        f32 = np.frombuffer(raw[: len(raw) - (len(raw) % 4)], dtype="<f4")
        if f32.size >= 2:
            iq = f32.reshape(-1, 2)
            x = iq[:, 0] + 1j * iq[:, 1]

            analyze_signal(x, "float32 IQ")

    print()
if __name__ == "__main__":
    inspect_dat_head(DAT_PATH, head_bytes=HEAD_BYTES, preview_count=PREVIEW_COUNT)
