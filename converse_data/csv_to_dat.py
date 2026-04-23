#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import struct
from array import array
from collections import deque


def iter_csv_floats(csv_path: str, read_chunk_size: int = 1 << 20):
    """
    Đọc CSV dạng 1 dòng rất dài theo kiểu streaming, tránh load toàn bộ vào RAM.
    Trả ra từng float theo đúng thứ tự trong file.
    """
    carry = ""
    with open(csv_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(read_chunk_size)
            if not chunk:
                break
            text = carry + chunk
            parts = text.split(",")
            carry = parts.pop()  # phần có thể chưa hoàn chỉnh ở cuối chunk

            for token in parts:
                token = token.strip()
                if token:
                    yield float(token)

        # xử lý phần còn lại
        carry = carry.strip()
        if carry:
            yield float(carry)


def convert_csv_to_dat(csv_path: str, dat_path: str, write_buffer_size: int = 1_000_000):
    """
    Convert CSV số thực -> .dat float32 little-endian.
    Giữ nguyên thứ tự dữ liệu.
    """
    count = 0
    first12 = []
    last12 = deque(maxlen=12)

    with open(dat_path, "wb") as fout:
        buf = array("f")  # float32 buffer

        for v in iter_csv_floats(csv_path):
            # lưu mẫu kiểm tra
            if len(first12) < 12:
                first12.append(v)
            last12.append(v)

            buf.append(v)
            count += 1

            if len(buf) >= write_buffer_size:
                buf.tofile(fout)
                buf = array("f")

        if buf:
            buf.tofile(fout)

    # kiểm tra hợp lệ cho IQ interleaved
    if count % 2 != 0:
        raise ValueError(
            f"So luong gia tri la {count} (le), khong hop le cho IQ interleaved (can so chan)."
        )

    size_bytes = os.path.getsize(dat_path)
    expected_bytes = count * 4  # float32 = 4 bytes

    print(f"CSV: {csv_path}")
    print(f"DAT: {dat_path}")
    print(f"VALUES_WRITTEN: {count}")
    print(f"EVEN_COUNT: {count % 2 == 0}")
    print(f"SIZE_BYTES: {size_bytes}")
    print(f"EXPECTED_BYTES: {expected_bytes}")
    print("FIRST12_CSV:", ",".join(f"{x:.6f}" for x in first12))
    print("LAST12_CSV :", ",".join(f"{x:.6f}" for x in last12))

    # đọc ngược từ DAT để xác nhận đầu/cuối
    with open(dat_path, "rb") as fin:
        first_raw = fin.read(12 * 4)
        first_dat = struct.unpack("<12f", first_raw)

        fin.seek(-12 * 4, os.SEEK_END)
        last_raw = fin.read(12 * 4)
        last_dat = struct.unpack("<12f", last_raw)

    print("FIRST12_DAT:", ",".join(f"{x:.6f}" for x in first_dat))
    print("LAST12_DAT :", ",".join(f"{x:.6f}" for x in last_dat))

    if size_bytes != expected_bytes:
        raise RuntimeError("Kich thuoc DAT khong khop so byte du kien.")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV RF IQ -> DAT float32")
    parser.add_argument("-i", "--input", required=True, help="Duong dan file .csv")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Duong dan file .dat (mac dinh: cung ten voi input, duoi .dat)",
    )
    args = parser.parse_args()

    csv_path = args.input
    dat_path = args.output if args.output else os.path.splitext(csv_path)[0] + ".dat"

    convert_csv_to_dat(csv_path, dat_path)


if __name__ == "__main__":
    main()