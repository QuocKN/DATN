from __future__ import annotations

from typing import Generator

import numpy as np


def iter_iq_chunks_from_dat(
    dat_path: str,
    chunk_size: int,
    dat_format: str = "float32_iq",
    normalize_int16: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield complex IQ chunks from a DAT file."""
    if dat_format not in {"float32_iq", "int16_iq"}:
        raise ValueError("dat_format must be 'float32_iq' or 'int16_iq'")

    if dat_format == "float32_iq":
        dtype = np.dtype("<f4")
        bytes_per_scalar = 4
    else:
        dtype = np.dtype("<i2")
        bytes_per_scalar = 2

    total_read = 0
    with open(dat_path, "rb") as f:
        while True:
            if max_iq_samples is not None and total_read >= max_iq_samples:
                break

            samples_to_read = chunk_size
            if max_iq_samples is not None:
                samples_to_read = min(samples_to_read, max_iq_samples - total_read)
                if samples_to_read <= 0:
                    break

            bytes_to_read = 2 * samples_to_read * bytes_per_scalar
            raw_bytes = f.read(bytes_to_read)
            if not raw_bytes:
                break

            data = np.frombuffer(raw_bytes, dtype=dtype)
            if data.size % 2 != 0:
                data = data[:-1]
            if data.size == 0:
                break

            iq_pairs = data.reshape(-1, 2)
            iq_count = iq_pairs.shape[0]

            if dat_format == "float32_iq":
                i_values = iq_pairs[:, 0].astype(np.float32, copy=False)
                q_values = iq_pairs[:, 1].astype(np.float32, copy=False)
            else:
                i_values = iq_pairs[:, 0].astype(np.float32)
                q_values = iq_pairs[:, 1].astype(np.float32)
                if normalize_int16:
                    i_values /= 32768.0
                    q_values /= 32768.0

            total_read += iq_count
            yield i_values + 1j * q_values
