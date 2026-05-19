from __future__ import annotations

from pathlib import Path

import numpy as np


def read_interleaved_iq(
    path: str | Path,
    dtype: str,
    max_iq_samples: int | None = None,
    normalize: bool = False,
) -> np.ndarray:
    count = -1 if max_iq_samples is None else max_iq_samples * 2

    if dtype == "float32_iq":
        data = np.fromfile(path, dtype=np.float32, count=count)
        scale = 1.0
    elif dtype == "int16_iq":
        data = np.fromfile(path, dtype=np.int16, count=count).astype(np.float32)
        scale = 32768.0 if normalize else 1.0
    else:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    if data.size % 2 != 0:
        data = data[:-1]

    return (data[0::2] / scale) + 1j * (data[1::2] / scale)


def read_float32_iq(path: str | Path, max_iq_samples: int | None = None) -> np.ndarray:
    return read_interleaved_iq(path, dtype="float32_iq", max_iq_samples=max_iq_samples)


def read_int16_iq(path: str | Path, max_iq_samples: int | None = None, normalize: bool = False) -> np.ndarray:
    return read_interleaved_iq(
        path,
        dtype="int16_iq",
        max_iq_samples=max_iq_samples,
        normalize=normalize,
    )
