from __future__ import annotations

from typing import Generator

import numpy as np


def iter_iq_chunks_from_bin(
    bin_path: str,
    chunk_size: int,
    normalize: bool = False,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield complex IQ chunks from an int16 interleaved binary file.

    Binary layout: I_0, Q_0, I_1, Q_1, ...
    """
    total_read = 0
    with open(bin_path, "rb") as f:
        while True:
            if max_iq_samples is not None and total_read >= max_iq_samples:
                break

            raw_bytes = f.read(2 * chunk_size * 2)
            if not raw_bytes:
                break

            int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
            if int16_data.size % 2 != 0:
                int16_data = int16_data[:-1]
            if int16_data.size == 0:
                break

            iq_pairs = int16_data.reshape(-1, 2)
            i_values = iq_pairs[:, 0].astype(np.float32)
            q_values = iq_pairs[:, 1].astype(np.float32)

            if normalize:
                i_values /= 32768.0
                q_values /= 32768.0

            total_read += iq_pairs.shape[0]
            yield i_values + 1j * q_values
