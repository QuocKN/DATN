from __future__ import annotations

from typing import Any, Generator

import numpy as np


def _is_hdf5_mat(mat_path: str) -> bool:
    with open(mat_path, "rb") as handle:
        signature = handle.read(8)
    return signature == b"\x89HDF\r\n\x1a\n"


def _to_complex_iq(data: Any) -> np.ndarray | None:
    arr = np.asarray(data)
    if arr.size == 0:
        return None

    arr = np.squeeze(arr)

    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False).reshape(-1)

    if not np.issubdtype(arr.dtype, np.number):
        return None

    if arr.ndim >= 1 and arr.shape[-1] == 2:
        reshaped = arr.reshape(-1, 2).astype(np.float32, copy=False)
        return reshaped[:, 0] + 1j * reshaped[:, 1]

    if arr.ndim == 2 and arr.shape[0] == 2:
        reshaped = arr.astype(np.float32, copy=False)
        return reshaped[0, :] + 1j * reshaped[1, :]

    return None


def _find_iq_pair_names(keys: list[str], preferred_base: str | None = None) -> tuple[str, str, str]:
    if preferred_base:
        key_i = f"{preferred_base}_I"
        key_q = f"{preferred_base}_Q"
        if key_i in keys and key_q in keys:
            return preferred_base, key_i, key_q

    for key_i in keys:
        if key_i.startswith("__"):
            continue
        if not key_i.upper().endswith("_I"):
            continue
        key_q = key_i[:-2] + "_Q"
        if key_q in keys:
            return key_i[:-2], key_i, key_q

    raise ValueError("Cannot find paired *_I and *_Q variables in MAT file.")


def _pick_iq_key_from_dict(data_dict: dict[str, Any]) -> tuple[str, np.ndarray]:
    candidates: list[tuple[int, str, np.ndarray]] = []

    try:
        base, key_i, key_q = _find_iq_pair_names(list(data_dict.keys()), None)
        i_arr = np.squeeze(np.asarray(data_dict[key_i])).astype(np.float32, copy=False).reshape(-1)
        q_arr = np.squeeze(np.asarray(data_dict[key_q])).astype(np.float32, copy=False).reshape(-1)
        if i_arr.shape == q_arr.shape and i_arr.size >= 1024:
            return base, i_arr + 1j * q_arr
    except Exception:
        pass

    for key, value in data_dict.items():
        if key.startswith("__"):
            continue
        iq = _to_complex_iq(value)
        if iq is None or iq.size < 1024:
            continue

        score = 0
        key_l = key.lower()
        if "iq" in key_l:
            score += 100
        if "rf" in key_l:
            score += 40
        if "signal" in key_l:
            score += 20
        if "data" in key_l:
            score += 10
        if np.iscomplexobj(iq):
            score += 30
        score += min(iq.size // 1000, 50)
        candidates.append((score, key, iq))

    if not candidates:
        raise ValueError("No IQ-like variable found in MAT file.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_key, best_iq = candidates[0]
    return best_key, best_iq


def iter_iq_chunks_from_mat(
    mat_path: str,
    chunk_size: int,
    mat_iq_key: str | None = None,
    max_iq_samples: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield complex IQ chunks from a MAT file."""
    def _iter_from_h5() -> Generator[np.ndarray, None, None]:
        import h5py

        with h5py.File(mat_path, "r") as handle:
            keys = list(handle.keys())
            base, key_i, key_q = _find_iq_pair_names(keys, mat_iq_key)
            ds_i = handle[key_i]
            ds_q = handle[key_q]

            total = int(np.prod(ds_i.shape))
            if int(np.prod(ds_q.shape)) != total:
                raise ValueError(f"'{key_i}' and '{key_q}' have mismatched lengths.")

            if max_iq_samples is not None:
                total = min(total, max_iq_samples)

            print(f"Using MAT variable pair: {base} ({key_i}, {key_q})")
            print(f"Total IQ samples: {total}")

            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                i_part = np.asarray(ds_i[start:end]).reshape(-1).astype(np.float32, copy=False)
                q_part = np.asarray(ds_q[start:end]).reshape(-1).astype(np.float32, copy=False)
                yield i_part + 1j * q_part

    if _is_hdf5_mat(mat_path):
        yield from _iter_from_h5()
        return

    import scipy.io as sio

    try:
        data_dict = sio.loadmat(mat_path)
    except NotImplementedError as exc:
        # Some MATLAB v7.3 files may bypass header sniffing; force HDF5 path.
        if "Please use HDF reader for matlab v7.3 files" in str(exc):
            yield from _iter_from_h5()
            return
        raise
    if mat_iq_key is not None:
        key_i = f"{mat_iq_key}_I"
        key_q = f"{mat_iq_key}_Q"
        if key_i in data_dict and key_q in data_dict:
            i_arr = np.squeeze(np.asarray(data_dict[key_i])).astype(np.float32, copy=False).reshape(-1)
            q_arr = np.squeeze(np.asarray(data_dict[key_q])).astype(np.float32, copy=False).reshape(-1)
            if i_arr.shape != q_arr.shape:
                raise ValueError(f"'{key_i}' and '{key_q}' have mismatched lengths.")
            key_name = mat_iq_key
            iq_all = i_arr + 1j * q_arr
        else:
            if mat_iq_key not in data_dict:
                raise KeyError(f"Key '{mat_iq_key}' not found in MAT file.")
            iq_all = _to_complex_iq(data_dict[mat_iq_key])
            if iq_all is None:
                raise ValueError(f"Key '{mat_iq_key}' is not a supported IQ variable.")
            key_name = mat_iq_key
    else:
        key_name, iq_all = _pick_iq_key_from_dict(data_dict)

    if max_iq_samples is not None:
        iq_all = iq_all[:max_iq_samples]

    print(f"Using MAT variable: {key_name}")
    print(f"Total IQ samples: {iq_all.size}")

    for start in range(0, iq_all.size, chunk_size):
        yield iq_all[start : start + chunk_size]
