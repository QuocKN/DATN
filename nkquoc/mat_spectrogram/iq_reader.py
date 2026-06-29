from __future__ import annotations

from typing import Any, Generator

import numpy as np


def _is_hdf5_mat(mat_path: str) -> bool:
    with open(mat_path, "rb") as handle:
        signature = handle.read(8)
    return signature == b"\x89HDF\r\n\x1a\n"


def _has_real_imag_fields(dtype: np.dtype) -> bool:
    names = dtype.names or ()
    return "real" in names and "imag" in names


def _to_complex_iq(data: Any) -> np.ndarray | None:
    arr = np.asarray(data)
    if arr.size == 0:
        return None

    arr = np.squeeze(arr)

    if _has_real_imag_fields(arr.dtype):
        real = arr["real"].astype(np.float32, copy=False).reshape(-1)
        imag = arr["imag"].astype(np.float32, copy=False).reshape(-1)
        return (real + 1j * imag).astype(np.complex64, copy=False)

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


def _single_sample_axis(shape: tuple[int, ...]) -> int | None:
    non_singleton = [axis for axis, size in enumerate(shape) if size > 1]
    if len(non_singleton) == 1:
        return non_singleton[0]
    return None


def _h5_iq_layout(dataset: Any) -> tuple[str, int] | None:
    shape = tuple(int(size) for size in dataset.shape)
    if not shape or int(np.prod(shape)) == 0:
        return None

    dtype = np.dtype(dataset.dtype)
    if _has_real_imag_fields(dtype) or np.issubdtype(dtype, np.complexfloating):
        return "complex_samples", int(np.prod(shape))

    if not np.issubdtype(dtype, np.number):
        return None

    if len(shape) >= 1 and shape[-1] == 2:
        return "last_dim_pair", int(np.prod(shape[:-1]))

    if len(shape) == 2 and shape[0] == 2:
        return "first_dim_pair", int(shape[1])

    return None


def _score_h5_iq_candidate(name: str, layout: str, total: int) -> int:
    score = min(total // 1000, 50)
    name_l = name.lower()
    if "iq" in name_l:
        score += 100
    if "rf" in name_l:
        score += 40
    if "samp" in name_l:
        score += 40
    if "signal" in name_l:
        score += 20
    if "data" in name_l:
        score += 10
    if layout == "complex_samples":
        score += 50
    return score


def _pick_h5_iq_dataset(handle: Any, mat_iq_key: str | None) -> tuple[str, Any, str, int]:
    import h5py

    if mat_iq_key is not None:
        if mat_iq_key not in handle:
            raise KeyError(f"Key '{mat_iq_key}' not found in MAT file.")
        dataset = handle[mat_iq_key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"Key '{mat_iq_key}' is not a dataset.")
        layout = _h5_iq_layout(dataset)
        if layout is None:
            raise ValueError(f"Key '{mat_iq_key}' is not a supported IQ variable.")
        layout_name, total = layout
        return mat_iq_key, dataset, layout_name, total

    candidates: list[tuple[int, str, Any, str, int]] = []

    def collect(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        layout = _h5_iq_layout(obj)
        if layout is None:
            return
        layout_name, total = layout
        if total < 1024:
            return
        score = _score_h5_iq_candidate(name, layout_name, total)
        candidates.append((score, name, obj, layout_name, total))

    handle.visititems(collect)

    if not candidates:
        raise ValueError("No IQ-like variable found in HDF5 MAT file.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, name, dataset, layout_name, total = candidates[0]
    return name, dataset, layout_name, total


def _read_h5_flat_chunk(dataset: Any, start: int, end: int) -> np.ndarray:
    shape = tuple(int(size) for size in dataset.shape)
    axis = _single_sample_axis(shape)
    if axis is not None:
        index: list[int | slice] = [0] * len(shape)
        index[axis] = slice(start, end)
        return np.asarray(dataset[tuple(index)]).reshape(-1)

    data = np.asarray(dataset[...]).reshape(-1)
    return data[start:end]


def _read_h5_last_dim_pair_chunk(dataset: Any, start: int, end: int) -> np.ndarray:
    sample_shape = tuple(int(size) for size in dataset.shape[:-1])
    axis = _single_sample_axis(sample_shape)
    if axis is not None:
        index: list[int | slice] = [0] * len(sample_shape)
        index[axis] = slice(start, end)
        index.append(slice(None))
        arr = np.asarray(dataset[tuple(index)]).reshape(-1, 2)
    else:
        arr = np.asarray(dataset[...]).reshape(-1, 2)[start:end]

    return arr[:, 0].astype(np.float32, copy=False) + 1j * arr[:, 1].astype(np.float32, copy=False)


def _read_h5_first_dim_pair_chunk(dataset: Any, start: int, end: int) -> np.ndarray:
    if len(dataset.shape) == 2:
        arr = np.asarray(dataset[:, start:end])
    else:
        arr = np.asarray(dataset[...]).reshape(2, -1)[:, start:end]
    return arr[0].astype(np.float32, copy=False).reshape(-1) + 1j * arr[1].astype(np.float32, copy=False).reshape(-1)


def _read_h5_iq_chunk(dataset: Any, layout: str, start: int, end: int) -> np.ndarray:
    if layout == "complex_samples":
        iq = _to_complex_iq(_read_h5_flat_chunk(dataset, start, end))
        if iq is None:
            raise ValueError("Selected HDF5 dataset cannot be converted to complex IQ.")
        return iq
    if layout == "last_dim_pair":
        return _read_h5_last_dim_pair_chunk(dataset, start, end)
    if layout == "first_dim_pair":
        return _read_h5_first_dim_pair_chunk(dataset, start, end)
    raise ValueError(f"Unsupported HDF5 IQ layout: {layout}")


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
            try:
                base, key_i, key_q = _find_iq_pair_names(keys, mat_iq_key)
            except ValueError:
                dataset_name, dataset, layout, total = _pick_h5_iq_dataset(handle, mat_iq_key)
                if max_iq_samples is not None:
                    total = min(total, max_iq_samples)

                print(f"Using MAT variable: {dataset_name}")
                print(f"Total IQ samples: {total}")

                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    yield _read_h5_iq_chunk(dataset, layout, start, end)
                return

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
                i_part = _read_h5_flat_chunk(ds_i, start, end).astype(np.float32, copy=False)
                q_part = _read_h5_flat_chunk(ds_q, start, end).astype(np.float32, copy=False)
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
