#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print the structure of an HDF5 file without loading large datasets into memory.

Usage:
    python nkquoc/inspect_h5_structure.py /path/to/file.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PiB"


def safe_repr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def print_attrs(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
    if not obj.attrs:
        return
    print(f"{indent}attrs:")
    for key, value in obj.attrs.items():
        print(f"{indent}  {key}: {safe_repr(value)}")


def dataset_sample(dataset: h5py.Dataset, sample_count: int) -> str:
    if sample_count <= 0:
        return "<disabled>"
    try:
        if dataset.ndim == 0:
            sample = dataset[()]
        elif dataset.shape[0] == 0:
            sample = []
        else:
            sample = dataset[: min(dataset.shape[0], sample_count)]
        array = np.asarray(sample)
        if array.size > sample_count:
            array = array.ravel()[:sample_count]
        return repr(array)
    except Exception as exc:
        return f"<failed: {exc}>"


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def attrs_to_dict(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {key: json_safe(value) for key, value in obj.attrs.items()}


def dataset_sample_value(dataset: h5py.Dataset, sample_count: int) -> Any:
    if sample_count <= 0:
        return []
    if dataset.ndim == 0:
        return json_safe(dataset[()])
    if dataset.shape[0] == 0:
        return []

    sample = dataset[: min(dataset.shape[0], sample_count)]
    array = np.asarray(sample)
    if array.size > sample_count:
        array = array.ravel()[:sample_count]
    return json_safe(array)


def build_h5_structure(path: Path, sample_count: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as handle:
        result: dict[str, Any] = {
            "file": str(path),
            "file_size_bytes": path.stat().st_size,
            "file_size_human": format_bytes(path.stat().st_size),
            "root_attrs": attrs_to_dict(handle),
            "groups": [],
            "datasets": [],
            "summary": {
                "groups": 0,
                "datasets": 0,
                "logical_dataset_size_bytes": 0,
                "logical_dataset_size_human": "0.00 B",
            },
        }

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                result["groups"].append(
                    {
                        "path": f"/{name}",
                        "attrs": attrs_to_dict(obj),
                    }
                )
                return

            size_bytes = obj.size * obj.dtype.itemsize if obj.dtype.itemsize > 0 else 0
            result["datasets"].append(
                {
                    "path": f"/{name}",
                }
            )

        handle.visititems(visit)

        result["summary"] = {
            "groups": len(result["groups"]),
            "datasets": len(result["datasets"]),
        }
        return result


def inspect_h5(path: Path, sample_count: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as handle:
        print(f"FILE: {path}")
        print(f"FILE_SIZE: {format_bytes(path.stat().st_size)}")
        print()

        print("ROOT")
        print_attrs(handle, "  ")
        print()

        group_count = 0
        dataset_count = 0
        logical_size = 0

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            nonlocal group_count, dataset_count, logical_size

            if isinstance(obj, h5py.Group):
                group_count += 1
                print(f"GROUP /{name}")
                print_attrs(obj, "  ")
                return

            dataset_count += 1
            size_bytes = obj.size * obj.dtype.itemsize if obj.dtype.itemsize > 0 else 0
            logical_size += size_bytes

            print(f"DATASET /{name}")

        handle.visititems(visit)

        print()
        print("SUMMARY")
        print(f"  groups: {group_count}")
        print(f"  datasets: {dataset_count}")
        print(f"  logical_dataset_size: {format_bytes(logical_size)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HDF5 file structure safely.")
    parser.add_argument("h5_path", type=Path, help="Path to the .h5/.hdf5 file")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="Number of values to print from the beginning of each dataset",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write the HDF5 structure to this JSON file instead of printing text",
    )
    args = parser.parse_args()

    if args.json_out:
        structure = build_h5_structure(args.h5_path, args.sample_count)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as output:
            json.dump(structure, output, ensure_ascii=False, indent=2)
        print(f"Wrote JSON structure: {args.json_out}")
        return

    inspect_h5(args.h5_path, args.sample_count)


if __name__ == "__main__":
    main()
