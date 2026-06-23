from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
DEFAULT_SOURCE_ROOT = Path(r"F:\Data_22_6_spectrogram_0_05")
DEFAULT_OUTPUT_ROOT = Path(r"F:\Data_22_6_spectrogram_0_05\balanced")
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
DEFAULT_DRONE_TO_NON_RATIO = 1.017
DRONE_CONDITIONS = ("BLUE", "BOTH", "CLEAN", "WIFI")


@dataclass(frozen=True)
class NonDroneGroup:
    group_id: str
    source_folder: str
    condition: str
    paths: List[Path]


@dataclass(frozen=True)
class DroneCollection:
    available: Dict[str, Dict[str, List[Path]]]
    layout: str
    relative_base_by_split: Mapping[str, Path]
    target_mode: str


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def non_drone_condition(source_folder: str) -> str:
    name = source_folder.lower()
    if "wifi" in name and "blue" in name:
        return "bluetooth_wifi_env"
    if "blue" in name:
        return "bluetooth_env"
    return "env"


def kaggle_group_id(path: Path) -> str:
    stem = path.stem
    if "_spectrogram_" in stem:
        return stem.split("_spectrogram_", 1)[0]
    return stem


def collect_non_drone_groups(non_drone_root: Path) -> List[NonDroneGroup]:
    groups: List[NonDroneGroup] = []
    for source_dir in sorted(path for path in non_drone_root.iterdir() if path.is_dir()):
        paths = collect_images(source_dir)
        condition = non_drone_condition(source_dir.name)
        if source_dir.name == "non_drone_kaggle_fixed":
            by_source: Dict[str, List[Path]] = defaultdict(list)
            for path in paths:
                by_source[kaggle_group_id(path)].append(path)
            for group_id, group_paths in sorted(by_source.items()):
                groups.append(
                    NonDroneGroup(
                        group_id=f"{source_dir.name}/{group_id}",
                        source_folder=source_dir.name,
                        condition=condition,
                        paths=sorted(group_paths),
                    )
                )
            continue

        groups.append(
            NonDroneGroup(
                group_id=source_dir.name,
                source_folder=source_dir.name,
                condition=condition,
                paths=paths,
            )
        )
    return groups


def allocate_equal(total: int, names: Sequence[str]) -> Dict[str, int]:
    base = total // len(names)
    remainder = total - base * len(names)
    return {name: base + (1 if index < remainder else 0) for index, name in enumerate(names)}


def allocate_by_ratio(total: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def ordered_drone_groups(names: Iterable[str]) -> List[str]:
    name_set = set(names)
    ordered = [name for name in DRONE_CONDITIONS if name in name_set]
    ordered.extend(sorted(name for name in name_set if name not in DRONE_CONDITIONS))
    return ordered


def split_paths_by_ratio(
    paths: Sequence[Path],
    ratios: Mapping[str, float],
    rng: random.Random,
) -> Dict[str, List[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)

    counts = allocate_by_ratio(len(shuffled), ratios)
    split_paths: Dict[str, List[Path]] = {}
    start = 0
    for split in SPLITS:
        end = start + counts[split]
        split_paths[split] = sorted(shuffled[start:end])
        start = end
    return split_paths


def collect_drone_images(drone_root: Path, rng: random.Random) -> DroneCollection:
    split_dirs = [drone_root / split for split in SPLITS if (drone_root / split).is_dir()]
    if split_dirs:
        group_names = {
            path.name
            for split_dir in split_dirs
            for path in split_dir.iterdir()
            if path.is_dir()
        }
        if group_names:
            available = {
                group_name: {
                    split: collect_images(drone_root / split / group_name)
                    for split in SPLITS
                }
                for group_name in ordered_drone_groups(group_names)
            }
            return DroneCollection(
                available=available,
                layout="split_group",
                relative_base_by_split={split: drone_root / split for split in SPLITS},
                target_mode="equal",
            )

        available = {
            "drone": {
                split: collect_images(drone_root / split)
                for split in SPLITS
            }
        }
        return DroneCollection(
            available=available,
            layout="split_flat",
            relative_base_by_split={split: drone_root / split for split in SPLITS},
            target_mode="proportional",
        )

    group_paths: Dict[str, List[Path]] = {}
    direct_paths = sorted(
        path
        for path in drone_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if direct_paths:
        group_paths["drone"] = direct_paths

    for source_dir in sorted(path for path in drone_root.iterdir() if path.is_dir()):
        paths = collect_images(source_dir)
        if paths:
            group_paths[source_dir.name] = paths

    if not group_paths:
        raise RuntimeError(f"No drone images found under {drone_root}")

    available = {
        group_name: split_paths_by_ratio(paths, DEFAULT_SPLIT_RATIOS, rng)
        for group_name, paths in sorted(group_paths.items())
    }
    return DroneCollection(
        available=available,
        layout="folder_group_generated_split",
        relative_base_by_split={split: drone_root for split in SPLITS},
        target_mode="proportional",
    )


def allocate_target_by_group(group_totals: Mapping[str, int], target: int) -> Dict[str, int]:
    total = sum(group_totals.values())
    if target > total:
        raise ValueError(f"Target {target} is larger than available total {total}")
    raw = {name: group_totals[name] * target / total for name in group_totals}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = target - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order:
        if remainder <= 0:
            break
        if counts[name] < group_totals[name]:
            counts[name] += 1
            remainder -= 1
    if remainder != 0:
        raise RuntimeError("Could not allocate exact group counts")
    return counts


def group_counts(records: Iterable[Mapping[str, str]], *keys: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        compound_key = "/".join(str(record[key]) for key in keys)
        counts[compound_key] += 1
    return dict(sorted(counts.items()))


def assign_non_drone_groups(
    groups: Sequence[NonDroneGroup],
    split_targets: Mapping[str, int],
    rng: random.Random,
) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    assigned_counts = {split: 0 for split in SPLITS}
    targets = {split: max(1, split_targets[split]) for split in SPLITS}

    def choose_split(size: int) -> str:
        return min(
            SPLITS,
            key=lambda split: (
                (assigned_counts[split] + size) / targets[split],
                assigned_counts[split],
                SPLITS.index(split),
            ),
        )

    # Keep interference/background conditions in train when possible so the model learns them as negatives.
    interference_groups = sorted(
        (g for g in groups if g.condition != "env"),
        key=lambda item: (-len(item.paths), item.group_id),
    )
    for group in interference_groups:
        size = len(group.paths)
        split = "train" if assigned_counts["train"] + size <= targets["train"] else choose_split(size)
        assignments[group.group_id] = split
        assigned_counts[split] += size

    env_groups = sorted(
        (g for g in groups if g.condition == "env"),
        key=lambda item: (-len(item.paths), item.group_id),
    )
    rng.shuffle(env_groups)
    env_groups.sort(key=lambda item: -len(item.paths))

    for group in env_groups:
        size = len(group.paths)
        split = choose_split(size)
        assignments[group.group_id] = split
        assigned_counts[split] += size

    if set(assignments) != {group.group_id for group in groups}:
        raise RuntimeError("Some non-drone groups were not assigned")

    return assignments


def copy_record(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_dataset(
    source_root: Path,
    output_root: Path,
    seed: int,
    clean: bool,
    drone_to_non_ratio: float = DEFAULT_DRONE_TO_NON_RATIO,
) -> dict:
    rng = random.Random(seed)
    drone_root = source_root / "Drone"
    non_drone_root = source_root / "Non_drone"

    if not drone_root.exists() or not non_drone_root.exists():
        raise FileNotFoundError(f"Expected Drone and Non_drone under {source_root}")
    if clean and output_root.exists():
        if output_root.resolve() == source_root.resolve() or output_root.resolve() in source_root.resolve().parents:
            raise RuntimeError(f"Refusing to clean unsafe output path: {output_root}")
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output folder already exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    drone_collection = collect_drone_images(drone_root, rng)
    drone_available = drone_collection.available
    drone_groups = tuple(drone_available)

    non_groups = collect_non_drone_groups(non_drone_root)
    non_by_condition: Dict[str, List[Path]] = defaultdict(list)
    non_by_source_folder: Dict[str, List[Path]] = defaultdict(list)
    for group in non_groups:
        non_by_condition[group.condition].extend(group.paths)
        non_by_source_folder[group.source_folder].extend(group.paths)

    total_non = sum(len(paths) for paths in non_by_condition.values())
    if total_non == 0:
        raise RuntimeError(f"No non-drone images found under {non_drone_root}")
    if drone_to_non_ratio <= 0:
        raise ValueError(f"drone_to_non_ratio must be greater than 0, got {drone_to_non_ratio}")

    target_drone_total = max(1, int(round(total_non * drone_to_non_ratio)))
    drone_group_totals = {
        group_name: sum(len(paths) for paths in split_paths.values())
        for group_name, split_paths in drone_available.items()
    }
    if drone_collection.target_mode == "equal":
        drone_condition_targets = allocate_equal(target_drone_total, drone_groups)
    else:
        drone_condition_targets = allocate_target_by_group(drone_group_totals, target_drone_total)
    drone_targets: Dict[str, Dict[str, int]] = {}
    for condition, total in drone_condition_targets.items():
        drone_targets[condition] = allocate_by_ratio(total, DEFAULT_SPLIT_RATIOS)

    non_split_targets = allocate_by_ratio(total_non, DEFAULT_SPLIT_RATIOS)

    for condition, split_counts in drone_targets.items():
        for split, target in split_counts.items():
            available = len(drone_available[condition][split])
            if target > available:
                raise RuntimeError(
                    f"Not enough drone images for {condition}/{split}: "
                    f"need {target}, available {available}"
                )

    non_group_assignments = assign_non_drone_groups(non_groups, non_split_targets, rng)

    records = []

    for condition in drone_groups:
        for split in SPLITS:
            chosen = sorted(rng.sample(drone_available[condition][split], drone_targets[condition][split]))
            for src in chosen:
                rel = src.relative_to(drone_collection.relative_base_by_split[split])
                dst = output_root / "drone" / split / rel
                copy_record(src, dst)
                records.append(
                    {
                        "split": split,
                        "label": "drone",
                        "binary_label": 1,
                        "condition": condition,
                        "source_path": str(src),
                        "output_path": str(dst),
                        "output_relative_path": str(dst.relative_to(output_root)),
                    }
                )

    for group in sorted(non_groups, key=lambda item: item.group_id):
        split = non_group_assignments[group.group_id]
        for src in group.paths:
            rel = src.relative_to(non_drone_root)
            dst = output_root / "non_drone" / split / group.condition / rel
            copy_record(src, dst)
            records.append(
                {
                    "split": split,
                    "label": "non_drone",
                    "binary_label": 0,
                    "condition": group.condition,
                    "source_folder": group.source_folder,
                    "source_group": group.group_id,
                    "source_path": str(src),
                    "output_path": str(dst),
                    "output_relative_path": str(dst.relative_to(output_root)),
                }
            )

    records = sorted(records, key=lambda item: (item["split"], item["label"], item["condition"], item["output_relative_path"]))

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "split_ratios_requested": DEFAULT_SPLIT_RATIOS,
        "drone_to_non_ratio_requested": drone_to_non_ratio,
        "drone_layout": drone_collection.layout,
        "drone_target_mode": drone_collection.target_mode,
        "target_counts_requested": {
            "drone": target_drone_total,
            "non_drone": total_non,
        },
        "policy": {
            "all_non_drone_images_used_once": True,
            "drone_total_matches_non_drone_total": target_drone_total == total_non,
            "drone_conditions_balanced": drone_collection.target_mode == "equal",
            "drone_folder_groups_split_generated": drone_collection.layout == "folder_group_generated_split",
            "non_drone_conditions_kept_with_all_available_images": True,
            "non_drone_group_split": "folder-level approximate split, except non_drone_kaggle_fixed grouped by filename prefix before _spectrogram_",
            "non_drone_interference_groups_prefer_train": True,
            "copy_mode": "copy2",
        },
        "original_counts": {
            "drone_by_split_condition": {
                f"{split}/{condition}": len(drone_available[condition][split])
                for split in SPLITS
                for condition in drone_groups
            },
            "drone_by_group": drone_group_totals,
            "non_drone_by_source_folder": {
                name: len(paths) for name, paths in sorted(non_by_source_folder.items())
            },
            "non_drone_by_condition": {
                name: len(paths) for name, paths in sorted(non_by_condition.items())
            },
            "non_drone_groups": {
                group.group_id: {
                    "condition": group.condition,
                    "source_folder": group.source_folder,
                    "count": len(group.paths),
                }
                for group in sorted(non_groups, key=lambda item: item.group_id)
            },
        },
        "created_counts": {
            "by_split_label": group_counts(records, "split", "label"),
            "drone_by_condition_split": {
                f"{condition}/{split}": drone_targets[condition][split]
                for condition in drone_groups
                for split in SPLITS
            },
            "non_drone_by_condition_split": {
                f"{condition}/{split}": sum(
                    len(group.paths)
                    for group in non_groups
                    if group.condition == condition and non_group_assignments[group.group_id] == split
                )
                for condition in sorted(non_by_condition)
                for split in SPLITS
            },
            "non_drone_group_assignments": dict(sorted(non_group_assignments.items())),
            "by_label": group_counts(records, "label"),
            "by_condition": group_counts(records, "label", "condition"),
            "by_source_group": group_counts(
                (record for record in records if record["label"] == "non_drone"),
                "split",
                "source_group",
            ),
            "total": len(records),
        },
        "records": records,
    }

    manifest_path = output_root / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=True)

    summary_path = output_root / "split_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(f"source_root: {source_root}\n")
        file.write(f"output_root: {output_root}\n")
        file.write(f"seed: {seed}\n")
        file.write("\npolicy:\n")
        for key, value in manifest["policy"].items():
            file.write(f"- {key}: {value}\n")
        file.write("\ncreated_counts.by_split_label:\n")
        for key, value in manifest["created_counts"]["by_split_label"].items():
            file.write(f"- {key}: {value}\n")
        file.write("\ncreated_counts.by_label:\n")
        for key, value in manifest["created_counts"]["by_label"].items():
            file.write(f"- {key}: {value}\n")
        file.write("\ncreated_counts.by_condition:\n")
        for key, value in manifest["created_counts"]["by_condition"].items():
            file.write(f"- {key}: {value}\n")
        file.write(f"\nmanifest: {manifest_path}\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced drone/non_drone spectrogram dataset.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Delete output root before creating the dataset")
    parser.add_argument(
        "--drone-to-non-ratio",
        type=float,
        default=DEFAULT_DRONE_TO_NON_RATIO,
        help="Target drone:non_drone ratio. For example, 2.0 creates twice as many drone images as non_drone.",
    )
    args = parser.parse_args()

    manifest = build_dataset(
        args.source_root,
        args.output_root,
        args.seed,
        args.clean,
        args.drone_to_non_ratio,
    )
    print(json.dumps(manifest["created_counts"], indent=2, ensure_ascii=True))
    print(f"Saved manifest: {args.output_root / 'split_manifest.json'}")
    print(f"Saved summary: {args.output_root / 'split_summary.txt'}")


if __name__ == "__main__":
    main()
