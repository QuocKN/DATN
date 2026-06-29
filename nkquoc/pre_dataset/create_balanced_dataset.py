from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_clean_dir(path: Path, clean: bool) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def sample_paths(paths: List[Path], k: int, rng: random.Random) -> List[Path]:
    if k >= len(paths):
        return list(paths)
    return sorted(rng.sample(paths, k))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced binary dataset from existing drone/non_drone roots")
    parser.add_argument("--drone-root", type=str, required=True)
    parser.add_argument("--non-drone-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True, help="Output root. Will create: <out-root>/drone and <out-root>/non_drone")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Delete output root before creating dataset")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    out_root = Path(args.out_root)
    drone_out = out_root / "drone"
    non_drone_out = out_root / "non_drone"

    if out_root.exists() and args.clean:
        shutil.rmtree(out_root)
    drone_out.mkdir(parents=True, exist_ok=True)
    non_drone_out.mkdir(parents=True, exist_ok=True)

    drone_root = Path(args.drone_root)
    non_drone_root = Path(args.non_drone_root)

    summary: Dict[str, Dict[str, int]] = {"original_counts": {}, "balanced_counts": {}}

    for split in SPLITS:
        drone_split = collect_images(drone_root / split)
        non_drone_split = collect_images(non_drone_root / split)

        if not drone_split or not non_drone_split:
            raise RuntimeError(
                f"Missing data in split '{split}': drone={len(drone_split)}, non_drone={len(non_drone_split)}"
            )

        target = min(len(drone_split), len(non_drone_split))
        chosen_drone = sample_paths(drone_split, target, rng)
        chosen_non_drone = sample_paths(non_drone_split, target, rng)

        drone_dst_split = drone_out / split
        non_drone_dst_split = non_drone_out / split
        ensure_clean_dir(drone_dst_split, clean=True)
        ensure_clean_dir(non_drone_dst_split, clean=True)

        for i, src in enumerate(chosen_drone):
            dst = drone_dst_split / f"drone_{i:06d}{src.suffix.lower()}"
            transfer(src, dst, args.mode)

        for i, src in enumerate(chosen_non_drone):
            dst = non_drone_dst_split / f"non_drone_{i:06d}{src.suffix.lower()}"
            transfer(src, dst, args.mode)

        summary["original_counts"][split] = {"drone": len(drone_split), "non_drone": len(non_drone_split)}
        summary["balanced_counts"][split] = {"drone": len(chosen_drone), "non_drone": len(chosen_non_drone)}

    summary.update(
        {
            "seed": args.seed,
            "mode": args.mode,
            "drone_root": str(drone_root),
            "non_drone_root": str(non_drone_root),
            "drone_out_root": str(drone_out),
            "non_drone_out_root": str(non_drone_out),
        }
    )

    summary_path = out_root / "balance_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
