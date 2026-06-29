from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LINEAR_PROBE_DIR = Path(__file__).resolve().parents[1]
if str(LINEAR_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(LINEAR_PROBE_DIR))

from common import (
    build_training_summary,
    prepare_binary_paths,
    print_dataset_summary,
    print_training_summary,
    save_training_artifact,
    select_device,
    train_default_svm,
)

SIGLIP_MODEL = "google/siglip-base-patch16-224"
FEATURE_EXTRACTOR = "siglip_base_patch16_224"


def load_feature_model(device: torch.device, model_name: str):
    try:
        from transformers import AutoImageProcessor, SiglipVisionModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'transformers'. Install it first: pip install transformers"
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor


def load_grayscale_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").convert("RGB")


def extract_embeddings(
    image_paths: Sequence[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Extract embeddings", unit="batch"):
            batch_paths = image_paths[start : start + batch_size]
            images = [load_grayscale_rgb(path) for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values)
            features = outputs.pooler_output.detach().cpu().numpy()
            embeddings.append(features)

    if not embeddings:
        raise ValueError("No image found for embedding extraction")
    return np.concatenate(embeddings, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a default SVM classifier on SigLIP embeddings for drone spectrogram detection"
    )
    parser.add_argument(
        "--drone-root",
        type=str,
        help="Path to drone spectrogram images",
        default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/drone",
    )
    parser.add_argument(
        "--non-drone-root",
        type=str,
        help="Path to non-drone spectrogram images",
        default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/non_drone",
    )
    parser.add_argument(
        "--artifact-out",
        type=str,
        help="Output .joblib for trained classifier",
        default="linear_probe/SigLIP/trained_classifier.joblib",
    )
    parser.add_argument("--siglip-model", type=str, default=SIGLIP_MODEL, help="HuggingFace SigLIP model id")
    parser.add_argument("--max-drone", type=int, default=3000, help="Max number of drone images used")
    parser.add_argument("--max-non-drone", type=int, default=3000, help="Max number of non-drone images used")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    drone_paths, non_drone_paths, all_paths, labels = prepare_binary_paths(
        args.drone_root,
        args.non_drone_root,
        max_drone=args.max_drone,
        max_non_drone=args.max_non_drone,
        seed=args.seed,
    )
    print_dataset_summary(device, drone_paths, non_drone_paths)

    feature_model, processor = load_feature_model(device, args.siglip_model)
    embeddings = extract_embeddings(
        all_paths,
        feature_model,
        processor,
        device,
        batch_size=args.batch_size,
    )

    results, best = train_default_svm(
        embeddings,
        labels,
        val_size=args.val_size,
        seed=args.seed,
    )
    summary = build_training_summary(
        FEATURE_EXTRACTOR,
        embeddings,
        labels,
        args,
        results,
        best,
        device,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
        extra={"siglip_model": args.siglip_model},
    )
    artifact_path = save_training_artifact(best, summary, args.artifact_out)
    print_training_summary(summary, artifact_path)


if __name__ == "__main__":
    main()
