from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_SIZE = 224


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_feature_model(device: torch.device) -> torch.nn.Module:
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        base = models.resnet18(weights=None)

    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


def extract_embeddings_from_images(
    image_paths: List[Path],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    desc: str,
) -> np.ndarray:
    transform = build_transform(image_size=image_size)
    embs = []
    for path in tqdm(image_paths, desc=desc, unit="img"):
        image = load_rgb_image(path)
        tensor = cast(torch.Tensor, transform(image))
        x = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).flatten(1).squeeze(0).detach().cpu().numpy()
        embs.append(emb)

    if not embs:
        raise ValueError(f"No images found for embedding extraction ({desc})")

    return np.asarray(embs)


def save_artifact(
    path: Path,
    scaler: StandardScaler,
    centroid: np.ndarray,
    threshold: float,
    threshold_percentile: float,
    image_size: int,
) -> None:
    payload = {
        "scaler": scaler,
        "centroid": centroid,
        "threshold": float(threshold),
        "threshold_percentile": float(threshold_percentile),
        "threshold_source": "valid",
        "image_size": int(image_size),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: Path) -> Tuple[StandardScaler, np.ndarray, float, int]:
    payload = joblib.load(path)
    scaler = cast(StandardScaler, payload["scaler"])
    centroid = np.asarray(payload["centroid"])
    threshold_percentile = float(payload.get("threshold_percentile", 95.0))
    image_size = int(payload.get("image_size", IMAGE_SIZE))
    return scaler, centroid, threshold_percentile, image_size


def calibrate_threshold_from_valid(
    artifact_in: Path,
    valid_root: Path,
    artifact_out: Path,
    threshold_percentile: float,
    image_size_override: int | None,
    device_name: str,
) -> tuple[int, float, int]:
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")

    scaler, centroid, old_threshold_percentile, image_size = load_artifact(artifact_in)
    if image_size_override is not None and image_size_override != image_size:
        print(
            f"[WARN] artifact image_size={image_size} differs from --image-size={image_size_override}. "
            f"Using --image-size for valid preprocessing."
        )
        image_size = image_size_override

    if threshold_percentile < 0:
        threshold_percentile = old_threshold_percentile

    feature_model = load_feature_model(device)

    valid_paths = collect_image_paths(valid_root)
    valid_embeddings = extract_embeddings_from_images(
        valid_paths,
        feature_model,
        device,
        image_size,
        desc="Valid embeddings",
    )
    valid_embeddings = scaler.transform(valid_embeddings)
    valid_distances = np.linalg.norm(valid_embeddings - centroid, axis=1)
    threshold = float(np.percentile(valid_distances, threshold_percentile))

    save_artifact(
        path=artifact_out,
        scaler=scaler,
        centroid=centroid,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        image_size=image_size,
    )

    return len(valid_paths), threshold, image_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-calibrate threshold from valid images using an existing one-class artifact")
    parser.add_argument("--artifact-in", type=str, required=True, help="Existing one-class artifact .joblib to reuse scaler/centroid")
    parser.add_argument("--valid-root", type=str, required=True, help="Valid folder used to compute threshold")
    parser.add_argument("--artifact-out", type=str, default=None, help="Output artifact path (default: overwrite artifact-in)")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=-1.0,
        help="Percentile for threshold on valid distances. Use negative value to keep artifact's saved percentile.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Optional override for valid image preprocessing size. 0 means use artifact image_size.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact_in = Path(args.artifact_in)
    valid_root = Path(args.valid_root)
    artifact_out = Path(args.artifact_out) if args.artifact_out else artifact_in
    image_size_override = args.image_size if args.image_size > 0 else None

    if not artifact_in.exists():
        raise FileNotFoundError(f"artifact not found: {artifact_in}")
    if not valid_root.exists():
        raise FileNotFoundError(f"valid root not found: {valid_root}")

    valid_count, threshold, used_image_size = calibrate_threshold_from_valid(
        artifact_in=artifact_in,
        valid_root=valid_root,
        artifact_out=artifact_out,
        threshold_percentile=args.threshold_percentile,
        image_size_override=image_size_override,
        device_name=args.device,
    )

    print(f"Saved artifact: {artifact_out}")
    print(f"Loaded artifact: {artifact_in}")
    print(f"Valid images: {valid_count}")
    if args.threshold_percentile >= 0:
        used_percentile = args.threshold_percentile
    else:
        used_percentile = float(joblib.load(artifact_out).get("threshold_percentile", 95.0))
    print(f"Image size used: {used_image_size}")
    print(f"Threshold: {threshold:.6f} (percentile={used_percentile:.2f}, source=valid)")


if __name__ == "__main__":
    main()
