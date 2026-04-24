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
) -> np.ndarray:
    transform = build_transform(image_size=image_size)
    embs = []
    for path in tqdm(image_paths, desc="Train embeddings", unit="img"):
        image = load_rgb_image(path)
        tensor = cast(torch.Tensor, transform(image))
        x = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).flatten(1).squeeze(0).detach().cpu().numpy()
        embs.append(emb)

    if not embs:
        raise ValueError("No training images found for embedding extraction")

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
        "image_size": int(image_size),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def fit_oneclass_artifact(
    train_root: Path,
    artifact_out: Path,
    threshold_percentile: float,
    image_size: int,
    device_name: str,
) -> Tuple[int, float]:
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")

    feature_model = load_feature_model(device)
    train_paths = collect_image_paths(train_root)
    train_embeddings = extract_embeddings_from_images(train_paths, feature_model, device, image_size)

    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    centroid = train_embeddings.mean(axis=0)
    train_distances = np.linalg.norm(train_embeddings - centroid, axis=1)
    threshold = float(np.percentile(train_distances, threshold_percentile))

    save_artifact(
        path=artifact_out,
        scaler=scaler,
        centroid=centroid,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        image_size=image_size,
    )

    return len(train_paths), threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-class centroid artifact from spectrogram images")
    parser.add_argument("--train-root", type=str, required=True, help="Train folder containing positive drone spectrogram images")
    parser.add_argument("--artifact-out", type=str, required=True, help="Where to save one-class artifact .joblib")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_root = Path(args.train_root)
    artifact_out = Path(args.artifact_out)

    if not train_root.exists():
        raise FileNotFoundError(f"train root not found: {train_root}")

    train_count, threshold = fit_oneclass_artifact(
        train_root=train_root,
        artifact_out=artifact_out,
        threshold_percentile=args.threshold_percentile,
        image_size=args.image_size,
        device_name=args.device,
    )

    print(f"Saved artifact: {artifact_out}")
    print(f"Train images: {train_count}")
    print(f"Threshold: {threshold:.6f} (percentile={args.threshold_percentile:.2f})")


if __name__ == "__main__":
    main()
