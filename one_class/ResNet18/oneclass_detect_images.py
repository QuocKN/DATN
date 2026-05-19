from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly instead of passing command-line arguments.
ARTIFACT_IN = "oneclass_artifact.joblib"
SOURCE_DIR = "E:\\DATN_DATA\\RF\\RF Control and Video Signal\\spectrograms"
OUTPUT_JSON = "Report/RFUAV/DJI_Inspire/detect_threshold_results.json"
OUTPUT_CHART = "Report/RFUAV/DJI_Inspire/threshold_chart.png"
IMAGE_SIZE = 224
DEVICE = "cuda:0"


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_transform(image_size: int = 224):
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

    # Global pooled penultimate feature (512-dim)
    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


def extract_embeddings(
    image_paths: List[Path],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> np.ndarray:
    transform = build_transform(image_size=image_size)
    embs = []
    for path in tqdm(image_paths, desc="Extract embeddings", unit="img"):
        image = load_rgb_image(path)
        tensor = cast(torch.Tensor, transform(image))
        x = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).flatten(1).squeeze(0).detach().cpu().numpy()
        embs.append(emb)

    if not embs:
        raise ValueError("No image found for embedding extraction")
    return np.asarray(embs)


def load_artifact(path: Path) -> Tuple[StandardScaler, np.ndarray, float, float, int]:
    payload = joblib.load(path)
    scaler = cast(StandardScaler, payload["scaler"])
    centroid = np.asarray(payload["centroid"])
    threshold = float(payload["threshold"])
    threshold_percentile = float(payload.get("threshold_percentile", 95.0))
    image_size = int(payload.get("image_size", 224))
    return scaler, centroid, threshold, threshold_percentile, image_size


def visualize_threshold_distribution(
    distances: np.ndarray,
    threshold: float,
    threshold_percentile: float,
    output_path: Path,
    drone_count: int,
    total: int,
) -> None:
    """Create a visualization showing distance distribution and threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with threshold line
    ax1 = axes[0]
    ax1.hist(distances, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.3f}")
    ax1.set_xlabel("Distance from Centroid", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distance Distribution", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Line chart showing distance progression
    ax2 = axes[1]
    drone_mask = distances <= threshold

    # Plot drone samples in green
    drone_indices = np.where(drone_mask)[0]
    ax2.plot(drone_indices, distances[drone_mask], color="green", linewidth=1.5, alpha=0.7, label="Drone")
    ax2.scatter(drone_indices, distances[drone_mask], color="green", s=20, alpha=0.6)

    # Plot non-drone samples in orange
    non_drone_indices = np.where(~drone_mask)[0]
    ax2.plot(non_drone_indices, distances[~drone_mask], color="orange", linewidth=1.5, alpha=0.7, label="Non-drone")
    ax2.scatter(non_drone_indices, distances[~drone_mask], color="orange", s=20, alpha=0.6)

    ax2.axhline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.3f}")
    ax2.set_xlabel("Image Index", fontsize=11)
    ax2.set_ylabel("Distance from Centroid", fontsize=11)
    ax2.set_title("Distance Progression", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add overall title with statistics
    fig.suptitle(
        f"One-Class Detection Results | Threshold: {threshold_percentile}th percentile | "
        f"Drone: {drone_count}/{total} ({100*drone_count/max(total, 1):.1f}%)",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {output_path}")
    plt.close()


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")

    artifact_in = Path(ARTIFACT_IN)
    source_root = Path(SOURCE_DIR)

    if not artifact_in.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_in}")

    source_paths = collect_image_paths(source_root)

    model = load_feature_model(device)
    scaler, centroid, threshold, threshold_percentile, saved_image_size = load_artifact(artifact_in)
    if saved_image_size != IMAGE_SIZE:
        print(
            f"[WARN] artifact image_size={saved_image_size} differs from IMAGE_SIZE={IMAGE_SIZE}. "
            f"Using IMAGE_SIZE for image preprocessing."
        )

    source_embeddings = extract_embeddings(source_paths, model, device, IMAGE_SIZE)
    source_embeddings = scaler.transform(source_embeddings)
    source_distances = np.linalg.norm(source_embeddings - centroid, axis=1)

    predictions = []
    for path, dist in zip(source_paths, source_distances):
        is_drone = bool(dist <= threshold)
        predictions.append(
            {
                "image": str(path),
                "distance": float(dist),
                "threshold": threshold,
                "prediction": "drone" if is_drone else "non_drone",
            }
        )

    drone_count = int(sum(1 for p in predictions if p["prediction"] == "drone"))
    total = len(predictions)

    summary = {
        "method": "one_class_centroid",
        "embedding_backbone": "resnet18_penultimate",
        "artifact_in": str(artifact_in),
        "source_images": total,
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": total - drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "mean_distance": float(np.mean(source_distances)) if total else None,
        "std_distance": float(np.std(source_distances)) if total else None,
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_path}")

    # Generate visualization chart
    chart_path = Path(OUTPUT_CHART)
    visualize_threshold_distribution(
        source_distances,
        threshold,
        threshold_percentile,
        chart_path,
        drone_count,
        total,
    )


if __name__ == "__main__":
    main()
