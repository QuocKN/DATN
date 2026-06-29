from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
FEATURE_EXTRACTOR = "efficientnet_b2_imagenet_penultimate"
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

# Edit these values directly.
ARTIFACT_IN = "linear_probe/EfficientNet_B2/trained_classifier.joblib"
SOURCE_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/spectrograms_30"
OUTPUT_JSON = "linear_probe/EfficientNet_B2/report/test/results.json"
OUTPUT_CHART = "linear_probe/EfficientNet_B2/report/test/results_chart.png"
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_WORKERS = 4


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image)


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_feature_model(device: torch.device) -> nn.Module:
    try:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.efficientnet_b2(weights=None)

    model.classifier = nn.Identity()
    model.eval()
    model.to(device)
    return model


def extract_embeddings(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = SpectrogramDataset(image_paths, image_size=image_size, mean=mean, std=std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="Extract embeddings", unit="batch"):
            images = images.to(device, non_blocking=True)
            batch_embeddings = model(images).flatten(1).cpu().numpy()
            embeddings.append(batch_embeddings)

    if not embeddings:
        raise ValueError("No image found for embedding extraction")
    return np.concatenate(embeddings, axis=0)


def visualize_detection_results(
    scores: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    drone_count: int,
    total: int,
    model_name: str | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#f7f9fc")

    drone_mask = pred == 1
    non_drone_mask = pred == 0

    ax1 = axes[0, 0]
    if np.any(drone_mask):
        ax1.hist(scores[drone_mask], bins=30, alpha=0.75, color="#2b8a6e", edgecolor="black", label="Predicted drone")
    if np.any(non_drone_mask):
        ax1.hist(scores[non_drone_mask], bins=30, alpha=0.75, color="#c94949", edgecolor="black", label="Predicted non-drone")
    ax1.axvline(0.5, color="#111827", linestyle="--", linewidth=2, label="Decision threshold: 0.5")
    ax1.set_xlabel("Drone score")
    ax1.set_ylabel("Number of images")
    ax1.set_title("Score Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[0, 1]
    indices = np.arange(total)
    if np.any(drone_mask):
        ax2.scatter(indices[drone_mask], scores[drone_mask], s=18, color="#2b8a6e", alpha=0.75, label="Predicted drone")
    if np.any(non_drone_mask):
        ax2.scatter(indices[non_drone_mask], scores[non_drone_mask], s=18, color="#c94949", alpha=0.75, label="Predicted non-drone")
    ax2.axhline(0.5, color="#111827", linestyle="--", linewidth=2)
    ax2.set_xlabel("Image index")
    ax2.set_ylabel("Drone score")
    ax2.set_title("Score by Image Order", fontweight="bold")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[1, 0]
    counts = [drone_count, total - drone_count]
    ax3.pie(
        counts,
        labels=["Drone", "Non-drone"],
        colors=["#2b8a6e", "#c94949"],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax3.set_title("Prediction Split", fontweight="bold")

    ax4 = axes[1, 1]
    drone_rate = drone_count / max(total, 1)
    metric_names = ["Source images", "Drone", "Non-drone", "Drone rate"]
    display_values = [str(total), str(drone_count), str(total - drone_count), f"{100 * drone_rate:.1f}%"]
    bars = ax4.bar(
        metric_names,
        [total, drone_count, total - drone_count, total * drone_rate],
        color=["#357ABD", "#2b8a6e", "#c94949", "#d9902f"],
    )
    ax4.set_title("Detection Summary", fontweight="bold")
    ax4.set_ylabel("Count-scaled value")
    ax4.grid(axis="y", alpha=0.3)
    for bar, text in zip(bars, display_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.015, 0.5),
            text,
            ha="center",
            fontweight="bold",
        )

    fig.suptitle(
        f"EfficientNet-B2 Linear Probe | Model: {model_name or 'unknown'} | "
        f"Drone: {drone_count}/{total} ({100 * drone_rate:.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def main() -> None:
    artifact_path = Path(ARTIFACT_IN)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    payload = joblib.load(artifact_path)
    if payload.get("artifact_type") != "linear_probe_classifier":
        raise ValueError("This script expects a linear_probe_classifier artifact")

    train_summary = payload.get("summary", {})
    feature_extractor = train_summary.get("feature_extractor")
    if feature_extractor != FEATURE_EXTRACTOR:
        raise ValueError(
            "This detector expects an EfficientNet-B2 artifact, "
            f"but artifact feature_extractor={feature_extractor!r}"
        )

    image_size = int(train_summary.get("image_size", 260))
    mean = list(train_summary.get("image_mean", DEFAULT_MEAN))
    std = list(train_summary.get("image_std", DEFAULT_STD))
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")

    source_paths = collect_image_paths(Path(SOURCE_DIR))
    model = load_feature_model(device)
    embeddings = extract_embeddings(
        source_paths,
        model,
        device,
        image_size=image_size,
        mean=mean,
        std=std,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    classifier = payload["model"]
    pred = classifier.predict(embeddings).astype(int)
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(embeddings)[:, 1]
    elif hasattr(classifier, "decision_function"):
        raw_scores = classifier.decision_function(embeddings)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        scores = pred.astype(np.float32)

    predictions = [
        {
            "image": str(path),
            "prediction": "drone" if int(label) == 1 else "non_drone",
            "drone_score": float(score),
        }
        for path, label, score in zip(source_paths, pred, scores)
    ]

    drone_count = int(np.sum(pred == 1))
    total = int(len(pred))
    summary = {
        "method": "linear_probe_classifier",
        "feature_extractor": FEATURE_EXTRACTOR,
        "model_name": payload.get("model_name"),
        "artifact_in": str(artifact_path),
        "source_images": total,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": total - drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "score_mean": float(np.mean(scores)) if total else None,
        "score_std": float(np.std(scores)) if total else None,
        "train_summary": train_summary,
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump({"summary": summary, "predictions": predictions}, output_file, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_path}")
    visualize_detection_results(
        scores=scores,
        pred=pred,
        output_path=Path(OUTPUT_CHART),
        drone_count=drone_count,
        total=total,
        model_name=payload.get("model_name"),
    )


if __name__ == "__main__":
    main()
