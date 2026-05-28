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
TIMM_MODEL_MAP = {
    "convnext_v2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnext_v2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnext_v2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
}

# Edit these values directly.
ARTIFACT_IN = "linear_probe/ConvNext_V2/trained_classifier.joblib"
SOURCE_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/spectrograms_refactor"  # --- IGNORE ---
OUTPUT_JSON = "linear_probe/ConvNext_V2/report/Tu_thu/2toan/results_refactor.json"
OUTPUT_CHART = "linear_probe/ConvNext_V2/report/Tu_thu/2toan/results_chart_refactor.png"
IMAGE_SIZE = 224
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_WORKERS = 4


class SpectrogramDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int) -> None:
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _build_convnext_v2(backbone_name: str) -> torch.nn.Module:
    if hasattr(models, backbone_name):
        ctor = getattr(models, backbone_name)
        enum_name = "".join([part.capitalize() for part in backbone_name.split("_")]) + "_Weights"
        weights_enum = getattr(models, enum_name, None)

        try:
            if weights_enum is not None and hasattr(weights_enum, "IMAGENET1K_V1"):
                return ctor(weights=weights_enum.IMAGENET1K_V1)
            return ctor(weights="DEFAULT")
        except Exception:
            return ctor(weights=None)

    try:
        import timm
    except Exception as exc:
        raise RuntimeError(
            f"Your torchvision does not provide '{backbone_name}', and 'timm' is not installed. "
            "Install one of these: "
            "1) upgrade torchvision with ConvNeXt V2 support, or "
            "2) pip install timm"
        ) from exc

    timm_model_name = TIMM_MODEL_MAP[backbone_name]
    try:
        return timm.create_model(timm_model_name, pretrained=True, num_classes=0, global_pool="avg")
    except Exception:
        return timm.create_model(timm_model_name, pretrained=False, num_classes=0, global_pool="avg")


def load_feature_model(device: torch.device, backbone_name: str) -> torch.nn.Module:
    base = _build_convnext_v2(backbone_name)
    base.classifier = nn.Identity()
    base.eval()
    base.to(device)
    return base


def extract_embeddings(
    image_paths: Sequence[Path],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = SpectrogramDataset(image_paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    embs: List[np.ndarray] = []
    with torch.no_grad():
        for x in tqdm(loader, desc="Extract embeddings", unit="batch"):
            x = x.to(device, non_blocking=True)
            emb = model(x).flatten(1).detach().cpu().numpy()
            embs.append(emb)

    if not embs:
        raise ValueError("No image found for embedding extraction")

    return np.concatenate(embs, axis=0)


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
    labels = ["Drone", "Non-drone"]
    colors = ["#2b8a6e", "#c94949"]
    ax3.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax3.set_title("Prediction Split", fontweight="bold")

    ax4 = axes[1, 1]
    metric_names = ["Source images", "Drone", "Non-drone", "Drone rate"]
    metric_values = [total, drone_count, total - drone_count, drone_count / max(total, 1)]
    display_values = [str(total), str(drone_count), str(total - drone_count), f"{100 * metric_values[-1]:.1f}%"]
    bar_values = [total, drone_count, total - drone_count, total * metric_values[-1]]
    bars = ax4.bar(metric_names, bar_values, color=["#357ABD", "#2b8a6e", "#c94949", "#d9902f"])
    ax4.set_title("Detection Summary", fontweight="bold")
    ax4.set_ylabel("Count-scaled value")
    ax4.grid(axis="y", alpha=0.3)
    for bar, text in zip(bars, display_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total * 0.015, 0.5), text, ha="center", fontweight="bold")

    fig.suptitle(
        f"Linear Probe Detection Results | Model: {model_name or 'unknown'} | "
        f"Drone: {drone_count}/{total} ({100 * drone_count / max(total, 1):.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def main() -> None:
    artifact_in = Path(ARTIFACT_IN)
    source_root = Path(SOURCE_DIR)

    if not artifact_in.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_in}")

    payload = joblib.load(artifact_in)
    if payload.get("artifact_type") != "linear_probe_classifier":
        raise ValueError("This script expects a linear_probe_classifier artifact")

    clf = payload["model"]
    summary_train = payload.get("summary", {})
    feature_extractor = summary_train.get("feature_extractor", "")
    if not feature_extractor.startswith("convnext_v2_"):
        raise ValueError(
            "This detector expects a ConvNeXt V2 artifact, "
            f"but artifact feature_extractor={feature_extractor!r}"
        )

    backbone_name = feature_extractor.replace("_imagenet_penultimate", "")

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")

    source_paths = collect_image_paths(source_root)
    model = load_feature_model(device=device, backbone_name=backbone_name)
    embeddings = extract_embeddings(
        source_paths,
        model,
        device,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    pred = clf.predict(embeddings)
    pred = pred.astype(int)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(embeddings)[:, 1]
    elif hasattr(clf, "decision_function"):
        raw = clf.decision_function(embeddings)
        scores = 1.0 / (1.0 + np.exp(-raw))
    else:
        scores = pred.astype(np.float32)

    predictions = []
    for path, y_hat, score in zip(source_paths, pred, scores):
        predictions.append(
            {
                "image": str(path),
                "prediction": "drone" if int(y_hat) == 1 else "non_drone",
                "drone_score": float(score),
            }
        )

    drone_count = int(np.sum(pred == 1))
    total = int(len(pred))

    summary = {
        "method": "linear_probe_classifier",
        "model_name": payload.get("model_name"),
        "artifact_in": str(artifact_in),
        "source_images": total,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": total - drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "score_mean": float(np.mean(scores)) if total else None,
        "score_std": float(np.std(scores)) if total else None,
        "train_summary": summary_train,
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_path}")

    chart_path = Path(OUTPUT_CHART)
    visualize_detection_results(
        scores=scores,
        pred=pred,
        output_path=chart_path,
        drone_count=drone_count,
        total=total,
        model_name=payload.get("model_name"),
    )


if __name__ == "__main__":
    main()
