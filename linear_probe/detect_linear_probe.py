#detect

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

def add_common_import_paths() -> None:
    if "__file__" in globals():
        candidates = [Path(__file__).resolve().parent]
    else:
        cwd = Path.cwd()
        candidates = [cwd, cwd / "linear_probe"]

    # Optional Kaggle dataset path that contains common.py. Keep harmless locally.
    candidates.append(Path("/kaggle/input/datasets/quoclop/linear-probe-base"))

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


add_common_import_paths()

from common import IMAGENET_MEAN, IMAGENET_STD, collect_image_paths, extract_embeddings, select_device


OUTPUT_JSON = r"/kaggle/working/linear_probe_results.json"
OUTPUT_CHART = r"/kaggle/working/linear_probe_results_chart.png"

# Set to None to use the selected model default artifact.
ARTIFACT_IN_OVERRIDE: str | None = None
BATCH_SIZE_OVERRIDE: int | None = None

USE_GPU = True
GPU_INDEX = 0
DEVICE = f"cuda:{GPU_INDEX}" if USE_GPU and torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

DINO_MODEL = "dinov2_vits14"


@dataclass(frozen=True)
class DetectConfig:
    feature_extractor: str
    default_artifact: str
    image_size: int
    batch_size: int
    load_model: Callable[[torch.device], torch.nn.Module]


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def device_status_lines() -> list[str]:
    if torch.cuda.is_available() and DEVICE.startswith("cuda"):
        device_index = torch.device(DEVICE).index or 0
        return [
            "CUDA available: yes",
            f"GPU: {torch.cuda.get_device_name(device_index)}",
        ]
    return ["CUDA available: no, using CPU"]


def load_dinov2(device: torch.device) -> torch.nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    except Exception as exc:
        raise RuntimeError(
            "Could not load DINOv2 from torch.hub. The first run needs internet access "
            "to download facebookresearch/dinov2 weights."
        ) from exc

    model.eval()
    model.to(device)
    return model


def load_swin_small(device: torch.device) -> torch.nn.Module:
    try:
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.swin_v2_s(weights=None)

    if hasattr(model, "head"):
        model.head = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()

    model.eval()
    model.to(device)
    return model


def load_vit_b16(device: torch.device) -> torch.nn.Module:
    try:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.vit_b_16(weights=None)

    model.heads = nn.Identity()
    model.eval()
    model.to(device)
    return model


def load_resnet50(device: torch.device) -> torch.nn.Module:
    try:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        base = models.resnet50(weights=None)

    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


def load_efficientnet_b2(device: torch.device) -> torch.nn.Module:
    try:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.efficientnet_b2(weights=None)

    model.classifier = nn.Identity()
    model.eval()
    model.to(device)
    return model


def load_vgg13bn(device: torch.device) -> torch.nn.Module:
    try:
        model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.vgg13_bn(weights=None)

    model.classifier[6] = nn.Identity()
    model.eval()
    model.to(device)
    return model


MODEL_CONFIGS = {
    "dinov2": DetectConfig(
        feature_extractor=f"{DINO_MODEL}_imagenet_ssl",
        default_artifact="/kaggle/input/datasets/quoclop/model-svm/Feature_extraction/dinov2_trained_classifier.joblib",
        image_size=224,
        batch_size=64,
        load_model=load_dinov2,
    ),
    "swin_small": DetectConfig(
        feature_extractor="swin_v2_s_imagenet_penultimate",
        default_artifact="/kaggle/input/datasets/quoclop/model-svm/Feature_extraction/swin_small_trained_classifier.joblib",
        image_size=224,
        batch_size=64,
        load_model=load_swin_small,
    ),
    "vit_b16": DetectConfig(
        feature_extractor="vit_b_16_imagenet_cls_token",
        default_artifact="/kaggle/input/datasets/quoclop/model-svm/Feature_extraction/vit_b16_trained_classifier.joblib",
        image_size=224,
        batch_size=32,
        load_model=load_vit_b16,
    ),
    "resnet50": DetectConfig(
        feature_extractor="resnet50_imagenet_v2_penultimate",
        default_artifact="/kaggle/input/datasets/quoclop/model-svm/Feature_extraction/resnet50_trained_classifier.joblib",
        image_size=224,
        batch_size=128,
        load_model=load_resnet50,
    ),
    "efficientnet_b2": DetectConfig(
        feature_extractor="efficientnet_b2_imagenet_penultimate",
        default_artifact="linear_probe/EfficientNet_B2/trained_classifier.joblib",
        image_size=260,
        batch_size=128,
        load_model=load_efficientnet_b2,
    ),
    "vgg13bn": DetectConfig(
        feature_extractor="vgg13_bn_imagenet_penultimate",
        default_artifact="linear_probe/VGG13_BN/trained_classifier.joblib",
        image_size=224,
        batch_size=32,
        load_model=load_vgg13bn,
    ),
}


def get_drone_scores(classifier, embeddings: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(embeddings)[:, 1]
    if hasattr(classifier, "decision_function"):
        raw_scores = classifier.decision_function(embeddings)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    return pred.astype(np.float32)


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
    ax3.pie(
        [drone_count, total - drone_count],
        labels=["Drone", "Non-drone"],
        colors=["#2b8a6e", "#c94949"],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax3.set_title("Prediction Split", fontweight="bold")

    ax4 = axes[1, 1]
    drone_rate = drone_count / max(total, 1)
    display_values = [str(total), str(drone_count), str(total - drone_count), f"{100 * drone_rate:.1f}%"]
    bars = ax4.bar(
        ["Source images", "Drone", "Non-drone", "Drone rate"],
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
        f"Linear Probe Detection Results | Model: {model_name or 'unknown'} | "
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
    import joblib

    configure_torch_runtime()

    if MODEL_TO_DETECT not in MODEL_CONFIGS:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown MODEL_TO_DETECT={MODEL_TO_DETECT!r}. Choose one of: {choices}")

    config = MODEL_CONFIGS[MODEL_TO_DETECT]
    artifact_path = Path(ARTIFACT_IN_OVERRIDE or config.default_artifact)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    payload = joblib.load(artifact_path)
    if payload.get("artifact_type") != "linear_probe_classifier":
        raise ValueError("This script expects a linear_probe_classifier artifact")

    train_summary = payload.get("summary", {})
    feature_extractor = train_summary.get("feature_extractor")
    if feature_extractor and feature_extractor != config.feature_extractor:
        raise ValueError(
            f"This detector expects {config.feature_extractor!r}, "
            f"but artifact feature_extractor={feature_extractor!r}"
        )

    image_size = int(train_summary.get("image_size", config.image_size))
    image_mean = list(train_summary.get("image_mean", IMAGENET_MEAN))
    image_std = list(train_summary.get("image_std", IMAGENET_STD))
    batch_size = BATCH_SIZE_OVERRIDE or config.batch_size
    device = select_device(DEVICE)

    print(f"Selected model: {MODEL_TO_DETECT}")
    for line in device_status_lines():
        print(line)
    print(f"Artifact: {artifact_path}")
    print(f"Source: {SOURCE_DIR}")

    source_paths = collect_image_paths(Path(SOURCE_DIR))
    model = config.load_model(device)
    embeddings = extract_embeddings(
        source_paths,
        model,
        device,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        image_mean=image_mean,
        image_std=image_std,
    )

    classifier = payload["model"]
    pred = classifier.predict(embeddings).astype(int)
    scores = get_drone_scores(classifier, embeddings, pred)

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
        "selected_model": MODEL_TO_DETECT,
        "feature_extractor": config.feature_extractor,
        "model_name": payload.get("model_name"),
        "artifact_in": str(artifact_path),
        "source_dir": str(SOURCE_DIR),
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
# Change this constant to choose the detector backbone:
# "dinov2", "swin_small", "vit_b16", "resnet50", "efficientnet_b2", or "vgg13bn".
MODEL_TO_DETECT = "dinov2"

# Edit these values directly.
SOURCE_DIR = r"/kaggle/input/datasets/kqucnguyn/data-test-spectrograms/drone_evaluate/drone_evaluate"

if __name__ == "__main__":
    main()
