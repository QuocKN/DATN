from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Mapping, Sequence

import cv2
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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Edit these values directly if you do not want to pass CLI arguments.
MODE = "dinov2"
CHECKPOINT_IN = None
SOURCE_MODE = "drone_indoor"  # Change to "drone_outdoor" when needed.
SOURCE_DIR = None
OUTPUT_JSON = None
OUTPUT_CHART = None
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_WORKERS = 4

DATA_TEST_SPECTROGRAM_ROOT = "/kaggle/input/datasets/kqucnguyn/data-test-spectrograms"
SOURCE_DIR_DRONE_EVALUATE = f"{DATA_TEST_SPECTROGRAM_ROOT}/drone_evaluate/drone_evaluate"
SOURCE_DIR_DRONE_INDOOR = f"{SOURCE_DIR_DRONE_EVALUATE}/indoor"
SOURCE_DIR_DRONE_OUTDOOR = f"{SOURCE_DIR_DRONE_EVALUATE}/outdoor"
SOURCE_DIR_NON_DRONE_EVALUATE = f"{DATA_TEST_SPECTROGRAM_ROOT}/non_drone_evalueate/non_drone_evalueate"
SOURCE_DIR_NON_DRONE_INDOOR = f"{SOURCE_DIR_NON_DRONE_EVALUATE}/indoor"
SOURCE_DIR_NON_DRONE_OUTDOOR = f"{SOURCE_DIR_NON_DRONE_EVALUATE}/outdoor"
SOURCE_DIR_DRONE_RF = f"{DATA_TEST_SPECTROGRAM_ROOT}/Drone_RF_spectrograms/Drone_RF_spectrograms"
SOURCE_DIR_RFUAV = f"{DATA_TEST_SPECTROGRAM_ROOT}/RFUAV_spectrograms/RFUAV_spectrograms"

SOURCE_DIR_CONFIGS = {
    "drone": SOURCE_DIR_DRONE_EVALUATE,
    "drone_indoor": SOURCE_DIR_DRONE_INDOOR,
    "drone_outdoor": SOURCE_DIR_DRONE_OUTDOOR,
    "non_drone": SOURCE_DIR_NON_DRONE_EVALUATE,
    "non_drone_indoor": SOURCE_DIR_NON_DRONE_INDOOR,
    "non_drone_outdoor": SOURCE_DIR_NON_DRONE_OUTDOOR,
    "drone_rf": SOURCE_DIR_DRONE_RF,
    "rfuav": SOURCE_DIR_RFUAV,
}

SOURCE_MODE_ALIASES = {
    "default": "default",
    "model_default": "default",
    "drone": "drone",
    "drone_indoor": "drone_indoor",
    "drone_outdoor": "drone_outdoor",
    "drone_all": "drone",
    "drone_evaluate": "drone",
    "non_drone": "non_drone",
    "non_drone_indoor": "non_drone_indoor",
    "non_drone_outdoor": "non_drone_outdoor",
    "non_drone_all": "non_drone",
    "non_drone_evaluate": "non_drone",
    "nondrone": "non_drone",
    "nondrone_all": "non_drone",
    "nondrone_indoor": "non_drone_indoor",
    "nondrone_outdoor": "non_drone_outdoor",
    "drone_rf": "drone_rf",
    "drone_rf_spectrograms": "drone_rf",
    "rfuav": "rfuav",
    "rfuav_spectrograms": "rfuav",
}


@dataclass(frozen=True)
class BuildResult:
    model: nn.Module
    class_names: List[str]
    class_to_idx: dict
    image_size: int
    image_mean: List[float]
    image_std: List[float]
    image_preprocess: str
    extra_summary: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    method: str
    default_model_name: str
    default_checkpoint: str
    default_source_dir: str
    default_output_json: str
    default_output_chart: str
    default_image_size: int
    build_model: Callable[[dict, torch.device, "ModelConfig"], BuildResult]


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim != 2:
            features = features.flatten(1)
        return self.head(features)


class SobelEdge3Channel:
    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)
        edge3 = np.stack([edge, edge, edge], axis=-1)
        edge3 = (edge3 * 255).astype(np.uint8)
        return Image.fromarray(edge3)


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        image_preprocess: str,
    ) -> None:
        self.image_paths = list(image_paths)
        transform_steps: list[Callable] = [transforms.Resize((image_size, image_size))]
        if image_preprocess == "sobel_edge_3channel":
            transform_steps.append(SobelEdge3Channel())
        elif image_preprocess == "grayscale_rgb":
            transform_steps.append(transforms.Grayscale(num_output_channels=3))
        else:
            raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")
        return self.transform(image)


def class_metadata(checkpoint: Mapping) -> tuple[List[str], dict]:
    class_names = list(checkpoint.get("class_names", ["non_drone", "drone"]))
    class_to_idx = dict(checkpoint.get("class_to_idx", {name: idx for idx, name in enumerate(class_names)}))
    return class_names, class_to_idx


def load_dinov2_backbone(model_name: str, device: torch.device) -> nn.Module:
    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone.to(device)
    return backbone


@torch.no_grad()
def infer_feature_dim(backbone: nn.Module, image_size: int, device: torch.device) -> int:
    backbone.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = backbone(dummy)
    if features.ndim != 2:
        features = features.flatten(1)
    return int(features.shape[1])


def build_dinov2_model(checkpoint: dict, device: torch.device, config: ModelConfig) -> BuildResult:
    class_names, class_to_idx = class_metadata(checkpoint)
    image_size = int(checkpoint.get("image_size", config.default_image_size))
    dino_model = checkpoint.get("dino_model", "dinov2_vits14")

    backbone = load_dinov2_backbone(dino_model, device)
    feature_dim = infer_feature_dim(backbone, image_size, device)
    model = DINOv2Classifier(backbone=backbone, feature_dim=feature_dim, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return BuildResult(
        model=model,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        image_preprocess="grayscale_rgb",
        extra_summary={"backbone_name": dino_model, "dino_model": dino_model},
    )


def build_efficientnet_b2_model(checkpoint: dict, device: torch.device, config: ModelConfig) -> BuildResult:
    class_names, class_to_idx = class_metadata(checkpoint)
    image_size = int(checkpoint.get("image_size", config.default_image_size))
    image_mean = list(checkpoint.get("image_mean", IMAGENET_MEAN))
    image_std = list(checkpoint.get("image_std", IMAGENET_STD))
    backbone_name = checkpoint.get("backbone_name", "efficientnet_b2")
    if backbone_name != "efficientnet_b2":
        raise ValueError(f"Unsupported backbone_name in checkpoint: {backbone_name}")

    try:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return BuildResult(
        model=model,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        image_preprocess="grayscale_rgb",
        extra_summary={"backbone_name": backbone_name},
    )


def build_resnet50_model(checkpoint: dict, device: torch.device, config: ModelConfig) -> BuildResult:
    class_names, class_to_idx = class_metadata(checkpoint)
    image_size = int(checkpoint.get("image_size", config.default_image_size))
    image_mean = list(checkpoint.get("image_mean", IMAGENET_MEAN))
    image_std = list(checkpoint.get("image_std", IMAGENET_STD))
    image_preprocess = "sobel_edge_3channel"
    checkpoint_image_preprocess = checkpoint.get("image_preprocess")
    if checkpoint_image_preprocess not in (None, image_preprocess):
        print(
            f"Warning: checkpoint image_preprocess={checkpoint_image_preprocess!r}; "
            f"using {image_preprocess!r} in this detector."
        )

    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return BuildResult(
        model=model,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        image_preprocess=image_preprocess,
        extra_summary={
            "backbone_name": checkpoint.get("backbone_name", "resnet50"),
            "checkpoint_image_preprocess": checkpoint_image_preprocess,
        },
    )


def build_swin_small_model(checkpoint: dict, device: torch.device, config: ModelConfig) -> BuildResult:
    class_names, class_to_idx = class_metadata(checkpoint)
    image_size = int(checkpoint.get("image_size", config.default_image_size))
    backbone_name = checkpoint.get("backbone_name", "swin_v2_s")
    if backbone_name != "swin_v2_s":
        raise ValueError(f"Unsupported backbone_name in checkpoint: {backbone_name}")

    try:
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.swin_v2_s(weights=None)
    model.head = nn.Linear(model.head.in_features, len(class_names))
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return BuildResult(
        model=model,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        image_preprocess="grayscale_rgb",
        extra_summary={"backbone_name": backbone_name},
    )


def build_vit_b16_model(checkpoint: dict, device: torch.device, config: ModelConfig) -> BuildResult:
    class_names, class_to_idx = class_metadata(checkpoint)
    image_size = int(checkpoint.get("image_size", config.default_image_size))

    try:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(class_names))
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return BuildResult(
        model=model,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        image_preprocess="grayscale_rgb",
        extra_summary={"backbone_name": checkpoint.get("backbone_name", "vit_b_16")},
    )


def build_vgg13_model(checkpoint: dict, device: torch.device, config: ModelConfig) -> BuildResult:
    class_names, class_to_idx = class_metadata(checkpoint)
    image_size = int(checkpoint.get("image_size", config.default_image_size))

    try:
        model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.vgg13_bn(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return BuildResult(
        model=model,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        image_preprocess="grayscale_rgb",
        extra_summary={"backbone_name": checkpoint.get("backbone_name", "vgg13_bn")},
    )


MODEL_CONFIGS = {
    "dinov2": ModelConfig(
        display_name="DINOv2",
        method="dinov2_finetuned_classifier",
        default_model_name="dinov2_vits14_binary_finetuned",
        default_checkpoint="/kaggle/input/datasets/kqucnguyn/model-dronedetect-final/Fine_tune_V8/DinoV2/balanced_dinov2_vits14_binary.pt",
        default_source_dir=SOURCE_DIR_NON_DRONE_OUTDOOR,
        default_output_json="fine_tune/DINOv2/report/test/results.json",
        default_output_chart="fine_tune/DINOv2/report/test/results_chart.png",
        default_image_size=224,
        build_model=build_dinov2_model,
    ),
    "efficientnet_b2": ModelConfig(
        display_name="EfficientNet-B2",
        method="efficientnet_b2_finetuned_classifier",
        default_model_name="efficientnet_b2_binary_finetuned",
        default_checkpoint="/kaggle/input/datasets/kqucnguyn/model-dronedetect-final/Fine_tune_V8/EfficientNet_B2/balanced_efficientnet_b2_binary.pt",
        default_source_dir=SOURCE_DIR_DRONE_EVALUATE,
        default_output_json="fine_tune/EfficientNet_B2/report/Tu_thu/results.json",
        default_output_chart="fine_tune/EfficientNet_B2/report/Tu_thu/results_chart.png",
        default_image_size=260,
        build_model=build_efficientnet_b2_model,
    ),
    "resnet50": ModelConfig(
        display_name="ResNet50",
        method="resnet50_finetuned_classifier",
        default_model_name="resnet50_binary_finetuned",
        default_checkpoint="/kaggle/input/datasets/kqucnguyn/model-dronedetect-final/Fine_tune_V8/Resnet50/balanced_resnet50_binary_v11.pt",
        default_source_dir=SOURCE_DIR_NON_DRONE_INDOOR,
        default_output_json="fine_tune/ResNet50/report/test/results.json",
        default_output_chart="fine_tune/ResNet50/report/test/results_chart.png",
        default_image_size=224,
        build_model=build_resnet50_model,
    ),
    "swin_small": ModelConfig(
        display_name="Swin V2 Small",
        method="swin_v2_s_finetuned_classifier",
        default_model_name="swin_v2_s_binary_finetuned",
        default_checkpoint="/kaggle/input/datasets/kqucnguyn/model-dronedetect-final/Fine_tune_V8/Swin_small/balanced_swin_small_binary.pt",
        default_source_dir=SOURCE_DIR_NON_DRONE_OUTDOOR,
        default_output_json="fine_tune/Swin_Small/report/test/results.json",
        default_output_chart="fine_tune/Swin_Small/report/test/results_chart.png",
        default_image_size=224,
        build_model=build_swin_small_model,
    ),
    "vit_b16": ModelConfig(
        display_name="ViT-B/16",
        method="vit_b16_finetuned_classifier",
        default_model_name="vit_b16_binary_finetuned",
        default_checkpoint="/kaggle/input/datasets/kqucnguyn/model-dronedetect-final/Fine_tune_V8/Vit_B16/balanced_vit_b16_binary.pt",
        default_source_dir=SOURCE_DIR_DRONE_INDOOR,
        default_output_json="fine_tune/ViT_B16/report/test/results.json",
        default_output_chart="fine_tune/ViT_B16/report/test/results_chart.png",
        default_image_size=224,
        build_model=build_vit_b16_model,
    ),
    "vgg13": ModelConfig(
        display_name="VGG13-BN",
        method="vgg13_bn_finetuned_classifier",
        default_model_name="vgg13_bn_binary_finetuned",
        default_checkpoint="/kaggle/input/datasets/kqucnguyn/model-dronedetect-final/Fine_tune_V8/VGG13_BN/balanced_vgg13_binary.pt",
        default_source_dir=SOURCE_DIR_NON_DRONE_OUTDOOR,
        default_output_json="fine_tune/VGG13/report/test/results.json",
        default_output_chart="fine_tune/VGG13/report/test/results_chart.png",
        default_image_size=224,
        build_model=build_vgg13_model,
    ),
}

MODE_ALIASES = {
    "dino": "dinov2",
    "dinov2": "dinov2",
    "efficientnet": "efficientnet_b2",
    "efficientnetb2": "efficientnet_b2",
    "efficientnet_b2": "efficientnet_b2",
    "resnet": "resnet50",
    "resnet50": "resnet50",
    "swin": "swin_small",
    "swin_s": "swin_small",
    "swin_small": "swin_small",
    "vit": "vit_b16",
    "vit_b16": "vit_b16",
    "vit_b_16": "vit_b16",
    "vgg13": "vgg13",
    "vgg13bn": "vgg13",
    "vgg13_bn": "vgg13",
}


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def normalize_mode(mode: str) -> str:
    key = mode.strip().lower().replace("-", "_")
    return MODE_ALIASES.get(key, key)


def normalize_source_mode(source_mode: str | None) -> str | None:
    if not source_mode:
        return None
    key = source_mode.strip().lower().replace("-", "_")
    return SOURCE_MODE_ALIASES.get(key, key)


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


@torch.no_grad()
def infer(
    image_paths: Sequence[Path],
    build_result: BuildResult,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SpectrogramDataset(
        image_paths,
        image_size=build_result.image_size,
        image_mean=build_result.image_mean,
        image_std=build_result.image_std,
        image_preprocess=build_result.image_preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    probs_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []

    build_result.model.eval()
    for x in tqdm(loader, desc="Infer", unit="batch"):
        x = x.to(device, non_blocking=True)
        logits = build_result.model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        probs_all.append(probs.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())

    if not probs_all:
        raise ValueError("No image found for inference")

    return np.concatenate(probs_all, axis=0), np.concatenate(pred_all, axis=0)


def source_label(source_root: Path) -> str:
    parts = source_root.parts[-2:]
    return "/".join(parts) if parts else str(source_root)


def visualize_detection_results(
    scores: np.ndarray,
    pred_binary: np.ndarray,
    output_path: Path,
    drone_count: int,
    total: int,
    model_name: str | None,
    config: ModelConfig,
    source_root: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#f7f9fc")

    drone_mask = pred_binary == 1
    non_drone_mask = pred_binary == 0

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
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.015, 0.5),
            text,
            ha="center",
            fontweight="bold",
        )

    fig.suptitle(
        f"{source_label(source_root)}\n\n"
        f"Fine-tuned {config.display_name} Detection Results | Model: {model_name or 'unknown'} | "
        f"Drone: {drone_count}/{total} ({100 * drone_count / max(total, 1):.1f}%)\n",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def resolve_value(value: str | None, fallback: str) -> str:
    return value if value else fallback


def resolve_source_dir(source_dir: str | None, source_mode: str | None, config: ModelConfig) -> tuple[str, str]:
    if source_dir:
        return source_dir, "custom"

    normalized_source_mode = normalize_source_mode(source_mode)
    if normalized_source_mode in (None, "default"):
        return config.default_source_dir, "model_default"

    if normalized_source_mode not in SOURCE_DIR_CONFIGS:
        choices = ", ".join(["default", *sorted(SOURCE_DIR_CONFIGS)])
        raise ValueError(f"Unknown source_mode={normalized_source_mode!r}. Choose one of: {choices}")

    return SOURCE_DIR_CONFIGS[normalized_source_mode], normalized_source_mode


def run_detection(
    mode: str,
    checkpoint_in: str | None,
    source_mode: str | None,
    source_dir: str | None,
    output_json: str | None,
    output_chart: str | None,
    device_name: str,
    batch_size: int,
    num_workers: int,
    save_chart: bool,
) -> None:
    mode = normalize_mode(mode)
    if mode not in MODEL_CONFIGS:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown mode={mode!r}. Choose one of: {choices}")

    config = MODEL_CONFIGS[mode]
    checkpoint_path = Path(resolve_value(checkpoint_in, config.default_checkpoint))
    resolved_source_dir, resolved_source_mode = resolve_source_dir(source_dir, source_mode, config)
    source_root = Path(resolved_source_dir)
    output_json_path = Path(resolve_value(output_json, config.default_output_json))
    output_chart_path = Path(resolve_value(output_chart, config.default_output_chart))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = select_device(device_name)
    print(f"Selected mode: {mode}")
    print(f"Selected source mode: {resolved_source_mode}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Source dir: {source_root}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    build_result = config.build_model(checkpoint, device, config)

    if "drone" not in build_result.class_to_idx:
        raise ValueError("Checkpoint class_to_idx does not contain 'drone'")
    drone_idx = int(build_result.class_to_idx["drone"])

    source_paths = collect_image_paths(source_root)
    probs, pred = infer(
        image_paths=source_paths,
        build_result=build_result,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    scores = probs[:, drone_idx]
    pred = pred.astype(int)
    idx_to_class = {v: k for k, v in build_result.class_to_idx.items()}

    predictions = []
    for path, y_hat, score in zip(source_paths, pred, scores):
        label_name = idx_to_class.get(int(y_hat), str(int(y_hat)))
        predictions.append(
            {
                "image": str(path),
                "prediction": label_name,
                "drone_score": float(score),
            }
        )

    drone_count = int(np.sum(pred == drone_idx))
    total = int(len(pred))
    non_drone_count = total - drone_count

    summary = {
        "mode": mode,
        "method": config.method,
        "model_name": checkpoint.get("model_name", config.default_model_name),
        "checkpoint_in": str(checkpoint_path),
        "source_mode": resolved_source_mode,
        "source_dir": str(source_root),
        "source_images": total,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": non_drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "score_mean": float(np.mean(scores)) if total else None,
        "score_std": float(np.std(scores)) if total else None,
        "class_names": build_result.class_names,
        "class_to_idx": build_result.class_to_idx,
        "image_size": build_result.image_size,
        "image_preprocess": build_result.image_preprocess,
        "image_mean": build_result.image_mean,
        "image_std": build_result.image_std,
        "best_valid_macro_f1": checkpoint.get("best_valid_macro_f1"),
    }
    summary.update(build_result.extra_summary)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_json_path}")

    if save_chart:
        visualize_detection_results(
            scores=scores,
            pred_binary=(pred == drone_idx).astype(np.int64),
            output_path=output_chart_path,
            drone_count=drone_count,
            total=total,
            model_name=checkpoint_path.stem,
            config=config,
            source_root=source_root,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fine-tuned detector by changing --mode.")
    mode_choices = ", ".join(sorted(MODEL_CONFIGS))
    source_mode_choices = ", ".join(["default", *sorted(SOURCE_DIR_CONFIGS)])
    parser.add_argument("--mode", default=MODE, help=f"Detector mode. Choices: {mode_choices}.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_IN, help="Override checkpoint path for the selected mode.")
    parser.add_argument(
        "--source-mode",
        default=SOURCE_MODE,
        help=f"Source directory mode. Choices: {source_mode_choices}. Uses the model default when omitted.",
    )
    parser.add_argument("--source-dir", default=SOURCE_DIR, help="Override source image directory path.")
    parser.add_argument("--output-json", default=OUTPUT_JSON, help="Override output JSON path.")
    parser.add_argument("--output-chart", default=OUTPUT_CHART, help="Override output chart path.")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation.")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    run_detection(
        mode=args.mode,
        checkpoint_in=args.checkpoint,
        source_mode=args.source_mode,
        source_dir=args.source_dir,
        output_json=args.output_json,
        output_chart=args.output_chart,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_chart=not args.no_chart,
    )


if __name__ == "__main__":
    main()
