#train
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models

import sys
from pathlib import Path

def add_common_import_paths() -> None:
    if "__file__" in globals():
        candidates = [Path(__file__).resolve().parent]
    else:
        cwd = Path.cwd()
        candidates = [cwd, cwd / "linear_probe"]

    candidates.append(Path("/kaggle/input/datasets/quoclop/linear-probe-base"))

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


add_common_import_paths()

from common import run_split_embedding_linear_probe


# Shared train settings.
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
NUM_WORKERS = 4
USE_GPU = True
GPU_INDEX = 0
DEVICE = f"cuda:{GPU_INDEX}" if USE_GPU and torch.cuda.is_available() else "cpu"
SEED = 42
SVM_CLASS_WEIGHT: str | dict | None = "balanced"

# Set to None to use the selected model default.
BATCH_SIZE_OVERRIDE: int | None = None
ARTIFACT_OUT_OVERRIDE: str | None = None

DINO_MODEL = "dinov2_vits14"


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
            f"CUDA available: yes",
            f"GPU: {torch.cuda.get_device_name(device_index)}",
        ]
    return ["CUDA available: no, using CPU"]


@dataclass(frozen=True)
class ModelConfig:
    feature_extractor: str
    image_size: int
    batch_size: int
    artifact_out: str
    load_model: Callable[[torch.device], torch.nn.Module]


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
    "dinov2": ModelConfig(
        feature_extractor=f"{DINO_MODEL}_imagenet_ssl",
        image_size=224,
        batch_size=64,
        artifact_out="linear_probe/DINOv2/trained_classifier.joblib",
        load_model=load_dinov2,
    ),
    "swin_small": ModelConfig(
        feature_extractor="swin_v2_s_imagenet_penultimate",
        image_size=224,
        batch_size=64,
        artifact_out="linear_probe/Swin_Small/trained_classifier.joblib",
        load_model=load_swin_small,
    ),
    "vit_b16": ModelConfig(
        feature_extractor="vit_b_16_imagenet_cls_token",
        image_size=224,
        batch_size=32,
        artifact_out="linear_probe/ViT_B16/trained_classifier.joblib",
        load_model=load_vit_b16,
    ),
    "resnet50": ModelConfig(
        feature_extractor="resnet50_imagenet_v2_penultimate",
        image_size=224,
        batch_size=128,
        artifact_out="linear_probe/ResNet50/trained_classifier.joblib",
        load_model=load_resnet50,
    ),
    "efficientnet_b2": ModelConfig(
        feature_extractor="efficientnet_b2_imagenet_penultimate",
        image_size=260,
        batch_size=128,
        artifact_out="linear_probe/EfficientNet_B2/trained_classifier.joblib",
        load_model=load_efficientnet_b2,
    ),
    "vgg13bn": ModelConfig(
        feature_extractor="vgg13_bn_imagenet_penultimate",
        image_size=224,
        batch_size=32,
        artifact_out="linear_probe/VGG13_BN/trained_classifier.joblib",
        load_model=load_vgg13bn,
    ),
}


def build_args(config: ModelConfig) -> SimpleNamespace:
    return SimpleNamespace(
        drone_root=DRONE_ROOT,
        non_drone_root=NON_DRONE_ROOT,
        artifact_out=ARTIFACT_OUT_OVERRIDE or config.artifact_out,
        max_drone=MAX_DRONE,
        max_non_drone=MAX_NON_DRONE,
        train_split=TRAIN_SPLIT,
        valid_split=VALID_SPLIT,
        test_split=TEST_SPLIT,
        image_size=config.image_size,
        batch_size=BATCH_SIZE_OVERRIDE or config.batch_size,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        seed=SEED,
    )


def main() -> None:
    configure_torch_runtime()

    if MODEL_TO_TRAIN not in MODEL_CONFIGS:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown MODEL_TO_TRAIN={MODEL_TO_TRAIN!r}. Choose one of: {choices}")

    config = MODEL_CONFIGS[MODEL_TO_TRAIN]
    args = build_args(config)
    run_split_embedding_linear_probe(
        args,
        feature_extractor=config.feature_extractor,
        load_feature_model=config.load_model,
        extra_lines=[f"Selected model: {MODEL_TO_TRAIN}", *device_status_lines()],
        extra_summary={"svm_class_weight": SVM_CLASS_WEIGHT},
        svm_class_weight=SVM_CLASS_WEIGHT,
    )

DRONE_ROOT = r"/kaggle/input/balanced-dataset-drone-chuan-full-non-done/drone/drone"
NON_DRONE_ROOT = "/kaggle/input/balanced-dataset-drone-chuan-full-non-done/non_drone_not11/non_drone"
MAX_DRONE = 2500
MAX_NON_DRONE = 2500
# Change this constant to choose the backbone:
# "dinov2", "swin_small", "vit_b16", "resnet50", "efficientnet_b2", or "vgg13bn".
MODEL_TO_TRAIN = "vgg13bn"

if __name__ == "__main__":
    main()
