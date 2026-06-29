from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

LINEAR_PROBE_DIR = Path(__file__).resolve().parents[1]
if str(LINEAR_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(LINEAR_PROBE_DIR))

from common import run_embedding_linear_probe

IMAGE_SIZE = 224
MODEL_CHOICES = ["convnext_v2_tiny", "convnext_v2_base", "convnext_v2_large"]
TIMM_MODEL_MAP = {
    "convnext_v2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnext_v2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnext_v2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a default SVM classifier on ConvNeXt V2 embeddings for drone spectrogram detection"
    )
    parser.add_argument(
        "--drone-root",
        type=str,
        help="Path to drone spectrogram images",
        default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/drone/train",
    )
    parser.add_argument(
        "--non-drone-root",
        type=str,
        help="Path to non-drone spectrogram images",
        default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/non_drone/train",
    )
    parser.add_argument(
        "--artifact-out",
        type=str,
        help="Output .joblib for trained classifier",
        default="linear_probe/ConvNext_V2/trained_classifier.joblib",
    )
    parser.add_argument("--backbone", type=str, default="convnext_v2_tiny", choices=MODEL_CHOICES)
    parser.add_argument("--max-drone", type=int, default=0, help="Max number of drone images used, 0 for all")
    parser.add_argument("--max-non-drone", type=int, default=0, help="Max number of non-drone images used, 0 for all")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_embedding_linear_probe(
        args,
        feature_extractor=f"{args.backbone}_imagenet_penultimate",
        load_feature_model=lambda device: load_feature_model(device, args.backbone),
        extra_lines=[f"Backbone: {args.backbone}"],
    )


if __name__ == "__main__":
    main()
