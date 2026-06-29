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

IMAGE_SIZE = 260
FEATURE_EXTRACTOR = "efficientnet_b2_imagenet_penultimate"


def load_feature_model(device: torch.device) -> nn.Module:
    try:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.efficientnet_b2(weights=None)

    model.classifier = nn.Identity()
    model.eval()
    model.to(device)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a default SVM classifier on EfficientNet-B2 embeddings for drone spectrogram detection"
    )
    parser.add_argument(
        "--drone-root",
        type=str,
        default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/drone/train",
    )
    parser.add_argument(
        "--non-drone-root",
        type=str,
        default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/non_drone/train",
    )
    parser.add_argument(
        "--artifact-out",
        type=str,
        default="linear_probe/EfficientNet_B2/trained_classifier.joblib",
    )
    parser.add_argument("--max-drone", type=int, default=0, help="Maximum drone images, 0 means all")
    parser.add_argument("--max-non-drone", type=int, default=0, help="Maximum non-drone images, 0 means all")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_embedding_linear_probe(args, FEATURE_EXTRACTOR, load_feature_model)


if __name__ == "__main__":
    main()
