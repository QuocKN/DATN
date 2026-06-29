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
FEATURE_EXTRACTOR = "resnet50_imagenet_v2_penultimate"


def load_feature_model(device: torch.device) -> torch.nn.Module:
    try:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        base = models.resnet50(weights=None)

    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a default SVM classifier on ResNet50 embeddings for drone spectrogram detection"
    )
    parser.add_argument(
        "--drone-root",
        type=str,
        help="Path to drone spectrogram images",
        default="E:\\DATN_DATA\\Spectrum\\DroneDetect_spectrogram_dataset\\train",
    )
    parser.add_argument(
        "--non-drone-root",
        type=str,
        help="Path to non-drone spectrogram images",
        default="E:\\DATN_DATA\\Spectrum\\non_drone\\train",
    )
    parser.add_argument(
        "--artifact-out",
        type=str,
        help="Output .joblib for trained classifier",
        default="linear_probe/ResNet50/trained_classifier.joblib",
    )
    parser.add_argument("--max-drone", type=int, default=1700, help="Max number of drone images used")
    parser.add_argument("--max-non-drone", type=int, default=1700, help="Max number of non-drone images used")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
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
