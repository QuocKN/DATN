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
FEATURE_EXTRACTOR = "swin_v2_s_imagenet_penultimate"


def load_feature_model(device: torch.device) -> torch.nn.Module:
    try:
        base = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
    except Exception:
        base = models.swin_v2_s(weights=None)

    if hasattr(base, "head"):
        base.head = nn.Identity()
    elif hasattr(base, "classifier"):
        base.classifier = nn.Identity()

    base.eval()
    base.to(device)
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a default SVM classifier on Swin V2 Small embeddings for drone spectrogram detection"
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
        default="linear_probe/Swin_Small/trained_classifier.joblib",
    )
    parser.add_argument("--max-drone", type=int, default=0, help="Max number of drone images used, 0 for all")
    parser.add_argument("--max-non-drone", type=int, default=0, help="Max number of non-drone images used, 0 for all")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_embedding_linear_probe(args, FEATURE_EXTRACTOR, load_feature_model)


if __name__ == "__main__":
    main()
