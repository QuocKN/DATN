from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly.
ARTIFACT_IN = "linear_probe_artifact.joblib"
SOURCE_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/RFUAV/DJI_Mavic_Mini/spectrograms"
OUTPUT_JSON = "Report/RFUAV/DJI_Mavic_Mini/linear_detect_results.json"
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


def load_feature_model(device: torch.device) -> torch.nn.Module:
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        base = models.resnet18(weights=None)

    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


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

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")

    source_paths = collect_image_paths(source_root)
    model = load_feature_model(device)
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
        # Monotonic mapping to [0,1] for readability only.
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


if __name__ == "__main__":
    main()
