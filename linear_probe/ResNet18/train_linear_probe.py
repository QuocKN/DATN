from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_SIZE = 224


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


def choose_subset(paths: Sequence[Path], max_count: int, seed: int) -> List[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return list(paths)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=max_count, replace=False)
    return [paths[i] for i in sorted(indices.tolist())]


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


@dataclass
class TrainResult:
    name: str
    pipeline: Pipeline
    report: dict
    macro_f1: float


def train_and_eval(
    name: str,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> TrainResult:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    macro_f1 = float(report["macro avg"]["f1-score"])
    return TrainResult(name=name, pipeline=pipeline, report=report, macro_f1=macro_f1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear classifiers on ResNet18 embeddings for drone spectrogram detection")
    parser.add_argument("--drone-root", type=str, required=True, help="Path to drone spectrogram images")
    parser.add_argument("--non-drone-root", type=str, required=True, help="Path to non-drone spectrogram images")
    parser.add_argument("--artifact-out", type=str, required=True, help="Output .joblib for trained classifier")
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

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    drone_paths = collect_image_paths(Path(args.drone_root))
    non_drone_paths = collect_image_paths(Path(args.non_drone_root))

    drone_paths = choose_subset(drone_paths, max_count=args.max_drone, seed=args.seed)
    non_drone_paths = choose_subset(non_drone_paths, max_count=args.max_non_drone, seed=args.seed + 1)

    if not drone_paths:
        raise ValueError("No drone images found")
    if not non_drone_paths:
        raise ValueError("No non-drone images found")

    print(f"Using drone images: {len(drone_paths)}")
    print(f"Using non-drone images: {len(non_drone_paths)}")
    if len(non_drone_paths) < 100:
        print("[WARN] Non-drone samples are very few. Keep class_weight='balanced' and add hard negatives soon.")

    all_paths = drone_paths + non_drone_paths
    y = np.array([1] * len(drone_paths) + [0] * len(non_drone_paths), dtype=np.int64)

    feature_model = load_feature_model(device)
    x = extract_embeddings(
        all_paths,
        feature_model,
        device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y,
    )

    candidates = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=args.seed,
            ),
        ),
        (
            "linear_svm",
            LinearSVC(
                class_weight="balanced",
                random_state=args.seed,
                max_iter=5000,
            ),
        ),
    ]

    results: List[TrainResult] = []
    for name, clf in candidates:
        results.append(train_and_eval(name, clf, x_train, y_train, x_val, y_val))

    best = max(results, key=lambda r: r.macro_f1)

    summary = {
        "feature_extractor": "resnet18_imagenet_penultimate",
        "image_size": args.image_size,
        "device": str(device),
        "n_total": int(len(y)),
        "n_drone": int(np.sum(y == 1)),
        "n_non_drone": int(np.sum(y == 0)),
        "val_size": float(args.val_size),
        "seed": int(args.seed),
        "models": {
            r.name: {
                "macro_f1": r.macro_f1,
                "drone_precision": float(r.report.get("1", {}).get("precision", 0.0)),
                "drone_recall": float(r.report.get("1", {}).get("recall", 0.0)),
                "drone_f1": float(r.report.get("1", {}).get("f1-score", 0.0)),
                "non_drone_precision": float(r.report.get("0", {}).get("precision", 0.0)),
                "non_drone_recall": float(r.report.get("0", {}).get("recall", 0.0)),
                "non_drone_f1": float(r.report.get("0", {}).get("f1-score", 0.0)),
            }
            for r in results
        },
        "selected_model": best.name,
    }

    payload = {
        "artifact_type": "linear_probe_classifier",
        "model_name": best.name,
        "model": best.pipeline,
        "summary": summary,
    }

    artifact_out = Path(args.artifact_out)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_out)

    print("=" * 80)
    print("Training summary")
    print(summary)
    print("=" * 80)
    print(f"Saved artifact: {artifact_out}")


if __name__ == "__main__":
    main()
