from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor, SiglipVisionModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'transformers'. Install it first: pip install transformers"
    ) from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
SIGLIP_MODEL = "google/siglip-base-patch16-224"
FEATURE_EXTRACTOR = "siglip_base_patch16_224"


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


def load_feature_model(device: torch.device, model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor


def extract_embeddings(
    image_paths: Sequence[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embs: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Extract embeddings", unit="batch"):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values)
            features = outputs.pooler_output.detach().cpu().numpy()
            embs.append(features)

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
    parser = argparse.ArgumentParser(description="Train linear classifiers on SigLIP embeddings for drone spectrogram detection")
    parser.add_argument("--drone-root", type=str, help="Path to drone spectrogram images", default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/drone")
    parser.add_argument("--non-drone-root", type=str, help="Path to non-drone spectrogram images", default="/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/non_drone")
    parser.add_argument("--artifact-out", type=str, help="Output .joblib for trained classifier", default="linear_probe/SigLIP/trained_classifier.joblib")
    parser.add_argument("--siglip-model", type=str, default=SIGLIP_MODEL, help="HuggingFace SigLIP model id")
    parser.add_argument("--max-drone", type=int, default=3000, help="Max number of drone images used")
    parser.add_argument("--max-non-drone", type=int, default=3000, help="Max number of non-drone images used")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--batch-size", type=int, default=64)
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

    all_paths = drone_paths + non_drone_paths
    y = np.array([1] * len(drone_paths) + [0] * len(non_drone_paths), dtype=np.int64)

    feature_model, processor = load_feature_model(device, args.siglip_model)
    x = extract_embeddings(
        all_paths,
        feature_model,
        processor,
        device,
        batch_size=args.batch_size,
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
        "feature_extractor": FEATURE_EXTRACTOR,
        "siglip_model": args.siglip_model,
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
