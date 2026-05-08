from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly instead of passing command-line arguments.
ARTIFACT_IN = "oneclass_artifact.joblib"
TEST_DIR = r"/home/quocnk/Documents/NKQuoc/Data/RF/DroneDetect/DroneDetect_spectrogram_dataset/test"
OUTPUT_JSON = "oneclass_test_results.json"
IMAGE_SIZE = 224
DEVICE = "cuda:0"


def collect_image_paths_with_labels(root: Path) -> Tuple[List[Path], List[str]]:
    """
    Collect image paths and infer labels from directory structure.
    All images in test set are drone (ground truth).
    """
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    
    image_paths = []
    labels = []
    
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # All images in test set are drone
            image_paths.append(p)
            labels.append("drone")
    
    return image_paths, labels


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_feature_model(device: torch.device) -> torch.nn.Module:
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        base = models.resnet18(weights=None)

    # Global pooled penultimate feature (512-dim)
    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


def extract_embeddings(
    image_paths: List[Path],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
) -> np.ndarray:
    transform = build_transform(image_size=image_size)
    embs = []
    for path in tqdm(image_paths, desc="Extract embeddings", unit="img"):
        try:
            image = load_rgb_image(path)
            tensor = cast(torch.Tensor, transform(image))
            x = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(x).flatten(1).squeeze(0).detach().cpu().numpy()
            embs.append(emb)
        except Exception as e:
            print(f"[WARN] Failed to process {path}: {e}")
            continue

    if not embs:
        raise ValueError("No image found for embedding extraction")
    return np.asarray(embs)


def load_artifact(path: Path) -> Tuple[StandardScaler, np.ndarray, float, float, int]:
    payload = joblib.load(path)
    scaler = cast(StandardScaler, payload["scaler"])
    centroid = np.asarray(payload["centroid"])
    threshold = float(payload["threshold"])
    threshold_percentile = float(payload.get("threshold_percentile", 95.0))
    image_size = int(payload.get("image_size", 224))
    return scaler, centroid, threshold, threshold_percentile, image_size


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    artifact_in = Path(ARTIFACT_IN)
    test_root = Path(TEST_DIR)

    if not artifact_in.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_in}")

    print("Collecting test images and labels...")
    test_paths, ground_truth_labels = collect_image_paths_with_labels(test_root)
    print(f"Found {len(test_paths)} test images")

    if not test_paths:
        raise ValueError("No test images found!")

    model = load_feature_model(device)
    scaler, centroid, threshold, threshold_percentile, saved_image_size = load_artifact(artifact_in)
    
    if saved_image_size != IMAGE_SIZE:
        print(
            f"[WARN] artifact image_size={saved_image_size} differs from IMAGE_SIZE={IMAGE_SIZE}. "
            f"Using IMAGE_SIZE for image preprocessing."
        )

    print("Extracting embeddings...")
    test_embeddings = extract_embeddings(test_paths, model, device, IMAGE_SIZE)
    test_embeddings = scaler.transform(test_embeddings)
    test_distances = np.linalg.norm(test_embeddings - centroid, axis=1)

    # Make predictions
    predictions = []
    predicted_labels = []
    for path, dist in zip(test_paths, test_distances):
        is_drone = bool(dist <= threshold)
        pred_label = "drone" if is_drone else "non_drone"
        predicted_labels.append(pred_label)
        predictions.append(
            {
                "image": str(path),
                "distance": float(dist),
                "threshold": threshold,
                "ground_truth": None,  # Will be filled below
                "prediction": pred_label,
            }
        )

    # Align ground truth with predictions (handling skipped images)
    if len(ground_truth_labels) != len(predicted_labels):
        print(f"[WARN] Ground truth count ({len(ground_truth_labels)}) != predictions count ({len(predicted_labels)})")
        # Try to reconstruct ground truth for successfully processed images
        valid_indices = []
        j = 0
        for i, path in enumerate(test_paths):
            if j < len(predicted_labels):
                valid_indices.append(i)
                j += 1
        ground_truth_labels = [ground_truth_labels[i] for i in valid_indices]

    # Add ground truth to predictions
    for i, pred in enumerate(predictions):
        if i < len(ground_truth_labels):
            pred["ground_truth"] = ground_truth_labels[i]

    # Calculate metrics
    y_true = np.array(ground_truth_labels)
    y_pred = np.array(predicted_labels)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['drone', 'non_drone'])

    drone_count = int(sum(1 for p in predictions if p["prediction"] == "drone"))
    total = len(predictions)
    drone_correct = int(sum(1 for p in predictions if p["prediction"] == "drone" and p["ground_truth"] == "drone"))
    non_drone_correct = int(sum(1 for p in predictions if p["prediction"] == "non_drone" and p["ground_truth"] == "non_drone"))

    summary = {
        "method": "one_class_centroid",
        "embedding_backbone": "resnet18_penultimate",
        "artifact_in": str(artifact_in),
        "test_images": total,
        "threshold_percentile": threshold_percentile,
        "threshold": float(threshold),
        "mean_distance": float(np.mean(test_distances)) if total else None,
        "std_distance": float(np.std(test_distances)) if total else None,
        
        # Predictions summary
        "predicted_drone_count": drone_count,
        "predicted_non_drone_count": total - drone_count,
        "predicted_drone_rate": float(drone_count / max(total, 1)),
        
        # Ground truth summary
        "ground_truth_drone_count": int(sum(1 for l in ground_truth_labels if l == "drone")),
        "ground_truth_non_drone_count": int(sum(1 for l in ground_truth_labels if l == "non_drone")),
        
        # Metrics
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "drone_correct": drone_correct,
        "non_drone_correct": non_drone_correct,
        
        # Per-class detailed metrics
        "drone_precision": float(report.get('drone', {}).get('precision', 0)),
        "drone_recall": float(report.get('drone', {}).get('recall', 0)),
        "drone_f1": float(report.get('drone', {}).get('f1-score', 0)),
        "non_drone_precision": float(report.get('non_drone', {}).get('precision', 0)),
        "non_drone_recall": float(report.get('non_drone', {}).get('recall', 0)),
        "non_drone_f1": float(report.get('non_drone', {}).get('f1-score', 0)),
        
        # Confusion matrix
        "confusion_matrix": {
            "drone_as_drone": int(conf_matrix[0, 0]),
            "drone_as_non_drone": int(conf_matrix[0, 1]),
            "non_drone_as_drone": int(conf_matrix[1, 0]),
            "non_drone_as_non_drone": int(conf_matrix[1, 1]),
        }
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=True)

    print("\n" + "="*60)
    print("TEST EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print("="*60)
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main()
