from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from utils.benchmark import Classify_Model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly instead of using command-line arguments.
METHOD = "oneclass"  # one of: all, classifier, oneclass
TEST_ROOT = "C:/Users/DiepHM/Documents/data/spectrograms/2toanbin"

# One-class settings
ARTIFACT_IN = "oneclass_artifact.joblib"
POSITIVE_LABEL = "drone"
POSITIVE_ONLY = True
# Evaluate extra threshold percentiles computed from test distances.
# Example: [90.0, 95.0, 97.0]. Keep [] to disable.
SWEEP_THRESHOLD_PERCENTILES: List[float] = [90.0, 95.0, 97.0]

# Classifier settings (used when METHOD is classifier or all)
CFG_PATH = ""
WEIGHTS_PATH = ""
BATCH_SIZE = 32

# Shared settings
IMAGE_SIZE = 224
DEVICE = "cuda:0"
OUTPUT_JSON = "benchmark_results.json"


def collect_image_paths(root: str | Path) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")

    image_paths: List[Path] = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            image_paths.append(path)

    return sorted(image_paths)


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_feature_model(device: torch.device) -> torch.nn.Module:
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        base = models.resnet18(weights=None)

    # Use global pooled penultimate features (512-d) for one-class centroid.
    model = nn.Sequential(*list(base.children())[:-1])
    model.eval()
    model.to(device)
    return model


def build_dino_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_classifier_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def get_model_class_names(cfg: Dict) -> List[str]:
    class_names = cfg.get("class_names", {})
    if isinstance(class_names, dict):
        return [name for name, _ in sorted(class_names.items(), key=lambda item: item[1])]
    if isinstance(class_names, Sequence):
        return list(class_names)
    raise TypeError("cfg['class_names'] must be a dict or sequence")


def extract_feature_embeddings(
    image_paths: Sequence[Path],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
) -> np.ndarray:
    transform = build_dino_transform(image_size=image_size)
    embeddings: List[np.ndarray] = []

    for path in tqdm(image_paths, desc="Extract embeddings", unit="img"):
        image = load_rgb_image(path)
        tensor = cast(torch.Tensor, transform(image))
        x = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).flatten(1).squeeze(0).detach().cpu().numpy()
        embeddings.append(feat)

    if not embeddings:
        raise ValueError("No images found for embedding extraction")

    return np.asarray(embeddings)


def load_oneclass_artifact(path: str | Path) -> tuple[StandardScaler, np.ndarray, float, float, int]:
    payload = joblib.load(path)
    scaler = cast(StandardScaler, payload["scaler"])
    centroid = np.asarray(payload["centroid"])
    threshold = float(payload["threshold"])
    threshold_percentile = float(payload.get("threshold_percentile", 95.0))
    image_size = int(payload.get("image_size", 224))
    return scaler, centroid, threshold, threshold_percentile, image_size


def evaluate_one_class(
    artifact_in: str,
    test_root: str,
    positive_label: str,
    device: torch.device,
    image_size: int = 224,
    positive_only: bool = False,
    sweep_threshold_percentiles: Sequence[float] | None = None,
) -> Dict[str, object]:
    test_root_path = Path(test_root)

    feature_model = load_feature_model(device)
    scaler, centroid, threshold, threshold_percentile, saved_image_size = load_oneclass_artifact(artifact_in)
    if saved_image_size != image_size:
        print(
            f"[WARN] artifact image_size={saved_image_size} differs from image_size={image_size}. "
            f"Using current image_size for preprocessing."
        )

    test_paths: List[Path]
    y_true: np.ndarray
    if positive_only:
        test_paths = collect_image_paths(test_root_path)
        y_true = np.ones(len(test_paths), dtype=int)
    else:
        test_dataset = datasets.ImageFolder(root=str(test_root_path), transform=None)
        test_class_names = test_dataset.classes
        if positive_label not in test_class_names:
            raise ValueError(
                f"positive_label='{positive_label}' not found in test classes: {test_class_names}"
            )
        test_paths = [Path(path) for path, _ in test_dataset.samples]
        y_true = np.array([1 if test_dataset.classes[label] == positive_label else 0 for _, label in test_dataset.samples])

    test_embeddings = extract_feature_embeddings(test_paths, feature_model, device, image_size=image_size)
    test_embeddings = scaler.transform(test_embeddings)
    test_distances = np.linalg.norm(test_embeddings - centroid, axis=1)
    y_pred = (test_distances <= threshold).astype(int)

    result: Dict[str, object] = {
        "method": "one_class_centroid",
        "embedding_backbone": "resnet18_penultimate",
        "eval_mode": "positive_only" if positive_only else "binary",
        "positive_label": positive_label,
        "artifact_in": str(artifact_in),
        "test_images": len(test_paths),
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
        "mean_distance": float(np.mean(test_distances)),
        "std_distance": float(np.std(test_distances)),
        "min_distance": float(np.min(test_distances)),
        "max_distance": float(np.max(test_distances)),
    }

    if positive_only:
        accept_rate = float(np.mean(y_pred == 1))
        result["accept_rate"] = accept_rate
        result["reject_rate"] = float(1.0 - accept_rate)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
        result["precision"] = float(precision)
        result["recall"] = float(recall)
        result["f1"] = float(f1)
        result["false_alarm_rate"] = float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        if len(np.unique(y_true)) > 1:
            result["roc_auc"] = float(roc_auc_score(y_true, -test_distances))

    sweep_percentiles = list(sweep_threshold_percentiles or [])
    if sweep_percentiles:
        sweep_rows: List[Dict[str, object]] = []
        for pct in sweep_percentiles:
            cur_threshold = float(np.percentile(test_distances, pct))
            cur_pred = (test_distances <= cur_threshold).astype(int)
            row: Dict[str, object] = {
                "threshold_percentile": float(pct),
                "threshold": cur_threshold,
            }

            if positive_only:
                accept_rate = float(np.mean(cur_pred == 1))
                row["accept_rate"] = accept_rate
                row["reject_rate"] = float(1.0 - accept_rate)
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true,
                    cur_pred,
                    average="binary",
                    zero_division=0,
                )
                row["accuracy"] = float(accuracy_score(y_true, cur_pred))
                row["precision"] = float(precision)
                row["recall"] = float(recall)
                row["f1"] = float(f1)
                row["false_alarm_rate"] = float(
                    ((cur_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
                )

            sweep_rows.append(row)

        result["threshold_sweep_on_test_distances"] = sweep_rows

    return result


def evaluate_classifier(
    cfg_path: str,
    weight_path: str,
    test_root: str,
    batch_size: int,
    device: str,
    image_size: int = 224,
) -> Dict[str, object]:
    classifier = Classify_Model(cfg=cfg_path, weight_path=weight_path, save=False)
    classifier.model.eval()

    dataset = datasets.ImageFolder(
        root=test_root,
        transform=build_classifier_transform(image_size=image_size),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_class_names = get_model_class_names(classifier.cfg)
    dataset_class_names = dataset.classes
    class_to_model_idx: Dict[int, int] = {}
    for idx, class_name in enumerate(dataset_class_names):
        if class_name not in model_class_names:
            raise ValueError(
                f"Test class '{class_name}' is not present in cfg class_names: {model_class_names}"
            )
        class_to_model_idx[idx] = model_class_names.index(class_name)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(classifier.device)
            outputs = classifier.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1).cpu().numpy().tolist()
            probs = probabilities.cpu().numpy().tolist()
            true_labels = [class_to_model_idx[int(label)] for label in labels.cpu().numpy().tolist()]

            y_true.extend(true_labels)
            y_pred.extend(preds)
            y_prob.extend(probs)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )

    result: Dict[str, object] = {
        "method": "classifier",
        "test_images": len(dataset),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(model_class_names)))).tolist(),
        "classes": model_class_names,
    }

    if len(np.unique(y_true_arr)) > 1:
        if y_prob_arr.shape[1] == 2:
            result["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr[:, 1]))
        else:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", average="macro")
            )

    return result


def main() -> None:
    if METHOD not in {"all", "classifier", "oneclass"}:
        raise ValueError("METHOD must be one of: all, classifier, oneclass")

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")
    results: Dict[str, object] = {"test_root": TEST_ROOT}

    if METHOD in {"all", "classifier"}:
        if not CFG_PATH or not WEIGHTS_PATH:
            raise ValueError("CFG_PATH and WEIGHTS_PATH are required for classifier benchmarking")
        results["classifier"] = evaluate_classifier(
            cfg_path=CFG_PATH,
            weight_path=WEIGHTS_PATH,
            test_root=TEST_ROOT,
            batch_size=BATCH_SIZE,
            device=str(device),
            image_size=IMAGE_SIZE,
        )

    if METHOD in {"all", "oneclass"}:
        if not ARTIFACT_IN:
            raise ValueError("ARTIFACT_IN is required for one-class benchmarking")
        results["oneclass"] = evaluate_one_class(
            artifact_in=ARTIFACT_IN,
            test_root=TEST_ROOT,
            positive_label=POSITIVE_LABEL,
            device=device,
            image_size=IMAGE_SIZE,
            positive_only=POSITIVE_ONLY,
            sweep_threshold_percentiles=SWEEP_THRESHOLD_PERCENTILES,
        )

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    print(json.dumps(results, indent=2, ensure_ascii=True))
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()