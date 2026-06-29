from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image)


@dataclass
class TrainResult:
    name: str
    pipeline: Any
    report: dict
    macro_f1: float


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


def select_device(device_name: str) -> torch.device:
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        return torch.device(device_name)
    return torch.device("cpu")


def prepare_binary_paths(
    drone_root: str,
    non_drone_root: str,
    max_drone: int,
    max_non_drone: int,
    seed: int,
) -> tuple[List[Path], List[Path], List[Path], np.ndarray]:
    drone_paths = choose_subset(collect_image_paths(Path(drone_root)), max_count=max_drone, seed=seed)
    non_drone_paths = choose_subset(
        collect_image_paths(Path(non_drone_root)),
        max_count=max_non_drone,
        seed=seed + 1,
    )

    if not drone_paths:
        raise ValueError("No drone images found")
    if not non_drone_paths:
        raise ValueError("No non-drone images found")

    all_paths = drone_paths + non_drone_paths
    labels = np.array([1] * len(drone_paths) + [0] * len(non_drone_paths), dtype=np.int64)
    return drone_paths, non_drone_paths, all_paths, labels


def make_labeled_paths(drone_paths: Sequence[Path], non_drone_paths: Sequence[Path]) -> tuple[List[Path], np.ndarray]:
    all_paths = list(drone_paths) + list(non_drone_paths)
    labels = np.array([1] * len(drone_paths) + [0] * len(non_drone_paths), dtype=np.int64)
    return all_paths, labels


def collect_split_paths(
    drone_root: str,
    non_drone_root: str,
    split_name: str,
    max_drone: int = 0,
    max_non_drone: int = 0,
    seed: int = 42,
    required: bool = True,
) -> tuple[List[Path], List[Path], List[Path], np.ndarray]:
    drone_split_root = Path(drone_root) / split_name
    non_drone_split_root = Path(non_drone_root) / split_name

    if not required and (not drone_split_root.exists() or not non_drone_split_root.exists()):
        return [], [], [], np.array([], dtype=np.int64)

    drone_paths = choose_subset(collect_image_paths(drone_split_root), max_count=max_drone, seed=seed)
    non_drone_paths = choose_subset(
        collect_image_paths(non_drone_split_root),
        max_count=max_non_drone,
        seed=seed + 1,
    )

    if required:
        if not drone_paths:
            raise ValueError(f"No drone images found in split: {drone_split_root}")
        if not non_drone_paths:
            raise ValueError(f"No non-drone images found in split: {non_drone_split_root}")

    all_paths, labels = make_labeled_paths(drone_paths, non_drone_paths)
    return drone_paths, non_drone_paths, all_paths, labels


def print_dataset_summary(
    device: torch.device,
    drone_paths: Sequence[Path],
    non_drone_paths: Sequence[Path],
    extra_lines: Sequence[str] | None = None,
    warn_small_non_drone: bool = True,
) -> None:
    for line in extra_lines or ():
        print(line)
    print(f"Device: {device}")
    print(f"Using drone images: {len(drone_paths)}")
    print(f"Using non-drone images: {len(non_drone_paths)}")
    if warn_small_non_drone and len(non_drone_paths) < 100:
        print("[WARN] Non-drone samples are very few. Default SVM is unweighted; add hard negatives soon.")


def print_split_summary(
    device: torch.device,
    split_counts: Mapping[str, tuple[int, int]],
    extra_lines: Sequence[str] | None = None,
) -> None:
    for line in extra_lines or ():
        print(line)
    print(f"Device: {device}")
    for split_name, (drone_count, non_drone_count) in split_counts.items():
        print(f"{split_name}: drone={drone_count}, non_drone={non_drone_count}, total={drone_count + non_drone_count}")


def extract_embeddings(
    image_paths: Sequence[Path],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray:
    dataset = SpectrogramDataset(
        image_paths,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="Extract embeddings", unit="batch"):
            images = images.to(device, non_blocking=True)
            features = feature_fn(model, images) if feature_fn is not None else model(images)
            embeddings.append(features.flatten(1).detach().cpu().numpy())

    if not embeddings:
        raise ValueError("No image found for embedding extraction")
    return np.concatenate(embeddings, axis=0)


def train_and_eval(
    name: str,
    classifier: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> TrainResult:
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return TrainResult(
        name=name,
        pipeline=pipeline,
        report=report,
        macro_f1=float(report["macro avg"]["f1-score"]),
    )


def train_default_svm(
    embeddings: np.ndarray,
    labels: np.ndarray,
    val_size: float,
    seed: int,
    svm_class_weight: str | dict | None = None,
) -> tuple[List[TrainResult], TrainResult]:
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    x_train, x_val, y_train, y_val = train_test_split(
        embeddings,
        labels,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    results = [
        train_and_eval("svm", SVC(class_weight=svm_class_weight), x_train, y_train, x_val, y_val),
    ]
    best = max(results, key=lambda result: result.macro_f1)
    return results, best


def train_default_svm_with_validation(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    valid_embeddings: np.ndarray,
    valid_labels: np.ndarray,
    svm_class_weight: str | dict | None = None,
) -> tuple[List[TrainResult], TrainResult]:
    from sklearn.svm import SVC

    results = [
        train_and_eval("svm", SVC(class_weight=svm_class_weight), train_embeddings, train_labels, valid_embeddings, valid_labels),
    ]
    best = max(results, key=lambda result: result.macro_f1)
    return results, best


def build_training_summary(
    feature_extractor: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    args: Any,
    results: Sequence[TrainResult],
    best: TrainResult,
    device: torch.device,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    summary = {
        "feature_extractor": feature_extractor,
        "embedding_dim": int(embeddings.shape[1]),
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        "device": str(device),
        "n_total": int(len(labels)),
        "n_drone": int(np.sum(labels == 1)),
        "n_non_drone": int(np.sum(labels == 0)),
        "val_size": float(args.val_size),
        "seed": int(args.seed),
        "models": {
            result.name: {
                "macro_f1": result.macro_f1,
                "drone_precision": float(result.report.get("1", {}).get("precision", 0.0)),
                "drone_recall": float(result.report.get("1", {}).get("recall", 0.0)),
                "drone_f1": float(result.report.get("1", {}).get("f1-score", 0.0)),
                "non_drone_precision": float(result.report.get("0", {}).get("precision", 0.0)),
                "non_drone_recall": float(result.report.get("0", {}).get("recall", 0.0)),
                "non_drone_f1": float(result.report.get("0", {}).get("f1-score", 0.0)),
            }
            for result in results
        },
        "selected_model": best.name,
    }
    if hasattr(args, "image_size"):
        summary["image_size"] = int(args.image_size)
    if extra:
        summary.update(dict(extra))
    return summary


def build_split_training_summary(
    feature_extractor: str,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
    args: Any,
    results: Sequence[TrainResult],
    best: TrainResult,
    device: torch.device,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
    test_labels: np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    test_labels = np.array([], dtype=np.int64) if test_labels is None else test_labels
    summary = {
        "feature_extractor": feature_extractor,
        "embedding_dim": int(train_embeddings.shape[1]),
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        "device": str(device),
        "split_mode": "predefined_train_valid_test",
        "train_split": getattr(args, "train_split", "train"),
        "valid_split": getattr(args, "valid_split", "valid"),
        "test_split": getattr(args, "test_split", "test"),
        "n_train": int(len(train_labels)),
        "n_train_drone": int(np.sum(train_labels == 1)),
        "n_train_non_drone": int(np.sum(train_labels == 0)),
        "n_valid": int(len(valid_labels)),
        "n_valid_drone": int(np.sum(valid_labels == 1)),
        "n_valid_non_drone": int(np.sum(valid_labels == 0)),
        "n_test": int(len(test_labels)),
        "n_test_drone": int(np.sum(test_labels == 1)),
        "n_test_non_drone": int(np.sum(test_labels == 0)),
        "seed": int(args.seed),
        "models": {
            result.name: {
                "macro_f1": result.macro_f1,
                "drone_precision": float(result.report.get("1", {}).get("precision", 0.0)),
                "drone_recall": float(result.report.get("1", {}).get("recall", 0.0)),
                "drone_f1": float(result.report.get("1", {}).get("f1-score", 0.0)),
                "non_drone_precision": float(result.report.get("0", {}).get("precision", 0.0)),
                "non_drone_recall": float(result.report.get("0", {}).get("recall", 0.0)),
                "non_drone_f1": float(result.report.get("0", {}).get("f1-score", 0.0)),
            }
            for result in results
        },
        "selected_model": best.name,
    }
    if hasattr(args, "image_size"):
        summary["image_size"] = int(args.image_size)
    if extra:
        summary.update(dict(extra))
    return summary


def save_training_artifact(best: TrainResult, summary: dict, artifact_out: str) -> Path:
    import joblib

    artifact_path = Path(artifact_out)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "linear_probe_classifier",
        "model_name": best.name,
        "model": best.pipeline,
        "summary": summary,
    }
    joblib.dump(payload, artifact_path)
    return artifact_path


def print_training_summary(summary: dict, artifact_path: Path) -> None:
    print("=" * 80)
    print("Training summary")
    print(summary)
    print("=" * 80)
    print(f"Saved artifact: {artifact_path}")


def run_embedding_linear_probe(
    args: Any,
    feature_extractor: str,
    load_feature_model: Callable[[torch.device], torch.nn.Module],
    extra_lines: Sequence[str] | None = None,
    extra_summary: Mapping[str, Any] | None = None,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
    svm_class_weight: str | dict | None = None,
) -> dict:
    device = select_device(args.device)
    drone_paths, non_drone_paths, all_paths, labels = prepare_binary_paths(
        args.drone_root,
        args.non_drone_root,
        max_drone=args.max_drone,
        max_non_drone=args.max_non_drone,
        seed=args.seed,
    )
    print_dataset_summary(device, drone_paths, non_drone_paths, extra_lines=extra_lines)

    feature_model = load_feature_model(device)
    embeddings = extract_embeddings(
        all_paths,
        feature_model,
        device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_mean=image_mean,
        image_std=image_std,
        feature_fn=feature_fn,
    )

    results, best = train_default_svm(
        embeddings,
        labels,
        val_size=args.val_size,
        seed=args.seed,
        svm_class_weight=svm_class_weight,
    )
    summary = build_training_summary(
        feature_extractor,
        embeddings,
        labels,
        args,
        results,
        best,
        device,
        image_mean=image_mean,
        image_std=image_std,
        extra=extra_summary,
    )
    artifact_path = save_training_artifact(best, summary, args.artifact_out)
    print_training_summary(summary, artifact_path)
    return summary


def run_split_embedding_linear_probe(
    args: Any,
    feature_extractor: str,
    load_feature_model: Callable[[torch.device], torch.nn.Module],
    extra_lines: Sequence[str] | None = None,
    extra_summary: Mapping[str, Any] | None = None,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
    svm_class_weight: str | dict | None = None,
) -> dict:
    device = select_device(args.device)

    train_drone, train_non_drone, train_paths, train_labels = collect_split_paths(
        args.drone_root,
        args.non_drone_root,
        split_name=args.train_split,
        max_drone=args.max_drone,
        max_non_drone=args.max_non_drone,
        seed=args.seed,
        required=True,
    )
    valid_drone, valid_non_drone, valid_paths, valid_labels = collect_split_paths(
        args.drone_root,
        args.non_drone_root,
        split_name=args.valid_split,
        seed=args.seed + 10,
        required=True,
    )
    test_drone, test_non_drone, _, test_labels = collect_split_paths(
        args.drone_root,
        args.non_drone_root,
        split_name=args.test_split,
        seed=args.seed + 20,
        required=False,
    )

    print_split_summary(
        device,
        {
            args.train_split: (len(train_drone), len(train_non_drone)),
            args.valid_split: (len(valid_drone), len(valid_non_drone)),
            args.test_split: (len(test_drone), len(test_non_drone)),
        },
        extra_lines=extra_lines,
    )

    feature_model = load_feature_model(device)
    train_embeddings = extract_embeddings(
        train_paths,
        feature_model,
        device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_mean=image_mean,
        image_std=image_std,
        feature_fn=feature_fn,
    )
    valid_embeddings = extract_embeddings(
        valid_paths,
        feature_model,
        device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_mean=image_mean,
        image_std=image_std,
        feature_fn=feature_fn,
    )

    results, best = train_default_svm_with_validation(
        train_embeddings,
        train_labels,
        valid_embeddings,
        valid_labels,
        svm_class_weight=svm_class_weight,
    )
    summary = build_split_training_summary(
        feature_extractor,
        train_embeddings,
        train_labels,
        valid_labels,
        args,
        results,
        best,
        device,
        image_mean=image_mean,
        image_std=image_std,
        test_labels=test_labels,
        extra=extra_summary,
    )
    artifact_path = save_training_artifact(best, summary, args.artifact_out)
    print_training_summary(summary, artifact_path)
    return summary
