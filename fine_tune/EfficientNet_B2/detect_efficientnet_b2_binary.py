from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly.
CHECKPOINT_IN = "fine_tune/EfficientNet_B2/efficientnet_b2_binary_runs/balanced_efficientnet_b2_binary_new.pt"
SOURCE_DIR = r"/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_dataset_drone_chuan_full_non_done/non_drone/test/env/non_drone_indoor_lab_spectrograms"
OUTPUT_JSON = "fine_tune/EfficientNet_B2/report/Tu_thu/results.json"
OUTPUT_CHART = "fine_tune/EfficientNet_B2/report/Tu_thu/results_chart.png"
IMAGE_SIZE = 260
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_WORKERS = 4
USE_PATCH_INFERENCE = False
PATCH_SIZE = 96
PATCH_STRIDE = 96
PATCH_AGGREGATION = "topk_mean"  # Options: max, topk_mean
PATCH_TOP_K = 3
RUN_FULL_IMAGE_IN_PATCH_MODE = False
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGE_MODE = "grayscale_rgb"
DEFAULT_IMAGE_PREPROCESS = "legacy_imagenet"
DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.0


class PercentileNormalizeTensor:
    def __init__(self, low: float, high: float, eps: float = 1e-6) -> None:
        self.low = low / 100.0
        self.high = high / 100.0
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        reference = x[:1].flatten()
        lo = torch.quantile(reference, self.low)
        hi = torch.quantile(reference, self.high)
        scale = torch.clamp(hi - lo, min=self.eps)
        return ((x - lo) / scale).clamp(0.0, 1.0)


def make_eval_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    image_preprocess: str,
    percentile_low: float,
    percentile_high: float,
) -> transforms.Compose:
    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if image_preprocess == "percentile":
        steps.append(PercentileNormalizeTensor(low=percentile_low, high=percentile_high))
    elif image_preprocess != "legacy_imagenet":
        raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")
    steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        image_preprocess: str,
        percentile_low: float,
        percentile_high: float,
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = make_eval_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            image_preprocess=image_preprocess,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")
        return self.transform(image)


class PatchSpectrogramDataset(Dataset):
    def __init__(
        self,
        patch_items: Sequence[tuple[Path, int, tuple[int, int, int, int]]],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        image_preprocess: str,
        percentile_low: float,
        percentile_high: float,
    ) -> None:
        self.patch_items = list(patch_items)
        self.transform = make_eval_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            image_preprocess=image_preprocess,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )

    def __len__(self) -> int:
        return len(self.patch_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, tuple[int, int, int, int]]:
        path, image_idx, box = self.patch_items[idx]
        with Image.open(path) as img:
            image = img.convert("RGB").crop(box)
        return self.transform(image), image_idx, box


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def sliding_window_boxes(width: int, height: int, patch_size: int, stride: int) -> List[tuple[int, int, int, int]]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("PATCH_SIZE and PATCH_STRIDE must be positive")
    patch_w = min(patch_size, width)
    patch_h = min(patch_size, height)

    def starts(length: int, patch_length: int) -> List[int]:
        if length <= patch_length:
            return [0]
        values = list(range(0, length - patch_length + 1, stride))
        last = length - patch_length
        if values[-1] != last:
            values.append(last)
        return values

    return [
        (left, top, left + patch_w, top + patch_h)
        for top in starts(height, patch_h)
        for left in starts(width, patch_w)
    ]


def build_patch_items(
    image_paths: Sequence[Path],
    patch_size: int,
    stride: int,
) -> tuple[List[tuple[Path, int, tuple[int, int, int, int]]], List[int]]:
    patch_items: List[tuple[Path, int, tuple[int, int, int, int]]] = []
    patch_counts: List[int] = []
    for image_idx, path in enumerate(image_paths):
        with Image.open(path) as img:
            width, height = img.size
        boxes = sliding_window_boxes(width, height, patch_size=patch_size, stride=stride)
        patch_counts.append(len(boxes))
        patch_items.extend((path, image_idx, box) for box in boxes)
    return patch_items, patch_counts


def build_model(checkpoint: dict, device: torch.device) -> tuple[nn.Module, List[str], dict, int, List[float], List[float], str, str, float, float]:
    class_names = checkpoint.get("class_names", ["non_drone", "drone"])
    class_to_idx = checkpoint.get("class_to_idx", {name: idx for idx, name in enumerate(class_names)})
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    mean = list(checkpoint.get("image_mean", DEFAULT_MEAN))
    std = list(checkpoint.get("image_std", DEFAULT_STD))
    backbone_name = checkpoint.get("backbone_name", "efficientnet_b2")
    image_preprocess = checkpoint.get("image_preprocess", DEFAULT_IMAGE_PREPROCESS)
    percentile_low = float(checkpoint.get("percentile_low", DEFAULT_PERCENTILE_LOW))
    percentile_high = float(checkpoint.get("percentile_high", DEFAULT_PERCENTILE_HIGH))

    if backbone_name != "efficientnet_b2":
        raise ValueError(f"Unsupported backbone_name in checkpoint: {backbone_name}")

    try:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.efficientnet_b2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, class_names, class_to_idx, image_size, mean, std, backbone_name, image_preprocess, percentile_low, percentile_high


@torch.no_grad()
def infer(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    image_preprocess: str,
    percentile_low: float,
    percentile_high: float,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SpectrogramDataset(
        image_paths,
        image_size=image_size,
        mean=mean,
        std=std,
        image_preprocess=image_preprocess,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    probs_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []

    model.eval()
    for x in tqdm(loader, desc="Infer", unit="batch"):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        probs_all.append(probs.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())

    if not probs_all:
        raise ValueError("No image found for inference")

    return np.concatenate(probs_all, axis=0), np.concatenate(pred_all, axis=0)


def aggregate_patch_scores(scores: Sequence[float], method: str, top_k: int) -> float:
    if not scores:
        return float("nan")
    values = np.asarray(scores, dtype=np.float32)
    if method == "max":
        return float(values.max())
    if method == "topk_mean":
        k = min(max(1, top_k), len(values))
        return float(np.sort(values)[-k:].mean())
    raise ValueError(f"Unsupported PATCH_AGGREGATION: {method}")


@torch.no_grad()
def infer_patches(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    image_preprocess: str,
    percentile_low: float,
    percentile_high: float,
    drone_idx: int,
    batch_size: int,
    num_workers: int,
    patch_size: int,
    stride: int,
    aggregation: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, List[dict]]:
    if not image_paths:
        raise ValueError("No image found for patch inference")

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    q_low = percentile_low / 100.0
    q_high = percentile_high / 100.0

    scores: List[float] = []
    pred: List[int] = []
    patch_summaries: List[dict] = []
    non_drone_idx = 0 if drone_idx != 0 else 1

    model.eval()
    for path in tqdm(image_paths, desc="Infer patch images", unit="image"):
        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(gray).unsqueeze(0)
        _, height, width = image_tensor.shape
        boxes = sliding_window_boxes(width, height, patch_size=patch_size, stride=stride)
        patch_details: List[dict] = []
        patch_scores: List[float] = []

        for start in range(0, len(boxes), batch_size):
            batch_boxes = boxes[start:start + batch_size]
            crops = [
                image_tensor[:, top:bottom, left:right]
                for left, top, right, bottom in batch_boxes
            ]
            x = torch.stack(crops, dim=0).to(device, non_blocking=True)
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)

            if image_preprocess == "percentile":
                flat = x.flatten(1)
                lo = torch.quantile(flat, q_low, dim=1, keepdim=True).view(-1, 1, 1, 1)
                hi = torch.quantile(flat, q_high, dim=1, keepdim=True).view(-1, 1, 1, 1)
                x = ((x - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(0.0, 1.0)
            elif image_preprocess != "legacy_imagenet":
                raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")

            x = x.repeat(1, 3, 1, 1)
            x = (x - mean_t) / std_t

            probs = torch.softmax(model(x), dim=1)[:, drone_idx].detach().cpu().numpy()
            for box, score in zip(batch_boxes, probs):
                score_float = float(score)
                patch_scores.append(score_float)
                patch_details.append({"box": list(box), "drone_score": score_float})

        image_score = aggregate_patch_scores(patch_scores, method=aggregation, top_k=top_k)
        scores.append(image_score)
        pred.append(drone_idx if image_score >= 0.5 else non_drone_idx)
        top_patches = sorted(patch_details, key=lambda item: item["drone_score"], reverse=True)[: max(1, top_k)]
        patch_summaries.append(
            {
                "patch_count": int(len(boxes)),
                "top_patches": top_patches,
            }
        )

    return np.asarray(scores, dtype=np.float32), np.asarray(pred, dtype=np.int64), patch_summaries


def visualize_detection_results(
    scores: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    drone_count: int,
    total: int,
    model_name: str | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#f7f9fc")

    drone_mask = pred == 1
    non_drone_mask = pred == 0

    ax1 = axes[0, 0]
    if np.any(drone_mask):
        ax1.hist(scores[drone_mask], bins=30, alpha=0.75, color="#2b8a6e", edgecolor="black", label="Predicted drone")
    if np.any(non_drone_mask):
        ax1.hist(scores[non_drone_mask], bins=30, alpha=0.75, color="#c94949", edgecolor="black", label="Predicted non-drone")
    ax1.axvline(0.5, color="#111827", linestyle="--", linewidth=2, label="Decision threshold: 0.5")
    ax1.set_xlabel("Drone score")
    ax1.set_ylabel("Number of images")
    ax1.set_title("Score Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[0, 1]
    indices = np.arange(total)
    if np.any(drone_mask):
        ax2.scatter(indices[drone_mask], scores[drone_mask], s=18, color="#2b8a6e", alpha=0.75, label="Predicted drone")
    if np.any(non_drone_mask):
        ax2.scatter(indices[non_drone_mask], scores[non_drone_mask], s=18, color="#c94949", alpha=0.75, label="Predicted non-drone")
    ax2.axhline(0.5, color="#111827", linestyle="--", linewidth=2)
    ax2.set_xlabel("Image index")
    ax2.set_ylabel("Drone score")
    ax2.set_title("Score by Image Order", fontweight="bold")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[1, 0]
    counts = [drone_count, total - drone_count]
    labels = ["Drone", "Non-drone"]
    colors = ["#2b8a6e", "#c94949"]
    ax3.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax3.set_title("Prediction Split", fontweight="bold")

    ax4 = axes[1, 1]
    metric_names = ["Source images", "Drone", "Non-drone", "Drone rate"]
    metric_values = [total, drone_count, total - drone_count, drone_count / max(total, 1)]
    display_values = [str(total), str(drone_count), str(total - drone_count), f"{100 * metric_values[-1]:.1f}%"]
    bar_values = [total, drone_count, total - drone_count, total * metric_values[-1]]
    bars = ax4.bar(metric_names, bar_values, color=["#357ABD", "#2b8a6e", "#c94949", "#d9902f"])
    ax4.set_title("Detection Summary", fontweight="bold")
    ax4.set_ylabel("Count-scaled value")
    ax4.grid(axis="y", alpha=0.3)

    path = Path(SOURCE_DIR)
    last_two_parts = path.parts[-2:]
    drone_name = "/".join(last_two_parts)

    for bar, text in zip(bars, display_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.015, 0.5),
            text,
            ha="center",
            fontweight="bold",
        )

    fig.suptitle(
        f"{drone_name}\n\n"
        f"Fine-tuned EfficientNet-B2 Detection Results | Model: {model_name or 'unknown'} | "
        f"Drone: {drone_count}/{total} ({100 * drone_count / max(total, 1):.1f}%)\n",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def main() -> None:
    checkpoint_path = Path(CHECKPOINT_IN)
    source_root = Path(SOURCE_DIR)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    (
        model,
        class_names,
        class_to_idx,
        image_size,
        mean,
        std,
        backbone_name,
        image_preprocess,
        percentile_low,
        percentile_high,
    ) = build_model(checkpoint, device)

    if "drone" not in class_to_idx:
        raise ValueError("Checkpoint class_to_idx does not contain 'drone'")
    drone_idx = int(class_to_idx["drone"])

    source_paths = collect_image_paths(source_root)
    if USE_PATCH_INFERENCE:
        full_scores = None
        full_pred = None
        if RUN_FULL_IMAGE_IN_PATCH_MODE:
            full_probs, full_pred = infer(
                image_paths=source_paths,
                model=model,
                device=device,
                image_size=image_size,
                mean=mean,
                std=std,
                image_preprocess=image_preprocess,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            full_scores = full_probs[:, drone_idx]

        scores, pred, patch_summaries = infer_patches(
            image_paths=source_paths,
            model=model,
            device=device,
            image_size=image_size,
            mean=mean,
            std=std,
            image_preprocess=image_preprocess,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            drone_idx=drone_idx,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            patch_size=PATCH_SIZE,
            stride=PATCH_STRIDE,
            aggregation=PATCH_AGGREGATION,
            top_k=PATCH_TOP_K,
        )
    else:
        full_probs, full_pred = infer(
            image_paths=source_paths,
            model=model,
            device=device,
            image_size=image_size,
            mean=mean,
            std=std,
            image_preprocess=image_preprocess,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        full_scores = full_probs[:, drone_idx]
        scores = full_scores.astype(np.float32)
        pred = full_pred.astype(int)
        patch_summaries = [{"patch_count": None, "top_patches": []} for _ in source_paths]

    pred = pred.astype(int)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    predictions = []
    for idx, (path, y_hat, score, patch_summary) in enumerate(zip(source_paths, pred, scores, patch_summaries)):
        label_name = idx_to_class.get(int(y_hat), str(int(y_hat)))
        predictions.append(
            {
                "image": str(path),
                "prediction": label_name,
                "drone_score": float(score),
                "full_image_drone_score": None if full_scores is None else float(full_scores[idx]),
                **patch_summary,
            }
        )

    drone_count = int(np.sum(pred == drone_idx))
    total = int(len(pred))
    non_drone_count = total - drone_count

    summary = {
        "method": "efficientnet_b2_finetuned_classifier",
        "model_name": checkpoint.get("model_name", "efficientnet_b2_binary_finetuned"),
        "backbone_name": backbone_name,
        "checkpoint_in": str(checkpoint_path),
        "source_images": total,
        "device": str(device),
        "inference_mode": "patch" if USE_PATCH_INFERENCE else "full_image",
        "run_full_image_in_patch_mode": RUN_FULL_IMAGE_IN_PATCH_MODE if USE_PATCH_INFERENCE else True,
        "full_image_score_mean": float(np.mean(full_scores)) if full_scores is not None and total else None,
        "full_image_score_std": float(np.std(full_scores)) if full_scores is not None and total else None,
        "patch_size": PATCH_SIZE if USE_PATCH_INFERENCE else None,
        "patch_stride": PATCH_STRIDE if USE_PATCH_INFERENCE else None,
        "patch_aggregation": PATCH_AGGREGATION if USE_PATCH_INFERENCE else None,
        "patch_top_k": PATCH_TOP_K if USE_PATCH_INFERENCE else None,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": non_drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "score_mean": float(np.mean(scores)) if total else None,
        "score_std": float(np.std(scores)) if total else None,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "image_mode": checkpoint.get("image_mode", IMAGE_MODE),
        "image_preprocess": image_preprocess,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "best_valid_macro_f1": checkpoint.get("best_valid_macro_f1"),
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_path}")

    chart_path = Path(OUTPUT_CHART)
    visualize_detection_results(
        scores=scores,
        pred=(pred == drone_idx).astype(np.int64),
        output_path=chart_path,
        drone_count=drone_count,
        total=total,
        model_name=Path(CHECKPOINT_IN).stem,
    )


if __name__ == "__main__":
    main()
