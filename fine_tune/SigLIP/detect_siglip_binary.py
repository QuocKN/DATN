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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor, SiglipVisionModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency 'transformers'. Install it first: pip install transformers") from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly.
CHECKPOINT_IN = "fine_tune/SigLIP/siglip_binary_runs/balanced_siglip_binary.pt"
SOURCE_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/spectrograms"
OUTPUT_JSON = "fine_tune/SigLIP/report/Tu_thu/2toan_results.json"
OUTPUT_CHART = "fine_tune/SigLIP/report/Tu_thu/2toan_results_chart.png"
IMAGE_SIZE = 224
DEVICE = "cuda:0"
BATCH_SIZE = 64
NUM_WORKERS = 4


class SigLIPClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        features = outputs.pooler_output
        return self.head(features)


class SpectrogramDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_model(checkpoint: dict, device: torch.device) -> tuple[nn.Module, List[str], dict, int, List[float], List[float], str]:
    siglip_model = checkpoint.get("siglip_model", "google/siglip-base-patch16-224")
    class_names = checkpoint.get("class_names", ["non_drone", "drone"])
    class_to_idx = checkpoint.get("class_to_idx", {name: idx for idx, name in enumerate(class_names)})
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))

    image_processor = AutoImageProcessor.from_pretrained(siglip_model)
    mean = list(image_processor.image_mean)
    std = list(image_processor.image_std)

    backbone = SiglipVisionModel.from_pretrained(siglip_model).to(device)
    feature_dim = int(backbone.config.hidden_size)
    model = SigLIPClassifier(backbone=backbone, feature_dim=feature_dim, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return model, class_names, class_to_idx, image_size, mean, std, siglip_model


@torch.no_grad()
def infer(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SpectrogramDataset(image_paths, image_size=image_size, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total * 0.015, 0.5), text, ha="center", fontweight="bold")

    fig.suptitle(
        f"{drone_name}\n\n"
        f"Fine-tuned SigLIP Detection Results | Model: {model_name or 'unknown'} | "
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, class_names, class_to_idx, image_size, mean, std, siglip_model = build_model(checkpoint, device)

    if "drone" not in class_to_idx:
        raise ValueError("Checkpoint class_to_idx does not contain 'drone'")
    drone_idx = int(class_to_idx["drone"])

    source_paths = collect_image_paths(source_root)
    probs, pred = infer(
        image_paths=source_paths,
        model=model,
        device=device,
        image_size=image_size,
        mean=mean,
        std=std,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    scores = probs[:, drone_idx]
    pred = pred.astype(int)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    predictions = []
    for path, y_hat, score in zip(source_paths, pred, scores):
        label_name = idx_to_class.get(int(y_hat), str(int(y_hat)))
        predictions.append(
            {
                "image": str(path),
                "prediction": label_name,
                "drone_score": float(score),
            }
        )

    drone_count = int(np.sum(pred == drone_idx))
    total = int(len(pred))
    non_drone_count = total - drone_count

    summary = {
        "method": "siglip_finetuned_classifier",
        "model_name": checkpoint.get("model_name", "siglip_binary_finetuned"),
        "siglip_model": siglip_model,
        "checkpoint_in": str(checkpoint_path),
        "source_images": total,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": non_drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "score_mean": float(np.mean(scores)) if total else None,
        "score_std": float(np.std(scores)) if total else None,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
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
