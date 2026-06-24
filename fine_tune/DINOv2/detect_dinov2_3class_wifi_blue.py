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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

# Edit these values directly.
CHECKPOINT_IN = "fine_tune/DINOv2/dinov2_binary_runs/balanced_dinov2_vits14_3class_wifi_blue.pt"
SOURCE_DIR =r"G:\DATN_DATA\RF\Tu_thu\signal_spectrograms_v1"
OUTPUT_JSON = "fine_tune/DINOv2/report/test/results_3class_wifi_blue.json"
OUTPUT_CHART = "fine_tune/DINOv2/report/test/results_3class_wifi_blue_chart.png"
IMAGE_SIZE = 224
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_WORKERS = 4
DEFAULT_CLASS_NAMES = ["non_drone", "drone", "wifi_blue"]


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class SpectrogramDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: Sequence[Path], image_size: int) -> None:
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_dinov2_backbone(model_name: str, device: torch.device) -> nn.Module:
    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone.to(device)
    return backbone


@torch.no_grad()
def infer_feature_dim(backbone: nn.Module, image_size: int, device: torch.device) -> int:
    backbone.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = backbone(dummy)
    if features.ndim != 2:
        features = features.flatten(1)
    return int(features.shape[1])


def build_model(checkpoint: dict, device: torch.device) -> tuple[nn.Module, List[str], dict, int]:
    dino_model = checkpoint["dino_model"]
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    class_to_idx = checkpoint.get("class_to_idx", {name: idx for idx, name in enumerate(class_names)})
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))

    backbone = load_dinov2_backbone(dino_model, device)
    feature_dim = infer_feature_dim(backbone, image_size, device)
    model = DINOv2Classifier(backbone=backbone, feature_dim=feature_dim, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, class_names, class_to_idx, image_size


@torch.no_grad()
def infer(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SpectrogramDataset(image_paths, image_size=image_size)
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
    probs: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    class_names: Sequence[str],
    class_counts: dict[str, int],
    model_name: str | None,
) -> None:
    total = int(len(pred))
    colors = {
        "non_drone": "#c94949",
        "drone": "#2b8a6e",
        "wifi_blue": "#357ABD",
    }
    fallback_colors = ["#c94949", "#2b8a6e", "#357ABD", "#d9902f", "#7a5195"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#f7f9fc")

    ax1 = axes[0, 0]
    for idx, class_name in enumerate(class_names):
        mask = pred == idx
        if np.any(mask):
            color = colors.get(class_name, fallback_colors[idx % len(fallback_colors)])
            ax1.hist(
                probs[mask, idx],
                bins=30,
                alpha=0.65,
                color=color,
                edgecolor="black",
                label=f"Predicted {class_name}",
            )
    ax1.set_xlabel("Predicted class probability")
    ax1.set_ylabel("Number of images")
    ax1.set_title("Confidence Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[0, 1]
    indices = np.arange(total)
    confidence = probs[np.arange(total), pred] if total else np.array([])
    for idx, class_name in enumerate(class_names):
        mask = pred == idx
        if np.any(mask):
            color = colors.get(class_name, fallback_colors[idx % len(fallback_colors)])
            ax2.scatter(indices[mask], confidence[mask], s=18, color=color, alpha=0.75, label=f"Predicted {class_name}")
    ax2.set_xlabel("Image index")
    ax2.set_ylabel("Prediction confidence")
    ax2.set_title("Confidence by Image Order", fontweight="bold")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[1, 0]
    counts = [class_counts.get(name, 0) for name in class_names]
    labels = [name.replace("_", " ").title() for name in class_names]
    pie_colors = [colors.get(name, fallback_colors[idx % len(fallback_colors)]) for idx, name in enumerate(class_names)]
    ax3.pie(
        counts,
        labels=labels,
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax3.set_title("Prediction Split", fontweight="bold")

    ax4 = axes[1, 1]
    metric_names = ["Source images"] + [name.replace("_", " ").title() for name in class_names]
    display_values = [str(total)] + [str(class_counts.get(name, 0)) for name in class_names]
    bar_values = [total] + [class_counts.get(name, 0) for name in class_names]
    bar_colors = ["#111827"] + pie_colors
    bars = ax4.bar(metric_names, bar_values, color=bar_colors)
    ax4.set_title("Detection Summary", fontweight="bold")
    ax4.set_ylabel("Count")
    ax4.grid(axis="y", alpha=0.3)
    ax4.tick_params(axis="x", rotation=15)

    path = Path(SOURCE_DIR)
    last_two_parts = path.parts[-2:]
    source_name = "/".join(last_two_parts)

    for bar, text in zip(bars, display_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total * 0.015, 0.5), text, ha="center", fontweight="bold")
    fig.suptitle(
        f"{source_name}\n\n"
        f"Fine-tuned DINOv2 3-Class Detection Results | Model: {model_name or 'unknown'} | "
        f"Images: {total}\n",
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
    model, class_names, class_to_idx, image_size = build_model(checkpoint, device)

    for class_name in class_names:
        if class_name not in class_to_idx:
            raise ValueError(f"Checkpoint class_to_idx does not contain {class_name!r}")

    source_paths = collect_image_paths(source_root)
    probs, pred = infer(
        image_paths=source_paths,
        model=model,
        device=device,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    pred = pred.astype(int)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    predictions = []
    for path, y_hat, prob_row in zip(source_paths, pred, probs):
        label_name = idx_to_class.get(int(y_hat), str(int(y_hat)))
        predictions.append(
            {
                "image": str(path),
                "prediction": label_name,
                "confidence": float(prob_row[int(y_hat)]),
                "scores": {class_name: float(prob_row[int(class_to_idx[class_name])]) for class_name in class_names},
            }
        )

    total = int(len(pred))
    class_counts = {
        class_name: int(np.sum(pred == int(class_to_idx[class_name])))
        for class_name in class_names
    }
    class_rates = {
        class_name: float(class_counts[class_name] / max(total, 1))
        for class_name in class_names
    }
    confidence = probs[np.arange(total), pred] if total else np.array([])

    summary = {
        "method": "dinov2_3class_wifi_blue_finetuned_classifier",
        "model_name": checkpoint.get("model_name", "dinov2_3class_wifi_blue_finetuned"),
        "checkpoint_in": str(checkpoint_path),
        "source_images": total,
        "class_counts": class_counts,
        "class_rates": class_rates,
        "confidence_mean": float(np.mean(confidence)) if total else None,
        "confidence_std": float(np.std(confidence)) if total else None,
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
        probs=probs,
        pred=pred,
        output_path=chart_path,
        class_names=class_names,
        class_counts=class_counts,
        model_name=Path(CHECKPOINT_IN).stem,
    )


if __name__ == "__main__":
    main()
