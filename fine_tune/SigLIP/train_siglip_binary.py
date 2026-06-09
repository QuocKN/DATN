from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor, SiglipVisionModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency 'transformers'. Install it first: pip install transformers") from exc

DRONE_ROOT = "/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/drone"
NON_DRONE_ROOT = "/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/non_drone"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_SIZE = 224
CLASS_NAMES = ["non_drone", "drone"]
SIGLIP_MODEL = "google/siglip-base-patch16-224"


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class BinarySpectrogramDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: transforms.Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label


class SigLIPClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        features = outputs.pooler_output
        return self.head(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def choose_subset(paths: Sequence[Path], max_count: int, seed: int) -> List[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return list(paths)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=max_count, replace=False)
    return [paths[i] for i in sorted(indices.tolist())]


def build_transforms(image_size: int, siglip_model: str) -> Tuple[transforms.Compose, transforms.Compose]:
    image_processor = AutoImageProcessor.from_pretrained(siglip_model)
    mean = image_processor.image_mean
    std = image_processor.image_std

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def build_samples(
    drone_root: Path,
    non_drone_root: Path,
    max_drone_train: int,
    max_non_drone_train: int,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    drone_train = choose_subset(collect_images(drone_root / "train"), max_drone_train, seed)
    drone_valid = collect_images(drone_root / "valid")
    drone_test = collect_images(drone_root / "test")

    non_drone_train = choose_subset(collect_images(non_drone_root / "train"), max_non_drone_train, seed + 1)
    non_drone_valid = collect_images(non_drone_root / "valid")
    non_drone_test = collect_images(non_drone_root / "test")

    train_samples = [Sample(p, 1) for p in drone_train] + [Sample(p, 0) for p in non_drone_train]
    valid_samples = [Sample(p, 1) for p in drone_valid] + [Sample(p, 0) for p in non_drone_valid]
    test_samples = [Sample(p, 1) for p in drone_test] + [Sample(p, 0) for p in non_drone_test]
    return train_samples, valid_samples, test_samples


def load_siglip_backbone(model_name: str, device: torch.device) -> nn.Module:
    try:
        backbone = SiglipVisionModel.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Could not load SigLIP vision model from Hugging Face. "
            "Ensure internet access and optional HF_TOKEN for higher rate limits."
        ) from exc
    backbone.to(device)
    return backbone


def build_model(model_name: str, freeze_backbone: bool, device: torch.device) -> SigLIPClassifier:
    backbone = load_siglip_backbone(model_name, device)
    feature_dim = int(backbone.config.hidden_size)

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    model = SigLIPClassifier(backbone=backbone, feature_dim=feature_dim, num_classes=len(CLASS_NAMES))
    model.to(device)
    return model


def count_labels(samples: Sequence[Sample]) -> dict:
    labels = [s.label for s in samples]
    return {CLASS_NAMES[label]: int(labels.count(label)) for label in range(len(CLASS_NAMES))}


def make_class_weights(samples: Sequence[Sample], device: torch.device) -> torch.Tensor:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_sample_weights(samples: Sequence[Sample]) -> torch.Tensor:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    sample_weights = class_weights[labels]
    return torch.tensor(sample_weights, dtype=torch.double)


def build_optimizer(model: SigLIPClassifier, backbone_lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in tqdm(loader, desc="Train", unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += batch_size

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true = []
    all_pred = []

    for x, y in tqdm(loader, desc="Eval", unit="batch"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((pred == y).sum().item())
        total_count += batch_size
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1), y_true, y_pred


def save_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP for binary drone/non-drone spectrogram detection")
    parser.add_argument("--drone-root", type=str, default=DRONE_ROOT)
    parser.add_argument("--non-drone-root", type=str, default=NON_DRONE_ROOT)
    parser.add_argument("--out-dir", type=str, default="fine_tune/SigLIP/siglip_binary_runs")
    parser.add_argument("--siglip-model", type=str, default=SIGLIP_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-drone-train", type=int, default=0, help="0 means use all drone train images")
    parser.add_argument("--max-non-drone-train", type=int, default=0, help="0 means use all non-drone train images")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-balanced-sampler", action="store_true", help="Use WeightedRandomSampler on train set.")
    parser.add_argument("--early-stop-patience", type=int, default=7, help="Stop if no valid_macro_f1 improvement for N epochs")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001, help="Minimum valid_macro_f1 gain to count as improvement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, valid_samples, test_samples = build_samples(
        drone_root=Path(args.drone_root),
        non_drone_root=Path(args.non_drone_root),
        max_drone_train=args.max_drone_train,
        max_non_drone_train=args.max_non_drone_train,
        seed=args.seed,
    )

    train_tf, eval_tf = build_transforms(args.image_size, args.siglip_model)
    train_dataset = BinarySpectrogramDataset(train_samples, train_tf)
    valid_dataset = BinarySpectrogramDataset(valid_samples, eval_tf)
    test_dataset = BinarySpectrogramDataset(test_samples, eval_tf)

    train_sampler = None
    if args.use_balanced_sampler:
        sample_weights = make_sample_weights(train_samples)
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }

    print(f"Device: {device}")
    print(f"SigLIP model: {args.siglip_model}")
    print(f"Balanced sampler: {args.use_balanced_sampler}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"train: {len(train_samples)} {count_labels(train_samples)}")
    print(f"valid: {len(valid_samples)} {count_labels(valid_samples)}")
    print(f"test: {len(test_samples)} {count_labels(test_samples)}")

    model = build_model(model_name=args.siglip_model, freeze_backbone=args.freeze_backbone, device=device)
    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_samples, device))

    best_valid_macro_f1 = -1.0
    best_path = out_dir / "siglip_binary_best.pt"
    history = []
    epochs_without_improvement = 0
    stopped_early = False
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        valid_loss, valid_acc, y_true_valid, y_pred_valid = evaluate(model, loaders["valid"], criterion, device)
        valid_macro_f1 = float(f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0))
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_macro_f1": valid_macro_f1,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        improvement = valid_macro_f1 - best_valid_macro_f1
        if improvement > args.early_stop_min_delta:
            best_valid_macro_f1 = valid_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": "siglip_binary_finetuned",
                    "siglip_model": args.siglip_model,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "class_to_idx": {"non_drone": 0, "drone": 1},
                    "image_size": args.image_size,
                    "args": vars(args),
                    "best_valid_macro_f1": best_valid_macro_f1,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                stopped_early = True
                print(
                    f"Early stopping at epoch {epoch}: no valid_macro_f1 improvement "
                    f"> {args.early_stop_min_delta} for {args.early_stop_patience} epochs."
                )
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm_array = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm = cm_array.tolist()
    confusion_matrix_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix(cm_array, CLASS_NAMES, confusion_matrix_path)
    summary = {
        "model_name": "siglip_binary_finetuned",
        "siglip_model": args.siglip_model,
        "args": vars(args),
        "drone_root": args.drone_root,
        "non_drone_root": args.non_drone_root,
        "checkpoint": str(best_path),
        "class_names": CLASS_NAMES,
        "class_to_idx": {"non_drone": 0, "drone": 1},
        "sample_counts": {
            "train": count_labels(train_samples),
            "valid": count_labels(valid_samples),
            "test": count_labels(test_samples),
        },
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_path": str(confusion_matrix_path),
        "history": history,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("\nTest summary")
    print(json.dumps({"test_loss": test_loss, "test_acc": test_acc, "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
