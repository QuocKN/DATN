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
    import timm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency 'timm'. Install it first: pip install timm") from exc

DRONE_ROOT = "/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/drone"
NON_DRONE_ROOT = "/home/quocnk/Documents/NKQuoc/Data/Spectrum/balanced_binary_dataset/non_drone"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_SIZE = 224
CLASS_NAMES = ["non_drone", "drone"]
CONVNEXTV2_MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        # Ensure file handles are closed promptly to avoid worker/file-descriptor issues across epochs.
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
        return self.transform(image), sample.label


class ConvNextV2BinaryClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features.float())


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


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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


def load_convnextv2_backbone(model_name: str, device: torch.device) -> nn.Module:
    model_candidates = [model_name]
    # If a pretrained tag suffix is invalid for local timm version, retry with plain architecture name.
    if "." in model_name:
        model_candidates.append(model_name.split(".", 1)[0])

    last_exc: Exception | None = None
    backbone = None
    for candidate in model_candidates:
        try:
            backbone = timm.create_model(candidate, pretrained=True, num_classes=0, global_pool="avg")
            if candidate != model_name:
                print(f"Fallback model name: {candidate} (from {model_name})")
            break
        except Exception as exc:
            last_exc = exc
        if backbone is not None:
            break

    if backbone is None:
        raise RuntimeError(
            f"Could not load pretrained ConvNeXtV2 weights for '{model_name}'; "
            "refusing to train from random initialization."
        ) from last_exc
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


def freeze_all_backbone_params(backbone: nn.Module) -> None:
    for param in backbone.parameters():
        param.requires_grad = False


def unfreeze_last_n_stages(backbone: nn.Module, train_last_n_stages: int) -> int:
    if train_last_n_stages <= 0:
        return 0

    if not hasattr(backbone, "stages"):
        raise ValueError("Backbone does not expose 'stages'; cannot apply stage-based unfreezing")

    stages = list(backbone.stages)
    total_stages = len(stages)
    n = min(train_last_n_stages, total_stages)
    for stage in stages[-n:]:
        for param in stage.parameters():
            param.requires_grad = True
    return n


def build_model(
    model_name: str,
    image_size: int,
    freeze_backbone: bool,
    train_last_n_stages: int,
    device: torch.device,
) -> tuple[ConvNextV2BinaryClassifier, int]:
    backbone = load_convnextv2_backbone(model_name=model_name, device=device)
    feature_dim = infer_feature_dim(backbone=backbone, image_size=image_size, device=device)

    unfrozen_stages = 0
    if train_last_n_stages > 0:
        freeze_all_backbone_params(backbone)
        unfrozen_stages = unfreeze_last_n_stages(backbone, train_last_n_stages)
    elif freeze_backbone:
        freeze_all_backbone_params(backbone)

    model = ConvNextV2BinaryClassifier(backbone=backbone, feature_dim=feature_dim, num_classes=len(CLASS_NAMES))
    model.to(device)
    return model, unfrozen_stages


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


def build_optimizer(
    model: ConvNextV2BinaryClassifier,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
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
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt V2 for binary drone/non-drone spectrogram detection")
    parser.add_argument("--drone-root", type=str, default=DRONE_ROOT)
    parser.add_argument("--non-drone-root", type=str, default=NON_DRONE_ROOT)
    parser.add_argument("--out-dir", type=str, default="fine_tune/ConvNext_V2/convnextv2_binary_runs")
    parser.add_argument("--convnextv2-model", type=str, default=CONVNEXTV2_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-drone-train", type=int, default=0, help="0 means use all drone train images")
    parser.add_argument("--max-non-drone-train", type=int, default=0, help="0 means use all non-drone train images")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head")
    parser.add_argument(
        "--train-last-n-stages",
        type=int,
        default=0,
        help="Freeze backbone then unfreeze the last N stages (overrides --freeze-backbone when > 0).",
    )
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

    train_tf, eval_tf = build_transforms(args.image_size)
    train_dataset = BinarySpectrogramDataset(train_samples, train_tf)
    valid_dataset = BinarySpectrogramDataset(valid_samples, eval_tf)
    test_dataset = BinarySpectrogramDataset(test_samples, eval_tf)

    train_sampler = None
    if args.use_balanced_sampler:
        sample_weights = make_sample_weights(train_samples)
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    pin_memory = device.type == "cuda"
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **common_loader_kwargs,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
    }

    print(f"Device: {device}")
    print(f"ConvNeXtV2 model: {args.convnextv2_model}")
    print(f"Balanced sampler: {args.use_balanced_sampler}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"train: {len(train_samples)} {count_labels(train_samples)}")
    print(f"valid: {len(valid_samples)} {count_labels(valid_samples)}")
    print(f"test: {len(test_samples)} {count_labels(test_samples)}")

    model, unfrozen_stages = build_model(
        model_name=args.convnextv2_model,
        image_size=args.image_size,
        freeze_backbone=args.freeze_backbone,
        train_last_n_stages=args.train_last_n_stages,
        device=device,
    )
    if args.train_last_n_stages > 0:
        print(f"Unfrozen last ConvNeXtV2 stages: {unfrozen_stages}")
    else:
        print(f"Freeze backbone: {args.freeze_backbone}")

    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = None if args.use_balanced_sampler else make_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_valid_macro_f1 = -1.0
    best_path = out_dir / "convnextv2_binary_best.pt"
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
                    "model_name": "convnextv2_base_binary_finetuned",
                    "convnextv2_model": args.convnextv2_model,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "class_to_idx": {"non_drone": 0, "drone": 1},
                    "image_size": args.image_size,
                    "image_mean": IMAGENET_MEAN,
                    "image_std": IMAGENET_STD,
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
        "model_name": "convnextv2_base_binary_finetuned",
        "convnextv2_model": args.convnextv2_model,
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
