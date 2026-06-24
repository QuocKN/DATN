#EffcientNet B2

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

DRONE_ROOT = "/kaggle/input/datasets/quoclop/balanced-dataset-drone-chuan-full-non-done/balanced_dataset_drone_chuan_full_non_done_1/balanced_dataset_drone_chuan_full_non_done/drone"
NON_DRONE_ROOT = "/kaggle/input/datasets/quoclop/balanced-dataset-drone-chuan-full-non-done/balanced_dataset_drone_chuan_full_non_done_1/balanced_dataset_drone_chuan_full_non_done/non_drone"
OUT_DIR = "fine_tune/EfficientNet_B2/efficientnet_b2_binary_runs/percentile_aug_holdout_both"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_SIZE = 260
CLASS_NAMES = ["non_drone", "drone"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_MODE = "grayscale_rgb"
IMAGE_PREPROCESS = "percentile"
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0
USE_DOMAIN_RANDOMIZATION = True
HOLDOUT_DRONE_CONDITION = ""
USE_BALANCED_SAMPLER = True


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
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
        return self.transform(image), sample.label


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


class RandomIntensityJitter:
    def __init__(self, brightness: float = 0.18, contrast: float = 0.35, p: float = 0.8) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        contrast = 1.0 + float(torch.empty(1).uniform_(-self.contrast, self.contrast))
        brightness = float(torch.empty(1).uniform_(-self.brightness, self.brightness))
        return (x * contrast + brightness).clamp(0.0, 1.0)


class RandomGamma:
    def __init__(self, gamma_min: float = 0.7, gamma_max: float = 1.5, p: float = 0.5) -> None:
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        gamma = float(torch.empty(1).uniform_(self.gamma_min, self.gamma_max))
        return x.clamp(0.0, 1.0).pow(gamma)


class RandomGaussianNoise:
    def __init__(self, std_max: float = 0.035, p: float = 0.5) -> None:
        self.std_max = std_max
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        std = float(torch.empty(1).uniform_(0.0, self.std_max))
        return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)


class RandomTimeFrequencyMask:
    def __init__(self, max_time_frac: float = 0.12, max_freq_frac: float = 0.10, p: float = 0.6) -> None:
        self.max_time_frac = max_time_frac
        self.max_freq_frac = max_freq_frac
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        _, height, width = x.shape
        fill = float(x.mean())
        x = x.clone()
        if torch.rand(()) < 0.5:
            mask_width = max(1, int(width * float(torch.empty(1).uniform_(0.02, self.max_time_frac))))
            start = int(torch.randint(0, max(width - mask_width + 1, 1), (1,)))
            x[:, :, start:start + mask_width] = fill
        if torch.rand(()) < 0.5:
            mask_height = max(1, int(height * float(torch.empty(1).uniform_(0.02, self.max_freq_frac))))
            start = int(torch.randint(0, max(height - mask_height + 1, 1), (1,)))
            x[:, start:start + mask_height, :] = fill
        return x


class RandomSpectrogramErase:
    def __init__(self, max_area_frac: float = 0.035, p: float = 0.35) -> None:
        self.max_area_frac = max_area_frac
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        _, height, width = x.shape
        area = height * width
        erase_area = max(1, int(area * float(torch.empty(1).uniform_(0.005, self.max_area_frac))))
        erase_height = int(np.sqrt(erase_area * float(torch.empty(1).uniform_(0.5, 2.0))))
        erase_height = min(max(1, erase_height), height)
        erase_width = min(max(1, erase_area // erase_height), width)
        top = int(torch.randint(0, max(height - erase_height + 1, 1), (1,)))
        left = int(torch.randint(0, max(width - erase_width + 1, 1), (1,)))
        x = x.clone()
        x[:, top:top + erase_height, left:left + erase_width] = float(x.mean())
        return x


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


def infer_drone_condition(path: Path) -> str:
    text = str(path).upper()
    for condition in ("CLEAN", "WIFI", "BLUE", "BOTH", "DRONE_TU_THU"):
        if f"/{condition}/" in text or f"__{condition}__" in path.name.upper():
            return condition
    return "UNKNOWN"


def choose_subset(paths: Sequence[Path], max_count: int, seed: int) -> List[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return list(paths)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=max_count, replace=False)
    return [paths[i] for i in sorted(indices.tolist())]


def build_transforms(
    image_size: int,
    image_preprocess: str,
    percentile_low: float,
    percentile_high: float,
    use_domain_randomization: bool,
) -> Tuple[transforms.Compose, transforms.Compose]:
    if image_preprocess not in {"legacy_imagenet", "percentile"}:
        raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")

    train_steps = []
    if use_domain_randomization:
        train_steps.append(transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.96, 1.04)))
    else:
        train_steps.append(transforms.Resize((image_size, image_size)))
    train_steps.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    eval_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]

    if image_preprocess == "percentile":
        percentile_tf = PercentileNormalizeTensor(low=percentile_low, high=percentile_high)
        train_steps.append(percentile_tf)
        eval_steps.append(percentile_tf)

    if use_domain_randomization:
        train_steps.extend(
            [
                RandomIntensityJitter(),
                RandomGamma(),
                RandomGaussianNoise(),
                RandomTimeFrequencyMask(),
                RandomSpectrogramErase(),
            ]
        )

    train_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    eval_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    train_tf = transforms.Compose(train_steps)
    eval_tf = transforms.Compose(eval_steps)
    return train_tf, eval_tf


def split_samples(samples: Sequence[Sample], val_size: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    train_idx, val_idx = train_test_split(
        np.arange(len(samples)),
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


def build_samples(
    drone_root: Path,
    non_drone_root: Path,
    max_drone_train: int,
    max_non_drone_train: int,
    seed: int,
    holdout_drone_condition: str,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    drone_train = collect_images(drone_root / "train")
    drone_valid = collect_images(drone_root / "valid")
    drone_test = collect_images(drone_root / "test")
    non_drone_train = collect_images(non_drone_root / "train")
    non_drone_valid = collect_images(non_drone_root / "valid") if (non_drone_root / "valid").exists() else []
    non_drone_test = collect_images(non_drone_root / "test")

    drone_train = choose_subset(drone_train, max_drone_train, seed)
    non_drone_train = choose_subset(non_drone_train, max_non_drone_train, seed + 1)

    if non_drone_valid:
        train_samples = [Sample(p, 1) for p in drone_train] + [Sample(p, 0) for p in non_drone_train]
        valid_samples = [Sample(p, 1) for p in drone_valid] + [Sample(p, 0) for p in non_drone_valid]
    else:
        train_pool = [Sample(p, 1) for p in drone_train] + [Sample(p, 0) for p in non_drone_train]
        train_samples, valid_from_train = split_samples(train_pool, val_size=0.2, seed=seed)
        valid_samples = valid_from_train + [Sample(p, 1) for p in drone_valid]

    test_samples = [Sample(p, 1) for p in drone_test] + [Sample(p, 0) for p in non_drone_test]

    holdout_condition = holdout_drone_condition.strip().upper()
    if holdout_condition:
        heldout_drone = [
            Sample(path, 1)
            for path in [*drone_train, *drone_valid, *drone_test]
            if infer_drone_condition(path) == holdout_condition
        ]
        if not heldout_drone:
            raise ValueError(f"No drone samples found for holdout condition: {holdout_condition}")
        train_samples = [
            sample
            for sample in train_samples
            if sample.label != 1 or infer_drone_condition(sample.path) != holdout_condition
        ]
        valid_samples = [sample for sample in valid_samples if sample.label == 0] + heldout_drone
        test_samples = [
            sample
            for sample in test_samples
            if sample.label != 1 or infer_drone_condition(sample.path) != holdout_condition
        ]
    return train_samples, valid_samples, test_samples


def freeze_all_backbone_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_feature_blocks(model: nn.Module, train_last_n_blocks: int) -> int:
    if train_last_n_blocks <= 0:
        return 0
    if not hasattr(model, "features"):
        raise ValueError("EfficientNet model does not expose 'features'; cannot unfreeze feature blocks")

    blocks = list(model.features)
    n = min(train_last_n_blocks, len(blocks))
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    return n


def build_model(freeze_backbone: bool, train_last_n_blocks: int) -> tuple[nn.Module, int]:
    try:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained EfficientNet-B2 ImageNet weights; refusing to train from random initialization."
        ) from exc

    unfrozen_blocks = 0
    if train_last_n_blocks > 0:
        freeze_all_backbone_params(model)
        unfrozen_blocks = unfreeze_last_n_feature_blocks(model, train_last_n_blocks)
    elif freeze_backbone:
        freeze_all_backbone_params(model)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model, unfrozen_blocks


def count_labels(samples: Sequence[Sample]) -> dict:
    labels = [s.label for s in samples]
    return {CLASS_NAMES[label]: int(labels.count(label)) for label in range(len(CLASS_NAMES))}


def count_drone_conditions(samples: Sequence[Sample]) -> dict:
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.label != 1:
            continue
        condition = infer_drone_condition(sample.path)
        counts[condition] = counts.get(condition, 0) + 1
    return dict(sorted(counts.items()))


def count_unique_images_by_class(samples: Sequence[Sample]) -> dict:
    unique_paths_by_label = {label: set() for label in range(len(CLASS_NAMES))}
    for sample in samples:
        unique_paths_by_label[sample.label].add(str(sample.path.resolve()))
    return {CLASS_NAMES[label]: len(unique_paths_by_label[label]) for label in range(len(CLASS_NAMES))}


def make_class_weights(samples: Sequence[Sample], device: torch.device) -> torch.Tensor:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_balanced_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    class_weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.1.")
    ]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier.1.")]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")

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
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B2 for binary drone/non-drone spectrogram detection")
    parser.add_argument("--drone-root", type=str, default=DRONE_ROOT)
    parser.add_argument("--non-drone-root", type=str, default=NON_DRONE_ROOT)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--image-preprocess", default=IMAGE_PREPROCESS, choices=["legacy_imagenet", "percentile"])
    parser.add_argument("--percentile-low", type=float, default=PERCENTILE_LOW)
    parser.add_argument("--percentile-high", type=float, default=PERCENTILE_HIGH)
    parser.add_argument("--disable-domain-randomization", action="store_true")
    parser.add_argument(
        "--holdout-drone-condition",
        default=HOLDOUT_DRONE_CONDITION,
        help="Leave one drone condition out of train and use it for validation, e.g. CLEAN, WIFI, BLUE, BOTH.",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-drone-train", type=int, default=0, help="0 means use all drone train images")
    parser.add_argument("--max-non-drone-train", type=int, default=0, help="0 means use all non-drone train images")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final classifier layer")
    parser.add_argument(
        "--train-last-n-blocks",
        type=int,
        default=0,
        help="Freeze backbone then unfreeze the last N EfficientNet feature blocks.",
    )
    parser.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        default=USE_BALANCED_SAMPLER,
        help="Use WeightedRandomSampler for class-balanced mini-batches in training loader",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        dest="use_balanced_sampler",
        action="store_false",
        help="Disable WeightedRandomSampler.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=7, help="Stop if no valid_macro_f1 improvement for N epochs")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001, help="Minimum valid_macro_f1 gain to count as improvement")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.image_preprocess == "percentile" and args.percentile_low >= args.percentile_high:
        raise ValueError("--percentile-low must be smaller than --percentile-high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, valid_samples, test_samples = build_samples(
        drone_root=Path(args.drone_root),
        non_drone_root=Path(args.non_drone_root),
        max_drone_train=args.max_drone_train,
        max_non_drone_train=args.max_non_drone_train,
        seed=args.seed,
        holdout_drone_condition=args.holdout_drone_condition,
    )

    use_domain_randomization = not args.disable_domain_randomization
    train_tf, eval_tf = build_transforms(
        args.image_size,
        image_preprocess=args.image_preprocess,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        use_domain_randomization=use_domain_randomization,
    )
    train_dataset = BinarySpectrogramDataset(train_samples, train_tf)
    train_sampler = make_balanced_sampler(train_samples) if args.use_balanced_sampler else None
    pin_memory = device.type == "cuda"
    common_loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **common_loader_kwargs,
        ),
        "valid": DataLoader(
            BinarySpectrogramDataset(valid_samples, eval_tf),
            batch_size=args.batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            BinarySpectrogramDataset(test_samples, eval_tf),
            batch_size=args.batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
    }

    print(f"Device: {device}")
    print("Model: efficientnet_b2")
    print(f"Classes: {CLASS_NAMES}")
    print(f"image_preprocess: {args.image_preprocess}")
    if args.image_preprocess == "percentile":
        print(f"percentile clip: p{args.percentile_low:g}-p{args.percentile_high:g}")
    print(f"use_domain_randomization: {use_domain_randomization}")
    print(f"holdout_drone_condition: {args.holdout_drone_condition or 'none'}")
    print(f"train: {len(train_samples)} {count_labels(train_samples)}")
    print(f"train drone conditions: {count_drone_conditions(train_samples)}")
    print(f"use_balanced_sampler: {args.use_balanced_sampler}")
    print(f"valid: {len(valid_samples)} {count_labels(valid_samples)}")
    print(f"valid drone conditions: {count_drone_conditions(valid_samples)}")
    print(f"test: {len(test_samples)} {count_labels(test_samples)}")
    print(f"test drone conditions: {count_drone_conditions(test_samples)}")

    model, unfrozen_blocks = build_model(
        freeze_backbone=args.freeze_backbone,
        train_last_n_blocks=args.train_last_n_blocks,
    )
    model = model.to(device)
    if args.train_last_n_blocks > 0:
        print(f"Unfrozen last EfficientNet feature blocks: {unfrozen_blocks}")
    else:
        print(f"Freeze backbone: {args.freeze_backbone}")

    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = None if args.use_balanced_sampler else make_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_valid_macro_f1 = -1.0
    best_path = out_dir / "balanced_efficientnet_b2_binary.pt"
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
                    "model_name": "efficientnet_b2_binary_finetuned",
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "class_to_idx": {"non_drone": 0, "drone": 1},
                    "image_size": args.image_size,
                    "image_mode": IMAGE_MODE,
                    "image_preprocess": args.image_preprocess,
                    "percentile_low": args.percentile_low,
                    "percentile_high": args.percentile_high,
                    "domain_randomization": use_domain_randomization,
                    "image_mean": IMAGENET_MEAN,
                    "image_std": IMAGENET_STD,
                    "backbone_name": "efficientnet_b2",
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
    train_counts = count_labels(train_samples)
    valid_counts = count_labels(valid_samples)
    test_counts = count_labels(test_samples)
    used_image_counts = {
        "train": count_unique_images_by_class(train_samples),
        "valid": count_unique_images_by_class(valid_samples),
        "test": count_unique_images_by_class(test_samples),
    }
    summary = {
        "model_name": "efficientnet_b2_binary_finetuned",
        "args": vars(args),
        "drone_root": args.drone_root,
        "non_drone_root": args.non_drone_root,
        "checkpoint": str(best_path),
        "class_names": CLASS_NAMES,
        "class_to_idx": {"non_drone": 0, "drone": 1},
        "image_mode": IMAGE_MODE,
        "image_preprocess": args.image_preprocess,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "domain_randomization": use_domain_randomization,
        "image_mean": IMAGENET_MEAN,
        "image_std": IMAGENET_STD,
        "backbone_name": "efficientnet_b2",
        "used_image_counts": used_image_counts,
        "train_images_used_total": len(train_samples),
        "train_images_used_by_class": train_counts,
        "sample_counts": {
            "train": train_counts,
            "valid": valid_counts,
            "test": test_counts,
        },
        "drone_condition_counts": {
            "train": count_drone_conditions(train_samples),
            "valid": count_drone_conditions(valid_samples),
            "test": count_drone_conditions(test_samples),
        },
        "sample_totals": {
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples),
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
