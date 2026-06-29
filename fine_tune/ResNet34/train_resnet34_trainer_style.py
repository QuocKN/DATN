from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm


DRONE_ROOT = "/kaggle/input/balanced-dataset-drone-chuan-full-non-done/balanced_dataset_drone_chuan_full_non_done/balanced_dataset_drone_chuan_full_non_done/drone"
NON_DRONE_ROOT = "/kaggle/input/balanced-dataset-drone-chuan-full-non-done/balanced_dataset_drone_chuan_full_non_done/balanced_dataset_drone_chuan_full_non_done/non_drone"
TRAIN_DIR = ""
VAL_DIR = ""
TEST_DIR = ""
OUT_DIR = "fine_tune/ResNet34/resnet34_binary_runs"
BEST_CHECKPOINT_NAME = "balanced_resnet34_binary.pt"
SUMMARY_NAME = "summary.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_SIZE = 224
CLASS_NAMES = ["non_drone", "drone"]
EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-5
WEIGHT_DECAY = 1e-4
DEVICE = "cuda:0"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_resnet34(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if pretrained:
        try:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            print("load pretrained weights")
        except Exception:
            print("failed to load pretrained weights")
            model = models.resnet34(pretrained=True)
    else:
        try:
            model = models.resnet34(weights=None)
            print("load default weights")
        except Exception:
            print("failed to load default weights")
            model = models.resnet34(pretrained=False)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def split_dir(class_root: Path, split: str) -> Path:
    candidates = [class_root / split]
    if split == "valid":
        candidates.append(class_root / "val")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class SplitClassDataset(Dataset):
    def __init__(self, drone_root: Path, non_drone_root: Path, split: str, transform: transforms.Compose) -> None:
        self.transform = transform
        self.classes = CLASS_NAMES
        self.class_to_idx = {"non_drone": 0, "drone": 1}

        drone_paths = collect_images(split_dir(drone_root, split))
        non_drone_paths = collect_images(split_dir(non_drone_root, split))
        self.samples = [(p, self.class_to_idx["non_drone"]) for p in non_drone_paths]
        self.samples += [(p, self.class_to_idx["drone"]) for p in drone_paths]

        if not self.samples:
            raise ValueError(f"No images found for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class ResNet34Trainer:
    def __init__(
        self,
        train_dir: str = "",
        val_dir: str = "",
        save_dir: str = OUT_DIR,
        test_dir: str = "",
        drone_root: str = DRONE_ROOT,
        non_drone_root: str = NON_DRONE_ROOT,
        weights: str = "",
        device: str = DEVICE,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_size: int = IMAGE_SIZE,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        shuffle: bool = True,
        grayscale_to_rgb: bool = True,
        best_checkpoint_name: str = BEST_CHECKPOINT_NAME,
    ) -> None:
        self.train_dir = Path(train_dir) if train_dir else None
        self.val_dir = Path(val_dir) if val_dir else None
        self.test_dir = Path(test_dir) if test_dir else None
        self.drone_root = Path(drone_root) if drone_root else None
        self.non_drone_root = Path(non_drone_root) if non_drone_root else None
        self.save_dir = Path(save_dir)
        self.weights = Path(weights) if weights else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.grayscale_to_rgb = grayscale_to_rgb
        self.best_checkpoint_name = best_checkpoint_name
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        self.best_acc = 0.0
        self.best_epoch = 0
        self.history = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self._setup()

    def _build_transforms(self) -> transforms.Compose:
        steps = [transforms.Resize((self.image_size, self.image_size))]
        if self.grayscale_to_rgb:
            steps.append(transforms.Grayscale(num_output_channels=3))
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(steps)

    def _make_dataset(self, split: str, imagefolder_dir: Path | None, transform: transforms.Compose) -> Dataset:
        if imagefolder_dir is not None:
            if not imagefolder_dir.exists():
                raise FileNotFoundError(f"{split} directory does not exist: {imagefolder_dir}")
            return datasets.ImageFolder(root=str(imagefolder_dir), transform=transform)

        if self.drone_root is None or self.non_drone_root is None:
            raise ValueError("Set train_dir/val_dir or set both drone_root and non_drone_root.")
        return SplitClassDataset(
            drone_root=self.drone_root,
            non_drone_root=self.non_drone_root,
            split=split,
            transform=transform,
        )

    def _make_optional_test_dataset(self, transform: transforms.Compose) -> Dataset | None:
        if self.test_dir is not None:
            return self._make_dataset("test", self.test_dir, transform)
        if self.drone_root is None or self.non_drone_root is None:
            return None
        if not split_dir(self.drone_root, "test").exists() or not split_dir(self.non_drone_root, "test").exists():
            return None
        return self._make_dataset("test", None, transform)

    def _setup(self) -> None:
        transform = self._build_transforms()
        train_dataset = self._make_dataset("train", self.train_dir, transform)
        val_dataset = self._make_dataset("valid", self.val_dir, transform)
        test_dataset = self._make_optional_test_dataset(transform)

        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError(
                "Train and validation class folders are different: "
                f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
            )

        self.class_names = train_dataset.classes
        self.class_to_idx = train_dataset.class_to_idx
        self.num_classes = len(self.class_names)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = (
            DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            if test_dataset is not None
            else None
        )

        self.model = build_resnet34(
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone,
        )
        self.model.to(self.device)

        if self.weights and self.weights.exists():
            checkpoint = torch.load(self.weights, map_location=self.device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Loaded weights: {self.weights}")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        print(f"Device: {self.device}")
        print(f"Classes: {self.class_to_idx}")
        print(f"Train images: {len(self.train_loader.dataset)}")
        print(f"Validation images: {len(self.val_loader.dataset)}")
        if self.test_loader is not None:
            print(f"Test images: {len(self.test_loader.dataset)}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    def train_one_epoch(self, epoch: int, num_epochs: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs} Train", unit="batch")
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += batch_size

            progress.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        return running_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader, desc: str) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        progress = tqdm(loader, desc=desc, unit="batch")
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            correct += int((preds == labels).sum().item())
            total += batch_size
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)

        return {
            "loss": running_loss / max(total, 1),
            "acc": correct / max(total, 1),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def validate(self) -> Dict[str, float]:
        return self.evaluate_loader(self.val_loader, "Validate")

    def test(self) -> Dict[str, float] | None:
        if self.test_loader is None:
            return None
        return self.evaluate_loader(self.test_loader, "Test")

    def _checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Dict:
        return {
            "model_name": "resnet34_binary_finetuned",
            "backbone_name": "resnet34",
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "class_to_idx": self.class_to_idx,
            "image_size": self.image_size,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "metrics": metrics,
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
        }

    def save_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        epoch_path = self.save_dir / f"resnet34_epoch_{epoch}.pt"
        torch.save(self._checkpoint(epoch, metrics), epoch_path)
        print(f"Saved epoch checkpoint: {epoch_path}")

        if metrics["acc"] > self.best_acc:
            self.best_acc = metrics["acc"]
            self.best_epoch = epoch
            best_path = self.save_dir / self.best_checkpoint_name
            torch.save(self._checkpoint(epoch, metrics), best_path)
            print(f"Saved best checkpoint: {best_path} | val_acc={metrics['acc']:.4f}")

    def train(self, num_epochs: int) -> Dict:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch, num_epochs)
            val_metrics = self.validate()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(row)
            print(json.dumps(row, indent=2))
            self.save_model(epoch, val_metrics)

        test_metrics = self.test()
        summary = {
            "model_name": "resnet34_binary_finetuned",
            "backbone_name": "resnet34",
            "class_names": self.class_names,
            "class_to_idx": self.class_to_idx,
            "checkpoint": str(self.save_dir / self.best_checkpoint_name),
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "test_metrics": test_metrics,
            "history": self.history,
        }
        summary_path = self.save_dir / SUMMARY_NAME
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        print(f"Saved summary: {summary_path}")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainer-style fine-tune script for ResNet34 classification")
    parser.add_argument("--drone-root", type=str, default=DRONE_ROOT)
    parser.add_argument("--non-drone-root", type=str, default=NON_DRONE_ROOT)
    parser.add_argument("--train-dir", type=str, default=TRAIN_DIR, help="Optional ImageFolder train directory")
    parser.add_argument("--val-dir", type=str, default=VAL_DIR, help="Optional ImageFolder validation directory")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR, help="Optional ImageFolder test directory")
    parser.add_argument("--out-dir", "--save-dir", dest="save_dir", type=str, default=OUT_DIR)
    parser.add_argument("--weights", type=str, default="", help="Optional checkpoint/state_dict to continue training from")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--best-checkpoint-name", type=str, default=BEST_CHECKPOINT_NAME)
    parser.add_argument("--no-pretrained", action="store_true", help="Train ResNet34 from random initialization")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final fc layer")
    parser.add_argument("--no-grayscale-to-rgb", action="store_true", help="Keep input image colors unchanged")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable train DataLoader shuffling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    trainer = ResNet34Trainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        save_dir=args.save_dir,
        test_dir=args.test_dir,
        drone_root=args.drone_root,
        non_drone_root=args.non_drone_root,
        weights=args.weights,
        device=args.device,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        shuffle=not args.no_shuffle,
        grayscale_to_rgb=not args.no_grayscale_to_rgb,
        best_checkpoint_name=args.best_checkpoint_name,
    )
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
