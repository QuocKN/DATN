from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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

try:
    import timm
except Exception:
    timm = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
EXCLUDE_NAME_KEYWORDS = {
    "results_chart",
    "confusion",
    "threshold_chart",
    "architecture",
    "visualization",
}
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

# Edit these values directly, or override them with command line arguments.
MODEL_ARCH = "efficientnet_b2"
CHECKPOINT_IN = "fine_tune/EfficientNet_B2/efficientnet_b2_binary_runs/balanced_efficientnet_b2_binary.pt"
SOURCE_PATH = r"F:\DroneDetect_V2\CLEAN\MP2_FY\MAV_0010_01_spectrograms"
OUTPUT_DIR = "fine_tune/grad_cam_outputs/efficientnet_b2"

TARGET_CLASS = "drone"  # Use "predicted" to explain the predicted class instead.
TARGET_LAYER = ""  # Leave empty for a good default layer per CNN architecture.
DEVICE = "cuda:0"
BATCH_SIZE = 64
NUM_WORKERS = 4
TOP_K = 16
SORT_BY = "drone_score"  # Options: drone_score, confidence, path
OVERLAY_ALPHA = 0.45


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

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image)


class ConvNextV2BinaryClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features.float())


@dataclass
class ModelBundle:
    model: nn.Module
    arch: str
    class_names: List[str]
    class_to_idx: Dict[str, int]
    image_size: int
    mean: List[float]
    std: List[float]


@dataclass
class PredictionRecord:
    path: Path
    pred_idx: int
    pred_label: str
    confidence: float
    drone_score: float
    probabilities: Dict[str, float]


def is_probable_spectrogram(path: Path) -> bool:
    stem = path.stem.lower()
    return not any(keyword in stem for keyword in EXCLUDE_NAME_KEYWORDS)


def collect_image_paths(source: Path) -> List[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file extension: {source}")
        return [source]
    return sorted(
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and is_probable_spectrogram(p)
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def infer_feature_dim(backbone: nn.Module, image_size: int, device: torch.device) -> int:
    backbone.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = backbone(dummy)
    if features.ndim != 2:
        features = features.flatten(1)
    return int(features.shape[1])


def infer_arch_from_checkpoint(checkpoint: dict, checkpoint_path: Path) -> str:
    model_name = str(checkpoint.get("model_name", "")).lower()
    path_text = str(checkpoint_path).lower()
    if "efficientnet_b2" in model_name or "efficientnet_b2" in path_text:
        return "efficientnet_b2"
    if "resnet18" in model_name or "resnet18" in path_text:
        return "resnet18"
    if "resnet50" in model_name or "resnet50" in path_text:
        return "resnet50"
    if "vgg13" in model_name or "vgg13" in path_text:
        return "vgg13_bn"
    if "convnextv2" in model_name or "convnextv2" in path_text or "convnext_v2" in path_text:
        return "convnextv2"
    raise ValueError("Could not infer MODEL_ARCH. Set --arch explicitly.")


def build_model(arch: str, checkpoint: dict, checkpoint_path: Path, device: torch.device) -> ModelBundle:
    arch = infer_arch_from_checkpoint(checkpoint, checkpoint_path) if arch == "auto" else arch
    class_names = list(checkpoint.get("class_names", ["non_drone", "drone"]))
    class_to_idx = dict(checkpoint.get("class_to_idx", {name: idx for idx, name in enumerate(class_names)}))
    image_size = int(checkpoint.get("image_size", 224))
    mean = list(checkpoint.get("image_mean", DEFAULT_MEAN))
    std = list(checkpoint.get("image_std", DEFAULT_STD))
    num_classes = len(class_names)

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "vgg13_bn":
        model = models.vgg13_bn(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif arch == "convnextv2":
        if timm is None:
            raise RuntimeError("Missing dependency 'timm'. Install it first: pip install timm")
        convnextv2_model = checkpoint.get("convnextv2_model", "convnextv2_tiny.fcmae_ft_in22k_in1k")
        candidates = [convnextv2_model]
        if "." in convnextv2_model:
            candidates.append(convnextv2_model.split(".", 1)[0])

        backbone = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                backbone = timm.create_model(candidate, pretrained=False, num_classes=0, global_pool="avg")
                break
            except Exception as exc:
                last_error = exc
        if backbone is None:
            raise RuntimeError(f"Could not create ConvNeXtV2 model from '{convnextv2_model}'.") from last_error
        backbone.to(device)
        feature_dim = infer_feature_dim(backbone, image_size=image_size, device=device)
        model = ConvNextV2BinaryClassifier(backbone=backbone, feature_dim=feature_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return ModelBundle(model=model, arch=arch, class_names=class_names, class_to_idx=class_to_idx, image_size=image_size, mean=mean, std=std)


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        available = [module_name for module_name in modules.keys() if module_name]
        preview = ", ".join(available[:40])
        raise ValueError(f"Target layer '{name}' was not found. First available layers: {preview}")
    return modules[name]


def find_last_conv2d(module: nn.Module) -> nn.Module:
    last_conv: nn.Module | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("Could not find any Conv2d layer for Grad-CAM.")
    return last_conv


def choose_target_layer(model: nn.Module, arch: str, target_layer_name: str) -> tuple[str, nn.Module]:
    if target_layer_name:
        return target_layer_name, get_module_by_name(model, target_layer_name)
    if arch in {"resnet18", "resnet50"}:
        return "layer4[-1]", model.layer4[-1]
    if arch == "efficientnet_b2":
        return "features[-1]", model.features[-1]
    if arch == "vgg13_bn":
        return "last Conv2d in features", find_last_conv2d(model.features)
    if arch == "convnextv2":
        return "last Conv2d in backbone", find_last_conv2d(model.backbone)
    return "last Conv2d", find_last_conv2d(model)


def colorize_cam(cam: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    return (cmap(cam)[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    heatmap_rgb = colorize_cam(cam)
    overlay = (image_rgb.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha).clip(0, 255)
    return heatmap_rgb, overlay.astype(np.uint8)


def sanitize_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return stem[:90] if len(stem) > 90 else stem


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Target layer output must be a tensor for Grad-CAM.")
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target_idx: int, output_size: tuple[int, int]) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, target_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        if self.activations.ndim != 4 or self.gradients.ndim != 4:
            raise RuntimeError(
                f"Grad-CAM needs 4D feature maps, got activations={tuple(self.activations.shape)}, "
                f"gradients={tuple(self.gradients.shape)}."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam[0, 0].detach().cpu().numpy(), logits.detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def make_transform(image_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def predict_all(
    image_paths: Sequence[Path],
    bundle: ModelBundle,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> List[PredictionRecord]:
    dataset = SpectrogramDataset(image_paths, image_size=bundle.image_size, mean=bundle.mean, std=bundle.std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    idx_to_class = {idx: name for name, idx in bundle.class_to_idx.items()}
    drone_idx = bundle.class_to_idx.get("drone")
    records: List[PredictionRecord] = []
    offset = 0
    for x in tqdm(loader, desc="Scan images", unit="batch"):
        x = x.to(device, non_blocking=True)
        probs = torch.softmax(bundle.model(x), dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)
        for row_idx, pred_idx in enumerate(pred):
            path = image_paths[offset + row_idx]
            pred_label = idx_to_class.get(int(pred_idx), str(int(pred_idx)))
            confidence = float(probs[row_idx, int(pred_idx)])
            drone_score = float(probs[row_idx, int(drone_idx)]) if drone_idx is not None else float("nan")
            probabilities = {name: float(probs[row_idx, idx]) for name, idx in bundle.class_to_idx.items()}
            records.append(
                PredictionRecord(
                    path=path,
                    pred_idx=int(pred_idx),
                    pred_label=pred_label,
                    confidence=confidence,
                    drone_score=drone_score,
                    probabilities=probabilities,
                )
            )
        offset += len(pred)
    return records


def select_records(records: Sequence[PredictionRecord], sort_by: str, top_k: int) -> List[PredictionRecord]:
    if sort_by == "path":
        selected = list(records)
    elif sort_by == "confidence":
        selected = sorted(records, key=lambda item: item.confidence, reverse=True)
    elif sort_by == "drone_score":
        selected = sorted(records, key=lambda item: item.drone_score, reverse=True)
    else:
        raise ValueError(f"Unsupported SORT_BY: {sort_by}")
    return selected[:top_k] if top_k > 0 else selected


def resolve_target_idx(target_class: str, class_to_idx: Dict[str, int], predicted_idx: int) -> int:
    if target_class.lower() == "predicted":
        return int(predicted_idx)
    if target_class in class_to_idx:
        return int(class_to_idx[target_class])
    try:
        return int(target_class)
    except ValueError as exc:
        raise ValueError(f"TARGET_CLASS must be 'predicted', a class name, or a class index. Got: {target_class}") from exc


def save_panel(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("white")
    panels = [
        ("Input", image_rgb),
        ("Grad-CAM", cam),
        ("Heatmap", heatmap_rgb),
        ("Overlay", overlay_rgb),
    ]
    for ax, (label, data) in zip(axes, panels):
        if label == "Grad-CAM":
            ax.imshow(data, cmap="jet", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(data)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cam_focus_stats(cam: np.ndarray) -> Dict[str, float]:
    return {
        "cam_mean": float(cam.mean()),
        "cam_std": float(cam.std()),
        "hot_area_ratio_0_60": float((cam >= 0.60).mean()),
        "hot_area_ratio_0_75": float((cam >= 0.75).mean()),
    }


def generate_grad_cam_outputs(
    records: Sequence[PredictionRecord],
    bundle: ModelBundle,
    device: torch.device,
    output_dir: Path,
    target_class: str,
    target_layer_name: str,
    overlay_alpha: float,
) -> List[dict]:
    target_layer_desc, target_layer = choose_target_layer(bundle.model, bundle.arch, target_layer_name)
    grad_cam = GradCAM(bundle.model, target_layer)
    transform = make_transform(bundle.image_size, bundle.mean, bundle.std)
    idx_to_class = {idx: name for name, idx in bundle.class_to_idx.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    try:
        for index, record in enumerate(tqdm(records, desc="Grad-CAM", unit="image")):
            with Image.open(record.path) as img:
                image = img.convert("RGB").resize((bundle.image_size, bundle.image_size), Image.BILINEAR)
            image_rgb = np.asarray(image)
            x = transform(image).unsqueeze(0).to(device)
            target_idx = resolve_target_idx(target_class, bundle.class_to_idx, record.pred_idx)
            target_label = idx_to_class.get(target_idx, str(target_idx))
            cam, logits = grad_cam(x, target_idx=target_idx, output_size=(bundle.image_size, bundle.image_size))
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            heatmap_rgb, overlay_rgb = overlay_heatmap(image_rgb, cam, alpha=overlay_alpha)
            stem = f"{index:03d}_{record.pred_label}_target_{target_label}_{sanitize_stem(record.path)}"
            heatmap_path = output_dir / f"{stem}_heatmap.png"
            overlay_path = output_dir / f"{stem}_overlay.png"
            panel_path = output_dir / f"{stem}_panel.png"

            Image.fromarray(heatmap_rgb).save(heatmap_path)
            Image.fromarray(overlay_rgb).save(overlay_path)
            title = (
                f"{record.path.name} | pred={record.pred_label} ({record.confidence:.3f}) | "
                f"target={target_label} | drone_score={record.drone_score:.3f}"
            )
            save_panel(image_rgb, cam, heatmap_rgb, overlay_rgb, title=title, output_path=panel_path)

            result = {
                "image": str(record.path),
                "prediction": record.pred_label,
                "confidence": record.confidence,
                "drone_score": record.drone_score,
                "target_class": target_label,
                "target_index": int(target_idx),
                "probabilities": {name: float(probs[idx]) for name, idx in bundle.class_to_idx.items()},
                "target_layer": target_layer_desc,
                "heatmap": str(heatmap_path),
                "overlay": str(overlay_path),
                "panel": str(panel_path),
            }
            result.update(cam_focus_stats(cam))
            results.append(result)
    finally:
        grad_cam.close()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for fine-tuned CNN binary classifiers.")
    parser.add_argument("--arch", default=MODEL_ARCH, choices=["auto", "resnet18", "resnet50", "efficientnet_b2", "vgg13_bn", "convnextv2"])
    parser.add_argument("--checkpoint", default=CHECKPOINT_IN)
    parser.add_argument("--source", default=SOURCE_PATH, help="Image file or directory containing spectrogram images.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--target-class", default=TARGET_CLASS, help="Class name, class index, or 'predicted'.")
    parser.add_argument("--target-layer", default=TARGET_LAYER, help="Optional exact module name from model.named_modules().")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--sort-by", default=SORT_BY, choices=["drone_score", "confidence", "path"])
    parser.add_argument("--overlay-alpha", type=float, default=OVERLAY_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    bundle = build_model(args.arch, checkpoint, checkpoint_path, device)

    image_paths = collect_image_paths(source_path)
    if not image_paths:
        raise ValueError(f"No images found in: {source_path}")

    records = predict_all(
        image_paths=image_paths,
        bundle=bundle,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    selected_records = select_records(records, sort_by=args.sort_by, top_k=args.top_k)
    cam_results = generate_grad_cam_outputs(
        records=selected_records,
        bundle=bundle,
        device=device,
        output_dir=output_dir,
        target_class=args.target_class,
        target_layer_name=args.target_layer,
        overlay_alpha=args.overlay_alpha,
    )

    drone_scores = [record.drone_score for record in records if not np.isnan(record.drone_score)]
    summary = {
        "method": "grad_cam_cnn_binary",
        "arch": bundle.arch,
        "checkpoint": str(checkpoint_path),
        "source": str(source_path),
        "output_dir": str(output_dir),
        "scanned_images": len(records),
        "generated_images": len(cam_results),
        "sort_by": args.sort_by,
        "target_class": args.target_class,
        "class_names": bundle.class_names,
        "class_to_idx": bundle.class_to_idx,
        "image_size": bundle.image_size,
        "image_mean": bundle.mean,
        "image_std": bundle.std,
        "drone_score_mean": float(np.mean(drone_scores)) if drone_scores else None,
        "drone_score_std": float(np.std(drone_scores)) if drone_scores else None,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "grad_cam": cam_results}, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved Grad-CAM summary: {summary_path}")


if __name__ == "__main__":
    main()
