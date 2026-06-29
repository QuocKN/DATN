from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
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
DEFAULT_IMAGE_PREPROCESS = "sobel_edge_3channel"
DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.0

# Edit these values directly, or override them with command line arguments.
MODEL_ARCH = "resnet50"
CHECKPOINT_IN = r"d:\balanced_resnet50_binary_edge.pt"
SOURCE_PATH = r"E:\Data_22_6\drone_test"
OUTPUT_DIR = r"fine_tune/grad_cam_outputs/resnet50_2"

TARGET_CLASS = "both"  # Options: "drone", "predicted", or "both" for predicted CAM + target drone CAM.
TARGET_LAYER = ""  # Leave empty for a good default layer per CNN architecture.
DEVICE = "cuda:0"
BATCH_SIZE = 64
NUM_WORKERS = 4
TOP_K = 32
SORT_BY = "confidence"  # Options: drone_score, confidence, path
OVERLAY_ALPHA = 0.45
CASE_FILTER = "false_negative_drone"
SOURCE_BINARY_LABEL = "auto"
OCCLUSION_THRESHOLD = 0.60
OCCLUSION_FILL_MODE = "mean"
OCCLUSION_FILL_VALUE = 255
OCCLUSION_COMPARE_MODES = "white,black,mean,median"


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


class SobelEdge3Channel:
    def __call__(self, img: Image.Image) -> Image.Image:
        img = np.array(img.convert("RGB"))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)

        edge3 = np.stack([edge, edge, edge], axis=-1)
        edge3 = (edge3 * 255).astype(np.uint8)

        return Image.fromarray(edge3)


def make_tensor_transform(
    mean: Sequence[float],
    std: Sequence[float],
    image_preprocess: str,
    percentile_low: float,
    percentile_high: float,
) -> transforms.Compose:
    steps = [transforms.ToTensor()]
    if image_preprocess == "percentile":
        steps.append(PercentileNormalizeTensor(low=percentile_low, high=percentile_high))
    elif image_preprocess not in {"legacy_imagenet", "sobel_edge_3channel"}:
        raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")
    steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def make_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    image_preprocess: str,
    percentile_low: float,
    percentile_high: float,
) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if image_preprocess == "sobel_edge_3channel":
        steps.append(SobelEdge3Channel())
    elif image_preprocess in {"legacy_imagenet", "percentile"}:
        steps.append(transforms.Grayscale(num_output_channels=3))
    else:
        raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")
    steps.extend(
        make_tensor_transform(
            mean=mean,
            std=std,
            image_preprocess=image_preprocess,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        ).transforms
    )
    return transforms.Compose(steps)


def make_model_input_image(
    image: Image.Image,
    image_size: int,
    image_preprocess: str,
) -> Image.Image:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    if image_preprocess == "sobel_edge_3channel":
        return SobelEdge3Channel()(image)
    if image_preprocess in {"legacy_imagenet", "percentile"}:
        return image.convert("L").convert("RGB")
    raise ValueError(f"Unsupported image_preprocess: {image_preprocess}")


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
        self.transform = make_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            image_preprocess=image_preprocess,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
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
    image_preprocess: str
    percentile_low: float
    percentile_high: float


@dataclass
class PredictionRecord:
    path: Path
    pred_idx: int
    pred_label: str
    confidence: float
    drone_score: float
    probabilities: Dict[str, float]
    source_label: str | None
    source_binary_label: str | None


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


def infer_source_label(path: Path) -> str | None:
    text = str(path).upper()
    match = re.search(r"(?:^|__)(" + "|".join(["CLEAN", "WIFI", "DRONE", "NON_DRONE"]) + r")(?:__|$)", path.stem.upper())
    if match:
        return match.group(1).lower()
    for label in ("clean", "wifi", "drone", "non_drone"):
        if re.search(rf"(^|[^A-Z0-9]){re.escape(label.upper())}([^A-Z0-9]|$)", text):
            return label
    return None


def source_label_to_binary_label(source_label: str | None) -> str | None:
    if source_label == "drone":
        return "drone"
    if source_label in {"clean", "wifi", "non_drone"}:
        return "non_drone"
    return None


def resolve_source_binary_label(source_label: str | None, source_binary_label: str) -> str | None:
    if source_binary_label == "auto":
        return source_label_to_binary_label(source_label)
    if source_binary_label == "unknown":
        return None
    return source_binary_label


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
    image_preprocess = str(checkpoint.get("image_preprocess", DEFAULT_IMAGE_PREPROCESS))
    percentile_low = float(checkpoint.get("percentile_low", DEFAULT_PERCENTILE_LOW))
    percentile_high = float(checkpoint.get("percentile_high", DEFAULT_PERCENTILE_HIGH))
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
    return ModelBundle(
        model=model,
        arch=arch,
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_size=image_size,
        mean=mean,
        std=std,
        image_preprocess=image_preprocess,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )


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
    cam_uint8 = (cam * 255).clip(0, 255).astype(np.uint8)
    return np.repeat(cam_uint8[..., None], 3, axis=2)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    heatmap_rgb = colorize_cam(cam)
    overlay = (image_rgb.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha).clip(0, 255)
    return heatmap_rgb, overlay.astype(np.uint8)


def resolve_occlusion_fill_value(image_rgb: np.ndarray, fill_mode: str, fill_value: int) -> int:
    gray = image_rgb[..., 0]
    if fill_mode == "value":
        return int(np.clip(fill_value, 0, 255))
    if fill_mode == "white":
        return 255
    if fill_mode == "black":
        return 0
    if fill_mode == "mean":
        return int(np.clip(round(float(gray.mean())), 0, 255))
    if fill_mode == "median":
        return int(np.clip(round(float(np.median(gray))), 0, 255))
    raise ValueError(f"Unsupported occlusion fill mode: {fill_mode}")


def parse_occlusion_compare_modes(value: str) -> List[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"white", "black", "mean", "median", "value"}
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"Unsupported occlusion compare mode(s): {', '.join(invalid)}")
    return modes


def make_occluded_image(image_rgb: np.ndarray, cam: np.ndarray, threshold: float, fill_value: int) -> tuple[np.ndarray, np.ndarray]:
    mask = cam >= threshold
    occluded = image_rgb.copy()
    occluded[mask] = np.uint8(fill_value)
    return occluded, mask


@torch.no_grad()
def predict_image_probabilities(
    model: nn.Module,
    image_rgb: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    image = Image.fromarray(image_rgb)
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


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


@torch.no_grad()
def predict_all(
    image_paths: Sequence[Path],
    bundle: ModelBundle,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    source_binary_label: str,
) -> List[PredictionRecord]:
    dataset = SpectrogramDataset(
        image_paths,
        image_size=bundle.image_size,
        mean=bundle.mean,
        std=bundle.std,
        image_preprocess=bundle.image_preprocess,
        percentile_low=bundle.percentile_low,
        percentile_high=bundle.percentile_high,
    )
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
            source_label = infer_source_label(path)
            records.append(
                PredictionRecord(
                    path=path,
                    pred_idx=int(pred_idx),
                    pred_label=pred_label,
                    confidence=confidence,
                    drone_score=drone_score,
                    probabilities=probabilities,
                    source_label=source_label,
                    source_binary_label=resolve_source_binary_label(source_label, source_binary_label),
                )
            )
        offset += len(pred)
    return records


def filter_records_by_case(records: Sequence[PredictionRecord], case_filter: str) -> List[PredictionRecord]:
    if case_filter == "all":
        return list(records)
    if case_filter == "correct_drone":
        return [record for record in records if record.source_binary_label == "drone" and record.pred_label == "drone"]
    if case_filter == "correct_non_drone":
        return [record for record in records if record.source_binary_label == "non_drone" and record.pred_label == "non_drone"]
    if case_filter == "false_positive_drone":
        return [record for record in records if record.source_binary_label == "non_drone" and record.pred_label == "drone"]
    if case_filter == "false_negative_drone":
        return [record for record in records if record.source_binary_label == "drone" and record.pred_label != "drone"]
    if case_filter == "clean_correct":
        return [record for record in records if record.source_label == "clean" and record.pred_label == "non_drone"]
    if case_filter == "wifi_false_positive":
        return [record for record in records if record.source_label == "wifi" and record.pred_label == "drone"]
    raise ValueError(f"Unsupported CASE_FILTER: {case_filter}")


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
    target_class = target_class.lower()
    if target_class == "predicted":
        return int(predicted_idx)
    if target_class in class_to_idx:
        return int(class_to_idx[target_class])
    try:
        return int(target_class)
    except ValueError as exc:
        raise ValueError(
            f"TARGET_CLASS must be 'predicted', 'both', a class name, or a class index. Got: {target_class}"
        ) from exc


def resolve_target_specs(target_class: str, class_to_idx: Dict[str, int], predicted_idx: int) -> List[tuple[str, int]]:
    """Return one or more CAM targets.

    - drone: explain evidence for class drone
    - predicted: explain evidence for the predicted class
    - both: generate predicted class CAM first, then target drone CAM.
      This is equivalent to running ClassifierOutputTarget(target) twice:
      once for the predicted class and once for the drone class.
    """
    value = target_class.lower()
    if value == "both":
        pred_idx = int(predicted_idx)
        pred_name = next((name for name, idx in class_to_idx.items() if idx == pred_idx), "predicted")
        specs: List[tuple[str, int]] = [(f"predicted_{pred_name}", pred_idx)]
        if "drone" in class_to_idx:
            drone_idx = int(class_to_idx["drone"])
            if drone_idx != pred_idx:
                specs.append(("target_drone", drone_idx))
        return specs
    idx = resolve_target_idx(target_class, class_to_idx, predicted_idx)
    name = next((name for name, class_idx in class_to_idx.items() if class_idx == idx), str(idx))
    return [(name, idx)]


def save_panel(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    occluded_rgb: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("white")
    panels = [
        ("Model Input", image_rgb),
        ("Grad-CAM", cam),
        ("Heatmap", heatmap_rgb),
        ("Overlay", overlay_rgb),
        ("Occluded", occluded_rgb),
    ]
    for ax, (label, data) in zip(axes, panels):
        if label == "Grad-CAM":
            ax.imshow(data, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(data)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def cam_focus_stats(cam: np.ndarray) -> Dict[str, float]:
    return {
        "cam_mean": float(cam.mean()),
        "cam_std": float(cam.std()),
        "hot_area_ratio_0_60": float((cam >= 0.60).mean()),
        "hot_area_ratio_0_75": float((cam >= 0.75).mean()),
    }


def summarize_occlusion_tests(results: Sequence[dict], compare_modes: Sequence[str]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for mode in compare_modes:
        drone_drops = [
            float(result["occlusion_tests"][mode]["drone_score_drop"])
            for result in results
            if mode in result.get("occlusion_tests", {})
            and result["occlusion_tests"][mode]["drone_score_drop"] is not None
        ]
        target_drops = [
            float(result["occlusion_tests"][mode]["target_score_drop"])
            for result in results
            if mode in result.get("occlusion_tests", {})
            and result["occlusion_tests"][mode].get("target_score_drop") is not None
        ]
        scores_after = [
            float(result["occlusion_tests"][mode]["drone_score"])
            for result in results
            if mode in result.get("occlusion_tests", {})
        ]
        target_scores_after = [
            float(result["occlusion_tests"][mode]["target_score"])
            for result in results
            if mode in result.get("occlusion_tests", {})
        ]
        if not drone_drops:
            continue
        summary[mode] = {
            "mean_target_score_drop": float(np.mean(target_drops)) if target_drops else None,
            "min_target_score_drop": float(np.min(target_drops)) if target_drops else None,
            "max_target_score_drop": float(np.max(target_drops)) if target_drops else None,
            "mean_occluded_target_score": float(np.mean(target_scores_after)) if target_scores_after else None,
            "target_drop_gt_0_10_count": int(sum(drop > 0.10 for drop in target_drops)),
            "target_drop_gt_0_50_count": int(sum(drop > 0.50 for drop in target_drops)),
            "target_negative_drop_count": int(sum(drop < 0.0 for drop in target_drops)),
            "mean_drone_score_drop": float(np.mean(drone_drops)),
            "min_drone_score_drop": float(np.min(drone_drops)),
            "max_drone_score_drop": float(np.max(drone_drops)),
            "mean_occluded_drone_score": float(np.mean(scores_after)) if scores_after else None,
            "drone_drop_gt_0_10_count": int(sum(drop > 0.10 for drop in drone_drops)),
            "drone_drop_gt_0_50_count": int(sum(drop > 0.50 for drop in drone_drops)),
            "drone_negative_drop_count": int(sum(drop < 0.0 for drop in drone_drops)),
        }
    return summary


def count_record_values(records: Sequence[PredictionRecord], attr_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = getattr(record, attr_name)
        key = str(value) if value is not None else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarize_predictions(records: Sequence[PredictionRecord]) -> Dict[str, object]:
    known_records = [record for record in records if record.source_binary_label is not None]
    confusion_counts: Dict[str, int] = {}
    for record in known_records:
        key = f"{record.source_binary_label}->{record.pred_label}"
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

    correct_count = sum(record.pred_label == record.source_binary_label for record in known_records)
    false_positive_drone_count = sum(
        record.source_binary_label == "non_drone" and record.pred_label == "drone" for record in known_records
    )
    false_negative_drone_count = sum(
        record.source_binary_label == "drone" and record.pred_label != "drone" for record in known_records
    )
    return {
        "source_label_counts": count_record_values(records, "source_label"),
        "source_binary_label_counts": count_record_values(records, "source_binary_label"),
        "prediction_counts": count_record_values(records, "pred_label"),
        "known_label_count": len(known_records),
        "correct_count": int(correct_count),
        "accuracy": float(correct_count / len(known_records)) if known_records else None,
        "false_positive_drone_count": int(false_positive_drone_count),
        "false_negative_drone_count": int(false_negative_drone_count),
        "confusion_counts": dict(sorted(confusion_counts.items())),
    }


def summarize_predictions_by_source_label(records: Sequence[PredictionRecord]) -> Dict[str, dict]:
    source_labels = sorted({record.source_label or "unknown" for record in records})
    grouped_summary: Dict[str, dict] = {}
    for source_label in source_labels:
        group = [record for record in records if (record.source_label or "unknown") == source_label]
        drone_scores = [record.drone_score for record in group if not np.isnan(record.drone_score)]
        grouped_summary[source_label] = {
            "count": len(group),
            "prediction_counts": count_record_values(group, "pred_label"),
            "binary_label_counts": count_record_values(group, "source_binary_label"),
            "mean_drone_score": float(np.mean(drone_scores)) if drone_scores else None,
            "min_drone_score": float(np.min(drone_scores)) if drone_scores else None,
            "max_drone_score": float(np.max(drone_scores)) if drone_scores else None,
        }
    return grouped_summary


def generate_grad_cam_outputs(
    records: Sequence[PredictionRecord],
    bundle: ModelBundle,
    device: torch.device,
    output_dir: Path,
    target_class: str,
    target_layer_name: str,
    overlay_alpha: float,
    occlusion_threshold: float,
    occlusion_fill_mode: str,
    occlusion_fill_value: int,
    occlusion_compare_modes: Sequence[str],
) -> List[dict]:
    target_layer_desc, target_layer = choose_target_layer(bundle.model, bundle.arch, target_layer_name)
    grad_cam = GradCAM(bundle.model, target_layer)
    model_input_transform = make_tensor_transform(
        mean=bundle.mean,
        std=bundle.std,
        image_preprocess=bundle.image_preprocess,
        percentile_low=bundle.percentile_low,
        percentile_high=bundle.percentile_high,
    )
    idx_to_class = {idx: name for name, idx in bundle.class_to_idx.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    try:
        for index, record in enumerate(tqdm(records, desc="Grad-CAM", unit="image")):
            with Image.open(record.path) as img:
                image = make_model_input_image(
                    image=img,
                    image_size=bundle.image_size,
                    image_preprocess=bundle.image_preprocess,
                )
            image_rgb = np.asarray(image)
            x = model_input_transform(image).unsqueeze(0).to(device)
            drone_idx = bundle.class_to_idx.get("drone")

            for target_name, target_idx in resolve_target_specs(target_class, bundle.class_to_idx, record.pred_idx):
                target_label = idx_to_class.get(target_idx, str(target_idx))
                cam, logits = grad_cam(x, target_idx=target_idx, output_size=(bundle.image_size, bundle.image_size))
                logits_np = logits[0].detach().cpu().numpy()
                logits_by_class = {name: float(logits_np[idx]) for name, idx in bundle.class_to_idx.items()}
                non_drone_logit = logits_by_class.get("non_drone")
                drone_logit = logits_by_class.get("drone")
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

                heatmap_rgb, overlay_rgb = overlay_heatmap(image_rgb, cam, alpha=overlay_alpha)
                panel_fill_value = resolve_occlusion_fill_value(image_rgb, occlusion_fill_mode, occlusion_fill_value)
                occluded_rgb, occlusion_mask = make_occluded_image(
                    image_rgb,
                    cam,
                    threshold=occlusion_threshold,
                    fill_value=panel_fill_value,
                )
                occluded_probs = predict_image_probabilities(bundle.model, occluded_rgb, model_input_transform, device)
                target_score_before = float(probs[target_idx])
                target_score_after = float(occluded_probs[target_idx])
                drone_score_before = float(probs[drone_idx]) if drone_idx is not None else float("nan")
                drone_score_after = float(occluded_probs[drone_idx]) if drone_idx is not None else float("nan")
                occlusion_tests = {}
                for fill_mode in occlusion_compare_modes:
                    test_fill_value = resolve_occlusion_fill_value(image_rgb, fill_mode, occlusion_fill_value)
                    test_occluded_rgb, _ = make_occluded_image(
                        image_rgb,
                        cam,
                        threshold=occlusion_threshold,
                        fill_value=test_fill_value,
                    )
                    test_probs = predict_image_probabilities(bundle.model, test_occluded_rgb, model_input_transform, device)
                    test_target_score = float(test_probs[target_idx])
                    test_drone_score = float(test_probs[drone_idx]) if drone_idx is not None else float("nan")
                    occlusion_tests[fill_mode] = {
                        "fill_value": int(test_fill_value),
                        "target_score": test_target_score,
                        "target_score_drop": target_score_before - test_target_score,
                        "drone_score": test_drone_score,
                        "drone_score_drop": drone_score_before - test_drone_score if drone_idx is not None else None,
                        "probabilities": {name: float(test_probs[idx]) for name, idx in bundle.class_to_idx.items()},
                    }
                stem = f"{index:03d}_{record.pred_label}_cam_{target_name}_{sanitize_stem(record.path)}"
                panel_path = output_dir / f"{stem}_panel.png"

                title = (
                    f"{record.path.name} | pred={record.pred_label} ({record.confidence:.3f}) | "
                    f"target={target_name}/{target_label}\n"
                    f"logits before softmax: non_drone={format_optional_float(non_drone_logit)} | "
                    f"drone={format_optional_float(drone_logit)}\n"
                    f"softmax target={target_score_before:.3f}->{target_score_after:.3f} | "
                    f"softmax drone={drone_score_before:.3f}->{drone_score_after:.3f} | "
                    f"mask={(occlusion_mask.mean() * 100):.1f}%"
                )
                save_panel(image_rgb, cam, heatmap_rgb, overlay_rgb, occluded_rgb, title=title, output_path=panel_path)

                result = {
                    "image": str(record.path),
                    "source_label": record.source_label,
                    "source_binary_label": record.source_binary_label,
                    "prediction": record.pred_label,
                    "is_correct": record.pred_label == record.source_binary_label if record.source_binary_label is not None else None,
                    "confidence": record.confidence,
                    "drone_score": drone_score_before,
                    "occluded_drone_score": drone_score_after,
                    "drone_score_drop": drone_score_before - drone_score_after if drone_idx is not None else None,
                    "target_class": target_name,
                    "target_label": target_label,
                    "target_index": int(target_idx),
                    "logits": logits_by_class,
                    "non_drone_logit": non_drone_logit,
                    "drone_logit": drone_logit,
                    "probabilities": {name: float(probs[idx]) for name, idx in bundle.class_to_idx.items()},
                    "occluded_probabilities": {name: float(occluded_probs[idx]) for name, idx in bundle.class_to_idx.items()},
                    "target_score": target_score_before,
                    "occluded_target_score": target_score_after,
                    "target_score_drop": target_score_before - target_score_after,
                    "occlusion_threshold": float(occlusion_threshold),
                    "occlusion_fill_mode": occlusion_fill_mode,
                    "occlusion_fill_value": int(panel_fill_value),
                    "occlusion_tests": occlusion_tests,
                    "occlusion_area_ratio": float(occlusion_mask.mean()),
                    "target_layer": target_layer_desc,
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
    parser.add_argument("--target-class", default=TARGET_CLASS, help="Class name, class index, 'predicted', or 'both'.")
    parser.add_argument("--target-layer", default=TARGET_LAYER, help="Optional exact module name from model.named_modules().")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--sort-by", default=SORT_BY, choices=["drone_score", "confidence", "path"])
    parser.add_argument(
        "--source-binary-label",
        default=SOURCE_BINARY_LABEL,
        choices=["auto", "drone", "non_drone", "unknown"],
        help="Ground-truth binary label override. Use 'drone' when every source image is a drone sample.",
    )
    parser.add_argument(
        "--case-filter",
        default=CASE_FILTER,
        choices=[
            "all",
            "correct_drone",
            "correct_non_drone",
            "false_positive_drone",
            "false_negative_drone",
            "clean_correct",
            "wifi_false_positive",
        ],
    )
    parser.add_argument("--overlay-alpha", type=float, default=OVERLAY_ALPHA)
    parser.add_argument("--occlusion-threshold", type=float, default=OCCLUSION_THRESHOLD)
    parser.add_argument("--occlusion-fill-mode", default=OCCLUSION_FILL_MODE, choices=["white", "black", "mean", "median", "value"])
    parser.add_argument("--occlusion-fill-value", type=int, default=OCCLUSION_FILL_VALUE)
    parser.add_argument("--occlusion-compare-modes", default=OCCLUSION_COMPARE_MODES)
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
        source_binary_label=args.source_binary_label,
    )
    filtered_records = filter_records_by_case(records, args.case_filter)
    selected_records = select_records(filtered_records, sort_by=args.sort_by, top_k=args.top_k)
    occlusion_compare_modes = parse_occlusion_compare_modes(args.occlusion_compare_modes)
    cam_results = generate_grad_cam_outputs(
        records=selected_records,
        bundle=bundle,
        device=device,
        output_dir=output_dir,
        target_class=args.target_class,
        target_layer_name=args.target_layer,
        overlay_alpha=args.overlay_alpha,
        occlusion_threshold=args.occlusion_threshold,
        occlusion_fill_mode=args.occlusion_fill_mode,
        occlusion_fill_value=args.occlusion_fill_value,
        occlusion_compare_modes=occlusion_compare_modes,
    )

    drone_scores = [record.drone_score for record in records if not np.isnan(record.drone_score)]
    summary = {
        "method": "grad_cam_cnn_binary",
        "arch": bundle.arch,
        "checkpoint": str(checkpoint_path),
        "source": str(source_path),
        "output_dir": str(output_dir),
        "scanned_images": len(records),
        "case_filter": args.case_filter,
        "case_filtered_images": len(filtered_records),
        "generated_images": len(cam_results),
        "sort_by": args.sort_by,
        "target_class": args.target_class,
        "source_binary_label_mode": args.source_binary_label,
        "class_names": bundle.class_names,
        "class_to_idx": bundle.class_to_idx,
        "scanned_prediction_summary": summarize_predictions(records),
        "scanned_prediction_by_source_label": summarize_predictions_by_source_label(records),
        "selected_prediction_summary": summarize_predictions(selected_records),
        "selected_prediction_by_source_label": summarize_predictions_by_source_label(selected_records),
        "image_size": bundle.image_size,
        "image_mean": bundle.mean,
        "image_std": bundle.std,
        "image_preprocess": bundle.image_preprocess,
        "percentile_low": bundle.percentile_low,
        "percentile_high": bundle.percentile_high,
        "occlusion_threshold": args.occlusion_threshold,
        "occlusion_fill_mode": args.occlusion_fill_mode,
        "occlusion_fill_value": args.occlusion_fill_value,
        "occlusion_compare_modes": occlusion_compare_modes,
        "occlusion_summary": summarize_occlusion_tests(cam_results, occlusion_compare_modes),
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
