import os
import cv2
import numpy as np
from pathlib import Path

def add_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.005):
    out = img.copy()
    rnd = np.random.rand(*img.shape[:2])

    out[rnd < prob / 2] = 0
    out[rnd > 1 - prob / 2] = 255

    return out

def add_vertical_bursts(img, num_bursts=8, max_width=4, intensity=60):
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]

    for _ in range(num_bursts):
        x = np.random.randint(0, w)
        bw = np.random.randint(1, max_width + 1)
        val = np.random.uniform(20, intensity)

        out[:, x:x+bw] += val

    return np.clip(out, 0, 255).astype(np.uint8)

def add_horizontal_interference(img, num_lines=4, max_height=3, intensity=40):
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]

    for _ in range(num_lines):
        y = np.random.randint(0, h)
        lh = np.random.randint(1, max_height + 1)
        val = np.random.uniform(10, intensity)

        out[y:y+lh, :] += val

    return np.clip(out, 0, 255).astype(np.uint8)

def add_random_block_noise(img, num_blocks=6, max_block_size=40, intensity=50):
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]

    for _ in range(num_blocks):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)

        bw = np.random.randint(5, max_block_size)
        bh = np.random.randint(5, max_block_size)

        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        val = np.random.uniform(15, intensity)
        out[y1:y2, x1:x2] += val

    return np.clip(out, 0, 255).astype(np.uint8)

def augment_spectrogram(img):
    out = img.copy()

    if np.random.rand() < 0.7:
        out = add_gaussian_noise(out, sigma=np.random.uniform(5, 20))

    if np.random.rand() < 0.4:
        out = add_vertical_bursts(
            out,
            num_bursts=np.random.randint(3, 12),
            max_width=4,
            intensity=np.random.uniform(20, 70),
        )

    if np.random.rand() < 0.3:
        out = add_horizontal_interference(
            out,
            num_lines=np.random.randint(1, 5),
            max_height=3,
            intensity=np.random.uniform(15, 50),
        )

    if np.random.rand() < 0.3:
        out = add_random_block_noise(
            out,
            num_blocks=np.random.randint(2, 8),
            max_block_size=35,
            intensity=np.random.uniform(20, 60),
        )

    if np.random.rand() < 0.2:
        out = add_salt_pepper_noise(out, prob=np.random.uniform(0.001, 0.006))

    return out

def process_folder(input_dir, output_dir, num_aug_per_image=3):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = [".png", ".jpg", ".jpeg"]

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Lưu ảnh gốc
        cv2.imwrite(str(output_dir / img_path.name), img)

        # Tạo ảnh nhiễu
        for i in range(num_aug_per_image):
            aug = augment_spectrogram(img)
            out_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
            cv2.imwrite(str(output_dir / out_name), aug)

# Ví dụ dùng
process_folder(
    input_dir=r"E:\DATN_Data_21_6\dataset_drone_0_5_non_drone_fixed\dronenew\train\CLEAN\MP2_FY",
    output_dir=r"E:\DATN_Data_21_6\dataset_drone_0_5_non_drone_fixed\dronenew\train\CLEAN\MP2_FY_AUG",
    num_aug_per_image=3
)