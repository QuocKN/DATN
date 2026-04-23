import os
import time
import torch
import numpy as np
from PIL import Image
from collections import deque
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# ========================
# CONFIG
# ========================
TRAIN_DIR = r"C:\Users\DiepHM\Documents\Spectrogram_RFUAV\data\ImageSet-AllDrones-MatlabPipeline\train"
STREAM_DIR = r"C:\Users\DiepHM\Documents\Spectrogram_RFUAV\data\ImageSet-AllDrones-MatlabPipeline\valid"
WINDOW_SIZE = 30       # số chunk (≈ 0.6s nếu 20ms/chunk)
THRESHOLD_PERCENTILE = 95
SLEEP_TIME = 0.02      # giả lập 20ms
CACHE_DIR = r"C:\Users\DiepHM\Documents\data\Spectrogram_RFUAV\data\cache"
TRAIN_CACHE = os.path.join(CACHE_DIR, "X_train.npy")
VALID_CACHE = os.path.join(CACHE_DIR, "X_valid.npy")

# TEST_DIR = 

# ========================
# LOAD DINO
# ========================
print("Loading DINO...")
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval()

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========================
# LOAD IMAGE PATHS
# ========================
def load_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith((".jpg", ".png")):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_or_build_embeddings(image_paths, cache_path, label):
    if os.path.exists(cache_path):
        print(f"Loading cached {label} embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting {label} embeddings... total={len(image_paths)}")
    embeddings = []
    for path in tqdm(image_paths, desc=f"{label} embeddings", unit="img"):
        embeddings.append(extract_embedding(path))

    embeddings = np.array(embeddings)
    np.save(cache_path, embeddings)
    print(f"Saved {label} embeddings to {cache_path}")
    return embeddings

# ========================
# EXTRACT EMBEDDING
# ========================
def extract_embedding(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(x).squeeze().numpy()

    return feat


def main():
    ensure_cache_dir()

    # ========================
    # LOAD IMAGE PATHS
    # ========================
    train_paths = load_images(TRAIN_DIR)
    valid_paths = load_images(STREAM_DIR)
    print(f"Found {len(train_paths)} train images and {len(valid_paths)} valid images")

    # ========================
    # TRAIN: BUILD CENTROID
    # ========================
    X_train = load_or_build_embeddings(train_paths, TRAIN_CACHE, "train")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    centroid = X_train.mean(axis=0)

    # ========================
    # VALID: FIND THRESHOLD
    # ========================
    X_valid = load_or_build_embeddings(valid_paths, VALID_CACHE, "valid")
    X_valid = scaler.transform(X_valid)

    def compute_dist(X):
        return np.linalg.norm(X - centroid, axis=1)

    dist_valid = compute_dist(X_valid)

    threshold = np.percentile(dist_valid, THRESHOLD_PERCENTILE)
    print("Threshold =", threshold)
    # Lưu scaler, centroid, threshold ra file để dùng lại
    scaler_path = os.path.join(CACHE_DIR, 'scaler.joblib')
    centroid_path = os.path.join(CACHE_DIR, 'centroid.npy')
    threshold_path = os.path.join(CACHE_DIR, 'threshold.npy')
    joblib.dump(scaler, scaler_path)
    np.save(centroid_path, centroid)
    np.save(threshold_path, threshold)
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved centroid to {centroid_path}")
    print(f"Saved threshold to {threshold_path}")
    print("Train embeddings shape =", X_train.shape)
    print("Valid embeddings shape =", X_valid.shape)

    return scaler, centroid, threshold, valid_paths, X_train, X_valid

if __name__ == "__main__":
    scaler, centroid, threshold, valid_paths, X_train, X_valid = main()

    # # ========================
    # # REALTIME SIMULATION
    # # ========================
    # print("\n=== START REALTIME DETECTION ===")

    # buffer = deque(maxlen=WINDOW_SIZE)
    # detect_count = 0

    # for path in valid_paths:
    #     feat = extract_embedding(path)
    #     feat = scaler.transform([feat])[0]

    #     dist = np.linalg.norm(feat - centroid)
    #     buffer.append(dist)

    #     # đủ window
    #     if len(buffer) == WINDOW_SIZE:
    #         mean_dist = np.mean(buffer)

    #         if mean_dist < threshold:
    #             detect_count += 1
    #         else:
    #             detect_count = 0

    #         # trigger
    #         if detect_count >= 3:
    #             print(f"🚨 DRONE DETECTED at {path}")
    #             detect_count = 0  # reset tránh spam

    #         print(f"[INFO] mean_dist={mean_dist:.4f}")

    #     time.sleep(SLEEP_TIME)