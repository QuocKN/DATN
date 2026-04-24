import os
from graphic.RawDataProcessor import RawDataProcessor
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import joblib

# Đường dẫn file .dat và nơi lưu spectrogram tạm thời
# datapack = r'C:\Users\DiepHM\Documents\AIR_0110_00.dat'
# datapack = r'C:\Users\DiepHM\Documents\YUNZHUO%20H16\YUNZHUO H16\YUNZHUO_H16_1_0-1s.iq'
# datapack = r'C:\Users\DiepHM\Documents\Mavic_11.dat'
# datapack = r'C:\Users\DiepHM\Documents\DroneRFA_24-Dataset\DJI_mavic_pro_2G\DJI_mavic_pro_2G.dat'
datapack = r'C:\Users\DiepHM\Documents\data\DroneRF\AR drone\RF Data_10100_H\10100H_0.csv'
spectrogram_path = 'spectrogram_tmp.png'

# Đường dẫn tới scaler, centroid, threshold đã train (dùng lại từ compute_centroid.py)
import sys
sys.path.append('.')
from nkquoc.compute_centroid import transform, model, CACHE_DIR
import numpy as np

scaler_path = os.path.join(CACHE_DIR, 'scaler.joblib')
centroid_path = os.path.join(CACHE_DIR, 'centroid.npy')
threshold_path = os.path.join(CACHE_DIR, 'threshold.npy')
SPECTROGRAM_SIZE = 224


def save_spectrogram_image(aug, extent, out_path, cmap='jet'):
    dpi = 100
    fig = plt.figure(figsize=(SPECTROGRAM_SIZE / dpi, SPECTROGRAM_SIZE / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap=cmap)
    ax.axis('off')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def save_spectrogram(datapack, out_path):
    file_type = np.float32
    sample_rate = 100_000_000
    stft_point = 2048
    duration_time = 0.0005

    read_data = np.fromfile(datapack, dtype=file_type)
    data = read_data[::2] + read_data[1::2] * 1j

    from graphic.RawDataProcessor import STFT
    f, t, Zxx = STFT(data, stft_point=stft_point, fs=sample_rate, duration_time=duration_time, onside=False)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    aug = 10 * np.log10(np.abs(Zxx) + 1e-12)
    extent = [t.min(), t.max(), f.min(), f.max()]
    save_spectrogram_image(aug, extent, out_path)

def extract_embedding_from_image(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(x).squeeze().numpy()
    return feat

def load_scaler_centroid_threshold():
    from sklearn.preprocessing import StandardScaler
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        print('Scaler file not found!')
    centroid = np.load(centroid_path) if os.path.exists(centroid_path) else None
    threshold = np.load(threshold_path) if os.path.exists(threshold_path) else None
    return scaler, centroid, threshold

def check_drone(datapack):
    save_spectrogram(datapack, spectrogram_path)
    feat = extract_embedding_from_image(spectrogram_path)
    scaler, centroid, threshold = load_scaler_centroid_threshold()
    if scaler is None or centroid is None or threshold is None:
        print('Không tìm thấy scaler, centroid hoặc threshold!')
        return
    feat_scaled = scaler.transform([feat])[0]
    dist = np.linalg.norm(feat_scaled - centroid)
    print(f'Distance to centroid: {dist:.4f} | Threshold: {threshold:.4f}')
    if dist < threshold:
        print('==> Có drone!')
    else:
        print('==> Không có drone!')


# --- Streaming detection for large .dat file ---
from collections import deque
import time
import matplotlib.pyplot as plt

def stream_check_drone(datapack, chunk_duration=0.02, window_size=30, detect_consecutive=3, stft_point=2048, sample_rate=100e6, middle_freq=2400e6):
    """
    Quét file .dat lớn theo từng đoạn nhỏ (chunk), sinh spectrogram liên tục và phát hiện drone theo kiểu streaming.
    chunk_duration: thời lượng mỗi spectrogram (giây)
    window_size: số spectrogram gần nhất để tính trung bình
    detect_consecutive: số lần liên tiếp mean_dist < threshold để báo có drone
    """
    scaler, centroid, threshold = load_scaler_centroid_threshold()
    if scaler is None or centroid is None or threshold is None:
        print('Không tìm thấy scaler, centroid hoặc threshold!')
        return

    sample_rate = int(sample_rate)

    # Đọc file .dat thành mảng numpy
    file_type = np.float32
    data = np.fromfile(datapack, dtype=file_type)
    data = data[::2] + data[1::2] * 1j
    slice_point = int(sample_rate * chunk_duration)
    total_chunks = (len(data) // slice_point)
    print(f'Total chunks: {total_chunks}')

    buffer = deque(maxlen=window_size)
    detect_count = 0

    for i in range(total_chunks):
        chunk = data[i*slice_point:(i+1)*slice_point]
        if len(chunk) < slice_point:
            break
        # Sinh spectrogram cho chunk này
        from graphic.RawDataProcessor import STFT
        f, t, Zxx = STFT(chunk, stft_point=stft_point, fs=sample_rate, duration_time=chunk_duration, onside=False)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]
        save_spectrogram_image(aug, extent, spectrogram_path)

        # Trích embedding và tính khoảng cách
        feat = extract_embedding_from_image(spectrogram_path)
        feat_scaled = scaler.transform([feat])[0]
        dist = np.linalg.norm(feat_scaled - centroid)
        buffer.append(dist)

        # Khi buffer đủ window_size
        if len(buffer) == window_size:
            mean_dist = np.mean(buffer)
            if mean_dist < threshold:
                detect_count += 1
            else:
                detect_count = 0
            if detect_count >= detect_consecutive:
                print(f"🚨 DRONE DETECTED at chunk {i}")
                detect_count = 0
            print(f"[Chunk {i}] mean_dist={mean_dist:.4f} | threshold={threshold:.4f}")
        else:
            print(f"[Chunk {i}] dist={dist:.4f}")
        # time.sleep(0.01)  # Bỏ sleep nếu muốn chạy nhanh

    print('Streaming detection done.')

if __name__ == '__main__':
    # check_drone(datapack)
    stream_check_drone(datapack)