# Đồ án tốt nghiệp: Phát hiện UAV/Drone bằng tín hiệu RF sử dụng AI

## 1. Mục tiêu đề tài

Xây dựng hệ thống phát hiện drone dựa trên tín hiệu RF trong băng tần ISM 2.4 GHz.

Pipeline tổng quát:

```text
RF Signal (IQ)
      ↓
Tiền xử lý tín hiệu
      ↓
STFT
      ↓
Spectrogram
      ↓
Mô hình AI
      ↓
Drone / Non-Drone
```

Mục tiêu không chỉ đạt độ chính xác cao trên dataset mà còn phải hoạt động tốt trên dữ liệu thực tế tự thu.

---

# 2. Dữ liệu

## 2.1 Dataset công khai

### RFUAV

Nguồn:

- RFUAV Dataset
- DJI Mavic 3 Pro

Đã chuyển tín hiệu IQ thành spectrogram để huấn luyện.

### DroneRF

Dataset RF drone phổ biến.

Thông tin:

- Fs ≈ 60 MHz
- Các chế độ bay khác nhau
- Có tín hiệu nền (background)
- Có tín hiệu drone

---

## 2.2 Dữ liệu tự thu

Thiết bị:

- bladeRF
- GNU Radio

Thông số thường dùng:

```text
Center Frequency = 2.4375 GHz
Sample Rate = 28 MHz
Bandwidth = 28 MHz
Format = int16 IQ
```

Một số file:

```text
1toan.bin        (drone)
non_toan.bin     (background)
PHA_0000_01.dat
```

---

# 3. Pipeline xử lý

## Bước 1: Thu tín hiệu IQ

Dữ liệu IQ được lưu dưới dạng:

```text
I,Q,I,Q,...
```

Kiểu dữ liệu:

```text
int16
```

ADC của bladeRF:

```text
12-bit ADC
→ lưu dưới dạng int16
```

---

## Bước 2: Sinh Spectrogram

Sử dụng STFT.

Ví dụ:

```python
n_fft = 2048
hop_length = 1024
window = hann
fftshift = True
```

Tạo ảnh spectrogram:

```text
Time × Frequency × Power
```

---

## Bước 3: Huấn luyện AI

Bài toán hiện tại:

```text
Binary Classification

Drone
Non-Drone
```

---

# 4. Các mô hình đã thử nghiệm

## CNN truyền thống

### ResNet50

Fine-tune toàn bộ backbone.

Kết quả:

- Accuracy trên dataset cao
- Khả năng tổng quát hóa kém

Khi test trên dữ liệu tự thu:

```text
~30%
```

---

### EfficientNet-B2

Fine-tune toàn bộ backbone.

Kết quả:

```text
>95%
```

trên dataset.

Dữ liệu tự thu:

```text
>90%
```

---

## Vision Transformer

### DINOv2

Backbone:

```text
dinov2_vits14
```

Embedding:

```text
384 chiều
```

Có hai hướng khai thác.

### Hướng 1: Feature Extractor

```text
Spectrogram
    ↓
DINOv2
    ↓
384-d embedding
    ↓
SVM / Logistic / KNN
```

Kết quả:

```text
SVM Accuracy ≈ 89%
```

### Hướng 2: Fine-tune toàn bộ DINOv2

Kết quả trên dataset:

```text
Train ≈ 99%
Validation ≈ 100%
```

Có dấu hiệu overfitting nhẹ.

Tuy nhiên trên dữ liệu thực tế:

```text
~90%
```

vẫn tốt hơn CNN.

---

# 5. Bộ dữ liệu huấn luyện hiện tại

```text
Train: 3649
  - non_drone: 1387
  - drone: 2262

Validation: 873
  - non_drone: 173
  - drone: 700

Test: 798
  - non_drone: 174
  - drone: 624
```

Chiến lược chia dữ liệu:

```text
Split theo source
```

nhằm giảm data leakage.

---

# 6. Kết quả phân tích đặc trưng

Các chỉ số phân tách miền đặc trưng:

| Model           | Silhouette |
| --------------- | ---------- |
| ResNet50        | 0.11       |
| EfficientNet-B2 | 0.08       |
| DINOv2          | 0.50       |

Nhận xét:

- DINOv2 tạo không gian đặc trưng tách biệt hơn đáng kể.
- Có khả năng giải thích cho việc DINOv2 tổng quát hóa tốt hơn trên dữ liệu thực tế.

---

# 7. Giả thuyết nghiên cứu hiện tại

## Vì sao DINOv2 tốt hơn ResNet?

### ResNet

Chủ yếu học:

- Texture cục bộ
- Các mẫu nhiễu
- Đường sáng
- Vệt phổ

Do đó dễ phụ thuộc vào dataset.

### DINOv2

Self-Supervised Vision Transformer.

Có khả năng học:

- Cấu trúc phổ tổng thể
- Quan hệ dài hạn giữa các vùng tín hiệu
- Mẫu FHSS
- Phân bố năng lượng

Nên tổng quát hóa tốt hơn khi gặp dữ liệu mới.

---

# 8. Các vấn đề tín hiệu đã phát hiện

## 8.1 Spectrogram bị sọc dọc

Một số file drone xuất hiện sọc dọc mạnh.

Kết quả kiểm tra:

```text
12-bit clipping ≈ 4%
```

Biên độ:

```text
p90 << p95
p99 >> p90
```

Cho thấy:

```text
Burst rất mạnh
ADC saturation
```

---

## 8.2 Tín hiệu FHSS

Drone sử dụng:

```text
Frequency Hopping Spread Spectrum (FHSS)
```

Theo nguyên lý Fourier:

```text
Xung càng ngắn trong miền thời gian
→ phổ càng rộng trong miền tần số
```

Do đó:

```text
Burst mạnh
→ phổ rộng
→ sọc dọc trên spectrogram
```

---

# 9. Phân tích PSD và SNR

## PSD

Sử dụng:

```python
scipy.signal.welch
```

Mục đích:

- Xác định băng tín hiệu
- Đánh giá năng lượng phổ
- Ước lượng SNR

---

## Kết quả SNR

Ví dụ:

```text
Drone signal
vs
Noise reference
```

Kết quả:

```text
SNR ≈ 32 dB
```

Thông tin phát hiện:

```text
Signal bandwidth ≈ 11.3 MHz
Center ≈ -6.3 MHz
```

---

# 10. Hạn chế hiện tại của đồ án

Pipeline hiện tại gần như:

```text
IQ
 ↓
Spectrogram
 ↓
Fine-tune model
 ↓
Drone / Non-Drone
```

Đóng góp nghiên cứu còn hạn chế.

Các điểm yếu:

## Chỉ có 2 lớp

```text
Drone
Non-Drone
```

---

## Chưa mô hình hóa môi trường thực tế

Chưa có các lớp:

```text
Wifi
Bluetooth
Drone + Wifi
Drone + Bluetooth
Drone + Wifi + Bluetooth
```

---

## Chưa đánh giá OOD

Hiện tượng:

- Ảnh không liên quan
- Spectrogram lạ

vẫn bị dự đoán:

```text
Drone ≈ 0.95
```

---

## Chưa giải thích đầy đủ nguyên nhân DINOv2 tốt hơn

Cần bổ sung:

- t-SNE
- UMAP
- Attention Map
- Grad-CAM
- Phân tích domain shift

---

# 11. Hướng phát triển tiếp theo

## Hướng 1: Multi-class RF Environment

Thu thêm dữ liệu:

```text
Background
Wifi
Bluetooth
Drone
```

Xây dựng:

```text
4-class
```

hoặc:

```text
7-class
```

---

## Hướng 2: Domain Shift Evaluation

Train:

```text
RFUAV
```

Test:

```text
Dữ liệu bladeRF tự thu
```

So sánh:

- ResNet
- EfficientNet
- Swin
- ConvNeXt
- DINOv2

---

## Hướng 3: Explainable AI

Sử dụng:

- t-SNE
- UMAP
- Attention Visualization
- Grad-CAM

để giải thích mô hình học gì từ spectrogram RF.

---

# 12. Đóng góp nghiên cứu tiềm năng

Giá trị nghiên cứu lớn nhất của đề tài không nằm ở việc xây dựng bộ phân loại Drone/Non-Drone, mà nằm ở:

> Đánh giá khả năng tổng quát hóa của các backbone thị giác hiện đại (CNN, Vision Transformer và Self-Supervised Transformer) trên bài toán phát hiện drone từ RF spectrogram dưới điều kiện domain shift giữa dataset công khai và dữ liệu RF thực tế tự thu.

Đây là hướng có khả năng tạo ra đóng góp nghiên cứu rõ ràng và thuyết phục hơn trong báo cáo đồ án.
