# Context Báo cáo Đồ án Tốt nghiệp

## Thông tin chung

Tên đề tài:

"Ứng dụng trí tuệ nhân tạo trong phát hiện thiết bị bay không người lái dựa trên tín hiệu RF"

Loại đề tài:

- Đồ án tốt nghiệp theo hướng nghiên cứu.
- Chuyên ngành liên quan đến IoT, AI và xử lý tín hiệu.
- Mục tiêu là xây dựng hệ thống phát hiện UAV/Drone dựa trên tín hiệu RF ở băng tần 2.4 GHz.

---

# Mục tiêu nghiên cứu

Xây dựng quy trình:

```text
RF Signal
    ↓
Signal Processing
    ↓
Spectrogram
    ↓
Deep Learning
    ↓
Drone Detection
```

Hệ thống phải:

- Phát hiện được tín hiệu drone.
- Hoạt động trên dữ liệu thực tế.
- So sánh hiệu quả của nhiều kiến trúc AI khác nhau.
- Đánh giá khả năng tổng quát hóa khi dữ liệu huấn luyện và dữ liệu triển khai khác miền.

---

# Động lực nghiên cứu

Drone ngày càng được sử dụng rộng rãi trong:

- Giám sát
- Quay phim
- Vận chuyển
- Nông nghiệp

Tuy nhiên drone cũng tạo ra các nguy cơ:

- Xâm phạm quyền riêng tư
- Bay vào khu vực cấm
- Đe dọa an ninh

Các phương pháp phát hiện drone phổ biến:

- Radar
- Camera
- Âm thanh
- RF

Trong đó RF có ưu điểm:

- Chi phí thấp
- Hoạt động cả ngày và đêm
- Không phụ thuộc điều kiện ánh sáng
- Có thể phát hiện từ khoảng cách xa

---

# Khoảng trống nghiên cứu

Nhiều nghiên cứu hiện nay:

- Đạt độ chính xác rất cao trên dataset.
- Đánh giá chủ yếu trên dữ liệu được thu trong môi trường kiểm soát.
- Ít đánh giá khả năng hoạt động trên dữ liệu RF thực tế khác miền.

Ngoài ra:

- Phần lớn nghiên cứu chỉ báo cáo accuracy.
- Chưa phân tích sâu khả năng tổng quát hóa của các backbone AI.
- Chưa giải thích tại sao một số mô hình hoạt động tốt hơn trên dữ liệu thực tế.

Đây là khoảng trống mà đồ án hướng tới.

---

# Câu hỏi nghiên cứu

RQ1:

Liệu các mô hình thị giác hiện đại có thể học được đặc trưng của tín hiệu RF thông qua spectrogram hay không?

RQ2:

Kiến trúc nào phù hợp hơn cho bài toán nhận dạng drone từ RF spectrogram?

- CNN
- Vision Transformer
- Self-Supervised Vision Transformer

RQ3:

Mô hình nào có khả năng tổng quát hóa tốt hơn khi gặp dữ liệu RF ngoài tập huấn luyện?

RQ4:

Có thể giải thích sự khác biệt về khả năng tổng quát hóa thông qua không gian đặc trưng hay không?

---

# Giả thuyết nghiên cứu

H1:

Các mô hình Self-Supervised Vision Transformer có khả năng học đặc trưng RF tốt hơn CNN truyền thống.

H2:

Độ chính xác cao trên dataset không đồng nghĩa với khả năng hoạt động tốt trên dữ liệu thực tế.

H3:

Khả năng tổng quát hóa của mô hình liên quan đến mức độ phân tách của không gian đặc trưng.

---

# Dữ liệu sử dụng

## Dataset công khai

### RFUAV

Được sử dụng làm tập dữ liệu chính.

Đặc điểm:

- Drone DJI Mavic 3 Pro.
- Dữ liệu RF đã được công bố công khai.
- Chuyển đổi thành spectrogram để huấn luyện.

### DroneRF

Được sử dụng để tham khảo và đối chiếu.

---

## Dữ liệu tự thu

Thiết bị:

- bladeRF

Phần mềm:

- GNU Radio

Thông số:

```text
Center Frequency = 2.4375 GHz
Sample Rate = 28 MHz
Bandwidth = 28 MHz
```

Mục đích:

- Đánh giá khả năng tổng quát hóa của mô hình.
- Mô phỏng điều kiện triển khai thực tế.

---

# Quy trình nghiên cứu

## Bước 1

Thu thập tín hiệu RF.

Nguồn:

- Dataset công khai
- Thiết bị bladeRF

---

## Bước 2

Tiền xử lý tín hiệu.

Bao gồm:

- Đọc IQ
- Chuẩn hóa
- Kiểm tra clipping
- Kiểm tra burst

---

## Bước 3

Biến đổi tín hiệu sang miền thời gian – tần số.

Phương pháp:

```text
STFT
```

Đầu ra:

```text
Spectrogram
```

---

## Bước 4

Huấn luyện mô hình AI.

Các mô hình được khảo sát:

- ResNet50
- EfficientNet-B2
- DINOv2

---

## Bước 5

Đánh giá.

Tiêu chí:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Ngoài ra:

- Silhouette Score
- Domain Separation
- Visualization bằng t-SNE hoặc UMAP

---

# Kết quả nổi bật hiện tại

## ResNet50

- Accuracy trên dataset cao.
- Kết quả trên dữ liệu tự thu thấp.

Khoảng:

```text
~30%
```

---

## EfficientNet-B2

- Accuracy rất cao.
- Tổng quát hóa tốt.

Khoảng:

```text
>90%
```

trên dữ liệu tự thu.

---

## DINOv2

Cho kết quả tốt nhất.

Đặc biệt:

- Không gian đặc trưng phân tách rõ.
- Generalization tốt.
- Hiệu quả cao trên dữ liệu thực tế.

Ví dụ:

```text
Silhouette ≈ 0.50
```

cao hơn đáng kể so với:

```text
ResNet50 ≈ 0.11
EfficientNet-B2 ≈ 0.08
```

---

# Đóng góp của đồ án

## Đóng góp kỹ thuật

Xây dựng quy trình:

```text
RF Signal
    ↓
Spectrogram
    ↓
Deep Learning
    ↓
Drone Detection
```

trên dữ liệu RF thực tế.

---

## Đóng góp thực nghiệm

So sánh:

- CNN
- Vision Transformer
- Self-Supervised Transformer

trong cùng một bài toán RF.

---

## Đóng góp nghiên cứu

Phân tích khả năng tổng quát hóa của các backbone thị giác hiện đại trên dữ liệu RF.

Chứng minh rằng:

```text
Accuracy trên dataset
≠
Hiệu quả trên dữ liệu thực tế
```

và chỉ ra lợi thế của Self-Supervised Transformer trong điều kiện domain shift.

---

# Thông điệp chính của báo cáo

Kết quả nghiên cứu cho thấy việc đánh giá mô hình chỉ dựa trên tập dữ liệu huấn luyện và kiểm thử công khai là chưa đủ đối với các hệ thống phát hiện drone bằng RF.

Các mô hình Self-Supervised Vision Transformer như DINOv2 thể hiện khả năng tổng quát hóa tốt hơn đáng kể so với CNN truyền thống khi được triển khai trên dữ liệu RF thực tế, cho thấy tiềm năng ứng dụng của các mô hình học biểu diễn hiện đại trong các hệ thống phát hiện UAV ngoài môi trường phòng thí nghiệm.
