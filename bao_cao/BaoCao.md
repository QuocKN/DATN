**Chữ đậm**

_Chữ nghiêng_

~~Gạch Ngang~~

# Tiêu đề lớn

## Tiêu đề nhỏ

### Nhỏ hơn nữa

- Item 1
- Item 2
- Item 3

1. Bước 1
2. Bước 2
3. Bước 3

[Google](https://google.com)

Dùng lệnh `npm install`

```js
console.log("Hello world");
```

- [x] Hoàn thành
- [ ] Chưa làm

| Tên  | Tuổi |
| ---- | ---- |
| An   | 20   |
| Bình | 22   |

> Đây là trích dẫn

# 1toanbin

## Fine tune

### Not refactor

- Các model được train trên tập balanced
  - Dinov2 ViT s 14
    ![alt text](fine_tune/DINOv2/report/Tu_thu/2toan/results_chart.png)
  - Swin v2 s Tranformer Small
  - ResNet18
  - ResNet50

### Kết quả nhận diện trên dữ liệu tự thu

Mỗi bộ `drone1` và `drone2` gồm 200 ảnh spectrogram. Tỷ lệ dưới đây là số ảnh được mô hình nhận diện là drone.

| Mô hình                               |  Drone1 | Tỷ lệ Drone1 |  Drone2 | Tỷ lệ Drone2 |
| ------------------------------------- | ------: | -----------: | ------: | -----------: |
| DINOv2 ViT-S/14                       | 182/200 |          91% | 198/200 |          99% |
| ResNet18                              |  25/200 |        12.5% |  47/200 |        23.5% |
| ResNet50                              |  41/200 |        20.5% |  33/200 |        16.5% |
| EfficientNet-B2                       | 193/200 |        96.5% | 191/200 |        95.5% |
| VGG13-BN                              | 189/200 |        94.5% | 198/200 |          99% |
| Swin V2 Small                         |   2/200 |           1% |   3/200 |         1.5% |
| OpenCLIP ViT-B/16                     |  17/200 |         8.5% |  12/200 |           6% |
| ConvNeXt V2 Small (fine-tune 3 stage) |  99/200 |        49.5% | 122/200 |          61% |

Kết quả tốt nhất:

- `drone1`: EfficientNet-B2 đạt 96.5%.
- `drone2`: DINOv2 và VGG13-BN đạt 99%.
- VGG13-BN, EfficientNet-B2 và DINOv2 có khả năng tổng quát hóa tốt nhất trên hai bộ dữ liệu tự thu.
- Swin V2 Small suy giảm mạnh trên dữ liệu tự thu, chỉ đạt 1% và 1.5% trên hai tập.

  Feature Extraction-based Classification: <50% cho tất cả model

Vậy dataset drone của bạn không chỉ có drone sạch. Nó gồm:

CLEAN: drone không có Wi-Fi/Bluetooth gây nhiễu.
WIFI: drone cùng Wi-Fi.
BLUE: drone cùng Bluetooth.
BOTH: drone cùng Wi-Fi và Bluetooth.
Đây là điểm tốt vì lớp drone đã có nhiều điều kiện nhiễu. Tuy nhiên, lớp non-drone cũng phải có các trường hợp tương ứng:

Drone Non-drone đối chứng
Drone sạch Nhiễu nền
Drone + Wi-Fi Chỉ Wi-Fi
Drone + Bluetooth Chỉ Bluetooth
Drone + Wi-Fi + Bluetooth Wi-Fi + Bluetooth, không drone
Nếu thiếu các mẫu đối chứng, model có thể học nhầm:

Có Wi-Fi/Bluetooth mạnh → drone

thay vì học phần pattern riêng của drone.

Bộ non-drone tối thiểu chỉ cần thu ba loại:

Nhiễu nền.
Wi-Fi.
Bluetooth.
Wi-Fi + Bluetooth nếu có thời gian.
Mỗi loại khoảng 2 phút, cùng thiết bị và cấu hình thu. Tổng cộng 6–8 phút là đủ để làm thí nghiệm ban đầu.

Quan trọng nhất là sinh spectrogram với cấu hình thống nhất với drone dataset: 60 MHz, STFT 2048, cửa sổ 0.1 s, percentile 60. Nếu thiết bị chỉ hỗ trợ 28 MHz, nên sinh lại cả dữ liệu drone thực tế và non-drone ở 28 MHz, rồi giữ dataset drone cũ làm dữ liệu bổ sung.

11:41 AM
làm sao để biết thiết bị hỗ trợ max bao nhiêu
11:42 AM

Worked for 19s
Hiện máy không phát hiện bladeRF qua USB, nên chưa thể đọc trực tiếp giới hạn thiết bị.

Khi cắm thiết bị, chạy:

bladeRF-cli -p
bladeRF-cli -e "info"
Thử thiết lập sample rate:

bladeRF-cli -e "set samplerate rx 60M"
Thiết bị sẽ báo sample rate thực tế được chấp nhận hoặc báo lỗi nếu vượt giới hạn.

Có thể thử lần lượt:

bladeRF-cli -e "set samplerate rx 28M"
bladeRF-cli -e "set samplerate rx 40M"
bladeRF-cli -e "set samplerate rx 60M"
Lưu ý:

Sample rate tối đa phụ thuộc đúng phiên bản bladeRF, kết nối USB 3.0 và khả năng ghi dữ liệu của máy.
Thiết bị hỗ trợ 60 MS/s không có nghĩa máy luôn ghi liên tục ổn định ở mức đó.
Trong project, dữ liệu tự thu đang dùng 28 MS/s; đây thường là mức thực tế ổn định hơn.
Khi cắm thiết bị, lsusb phải xuất hiện bladeRF. Hiện chỉ thấy chuột và bàn phím.
11:42 AM
