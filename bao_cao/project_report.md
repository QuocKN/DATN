# Context chi tiet du an DATN

## 1. Ten de tai va dinh huong

Ten de tai:

> Nghiên cứu phương pháp phát hiện Drone dựa trên tín hiệu RF

Du an huong toi bai toan phat hien UAV/Drone trong mien tan so 2.4 GHz bang cach bien doi tin hieu IQ thanh anh spectrogram, sau do khai thac cac backbone thi giac may tinh de phan loai.

Pipeline tong quat cua codebase:

```text
RF Recording / IQ file
    -> Doc va chia chunk IQ
    -> Tien xu ly bien do / clipping / spike neu can
    -> STFT hai phia + fftshift
    -> Spectrogram PNG 224x224
    -> Feature Extraction-based Classification / fine-tune binary classifier
    -> Detection Result va bao cao JSON/PNG
```

Pham vi chinh hien tai la **Binary Classification**:

```text
non_drone = 0
drone     = 1
```

Ghi chu: Phan `SNREstimation` khong dua vao noi dung chinh; neu can se chuyen thanh phu luc ky thuat. Bao cao chinh chi tap trung vao pipeline spectrogram va AI.

---

## 2. Muc tieu nghien cuu

Muc tieu ky thuat:

- Xay dung quy trinh xu ly tin hieu RF tu du lieu IQ sang spectrogram.
- Huan luyen va danh gia cac mo hinh AI phat hien Drone Signal va Non-Drone Signal.
- So sanh hai chien luoc hoc chinh: fine-tune va Feature Extraction-based Classification.
- Kiem tra kha nang tong quat hoa khi mo hinh gap du lieu khac mien voi du lieu huan luyen.

Muc tieu nghien cuu:

- Khong chi bao cao accuracy tren dataset noi mien.
- Phan tich su khac biet giua ket qua test noi mien va ket qua tren du lieu tu thu/thuc te.
- Tim backbone phu hop hon cho RF spectrogram trong dieu kien domain shift.

---

## 3. Cau truc codebase

### 3.1. Nhom tien xu ly va sinh spectrogram

Thu muc chinh:

```text
nkquoc/
  base/
    iq_spectrogram_core.py
    iq_preprocessing.py
    iq_waveform_plotter.py
  bin_spectrogram/
  dat_spectrogram/
  mat_spectrogram/
  spectrogram_v1/
  pre_dataset/
  psd/
```

Vai tro:

- `nkquoc/base/iq_spectrogram_core.py`: tinh STFT, fftshift va luu spectrogram PNG.
- `nkquoc/base/iq_preprocessing.py`: kiem tra bien do IQ, clipping 12-bit, spike/burst va cac ham sua/lam mong xung bat thuong.
- `nkquoc/bin_spectrogram`, `nkquoc/dat_spectrogram`, `nkquoc/mat_spectrogram`: doc cac dinh dang IQ khac nhau va chuyen sang spectrogram.
- `nkquoc/pre_dataset`: cac script batch de tao dataset spectrogram tu raw dataset.
- `converse_data`: cac script chuyen doi dinh dang du lieu thuc nghiem.

Thong so spectrogram mac dinh trong code:

```text
Image size      = 224 x 224
Sample rate     = 28 MHz voi du lieu tu thu
STFT point      = 2048 trong converter chinh
Duration/chunk  = 0.05 s trong converter chinh
STFT window     = hamming
STFT mode       = two-sided
FFT shift       = co
Col denoise     = co trong code, khong trinh bay nhu buoc phuong phap chinh
Colormap        = jet
```

### 3.2. Nhom fine-tune supervised

Thu muc chinh:

```text
fine_tune/
  ResNet18/ (khong dua vao bao cao chinh)
  ResNet50/
  EfficientNet_B2/
  ConvNext_V2/
  Swin_Small/
  DINOv2/
```

Dac diem chung:

- Dau vao la anh spectrogram.
- Nhan mac dinh: `class_names = ["non_drone", "drone"]`.
- Dung `CrossEntropyLoss`.
- Dung `AdamW`, `CosineAnnealingLR`, early stopping theo `valid_macro_f1`.
- Co tuy chon class weights hoac `WeightedRandomSampler` de xu ly imbalance.
- Ket qua duoc luu thanh checkpoint `.pt`, summary `.json`, confusion matrix va chart.

Mot so backbone/dinh huong dang co:

- CNN: ResNet50, EfficientNet-B2.
- ConvNeXt: ConvNeXt V2.
- Transformer/self-supervised backbone: Swin Small, DINOv2.
- Cac backbone co thu muc nhung khong dua vao bao cao chinh: ResNet18, OpenCLIP va mot so thu muc thu nghiem phu.

### 3.3. Nhom Feature Extraction-based Classification

Thu muc chinh:

```text
linear_probe/
  ResNet18/
  ResNet50/
  EfficientNet_B2/
  ConvNext_Tiny/
  ConvNext_V2/
  Swin_Small/
  ViT_L_16/
  DINOv2/
```

Y tuong:

```text
Spectrogram
    -> Frozen feature extractor
    -> Embedding
    -> StandardScaler
    -> Logistic Regression / Linear SVM  (dung LR)
    -> Drone / Non-Drone
```

Trong `linear_probe/DINOv2/train_linear_probe_dinov2.py`, backbone mac dinh la `dinov2_vits14`, embedding 384 chieu, classifier ung vien gom logistic regression va linear SVM.

### 3.4. Nhom bao cao va ket qua

Thu muc chinh:

```text
bao_cao/
fine_tune/*/report/
linear_probe/*/report/
```

Cac file JSON/PNG trong `report` ghi lai ket qua detect tren tung tap nguon, vi du `Tu_thu`, `DroneDetect`, `Mavic_pro_2G`, `Inspire`, `Non_Drone`.

---

## 4. Du lieu va don vi mau

### 4.1. Don vi du lieu trong code

Theo `CONTEXT.md`, du an dung cac khai niem:

- **RF Recording**: file IQ lien tuc, tu do sinh ra nhieu spectrogram.
- **Spectrogram Sample**: mot anh spectrogram 224x224 tu mot doan tin hieu RF.
- **Drone Signal**: lop duong tinh.
- **Non-Drone Signal**: lop am tinh.
- **Dataset Split**: train, validation, test.
- **Detection Result**: nhan du doan va confidence cua tung sample.
- **Macro F1**: metric uu tien khi so sanh mo hinh.

### 4.2. Nguon du lieu trong repo

Code dang tham chieu cac nguon:

- Dataset public/balanced binary dataset trong cac duong dan Linux/Kaggle.
- Du lieu tu thu bang bladeRF/GNU Radio trong cac duong dan Windows nhu `G:\DATN_DATA`.
- Cac tap detect rieng: `Tu_thu`, `DroneDetect`, `RFUAV`, `Inspire`, `Mavic_pro_2G`, `MavicRC1`, `Non_Drone`.

Thong so thu tin hieu tu context hien co:

```text
Center Frequency = 2.4375 GHz
Sample Rate      = 28 MHz
Bandwidth        = 28 MHz
Format           = int16 IQ hoac float32 IQ tuy nguon
```

### 4.3. Dataset split xuat hien trong summary

Balanced binary dataset thuong co:

```text
Train: 2774
  non_drone: 1387
  drone:     1387

Valid: 346
  non_drone: 173
  drone:     173

Test: 348
  non_drone: 174
  drone:     174
```

Mot cau hinh DINOv2/RF tu thu co split lech lop hon:

```text
Train: 3649
  non_drone: 1387
  drone:     2262

Valid: 873
  non_drone: 173
  drone:     700

Test: 798
  non_drone: 174
  drone:     624
```

---

## 5. Pipeline xu ly tin hieu

### 5.1. Doc IQ

Du lieu IQ duoc doc theo dang:

```text
I, Q, I, Q, ...
```

Code ho tro nhieu dinh dang:

- `.bin`: thuong la int16 IQ tu bladeRF.
- `.dat`: co `float32_iq` hoac int16 tuy tham so.
- `.mat`/HDF5: thong qua cac script trong `mat_spectrogram` va `spectrogram_v1`.

### 5.2. Kiem tra va tien xu ly IQ

`iq_preprocessing.py` cho phep:

- In min/max cua I, Q va bien do.
- Tinh percentile 90/95/99/99.9 cua bien do.
- Kiem tra clipping theo ADC 12-bit voi nguong `-2048..2047`.
- Phat hien spike/burst bat thuong.
- Sua clipping bang noi suy.
- Nen spike bien do cao nhung giu pha.
- Lam mong cac run bien do cao ngan de giam soc doc trong spectrogram.

Phan nay co trong code de kiem tra chat luong tin hieu va giai thich cac anh spectrogram co soc doc manh, burst hoac saturation; tuy nhien khong trinh bay nhu mot buoc thuc nghiem chinh.

### 5.3. Sinh spectrogram

Ham chinh:

```text
compute_spectrogram()
save_spectrogram_image()
```

Quy trinh:

```text
IQ chunk
    -> cat segment theo sample_rate * duration_time
    -> scipy.signal.stft
    -> fftshift truc tan so
    -> magnitude dB = 10 * log10(abs(spectrum) + 1e-12)
    -> tuy chon tru nen cot theo percentile
    -> ve anh khong truc/label
    -> resize ve 224x224
```

Luu y: Do anh spectrogram duoc luu khong co truc va label, mo hinh hoc truc tiep tu mau nang luong thoi gian-tan so, khong hoc tu metadata hien thi.

---

## 6. Pipeline huan luyen

### 6.1. Fine-tune binary classifier

Pipeline chung:

```text
Spectrogram dataset
    -> Resize
    -> ToTensor
    -> Normalize theo ImageNet hoac backbone-specific mean/std
    -> Backbone pretrained
    -> Linear head 2 lop
    -> CrossEntropyLoss
    -> Macro F1 / Accuracy / Confusion Matrix
```

Chien luoc toi uu:

```text
Optimizer       = AdamW
Scheduler       = CosineAnnealingLR
Early stopping  = valid_macro_f1
Seed            = 42
Batch size      = 32 trong phan lon script
```

Mot so script co chien luoc freeze/unfreeze:

- DINOv2: co the freeze backbone hoac fine-tune N block cuoi.
- ConvNeXt V2: co the fine-tune N stage cuoi.
- EfficientNet-B2: co bien the train last N block.

### 6.2. Feature Extraction-based Classification

Muc dich:

- Kiem tra chat luong embedding cua backbone khi khong fine-tune end-to-end.
- Giam chi phi huan luyen.
- So sanh kha nang tach lop cua feature extractor.

Metric chinh:

- Macro F1.
- Precision/Recall/F1 tung lop.

## 7. Ket qua hien co trong repo

### 7.1. Ket qua noi mien tren balanced binary dataset

Nhieu summary cho thay ket qua noi mien gan nhu bao hoa:

| Model | Split test | Test accuracy | Test macro F1 | Ghi chu |
| --- | ---: | ---: | ---: | --- |
| ResNet50 fine-tune | 348 | 1.000 | 1.000 | `fine_tune/ResNet50/resnet50_binary_runs/balanced_summary.json` |
| EfficientNet-B2 fine-tune | 348 | 1.000 | 1.000 | `fine_tune/EfficientNet_B2/efficientnet_b2_binary_runs/summary.json` |
| DINOv2 ViT-S/14 fine-tune | 348 | 1.000 | 1.000 | `fine_tune/DINOv2/dinov2_binary_runs/balanced_summary.json` |
| ConvNeXt V2 fine-tune | 348 | 1.000 | 1.000 | `fine_tune/ConvNext_V2/convnextv2_binary_runs/balanced_summary.json` |

Nhan xet:

- Ket qua noi mien qua cao de lam bang chung duy nhat cho kha nang ung dung.
- Can dat trong bao cao rang test noi mien co the chua du kho, hoac train/valid/test co cung mien du lieu.
- Gia tri nghien cuu nen tap trung vao danh gia tren tap tu thu, tap nguon moi va cac case ngoai phan phoi.

### 7.2. Ket qua DINOv2 voi tap RF tu thu/khong can bang

File `fine_tune/DINOv2/dinov2_binary_runs/rf_tu_thu_summary.json` ghi:

```text
Train: 3649
Valid: 873
Test:  798

Test accuracy  = 0.994987
Test macro F1  = 0.992710
Confusion      = [[174, 0], [4, 620]]
```

Nhan xet:

- DINOv2 dat ket qua rat cao tren cau hinh co them tin hieu tu thu.
- Loi sai chinh trong summary nay la 4 mau drone bi du doan thanh non-drone.
- Day la ket qua manh hon so voi viec chi bao cao balanced dataset 100%, vi tap nay phan anh du lieu lech lop va gan thuc te hon.

### 7.3. Ket qua detect rieng le

Repo co nhieu file `results.json` va `results_chart.png` trong:

```text
fine_tune/*/report/
linear_probe/*/report/
```

Cac file nay phu hop de dua vao chuong thuc nghiem:

- So sanh ty le detect drone tren tung nguon du lieu.
- Quan sat confidence distribution.
- Phan tich false positive tren Non-Drone.
- Phan tich false negative tren DroneDetect/Tu_thu.

---

## 8. Cau hoi nghien cuu nen trinh bay

RQ1. Anh spectrogram tao tu IQ RF co du thong tin de phan biet Drone Signal va Non-Drone Signal hay khong? 

RQ2. Cac backbone pretrained tu thi giac may tinh co chuyen giao tot sang RF spectrogram hay khong?

RQ3. Ket qua noi mien 100% co phan anh kha nang hoat dong tren du lieu thuc te hay khong?

RQ4. Fine-tune end-to-end va Feature Extraction-based Classification khac nhau the nao ve do on dinh, chi phi va kha nang tong quat hoa?

RQ5. DINOv2, ConvNeXt va CNN truyen thong khac nhau the nao khi gap du lieu tu thu hoac du lieu ngoai phan phoi?

---

## 9. Dong gop cua du an

### 9.1. Dong gop ky thuat

- Xay dung pipeline doc IQ va sinh spectrogram tu nhieu dinh dang raw RF.
- Dong nhat dau ra spectrogram 224x224 de dung voi nhieu backbone AI.
- Xay dung va so sanh hai huong huan luyen chinh: fine-tune va Feature Extraction-based Classification.
- Tu dong luu checkpoint, summary JSON, confusion matrix va chart detect.

### 9.2. Dong gop thuc nghiem

- So sanh nhieu backbone tren cung bai toan Drone/Non-Drone.
- Co ket qua noi mien tren balanced dataset va ket qua tren tap RF tu thu.
- Co log/report rieng cho nhieu nguon du lieu, giup phan tich domain shift.

### 9.3. Dong gop nghien cuu

Thong diep nen nhan manh:

> Accuracy noi mien cao khong du de ket luan mo hinh phat hien drone RF co kha nang trien khai. Can danh gia them tren du lieu RF tu thu, du lieu khac nguon va cac truong hop non-drone gan voi moi truong thuc.

---

## 10. Han che hien tai

- Bai toan chinh van la binary classification, chua mo hinh hoa day du cac tac nhan RF nhu WiFi/Bluetooth.
- Nhieu script detect dung hang so hard-code cho checkpoint, source dir va output dir.
- Nhieu duong dan tuyet doi phu thuoc may ca nhan/Kaggle/Linux, gay kho tai lap tren may khac.
- Mot so ket qua noi mien dat 100%, can canh bao nguy co dataset qua de, cung mien, hoac co kha nang trung pattern nguon.
- Chua co mot entrypoint thong nhat cho toan pipeline tu raw IQ den report cuoi.
- Chua co file cau hinh chung cho dataset, split, model, checkpoint va output.
- Mot so file report co hau to `old`, `refactor`, `nhap`, hoac file trung lap; cac file nay duoc giu trong repo de truy vet, nhung khong dua vao bao cao chinh.

---

## 11. Huong phat trien de bao cao thuyet phuc hon

### 11.1. Chuan hoa thuc nghiem domain shift

Nen chia bang:

```text
Train domain: public/balanced dataset
Test domain : Tu_thu / DroneDetect / RFUAV / Non_Drone
```

Sau do so sanh:

- Accuracy/Macro F1 neu co nhan that.
- Drone detection rate neu tap chi gom mot nguon.
- False positive rate tren Non-Drone.
- Confidence distribution.

### 11.2. Mo rong multi-class

Mo rong multi-class khong dua vao bao cao chinh hien tai, nhung co the la huong phat trien sau:

```text
background
wifi
bluetooth
drone
drone + wifi
drone + bluetooth
drone + wifi + bluetooth
```

Chi nen dua vao bao cao khi co du lieu va ket qua thuc nghiem ro rang.

### 11.3. Giai thich mo hinh

Co the bo sung:

- t-SNE/UMAP embedding theo class va theo source domain.
- Confusion matrix theo tung nguon.
- Score distribution theo tap Tu_thu/Non_Drone.
- Grad-CAM/attention map neu can giai thich mo hinh nhin vao vung nao cua spectrogram.

---

## 12. De xuat cau truc bao cao

### Chuong 1. Gioi thieu

- Dong luc phat hien drone bang RF.
- Ly do chon spectrogram + deep learning.
- Van de domain shift giua dataset va du lieu thuc te.

### Chuong 2. Co so ly thuyet

- Tin hieu IQ.
- STFT va spectrogram.
- CNN, Vision Transformer, Transformer phan cap va self-supervised backbone.
- Metric: accuracy, precision, recall, F1, macro F1, confusion matrix.

### Chuong 3. Phuong phap de xuat

- Pipeline raw IQ -> spectrogram -> classifier.
- Tien xu ly IQ va kiem tra clipping/spike.
- Kien truc fine-tune.
- Feature Extraction-based Classification.

### Chuong 4. Thuc nghiem

- Mo ta dataset/split.
- Cau hinh huan luyen.
- Bang ket qua noi mien.
- Bang ket qua tren du lieu tu thu/khac mien.
- Phan tich loi va confidence.

### Chuong 5. Ket luan

- Tom tat ket qua.
- Han che.
- Huong phat trien: multi-class RF environment, OOD, explainability.

---

## 13. Quyet dinh pham vi bao cao

Cac quyet dinh sau duoc dung de gioi han pham vi bao cao chinh:

1. `SNREstimation/` khong nam trong noi dung chinh; neu can se dua vao phu luc ky thuat.
2. `one_class/ResNet18` khong su dung trong bao cao.
3. DroneRF khong dua vao bao cao vi khong co ket qua truc tiep trong code/result hien tai.
4. Cac file `old`, `refactor`, `nhap` va report trung lap duoc giu trong repo de truy vet, nhung khong dua vao noi dung bao cao.
5. Khong dat muc tieu chuan hoa CLI/config cho cac script detect trong pham vi bao cao nay.
6. Ket qua 100% tren balanced dataset van duoc trinh bay, nhung chi xem la ket qua noi mien va can kem canh bao.
7. Thu nghiem 3-class `wifi_blue` khong dua vao bao cao chinh.
