import os

folder_path = r"f:\Data_22_6_spectrogram_0_05\drone"  # Thay bằng đường dẫn của bạn

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

count = 0

for root, dirs, files in os.walk(folder_path):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            count += 1

print(f"Tổng số ảnh: {count}")