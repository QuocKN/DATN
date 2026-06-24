# import os

# folder_path = r"f:\Data_22_6_spectrogram_0_05\drone"  # Thay bằng đường dẫn của bạn

# image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

# count = 0

# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         ext = os.path.splitext(file)[1].lower()
#         if ext in image_extensions:
#             count += 1

# print(f"Tổng số ảnh: {count}")

from PIL import Image

img = Image.open(r"f:\DroneDetect_V2\BOTH\INS_ON\INS_1100_00_spectrograms\spectrogram_000019.png")

gray = img.convert("L")

gray.save(r"spectrogram__CLEAN__MP2_FY__MAV_0010_00__w00014gray.png")


# import cv2
# import numpy as np

# drone = cv2.imread(r"e:\DATN_Data_21_6\dataset\Drone_0_0_5_full\train\CLEAN\MP2_FY\spectrogram__CLEAN__MP2_FY__MAV_0010_01__w00027.png")
# noise = cv2.imread(r"g:\DATN_DATA\Spectrum\dataset\non_drone\train\bluetooth_wifi_env\home_outdoor2_spectrograms\spectrogram_000006.png")

# drone = drone.astype(np.float32) / 255.
# noise = noise.astype(np.float32) / 255.

# mixed = 0.6 * drone + 0.4 * noise

# mixed = np.clip(mixed, 0, 1)

# cv2.imwrite(
#     "mixed.png",
#     (mixed * 255).astype(np.uint8)
# )