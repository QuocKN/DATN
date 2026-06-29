# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt


# def rgb_to_gray_edge(image_path, save_path=None, show=True):

#     # Đọc ảnh RGB
#     img = np.array(Image.open(image_path).convert("RGB"))

#     # Gray
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
#     gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

#     # Sobel
#     gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

#     # Magnitude
#     edge = np.sqrt(gx * gx + gy * gy)
#     edge = edge / (edge.max() + 1e-6)

#     # Ghép thành 3 channel
#     out = np.stack(
#         [
#             gray,
#             gray,
#             edge,
#         ],
#         axis=-1,
#     )


#     out_uint8 = (out * 255).astype(np.uint8)

#     if save_path is not None:
#         Image.fromarray(out_uint8).save(save_path)

#     if show:

#         plt.figure(figsize=(14,4))

#         plt.subplot(1,4,1)
#         plt.title("Original")
#         plt.imshow(img)
#         plt.axis("off")

#         plt.subplot(1,4,2)
#         plt.title("Gray")
#         plt.imshow(gray,cmap="gray")
#         plt.axis("off")

#         plt.subplot(1,4,3)
#         plt.title("Edge")
#         plt.imshow(edge,cmap="gray")
#         plt.axis("off")

#         plt.subplot(1,4,4)
#         plt.title("CNN Input")
#         plt.imshow(out_uint8)
#         plt.axis("off")

#         plt.tight_layout()
#         plt.show()

#     return out_uint8

# rgb_to_gray_edge(
#     r"e:\Data_22_6\balanced_dataset_drone_chuan_full_non_done_fix_env_drone\drone\train\CLEAN\MP2_ON\spectrogram__CLEAN__MP2_ON__MAV_0000_00__w00061.png",
#     save_path="output.png",
#     show=True
# )

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def rgb_to_gray_edge(img):
    """
    Input:
        img: RGB image (numpy array HxWx3)
    Output:
        out_uint8: HxWx3
            Channel 0 = Gray
            Channel 1 = Gray
            Channel 2 = Edge
    """

    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

    # Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (edge.max() + 1e-6)
    edge3 = np.stack([edge, edge, edge], axis=-1)
    edge_uint8 = (edge3 * 255).astype(np.uint8)
    return edge_uint8

    # 3-channel
    # out = np.stack(
    #     [
    #         gray,
    #         gray,
    #         edge,
    #     ],
    #     axis=-1,
    # )

    # return (out * 255).astype(np.uint8)


def convert_folder(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    image_files = [
        p for p in input_root.rglob("*")
        if p.suffix.lower() in exts
    ]

    print(f"Found {len(image_files)} images.")

    for img_path in tqdm(image_files):

        # Giữ nguyên cấu trúc thư mục
        relative_path = img_path.relative_to(input_root)
        save_path = output_root / relative_path

        save_path.parent.mkdir(parents=True, exist_ok=True)

        img = np.array(Image.open(img_path).convert("RGB"))

        out = rgb_to_gray_edge(img)

        Image.fromarray(out).save(save_path)

    print("Done!")


if __name__ == "__main__":

    input_folder = r"e:\Data_22_6\balanced_dataset_drone_chuan_full_non_done_fix_env_drone\non_drone\train\env\c2_spectrograms"

    output_folder = r"E:\Data_22_6\balanced_dataset_drone_chuan_full_non_done_fix_env_drone\non_drone\train\env\c2_spectrograms_edge"

    convert_folder(input_folder, output_folder)