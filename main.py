from glob import glob

from model import Sobel
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


def np_img_to_tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img[0, ...]  # get the first element since it's batch form


def sobel_torch_version(img_np, torch_sobel):
    img_tensor = np_img_to_tensor(np.float32(img_np))
    img_edged = tensor_to_np_img(torch_sobel(img_tensor))
    img_edged = np.squeeze(img_edged)
    return img_edged


def main():
    img_dir = "sample-imgs/*"
    imgs = sorted(glob(img_dir))
    torch_sobel = Sobel()
    for img in imgs:
        rgb_orig = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        rgb_orig = cv2.resize(rgb_orig, (224, 224))
        tik = time.time()
        rgb_edged = sobel_torch_version(rgb_orig, torch_sobel=torch_sobel)
        tok = time.time()

        torch_time = tok-tik

        tik = time.time()
        rgb_edged_cv2_x = cv2.Sobel(rgb_orig, cv2.CV_64F, 1, 0, ksize=3)
        rgb_edged_cv2_y = cv2.Sobel(rgb_orig, cv2.CV_64F, 0, 1, ksize=3)
        tok = time.time()
        rgb_edged_cv2 = np.sqrt(np.square(rgb_edged_cv2_x), np.square(rgb_edged_cv2_y))
        cv2_time = tok-tik

        print(f"torch time: {torch_time}s, cv2 time: {cv2_time}s")

        rgb_orig = cv2.resize(rgb_orig, (222, 222))
        rgb_edged_cv2 = cv2.resize(rgb_edged_cv2, (222, 222))
        rgb_both = np.concatenate(
            [rgb_orig / 255, rgb_edged / np.max(rgb_edged), rgb_edged_cv2 / np.max(rgb_edged_cv2)], axis=1)

        plt.imshow(rgb_both, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
