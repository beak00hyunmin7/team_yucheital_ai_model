"""
텐서([-1,1] 정규화) <-> PNG 이미지 변환, 학습 중 미리보기 저장 유틸.
"""

import os

import numpy as np
from PIL import Image


def tensor_to_uint8(img_t):
    """(3,H,W), [-1,1] 텐서 -> (H,W,3) uint8 배열."""
    arr = img_t.detach().cpu().clamp(-1, 1).numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).round().astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def save_tensor_as_png(img_t, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(tensor_to_uint8(img_t)).save(path)


def save_triplet_png(input_t, pred_t, target_t, path):
    """등고선 입력 / 예측 배수 이미지 / 정답 배수 이미지를 가로로 이어붙여 저장."""
    imgs = [tensor_to_uint8(input_t), tensor_to_uint8(pred_t), tensor_to_uint8(target_t)]
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for im in imgs:
        canvas[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(canvas).save(path)
