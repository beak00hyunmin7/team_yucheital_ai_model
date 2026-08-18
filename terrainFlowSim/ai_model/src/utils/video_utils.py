"""
프레임 시퀀스 -> 영상 저장 유틸.

지금은 정지 이미지(등고선 -> 배수 이미지 1장) 모델만 만들지만, 추후
시계열 예측 모델(예: ConvLSTM, 시간축을 추가한 U-Net, diffusion 등)이
프레임을 연속으로 생성하게 되면 frames_to_video()로 그 프레임들을
그대로 영상(mp4)으로 인코딩할 수 있다.
"""

import os

import cv2
import numpy as np


def frames_to_video(frames, out_path, fps=15):
    """frames: (H, W, 3) uint8 RGB 배열의 리스트."""
    if not frames:
        raise ValueError("frames가 비어 있습니다.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
