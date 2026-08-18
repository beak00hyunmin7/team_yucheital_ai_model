"""
등고선 이미지 <-> 배수(수심) 이미지 짝을 읽는 범용 Dataset.

data/contours/<name>.png  (입력: 등고선 이미지)
data/flow/<name>.png      (타깃: 배수 시뮬레이션 결과 이미지, 예: 첨부한 파란 정사각형)

같은 파일명을 가진 두 이미지를 한 쌍으로 취급한다. 데이터셋은 아직 비어 있고
나중에 채워 넣을 예정이므로, 폴더가 비어 있으면 바로 알 수 있도록 에러 메시지를 남긴다.
"""

import glob
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_AUG_SUFFIX_RE = re.compile(r"^(.*)_(?:r[0-3]|f[0-3])(\.[^.]+)$")


def _group_key(name):
    """파일명에서 증강 접미사(_r0.._r3, _f0.._f3)를 뗀 원본 샘플 키.
    회전/반전으로 만든 같은 지형의 증강본들이 train/val에 나뉘어 들어가는
    데이터 누수를 막기 위해, split은 이 키 단위로 묶어서 수행한다."""
    m = _AUG_SUFFIX_RE.match(name)
    return m.group(1) if m else os.path.splitext(name)[0]


class ContourFlowDataset(Dataset):
    def __init__(self, contours_dir, flow_dir, image_size=256, augment=False, names=None):
        self.contours_dir = contours_dir
        self.flow_dir = flow_dir

        if names is not None:
            candidates = [os.path.join(contours_dir, n) for n in names]
        else:
            candidates = sorted(glob.glob(os.path.join(contours_dir, "*.*")))

        if not candidates:
            raise RuntimeError(
                f"'{contours_dir}'에 이미지가 없습니다. "
                f"등고선/배수 이미지 쌍을 각각 data/contours, data/flow에 같은 파일명으로 넣어주세요."
            )

        pairs = []
        for cpath in candidates:
            name = os.path.basename(cpath)
            fpath = os.path.join(flow_dir, name)
            if os.path.exists(cpath) and os.path.exists(fpath):
                pairs.append((cpath, fpath))
        if not pairs:
            raise RuntimeError(
                f"'{contours_dir}'와 '{flow_dir}' 사이에 파일명이 일치하는 쌍을 찾지 못했습니다."
            )
        self.pairs = pairs

        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cpath, fpath = self.pairs[idx]
        contour = Image.open(cpath).convert("RGB")
        flow = Image.open(fpath).convert("RGB")

        input_t = self.base_transform(contour)
        target_t = self.base_transform(flow)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                input_t = torch.flip(input_t, dims=[-1])
                target_t = torch.flip(target_t, dims=[-1])
            if torch.rand(1).item() < 0.5:
                input_t = torch.flip(input_t, dims=[-2])
                target_t = torch.flip(target_t, dims=[-2])

        return {"input": input_t, "target": target_t, "name": os.path.basename(cpath)}


def split_train_val(contours_dir, flow_dir, val_split=0.15, seed=42):
    import numpy as np

    contour_paths = sorted(glob.glob(os.path.join(contours_dir, "*.*")))
    names = [os.path.basename(p) for p in contour_paths if os.path.exists(os.path.join(flow_dir, os.path.basename(p)))]

    groups = {}
    for n in names:
        groups.setdefault(_group_key(n), []).append(n)
    group_keys = sorted(groups.keys())

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(group_keys))
    n_val_groups = max(1, int(len(group_keys) * val_split)) if len(group_keys) > 1 else 0
    val_group_idx = set(perm[:n_val_groups].tolist())

    train_names, val_names = [], []
    for i, gk in enumerate(group_keys):
        bucket = val_names if i in val_group_idx else train_names
        bucket.extend(groups[gk])
    return train_names, val_names
