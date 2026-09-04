"""
B안: 스칼라 고도격자 -> 스칼라 수심격자 회귀용 Dataset.

build_scalar_dataset.py 가 만든 data_scalar/ 를 읽는다.

입력  : 지형 고도격자, 샘플별 [min,max] -> [-1,1] 정규화 (절대고도 X, 국부기복 O)
출력  : 수심격자, log 정규화 후 [-1,1]   y = log1p(depth/ref)/log_max * 2 - 1
조건  : 강우량(mm) -> [0,1] (FiLM cond 스칼라)

D4 대칭(회전 90/180/270 + 좌우반전) 은 배수가 방향 대칭이므로 로딩 시 즉석 증강.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _norm_rain(rain_mm, lo, hi):
    return float(np.clip((rain_mm - lo) / (hi - lo), 0.0, 1.0))


def depth_to_y(depth_m, ref, log_max):
    """수심[m] -> [-1,1] 학습 타깃."""
    y01 = np.clip(np.log1p(np.maximum(depth_m, 0.0) / ref) / log_max, 0.0, 1.0)
    return y01 * 2.0 - 1.0


def y_to_depth(y, ref, log_max):
    """[-1,1] 예측 -> 수심[m] (inference 에서 사용)."""
    y01 = np.clip((np.asarray(y) + 1.0) / 2.0, 0.0, 1.0)
    return np.expm1(y01 * log_max) * ref


class ScalarFlowDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 128, augment: bool = False,
                 use_flowacc: bool = False):
        self.root = root
        self.image_size = image_size
        self.augment = augment
        self.use_flowacc = use_flowacc
        with open(os.path.join(root, "stats.json"), encoding="utf-8") as f:
            self.stats = json.load(f)
        self.names = list(self.stats[split])
        self.ref = self.stats["depth_ref_m"]
        self.log_max = self.stats["depth_log_max"]
        self.rain_lo = self.stats["rain_min_mm"]
        self.rain_hi = self.stats["rain_max_mm"]
        self.fa_log_max = self.stats.get("flowacc_log_max") or 8.5
        self.in_channels = 2 if use_flowacc else 1

    def __len__(self):
        return len(self.names)

    def _resize(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float()[None, None]  # (1,1,H,W)
        if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
            t = F.interpolate(t, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return t[0]  # (1,H,W)

    def __getitem__(self, idx):
        name = self.names[idx]
        d = np.load(os.path.join(self.root, "samples", name + ".npz"))
        terrain = d["terrain"].astype(np.float32)
        depth = d["depth"].astype(np.float32)
        rain_mm = float(d["rain_mm"])
        flowacc = d["flowacc"].astype(np.float32) if self.use_flowacc and "flowacc" in d else None

        if self.augment:
            k = int(torch.randint(0, 4, (1,)).item())
            flip = torch.rand(1).item() < 0.5
            terrain = np.rot90(terrain, k).copy()
            depth = np.rot90(depth, k).copy()
            if flowacc is not None:
                flowacc = np.rot90(flowacc, k).copy()   # D8 흐름누적은 D4 변환에 equivariant
            if flip:
                terrain = np.fliplr(terrain).copy()
                depth = np.fliplr(depth).copy()
                if flowacc is not None:
                    flowacc = np.fliplr(flowacc).copy()

        # 지형: 샘플별 min-max -> [-1,1]
        lo, hi = float(terrain.min()), float(terrain.max())
        terrain_n = (terrain - lo) / (hi - lo + 1e-6) * 2.0 - 1.0
        inp = self._resize(terrain_n)

        if self.use_flowacc:
            if flowacc is None:  # npz 에 없으면 즉석 계산
                from src.utils.flow_accum import d8_flow_accumulation
                flowacc = d8_flow_accumulation(terrain).astype(np.float32)
            fa_n = np.clip(np.log1p(flowacc) / self.fa_log_max, 0.0, 1.0) * 2.0 - 1.0
            inp = torch.cat([inp, self._resize(fa_n)], dim=0)   # (2, H, W)

        return {
            "input": inp,
            "target": self._resize(depth_to_y(depth, self.ref, self.log_max)),
            "cond": torch.tensor([_norm_rain(rain_mm, self.rain_lo, self.rain_hi)], dtype=torch.float32),
            "name": name,
        }
