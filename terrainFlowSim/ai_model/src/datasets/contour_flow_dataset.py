"""
등고선 이미지 <-> 배수(수심) 이미지 짝을 읽는 범용 Dataset.

data/contours/<name>.png  (입력: 등고선 이미지)
data/flow/<name>.png      (타깃: 배수 시뮬레이션 결과 이미지, 예: 첨부한 파란 정사각형)

같은 파일명을 가진 두 이미지를 한 쌍으로 취급한다. 데이터셋은 아직 비어 있고
나중에 채워 넣을 예정이므로, 폴더가 비어 있으면 바로 알 수 있도록 에러 메시지를 남긴다.
"""

import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

RAIN_MIN_MM = 20.0
RAIN_MAX_MM = 80.0
TIME_MIN_S = 0.0
TIME_MAX_S = 120.0  # 이 프로젝트 시뮬레이션은 total_time_s가 항상 120s로 고정됨


def normalize_rain(rain_mm):
    """강우량(mm) -> [0,1] 정규화 (FiLM 조건 스칼라용). prepare_samples.py/
    run_rain_variants.py가 20~80mm 범위에서 뽑으므로 그 범위 기준."""
    return max(0.0, min(1.0, (rain_mm - RAIN_MIN_MM) / (RAIN_MAX_MM - RAIN_MIN_MM)))


def normalize_time(t_s):
    """시뮬레이션 경과 시각(s) -> [0,1] 정규화 (FiLM 조건 스칼라용)."""
    return max(0.0, min(1.0, (t_s - TIME_MIN_S) / (TIME_MAX_S - TIME_MIN_S)))


def _group_key(name):
    """파일명에서 시점 접미사(_t0, _t1..), 증강 접미사(_r0.._r3, _f0.._f3),
    강우 변형 접미사(_rv1, _rv2..)를 순서대로 뗀 원본 지형 키. 회전/반전 증강본,
    같은 지형에 강우 조건만 다르게 돌린 _rv1/_rv2 변형, 같은 시뮬레이션의 다른
    시점(_t0.._t5)까지 전부 결국 같은 지형이므로 train/val 분리 시 같은 그룹으로
    묶어야 데이터 누수(같은 지형이 train/val에 나뉘어 들어가는 것)를 막을 수 있다.
    순서를 고정하지 않고 매번 끝에서부터 매칭되는 접미사를 순차로 떼어내므로,
    파일명에 접미사가 일부만 있어도(예전 방식, _t 접미사 없음) 안전하게 동작한다."""
    base = os.path.splitext(name)[0]
    base = re.sub(r"_t\d+$", "", base)
    base = re.sub(r"_(?:r[0-3]|f[0-3])$", "", base)
    base = re.sub(r"_rv\d+$", "", base)
    return base


class ContourFlowDataset(Dataset):
    def __init__(self, contours_dir, flow_dir, image_size=256, augment=False, names=None,
                 rain_dir=None, rain_values_json=None, time_values_json=None, elev_dir=None, elev_only=False):
        """rain_dir을 주면 등고선(3채널)에 강우 채널(1채널)을 이어붙여 4채널 입력을
        만든다 (data/rain/<name>.png, 균일한 값의 타일 이미지로 강우량을 인코딩) - 채널
        결합 방식, config_raincond.yaml에서 씀.

        rain_values_json / time_values_json을 주면 대신 강우량/시점을 이미지가
        아니라 스칼라 "cond"로 배치에 실어서 반환한다 - FiLM 방식(config_film.yaml,
        config_film_time.yaml)에서 씀. 둘 다 {파일명: 값} 형태의 JSON 경로이고,
        둘 다 주면 cond = [정규화된 강우량, 정규화된 시점] 2차원 벡터가 된다
        (강우량만 주면 기존과 동일하게 1차원).

        rain_dir=None, rain_values_json=None, time_values_json=None(기본)이면
        기존과 동일하게 3채널만 반환."""
        self.contours_dir = contours_dir
        self.flow_dir = flow_dir
        self.rain_dir = rain_dir

        # elev_dir: 고도 격자를 입력 4번째 채널로 붙인다 (scripts/build_elev_channel.py).
        #
        # 등고선 그림은 contourf(levels=20) 으로 고도를 20단계(한 칸 약 1.6m)로 양자화하는데,
        # 배수 물길은 그보다 얕게 파여 있어 같은 색 띠 안에 묻힌다. 실측으로 물길 위치
        # 일치 IoU 가 0.216 까지 떨어진다(scripts/diagnose_input_info.py). 즉 모델이 못 그리는
        # 구조가 애초에 입력에 없다. 원본 고도를 그대로 넣어 그 정보를 되살린다.
        self.elev_dir = elev_dir

        # elev_only: 등고선을 빼고 고도만 입력으로 준다 (in_channels=1).
        #
        # 고도를 4번째 채널로 "추가"만 했더니 모델이 사실상 무시했다 - 채널을 0으로 지워도
        # 출력 변화가 표준편차의 1.1%에 불과했다. 손실은 넓은 영역이 지배하고 등고선만으로도
        # 그건 맞출 수 있으니, 얇은 물길에만 쓸모 있는 채널을 배울 유인이 없었던 것이다.
        # 선택지를 없애면 모델은 고도를 쓸 수밖에 없고, 그래야 "입력 정보가 병목인가"를
        # 실제로 검증할 수 있다.
        self.elev_only = elev_only
        self.image_size = image_size

        self.rain_values = None
        if rain_values_json is not None:
            with open(rain_values_json, "r", encoding="utf-8") as f:
                self.rain_values = json.load(f)

        self.time_values = None
        if time_values_json is not None:
            with open(time_values_json, "r", encoding="utf-8") as f:
                self.time_values = json.load(f)

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
            if self.rain_values is not None and name not in self.rain_values:
                continue
            if self.time_values is not None and name not in self.time_values:
                continue
            if rain_dir is not None:
                rpath = os.path.join(rain_dir, name)
                if os.path.exists(cpath) and os.path.exists(fpath) and os.path.exists(rpath):
                    pairs.append((cpath, fpath, rpath))
            elif os.path.exists(cpath) and os.path.exists(fpath):
                pairs.append((cpath, fpath, None))
        if not pairs:
            raise RuntimeError(
                f"'{contours_dir}'와 '{flow_dir}'" + (f", '{rain_dir}'" if rain_dir else "") +
                " 사이에 파일명이 일치하는 쌍을 찾지 못했습니다."
            )
        self.pairs = pairs

        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])
        self.rain_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.augment = augment

    def _load_elev(self, name, size):
        """(1, size, size) 텐서로 고도를 읽는다. 값 범위는 다른 채널과 같은 [-1, 1].

        고도는 시점(_t0..)에 따라 변하지 않으므로 파일에는 시점 접미사가 없다.
        """
        key = re.sub(r"_t\d+$", "", os.path.splitext(name)[0])
        path = os.path.join(self.elev_dir, key + ".npy")
        arr = np.load(path).astype(np.float32)
        t = torch.from_numpy(arr)[None, None]
        if t.shape[-1] != size or t.shape[-2] != size:
            t = torch.nn.functional.interpolate(t, size=(size, size),
                                                 mode="bilinear", align_corners=False)
        return (t[0] - 0.5) / 0.5

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cpath, fpath, rpath = self.pairs[idx]
        flow = Image.open(fpath).convert("RGB")
        target_t = self.base_transform(flow)

        if self.elev_only:
            # 등고선 PNG 는 아예 열지 않는다 (입력에 안 쓰이므로)
            input_t = self._load_elev(os.path.basename(cpath), self.image_size)
        else:
            input_t = self.base_transform(Image.open(cpath).convert("RGB"))

        if self.elev_dir is not None and not self.elev_only:
            # 증강보다 먼저 붙인다 - 고도는 지형이라 등고선과 같이 뒤집혀야 한다
            # (강우 채널은 공간 구조가 없어서 증강 뒤에 붙는 것과 대비된다).
            input_t = torch.cat([input_t, self._load_elev(os.path.basename(cpath),
                                                           input_t.shape[-1])], dim=0)

        rain_t = None
        if rpath is not None:
            rain = Image.open(rpath).convert("L")
            rain_t = self.rain_transform(rain)  # (1, H, W)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                input_t = torch.flip(input_t, dims=[-1])
                target_t = torch.flip(target_t, dims=[-1])
                if rain_t is not None:
                    rain_t = torch.flip(rain_t, dims=[-1])
            if torch.rand(1).item() < 0.5:
                input_t = torch.flip(input_t, dims=[-2])
                target_t = torch.flip(target_t, dims=[-2])
                if rain_t is not None:
                    rain_t = torch.flip(rain_t, dims=[-2])

        if rain_t is not None:
            input_t = torch.cat([input_t, rain_t], dim=0)  # (4, H, W)

        out = {"input": input_t, "target": target_t, "name": os.path.basename(cpath)}
        cond_parts = []
        if self.rain_values is not None:
            cond_parts.append(normalize_rain(self.rain_values[os.path.basename(cpath)]))
        if self.time_values is not None:
            cond_parts.append(normalize_time(self.time_values[os.path.basename(cpath)]))
        if cond_parts:
            out["cond"] = torch.tensor(cond_parts, dtype=torch.float32)
        return out


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
