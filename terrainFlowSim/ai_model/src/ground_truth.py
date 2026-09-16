"""
학습 데이터 파일명 -> OpenFOAM 원본 정답(수심[m]) 조회.

웹앱의 검증 탭과 scripts/evaluate.py 가 "정답"을 같은 방식으로 얻게 하려고 분리했다.
두 곳이 각자 파싱/변환을 구현하면 한쪽만 고쳐져서 서로 다른 숫자를 내는 사고가 난다.

핵심 주의 두 가지:

1. **train/val 구분.** data/contours 에는 학습에 쓴 지형과 안 쓴 지형이 섞여 있다.
   웹앱 샘플 갤러리는 이걸 구분하지 않아서 30개 중 25개가 학습 지형이었다(데이터 누출).
   학습 지형에서의 성능은 val 대비 크게 부풀려진다(MAE 0.0069 vs 0.0115, IoU@5cm 0.58 vs 0.43).
   `split_of()` 로 항상 라벨을 붙일 것.

2. **방향.** 모델 출력(이미지)은 원본 terrain/depth 배열 대비 상하반전이다
   (contourf 와 imshow(origin="lower") 때문). 지표를 계산할 땐 `to_array_orientation()` 로
   되돌리고, 화면에 나란히 띄울 땐 정답도 같은 렌더러로 그려서 방향을 맞춘다.
"""

import os
import re
import sys
import json
import functools

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI_MODEL = os.path.abspath(os.path.join(_HERE, ".."))
_SCRIPTS = os.path.join(_AI_MODEL, "scripts")
for _p in (_AI_MODEL, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.render_fields import (depth_to_norm, render_depth_rgb,  # noqa: E402
                                     DEPTH_VMAX_M)
from prepare_openfoam_dataset import REGIONS, d4_transforms, _pick_timesteps  # noqa: E402

D4 = dict(d4_transforms())
NAME_RE = re.compile(r"^(seoul|busan|gangwon)_(sample_\d+(?:_rv\d+)?)_(r[0-3]|f[0-3])_t(\d+)$")

DEFAULT_CONTOURS = os.path.join(_AI_MODEL, "data", "contours")
DEFAULT_FLOW = os.path.join(_AI_MODEL, "data", "flow")
WET_THRESHOLDS = (0.01, 0.05)


def parse_name(filename):
    """'seoul_sample_0012_rv1_r0_t3.png' -> (region, sample, aug, t_idx) 또는 None."""
    m = NAME_RE.match(os.path.splitext(os.path.basename(filename))[0])
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


def group_key(filename):
    """같은 지형을 가리키는 키. 회전/반전(_r0.._f3), 강우변형(_rv1), 시점(_t0..) 접미사를 뗀다.

    학습 시 train/val 분리에 쓰는 것과 **같은 함수**를 쓴다. 웹앱이 자체 정규식으로
    흉내내던 `_base_terrain_id` 는 파일명에 `_t{idx}` 가 추가된 뒤로 증강 접미사를
    떼지 못해, 같은 지형의 회전본을 서로 다른 지형으로 세고 있었다.
    """
    from src.datasets.contour_flow_dataset import _group_key
    return _group_key(os.path.basename(filename))


def to_array_orientation(depth_img):
    """예측 결과(이미지 방향) -> terrain/depth 배열 방향. 근거는 evaluate.py 주석 참조."""
    return np.ascontiguousarray(np.flipud(depth_img))


@functools.lru_cache(maxsize=1)
def _splits(contours_dir=DEFAULT_CONTOURS, flow_dir=DEFAULT_FLOW, val_split=0.15, seed=42):
    """학습에 쓴 지형 / 안 쓴 지형 집합. 학습 때와 같은 함수·같은 seed 여야 한다."""
    from src.datasets.contour_flow_dataset import split_train_val, _group_key
    tr, va = split_train_val(contours_dir, flow_dir, val_split=val_split, seed=seed)
    return {_group_key(n) for n in tr}, {_group_key(n) for n in va}


def split_of(filename):
    """'train'(학습에 쓰임) / 'val'(안 쓰임) / None(데이터셋 밖)."""
    from src.datasets.contour_flow_dataset import _group_key
    try:
        train_g, val_g = _splits()
    except Exception:  # noqa: BLE001  (데이터 폴더가 없는 배포 환경)
        return None
    key = _group_key(os.path.basename(filename))
    if key in val_g:
        return "val"
    if key in train_g:
        return "train"
    return None


class GroundTruth:
    """(지형, 증강, 시점) -> 실제 수심[m]. 같은 샘플을 연속 조회할 때를 위해 1개 캐시."""

    def __init__(self):
        self._key = None
        self._steps = None
        self._meta = None

    def _load(self, region, sample):
        if (region, sample) == self._key:
            return
        sdir = os.path.join(REGIONS[region], sample)
        meta_path = os.path.join(sdir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        self._steps = _pick_timesteps(sdir, meta.get("total_time_s", 120.0))
        self._meta = meta
        self._key = (region, sample)

    def available(self, region, sample):
        return region in REGIONS and os.path.isdir(os.path.join(REGIONS[region], sample))

    def get(self, filename):
        """파일명으로 정답을 찾는다. -> dict 또는 None(원본 없음/이름 규칙 불일치)."""
        parsed = parse_name(filename)
        if not parsed:
            return None
        region, sample, aug, t_idx = parsed
        if not self.available(region, sample):
            return None
        self._load(region, sample)
        if t_idx >= len(self._steps):
            return None
        depth, t_val = self._steps[t_idx]
        return {
            "depth_m": np.clip(np.ascontiguousarray(D4[aug](depth)), 0.0, DEPTH_VMAX_M),
            "dx_m": float(self._meta.get("dx", 10.416666666666666)),
            "rain_mm": self._meta.get("rain_mm"),
            "time_s": float(t_val),
            "region": region,
            "sample": sample,
            "aug": aug,
            "t_idx": t_idx,
        }


def render_gt_png_bytes(depth_m):
    """정답 수심을 학습 타깃과 똑같은 방식으로 렌더링 (예측 이미지와 방향·색이 일치)."""
    import io
    from PIL import Image
    rgb = render_depth_rgb(depth_to_norm(depth_m), vmax=1.0)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def depth_from_prediction_image(pred_img, out_shape):
    """예측 PNG(PIL) -> 정답과 같은 격자의 수심[m] 배열(배열 방향).

    predict_depth_from_image 를 다시 부르면 추론이 한 번 더 도므로, 이미 받아둔
    예측 이미지를 재사용한다. 변환 경로(리사이즈 -> Blues 역LUT -> norm_to_depth)는
    inference_api 와 동일하다.
    """
    from PIL import Image
    from src.inference_api import _image_to_depth
    img = pred_img.resize((out_shape[1], out_shape[0]), Image.BICUBIC)
    return to_array_orientation(_image_to_depth(img))


def compare(pred_depth_m, gt_depth_m):
    """예측 수심 vs 정답 수심 지표. 둘 다 배열 방향(terrain 기준)이어야 한다."""
    p = np.asarray(pred_depth_m, dtype=np.float64)
    g = np.asarray(gt_depth_m, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape 불일치: 예측 {p.shape} vs 정답 {g.shape}")

    err = p - g
    pc, gc = p.ravel() - p.mean(), g.ravel() - g.mean()
    den = np.sqrt((pc * pc).sum() * (gc * gc).sum())
    r = float((pc * gc).sum() / den) if den > 1e-12 else None

    out = {
        "mae_m": float(np.abs(err).mean()),
        "rmse_m": float(np.sqrt((err ** 2).mean())),
        "bias_m": float(err.mean()),
        "r": r,
        "gt_max_m": float(g.max()),
        "pred_max_m": float(p.max()),
        "gt_mean_m": float(g.mean()),
        "pred_mean_m": float(p.mean()),
        "peak_ratio": float(p.max() / g.max()) if g.max() > 1e-9 else None,
    }
    pi = np.unravel_index(int(p.argmax()), p.shape)
    gi = np.unravel_index(int(g.argmax()), g.shape)
    out["peak_dist_px"] = float(np.hypot(pi[0] - gi[0], pi[1] - gi[1]))
    for thr in WET_THRESHOLDS:
        pm, gm = p >= thr, g >= thr
        union = np.logical_or(pm, gm).sum()
        out[f"iou_{int(thr * 100)}cm"] = (float(np.logical_and(pm, gm).sum() / union)
                                          if union else None)
    return out
