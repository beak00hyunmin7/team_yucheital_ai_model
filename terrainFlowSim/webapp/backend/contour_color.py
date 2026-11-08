"""
사용자가 올린 등고선 '이미지'의 색 체계를 학습 때 쓴 matplotlib `terrain` 체계로 바꾼다.

문제: 모델은 색 자체를 높이로 읽는다. `terrain` 은 파랑(낮음) -> 초록 -> 갈색 -> 흰색(높음)
인데, 사용자가 회색조(밝을수록 높음)나 viridis, 거꾸로 칠한 지도를 올리면 같은 색이 전혀 다른
높이로 읽혀서 높낮이가 뒤집힌 예측이 나온다.

해결: 올린 이미지의 색을 (선택한 또는 자동 감지한) 컬러맵의 LUT 에서 찾아 0~1 높이 값으로
복원하고, 학습 데이터와 똑같이 `render_contour_rgb` 로 다시 그려서 모델에 넣는다.
색만으로는 방향(예: viridis vs 뒤집힌 viridis)을 알 수 없으므로 사용자가 반전 여부를
고를 수 있게 하고, 어떻게 해석했는지 범례 색으로 돌려준다.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import distance_transform_edt, median_filter  # noqa: E402

WORK_SIZE = 256            # 학습 이미지(4in x 64dpi)와 같은 크기로 맞춰서 처리
LUT_N = 256

# (표시 이름, matplotlib 컬러맵 이름). "gray" 는 밝기를 그대로 높이로 쓴다.
SCHEMES: dict[str, tuple[str, str]] = {
    "terrain": ("학습 형식 (파랑·초록·갈색·흰색)", "terrain"),
    "gray": ("회색조", "gray"),
    "viridis": ("viridis (보라→노랑)", "viridis"),
    "jet": ("jet 무지개 (파랑→빨강)", "jet"),
    "gist_earth": ("gist_earth (검정·파랑·초록·갈색)", "gist_earth"),
    "rainbow": ("rainbow (보라→빨강)", "rainbow"),
    "turbo": ("turbo (남색→빨강)", "turbo"),
    "cividis": ("cividis (남색→노랑)", "cividis"),
    # 색 값이 아니라 '색상(hue)'으로 읽는다. 음영기복이 겹쳐 밝기가 변한 무지개색 지도
    # (예: NOAA 수심도)는 RGB 최근접 매칭이 23~36%밖에 못 맞추지만 hue 는 밝기에 둔감하다.
    "hue": ("무지개 색상 (음영·어두운 부분이 섞여도, 파랑→빨강)", "hue"),
}
AUTO = "auto"

# 회색조 판정: 채도 있는 픽셀이 이 비율보다 적으면 회색조로 본다
_GRAY_COLORED_FRAC = 0.015
# 컬러맵 일치도: 색이 LUT 에서 이만큼 이내(0~1 RGB 유클리드)면 '맞는 색'
_MATCH_DIST = 0.12
# 자동 감지에서 이 비율 이상이 맞아야 그 컬러맵으로 인정
# (학습 샘플·JPEG 최소 0.87, 합성 컬러맵 최소 0.84, 토지피복 지도 0.66 이하)
_MIN_MATCH_FRAC = 0.75
# 복원에 쓸 수 있는 픽셀이 이보다 적으면 변환 실패
_MIN_VALID_FRAC = 0.30
# hue 읽기: 채도·밝기가 충분한(=색이 있는) 픽셀이 이 비율 이상이어야 한다
_MIN_HUE_FRAC = 0.5
# 이웃 픽셀의 높이 값 차이가 이 이상이면 '튄다'(20단계 컬러맵의 한 단계는 0.05)
_JUMP = 0.15
# 튀는 이웃 비율이 이 이상이면 높이를 칠한 이미지가 아니라고 본다
# (학습 형식 최대 0.028, 실제 지형도 0.16~0.19 사이)
_MAX_ROUGHNESS = 0.07


class ColorSchemeError(ValueError):
    pass


@dataclass
class Conversion:
    image: Image.Image          # 모델에 넣을 (terrain 체계로 다시 그린) 이미지
    scheme: str                 # 최종 적용한 체계
    reversed: bool              # 높낮이 반전 적용 여부
    converted: bool             # False 면 원본을 그대로 통과시킴
    auto: bool                  # 자동 감지였는지
    match_frac: float | None    # 선택 체계와 색이 맞은 픽셀 비율
    legend: list[str]           # 해석한 '낮음 -> 높음' 색 (hex)
    warnings: list[str]


def _lut(name: str) -> np.ndarray:
    """(LUT_N, 3) float 0..1. 낮은 값 -> 높은 값."""
    if name == "gray":
        g = np.linspace(0.0, 1.0, LUT_N)
        return np.stack([g, g, g], axis=1)
    if name == "hue":
        from matplotlib.colors import hsv_to_rgb
        t = np.linspace(0.0, 1.0, LUT_N)
        return hsv_to_rgb(np.stack([(1.0 - t) * (240.0 / 360.0), np.ones(LUT_N), np.ones(LUT_N)], axis=1))
    return np.asarray(colormaps[name](np.linspace(0.0, 1.0, LUT_N))[:, :3], dtype=float)


def _nearest(arr: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 픽셀의 최근접 LUT 색 -> (값 0..1, 거리). arr: (H,W,3) 0..1."""
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    out_idx = np.empty(len(flat), dtype=np.int32)
    out_d = np.empty(len(flat), dtype=np.float32)
    step = 16384
    for s in range(0, len(flat), step):
        chunk = flat[s:s + step]
        d2 = ((chunk[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        i = d2.argmin(axis=1)
        out_idx[s:s + step] = i
        out_d[s:s + step] = np.sqrt(d2[np.arange(len(chunk)), i])
    return (out_idx / (LUT_N - 1)).reshape(h, w), out_d.reshape(h, w)


def _prep(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR),
                      dtype=float) / 255.0


def _colored_frac(arr: np.ndarray) -> float:
    """채도가 눈에 띄는 픽셀의 비율. 평균 채도로 보면 고지대가 대부분인(거의 흰색인)
    학습 형식 이미지가 회색조로 오인된다(busan_sample_0091 에서 실제로 발생)."""
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    return float(np.mean((mx - mn) / np.maximum(mx, 1e-6) > 0.15))


def _not_line(arr: np.ndarray) -> np.ndarray:
    """검은 등고선 선(어두운 픽셀)을 뺀 마스크. 선 색은 높이 정보가 아니다."""
    return arr.max(axis=2) > 0.28


def _decode(arr: np.ndarray, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """색 -> (높이 0..1, 믿을 만한 픽셀 마스크). gray 는 호출하지 않는다."""
    cmap = SCHEMES[scheme][1]
    keep = _not_line(arr)
    if cmap == "hue":
        from matplotlib.colors import rgb_to_hsv
        hsv = rgb_to_hsv(arr)
        deg = hsv[..., 0] * 360.0
        # 파랑(240°)=낮음 -> 빨강(0°)=높음. 보라(240~300°)는 가장 낮은 쪽, 300° 이상은 빨강 쪽.
        v = 1.0 - np.clip(deg, 0.0, 240.0) / 240.0
        v = np.where((deg > 240.0) & (deg < 300.0), 0.0, v)
        v = np.where(deg >= 300.0, 1.0, v)
        ok = keep & (hsv[..., 1] > 0.35) & (hsv[..., 2] > 0.25)
        return v, ok
    v, d = _nearest(arr, _lut(cmap))
    return v, keep & (d < _MATCH_DIST * 1.6)


def _roughness(arr: np.ndarray, scheme: str) -> float:
    """색->높이로 읽은 값이 이웃 픽셀끼리 얼마나 튀는지(0=매끈, 1=무작위).

    높이를 색으로 칠한 이미지는 이웃 높이가 연속이라 단계 하나(20단계면 0.05)씩만 변한다.
    반면 일반 지형도·토지피복도는 색이 숲(녹색)·시가지(흰색)·물(파랑)처럼 지물을 뜻해서,
    우연히 terrain 컬러맵과 색이 90% 넘게 맞아도(USGS 지형도 실측 94%) 높이로 읽으면
    이웃끼리 크게 튄다. 학습 샘플 200장은 0.028 이하(JPEG70·확대해도 0.016 이하),
    그 지형도는 0.16 이었다. 검은 등고선 선은 높이가 아니므로 뺀다.
    """
    cmap = SCHEMES[scheme][1]
    if cmap == "gray":
        return 0.0     # 회색조는 선(검정)과 낮은 지대를 구분할 수 없어 판정하지 않는다
    v, ok = _decode(arr, scheme)
    tot, big = 0, 0
    for a, b, ma, mb in ((v[:, 1:], v[:, :-1], ok[:, 1:], ok[:, :-1]),
                         (v[1:, :], v[:-1, :], ok[1:, :], ok[:-1, :])):
        m = ma & mb
        tot += int(m.sum())
        big += int((np.abs(a - b)[m] > _JUMP).sum())
    return big / tot if tot else 1.0


def detect_scheme(img: Image.Image) -> tuple[str, float, bool]:
    """(체계 이름, 일치 비율, 회색조 여부). 컬러맵 방향(뒤집힘)은 색만으로 알 수 없다."""
    arr = _prep(img)
    if _colored_frac(arr) < _GRAY_COLORED_FRAC:
        return "gray", 1.0, True
    keep = _not_line(arr)
    sub = arr[::2, ::2][keep[::2, ::2]]
    if sub.size == 0:
        return "terrain", 0.0, False
    best, best_frac = "terrain", -1.0
    for key, (_label, cmap) in SCHEMES.items():
        if cmap in ("gray", "hue"):
            continue
        _v, d = _nearest(sub[None, :, :], _lut(cmap))
        frac = float((d < _MATCH_DIST).mean())
        # 같은 점수면 학습 형식을 우선(앞에 있는 항목이 먼저 이김)
        if frac > best_frac + 1e-6:
            best, best_frac = key, frac

    if best_frac < _MIN_MATCH_FRAC:
        # 어떤 컬러맵과도 RGB 로는 안 맞는다. 음영이 겹친 무지개색 지도일 수 있으니 hue 로 한 번 더 본다.
        # 단, 채도 있는 픽셀이 충분하고(흰 바탕 위 지도 제외), 색상이 넓게 퍼져 있으며(단색 지물 제외),
        # 이웃끼리 매끈해야(지형도 제외) 인정한다.
        v, ok = _decode(arr, "hue")
        frac = float(ok.mean())
        if frac >= _MIN_HUE_FRAC:
            occ = np.histogram(v[ok], bins=10, range=(0, 1))[0] / ok.sum()
            if int((occ >= 0.03).sum()) >= 5 and _roughness(arr, "hue") <= _MAX_ROUGHNESS:
                return "hue", frac, False
    return best, best_frac, False


def _to_height(img: Image.Image, scheme: str) -> tuple[np.ndarray, float]:
    """이미지 -> (H,W) 0..1 높이, 일치 비율."""
    arr = _prep(img)
    cmap = SCHEMES[scheme][1]
    if cmap == "gray":
        h = arr.mean(axis=2)
        return h, 1.0

    v, good = _decode(arr, scheme)
    frac = float(good.mean())
    if frac < _MIN_VALID_FRAC:
        raise ColorSchemeError(
            f"이미지 색이 '{SCHEMES[scheme][0]}' 체계와 거의 맞지 않습니다"
            f"(일치 {frac * 100:.0f}%). 색 체계를 다시 고르거나 '등고선 데이터' 모드를 쓰세요.")
    if not good.all():
        # 선·범례·글씨처럼 맞지 않는 픽셀은 가장 가까운 정상 픽셀 값으로 메운다
        idx = distance_transform_edt(~good, return_distances=False, return_indices=True)
        v = v[idx[0], idx[1]]
    return v, frac


def _legend(scheme: str, reverse: bool, n: int = 7) -> list[str]:
    lut = _lut(SCHEMES[scheme][1])
    pos = np.linspace(0, LUT_N - 1, n).astype(int)
    cols = lut[pos][::-1] if reverse else lut[pos]
    return ["#%02x%02x%02x" % tuple(int(round(c * 255)) for c in col) for col in cols]


def convert_to_training_style(img: Image.Image, scheme: str = AUTO,
                              reverse: bool = False) -> Conversion:
    """등고선 이미지를 학습 형식으로 변환. 이미 학습 형식이면 원본 그대로 통과."""
    from src.utils.render_fields import render_contour_rgb  # 지연 import (ai_model 경로 필요)

    warnings: list[str] = []
    auto = scheme == AUTO
    gray_ambiguous = False
    frac_auto: float | None = None
    if auto:
        scheme, frac_auto, gray_ambiguous = detect_scheme(img)
        if gray_ambiguous:
            warnings.append(
                "회색조 이미지는 밝은 쪽이 높은지 낮은지 자동으로 알 수 없어 '밝을수록 높음'으로 "
                "가정했습니다. 결과의 높낮이가 반대로 보이면 '높낮이 반전'을 켜세요. "
                "또한 밝기가 높이가 아니라 빛·선이면(음영기복 hillshade, 흑백 등고선 선 도면) "
                "예측이 의미 없으니 고도값 자체(DEM)를 쓰세요.")
        elif scheme != "hue" and frac_auto is not None and frac_auto < _MIN_MATCH_FRAC:
            # 예전엔 경고만 띄우고 terrain 으로 가정해 원본을 그대로 통과시켰는데, 그러면 색이
            # 높이가 아닌 이미지(스캔 도면, 3D 투시도)도 아무 말 없이 얼룩 예측이 나왔다.
            raise ColorSchemeError(
                f"이미지 색이 알려진 어떤 색 체계와도 맞지 않습니다(가장 가까운 "
                f"'{SCHEMES[scheme][0]}'도 {frac_auto * 100:.0f}%만 일치). 높이가 색으로 칠해진 "
                "이미지가 아니거나(지형도·스캔 도면·3D 투시도) 팔레트가 특이한 경우입니다. "
                "팔레트를 알면 '이미지 색 체계'를 직접 고르고, 아니면 고도 격자(DEM)를 "
                "'등고선 데이터' 모드로 올리세요.")
    elif scheme not in SCHEMES:
        raise ColorSchemeError(f"알 수 없는 색 체계: {scheme}")

    if not gray_ambiguous:
        rough = _roughness(_prep(img), scheme)
        if rough > _MAX_ROUGHNESS:
            if auto:
                raise ColorSchemeError(
                    "이 이미지는 색이 높이를 나타내는 등고선 이미지가 아니라 일반 지형도·토지피복도로 "
                    f"보입니다(이웃 픽셀의 높이 값이 {rough * 100:.0f}% 튐, 높이 이미지는 3% 이하). "
                    "색이 숲·시가지·물 같은 지물을 뜻해서 색에서 높이를 읽을 수 없고, 이대로는 예측이 "
                    "의미 없습니다. 고도 격자(.asc/.tif/.npy/.csv 등 DEM)를 '등고선 데이터' 모드로 "
                    "올리거나, 높이를 색으로 칠한 이미지를 사용하세요.")
            warnings.append(
                f"선택한 '{SCHEMES[scheme][0]}' 체계로 읽으면 이웃 픽셀의 높이 값이 {rough * 100:.0f}% 튑니다"
                "(높이 이미지는 3% 이하). 일반 지형도처럼 색이 높이가 아닌 이미지면 예측이 의미 없습니다.")
    legend = _legend(scheme, reverse)
    if scheme == "terrain" and not reverse:
        # 이미 학습 형식. 건드리지 않는다.
        return Conversion(img.convert("RGB"), scheme, False, False, auto, frac_auto, legend, warnings)

    h, frac = _to_height(img, scheme)
    if reverse:
        h = 1.0 - h
    h = median_filter(h, size=3)
    lo, hi = np.percentile(h, [0.5, 99.5])
    if hi - lo < 0.02:
        raise ColorSchemeError("이미지에서 높이 변화를 읽지 못했습니다(거의 단색). 색 체계를 확인하세요.")
    h = np.clip((h - lo) / (hi - lo), 0.0, 1.0)

    # render_contour_rgb 는 행 0 을 아래(y=0)에 그리므로, 결과 이미지의 위/아래가
    # 사용자 이미지와 같도록 먼저 상하를 뒤집어 넣는다.
    z = np.flipud(h)
    ys, xs = np.mgrid[0:WORK_SIZE, 0:WORK_SIZE].astype(float)
    rgb = render_contour_rgb(xs, ys, z)
    return Conversion(Image.fromarray(rgb), scheme, bool(reverse), True, auto,
                      frac if not auto else (frac_auto if frac_auto is not None else frac),
                      legend, warnings)
