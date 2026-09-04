"""
고도(elevation)/수심(depth) 2D 배열 -> RGB 이미지 렌더링 공용 함수.
scripts/render_from_openfoam.py와 scripts/generate_dataset.py가 함께 쓴다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_contour_rgb(X, Y, Z, levels=20, figsize=(4, 4), dpi=64):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(X, Y, Z, levels=levels, cmap="terrain")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return _fig_to_rgb(fig)


# 수심 타깃 렌더링의 전역 기준 최대 수심 [m].
# (데이터셋 전체 max_depth 최댓값 0.628m을 덮는 값 -> 클리핑 손실 없음)
#
# 중요: 예전에는 render_depth_rgb(depth)를 vmax 없이 호출해서 "샘플마다 자기
# 자신의 최댓값"으로 정규화했는데, 이러면 절대 수심 정보가 통째로 사라진다.
# 실제로 같은 지형-다른 강우량 쌍 120개를 검사한 결과, "강우가 많은 쪽이 더
# 심각하게 렌더링되는" 비율이 57.5%(= 사실상 랜덤)에 불과했다. 즉 학습 타깃이
# 강우량 -> 침수 심각도 관계를 담고 있지 않아서, 어떤 아키텍처(FiLM/GAN/채널결합)로도
# 강우 조건화를 배울 수 없었다. 전역 vmax + sqrt 압축으로 바꾸면 같은 검사에서
# 99.2%로 올라간다.
DEPTH_VMAX_M = 0.6


def depth_to_norm(depth, vmax=DEPTH_VMAX_M):
    """수심[m] -> [0,1] 정규화 값. sqrt로 압축해서 얕은 구간의 해상도를 살린다.

    데이터셋 수심 분포가 한쪽으로 크게 치우쳐 있어(중앙값 0.13m, 최대 0.63m,
    20%는 0.05m 미만) 선형 정규화를 쓰면 대부분의 샘플이 8bit 색상 범위의
    아래쪽 몇 단계에 뭉친다. sqrt를 씌우면 절대 크기 순서(단조성)는 유지하면서
    얕은 구간이 색상 범위를 더 넓게 쓴다.
    """
    return np.sqrt(np.clip(depth, 0.0, vmax) / vmax)


def norm_to_depth(norm, vmax=DEPTH_VMAX_M):
    """depth_to_norm의 역변환: [0,1] 정규화 값 -> 수심[m] 근사치."""
    return (np.clip(norm, 0.0, 1.0) ** 2) * vmax


def render_depth_rgb(depth, vmax=None, figsize=(4, 4), dpi=64):
    """수심 배열을 Blues 컬러맵 RGB로 렌더링.

    vmax=None이면 예전처럼 샘플별 자체 정규화(하위호환용, 신규 학습에는 쓰지 말 것).
    신규 데이터 생성은 depth_to_norm()으로 먼저 정규화한 뒤 vmax=1.0으로 호출한다.
    """
    if vmax is None:
        vmax = max(float(depth.max()), 1e-6)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(depth, origin="lower", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_axis_off()
    return _fig_to_rgb(fig)


def render_rain_channel(rain_mm, rain_min=20.0, rain_max=80.0, size=8):
    """강우량(mm)을 [0,1]로 정규화해 균일한 값의 단일 채널(L모드) 타일로 렌더링.
    UnetGenerator 입력에서 등고선(3채널) 뒤에 이어붙이는 4번째 채널(강우 조건)로 쓴다.
    스스로 공간적 구조가 없는 값이라 작은 타일(기본 8x8)로 만들고, 로딩 시
    transforms.Resize로 학습 해상도에 맞게 늘어난다.

    rain_min/rain_max: 정규화 기준 범위. prepare_samples.py/run_rain_variants.py가
    강우량을 20~80mm 범위에서 뽑으므로 기본값을 그 범위로 맞췄다.
    """
    norm = np.clip((rain_mm - rain_min) / (rain_max - rain_min), 0.0, 1.0)
    val = np.uint8(round(norm * 255))
    return np.full((size, size), val, dtype=np.uint8)


def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb
