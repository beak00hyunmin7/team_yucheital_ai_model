"""
지형(DEM) 절차적 생성기 - "타입별 지형 구조를 랜덤 배치 후 매끈하게 다듬기" 방식.

순수 가우시안 언덕 하나의 고도/위치만 무작위로 뽑는 대신, 미리 정의한 지형
타입(봉우리/능선/평지/분지/안장) 중 몇 개를 무작위로 골라 랜덤한 위치·크기로
배치하고 더한 뒤, 가우시안 블러로 이어붙인 경계를 매끈하게 만든다.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

FEATURE_TYPES = ("peak", "ridge", "plateau", "basin", "saddle")


def _peak(X, Y, cx, cy, amp, sigma):
    return amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def _ridge(X, Y, cx, cy, amp, length, width, angle):
    Xc, Yc = X - cx, Y - cy
    ca, sa = np.cos(angle), np.sin(angle)
    Xr = Xc * ca + Yc * sa    # 능선 축 방향
    Yr = -Xc * sa + Yc * ca   # 능선 축에 수직인 방향
    return amp * np.exp(-(Xr ** 2) / (2 * length ** 2) - (Yr ** 2) / (2 * width ** 2))


def _plateau(X, Y, cx, cy, amp, radius, edge_softness):
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return amp * 0.5 * (1.0 - np.tanh((r - radius) / edge_softness))


def _basin(X, Y, cx, cy, amp, sigma):
    return -amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def _saddle(X, Y, cx, cy, amp, sigma, angle):
    sep = sigma * 1.6
    dx, dy = np.cos(angle) * sep / 2, np.sin(angle) * sep / 2
    g1 = _peak(X, Y, cx - dx, cy - dy, amp, sigma)
    g2 = _peak(X, Y, cx + dx, cy + dy, amp, sigma)
    return g1 + g2


def _sample_feature(ftype, domain_len, rng):
    cx = rng.uniform(0.2, 0.8) * domain_len
    cy = rng.uniform(0.2, 0.8) * domain_len
    angle = rng.uniform(0, np.pi)

    if ftype == "peak":
        amp = rng.uniform(3.0, 9.0)
        sigma = rng.uniform(0.08, 0.18) * domain_len
        return lambda X, Y: _peak(X, Y, cx, cy, amp, sigma)

    if ftype == "ridge":
        amp = rng.uniform(2.5, 7.0)
        length = rng.uniform(0.20, 0.40) * domain_len
        width = rng.uniform(0.05, 0.12) * domain_len
        return lambda X, Y: _ridge(X, Y, cx, cy, amp, length, width, angle)

    if ftype == "plateau":
        amp = rng.uniform(2.5, 6.0)
        radius = rng.uniform(0.10, 0.20) * domain_len
        edge = rng.uniform(0.02, 0.05) * domain_len
        return lambda X, Y: _plateau(X, Y, cx, cy, amp, radius, edge)

    if ftype == "basin":
        amp = rng.uniform(1.5, 4.0)   # 언덕보다 얕게 -> 지형 전체가 분지로 뒤집히지 않도록
        sigma = rng.uniform(0.10, 0.20) * domain_len
        return lambda X, Y: _basin(X, Y, cx, cy, amp, sigma)

    if ftype == "saddle":
        amp = rng.uniform(3.0, 7.0)
        sigma = rng.uniform(0.08, 0.14) * domain_len
        return lambda X, Y: _saddle(X, Y, cx, cy, amp, sigma, angle)

    raise ValueError(f"unknown feature type: {ftype}")


def generate_hill_terrain_v2(size=96, dx=1.0, n_features=None, feature_types=None,
                              smooth_sigma_px=None, seed=None):
    """
    size, dx: size x size 격자, 셀 크기 dx[m]
    n_features: 배치할 지형 개수 (None이면 2~4 중 랜덤)
    feature_types: 사용할 타입 목록 (None이면 FEATURE_TYPES 전체에서 랜덤 선택)
    smooth_sigma_px: 이어붙임을 다듬는 가우시안 블러 표준편차[px] (None이면 size의 3%)

    반환: z (size, size) 고도[m], X, Y 좌표 격자
    """
    rng = np.random.default_rng(seed)
    domain_len = size * dx
    x = np.arange(size) * dx
    y = np.arange(size) * dx
    X, Y = np.meshgrid(x, y)

    if n_features is None:
        n_features = int(rng.integers(2, 5))
    types_pool = feature_types or FEATURE_TYPES

    z = np.zeros((size, size))
    for _ in range(n_features):
        ftype = types_pool[rng.integers(0, len(types_pool))]
        feature_fn = _sample_feature(ftype, domain_len, rng)
        z += feature_fn(X, Y)

    sigma_px = smooth_sigma_px if smooth_sigma_px is not None else max(1.0, 0.03 * size)
    z = gaussian_filter(z, sigma=sigma_px)

    z -= z.min()  # 최저 고도를 0으로 맞춰 항상 지형이 지면 위로 솟아있게 함
    return z, X, Y


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        z, X, Y = generate_hill_terrain_v2(size=96, seed=i)
        cs = ax.contourf(X, Y, z, levels=20, cmap="terrain")
        ax.contour(X, Y, z, levels=20, colors="k", linewidths=0.3)
        ax.set_title(f"seed={i}")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("terrain_types_preview.png", dpi=120)
    print("saved terrain_types_preview.png")
