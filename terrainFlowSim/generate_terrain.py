"""
언덕 모양의 가상 지형(DEM) 생성기.

가우시안 형태의 언덕 1~3개를 랜덤한 위치/높이/퍼짐 정도로 배치해서
정사각형 격자 위의 고도(z) 배열을 만든다. 실제 DEM(GeoTIFF) 대신
이 프로젝트 1단계에서 학습 데이터를 빠르게 확보하기 위한 용도.
"""

import numpy as np


def generate_hill_terrain(size=128, dx=1.0, n_hills=None, seed=None):
    """
    size: 격자 한 변의 셀 개수 (size x size)
    dx: 셀 크기 [m]
    n_hills: 언덕 개수 (None이면 1~3 중 랜덤)
    seed: 재현성을 위한 랜덤 시드

    반환: z (size, size) 고도 배열 [m]
    """
    rng = np.random.default_rng(seed)
    domain_len = size * dx

    x = np.arange(size) * dx
    y = np.arange(size) * dx
    X, Y = np.meshgrid(x, y)

    if n_hills is None:
        n_hills = int(rng.integers(1, 4))  # 1~3개

    z = np.zeros((size, size))
    for _ in range(n_hills):
        cx = rng.uniform(0.25, 0.75) * domain_len
        cy = rng.uniform(0.25, 0.75) * domain_len
        amp = rng.uniform(3.0, 10.0)          # 언덕 높이 [m]
        sigma = rng.uniform(0.10, 0.22) * domain_len  # 언덕 퍼짐 정도
        z += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    return z, X, Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    z, X, Y = generate_hill_terrain(size=128, dx=1.0, seed=0)
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.contourf(X, Y, z, levels=20, cmap="terrain")
    ax.contour(X, Y, z, levels=20, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax, label="Elevation [m]")
    ax.set_title("Sample hill terrain (seed=0)")
    ax.set_aspect("equal")
    fig.savefig("terrain_preview.png", dpi=120)
    print("saved terrain_preview.png, z range:", z.min(), z.max())
