"""
pointcloud.npy(서울시 전체 등고선+표고점 산점 데이터)에서
지정한 중심 좌표 주변 96m x 96m(기본) 패치를 잘라내 1m 격자 DEM으로 보간한다.

가상 데이터(terrainFlowSim)와 같은 그리드 크기(96x96, dx=1m)를 맞춰서
같은 U-Net 입력 파이프라인에 바로 섞어 쓸 수 있게 한다.
"""

import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter

CLOUD_FILE = os.path.join(os.path.dirname(__file__), "pointcloud.npy")

_cloud = None
_tree = None


def _load():
    global _cloud, _tree
    if _cloud is None:
        _cloud = np.load(CLOUD_FILE)
        _tree = cKDTree(_cloud[:, :2])
    return _cloud, _tree


def get_bbox():
    cloud, _ = _load()
    return cloud[:, 0].min(), cloud[:, 0].max(), cloud[:, 1].min(), cloud[:, 1].max()


def extract_patch(cx, cy, size=96, dx=1.0, margin=25.0, min_points=200):
    """
    cx, cy: 패치 중심 좌표 (TM, m)
    size: 격자 한 변 셀 개수 (size x size)
    dx: 셀 크기 [m]
    반환: z (size, size) 고도 배열 [m], 또는 데이터 부족 시 None
    """
    cloud, tree = _load()
    half = size * dx / 2.0
    radius = half + margin

    idx = tree.query_ball_point([cx, cy], radius)
    if len(idx) < min_points:
        return None

    pts = cloud[idx]

    xs = cx + (np.arange(size) - size / 2.0 + 0.5) * dx
    ys = cy + (np.arange(size) - size / 2.0 + 0.5) * dx
    X, Y = np.meshgrid(xs, ys)

    Z = griddata(pts[:, :2], pts[:, 2], (X, Y), method="linear")

    if np.isnan(Z).any():
        Z_nearest = griddata(pts[:, :2], pts[:, 2], (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)

    # 등고선 기반 보간은 삼각망 경계 부근에 뾰족한 아티팩트(고립된 웅덩이)를
    # 만들 수 있어서, 3x3 중앙값 필터로 스파이크를 없앤 뒤 가벼운 가우시안
    # 스무딩으로 격자 특유의 각진 흔적을 완화한다.
    Z = median_filter(Z, size=3)
    Z = gaussian_filter(Z, sigma=1.0)

    # 가파른 지형(급경사)만 골라 부드러운 곡선으로 완화한다. 완만한 곳은
    # 그대로 두고, 국소 경사가 임계값을 넘는 지점만 반복적으로 주변과
    # 블렌딩해서 완만한 곡선으로 만든다 (OpenFOAM 메쉬 왜곡 방지 목적).
    Z = smooth_steep_regions(Z, dx)

    return Z.astype(np.float32)


def smooth_steep_regions(z, dx, max_slope=0.3, iterations=40, blend=0.25):
    z = z.copy()
    for _ in range(iterations):
        gy, gx = np.gradient(z, dx)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        mask = slope > max_slope
        if not mask.any():
            break
        smoothed = gaussian_filter(z, sigma=1.5)
        z = np.where(mask, z * (1 - blend) + smoothed * blend, z)
    return z


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax = get_bbox()
    print(f"bbox: x [{xmin:.0f}, {xmax:.0f}]  y [{ymin:.0f}, {ymax:.0f}]")

    # 기복이 있는 패치를 찾기 위해 격자 형태로 후보 중심을 스캔
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        cx = rng.uniform(xmin + 100, xmax - 100)
        cy = rng.uniform(ymin + 100, ymax - 100)
        z = extract_patch(cx, cy, size=96, dx=1.0)
        if z is None:
            continue
        relief = z.max() - z.min()
        if best is None or relief > best[2]:
            best = (cx, cy, relief, z)

    cx, cy, relief, z = best
    print(f"best patch center=({cx:.0f},{cy:.0f}) relief={relief:.1f}m "
          f"range=({z.min():.1f},{z.max():.1f})")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    c = ax.imshow(z, origin="lower", cmap="terrain")
    ax.set_title(f"Seoul terrain patch (relief {relief:.1f}m)")
    fig.colorbar(c, ax=ax, label="Elevation [m]")
    fig.savefig("seoul_patch_preview.png", dpi=120)
    print("saved seoul_patch_preview.png")
