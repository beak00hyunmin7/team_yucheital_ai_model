"""
pointcloud.npy(강원 태백시 일대 국가기본도 등고선 산점 데이터)에서
지정한 중심 좌표 주변 96m x 96m(기본) 패치를 잘라내 1m 격자 DEM으로 보간한다.

seoulTerrain/extract_patch.py와 동일한 로직 (좌표계는 EPSG:5179, Korea 2000
Unified CS — 서울 데이터의 좌표계와 다르므로 섞어 쓰지 말 것).
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
    cx, cy: 패치 중심 좌표 (EPSG:5179, m)
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

    Z = median_filter(Z, size=3)
    Z = gaussian_filter(Z, sigma=1.0)
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
