"""
D8 흐름 방향 + 흐름 누적(flow accumulation) 계산.

지형(DEM)에서 "각 셀에 상류로부터 몇 개의 셀이 흘러드는가"를 구한다.
값이 큰 곳 = 물길(배수 채널). AI 모델 입력에 이 채널을 더해 주면 모델이
배수망 위치를 스스로 학습하지 않아도 된다 (B4 실험).

team_yucheital_backend-main/app/services/hydrology.py 의 D8 로직을 옮기되,
실제 DEM 의 함몰지(pit)에서 흐름이 갇히지 않도록 priority-flood 로 먼저
함몰지를 메운다 (Barnes 2014, O(n log n)).
"""
from __future__ import annotations

import heapq

import numpy as np

_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_DIAG_DIST = np.sqrt(2.0)


def fill_depressions(elevation: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """priority-flood + epsilon: 함몰지를 메우되 아주 작은 경사를 남겨 흐름이 계속되게 한다."""
    h, w = elevation.shape
    filled = np.full_like(elevation, np.inf, dtype=np.float64)
    closed = np.zeros((h, w), dtype=bool)
    pq: list = []

    for y in range(h):
        for x in (0, w - 1):
            heapq.heappush(pq, (float(elevation[y, x]), y, x))
            closed[y, x] = True
    for x in range(w):
        for y in (0, h - 1):
            if not closed[y, x]:
                heapq.heappush(pq, (float(elevation[y, x]), y, x))
                closed[y, x] = True

    while pq:
        e, y, x = heapq.heappop(pq)
        filled[y, x] = e
        for dy, dx in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not closed[ny, nx]:
                closed[ny, nx] = True
                ne = max(float(elevation[ny, nx]), e + epsilon)
                heapq.heappush(pq, (ne, ny, nx))
    return filled


def d8_flow_accumulation(elevation: np.ndarray, fill: bool = True) -> np.ndarray:
    """(H, W) 고도 -> (H, W) 흐름 누적 (자기 자신 포함, 최소 1.0)."""
    z = elevation.astype(np.float64)
    if fill:
        z = fill_depressions(z)
    h, w = z.shape

    # 각 셀에서 가장 가파르게 내려가는 이웃 방향 (downstream)
    downstream = np.full(h * w, -1, dtype=np.int64)
    best_slope = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.indices((h, w))
    for dy, dx in _NEIGHBORS:
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        sub = slice(y0, y1), slice(x0, x1)
        nsub = slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx)
        dist = _DIAG_DIST if (dy and dx) else 1.0
        slope = (z[sub] - z[nsub]) / dist
        better = slope > best_slope[sub] + 1e-12
        idx = (yy[sub] + dy) * w + (xx[sub] + dx)
        ds_sub = downstream.reshape(h, w)[sub]
        bs_sub = best_slope[sub]
        ds_sub[better] = idx[better]
        bs_sub[better] = slope[better]
        downstream.reshape(h, w)[sub] = ds_sub
        best_slope[sub] = bs_sub

    # 고도 내림차순으로 상류 -> 하류 누적 (내리막 라우팅이라 비순환)
    acc = np.ones(h * w, dtype=np.float64)
    order = np.argsort(z.ravel(), kind="stable")[::-1]
    ds = downstream
    for src in order:
        t = ds[src]
        if t >= 0:
            acc[t] += acc[src]
    return acc.reshape(h, w)


def flowacc_channel(elevation: np.ndarray, log_norm: float = 9.2) -> np.ndarray:
    """모델 입력용 [-1, 1] 정규화 흐름누적 채널. log1p 후 log_norm 으로 나눔.

    log_norm 기본 9.2 ≈ log1p(96*96*1.1). 격자 크기가 크게 다르면 조정.
    """
    acc = d8_flow_accumulation(elevation)
    return np.clip(np.log1p(acc) / log_norm, 0.0, 1.0) * 2.0 - 1.0
