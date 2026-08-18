#!/usr/bin/env python3
"""
blockMesh로 만든 평평한 박스 메쉬(constant/polyMesh/points)를 terrain_raw.npy
(96x96, dx=DX_TERRAIN[m] 간격의 실제 지형 격자)로 warp한다.
"""

import sys
import numpy as np
import re

POINTS_FILE = "constant/polyMesh/points"
TERRAIN_FILE = "terrain_raw.npy"
DX_TERRAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0


def bilinear_sample(z, xq, yq):
    ny, nx = z.shape
    xq = np.clip(xq, 0.0, nx - 1.0)
    yq = np.clip(yq, 0.0, ny - 1.0)
    x0 = np.floor(xq).astype(int)
    y0 = np.floor(yq).astype(int)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    fx = xq - x0
    fy = yq - y0

    z00 = z[y0, x0]
    z10 = z[y0, x1]
    z01 = z[y1, x0]
    z11 = z[y1, x1]

    return (z00 * (1 - fx) * (1 - fy) + z10 * fx * (1 - fy)
            + z01 * (1 - fx) * fy + z11 * fx * fy)


def main():
    z = np.load(TERRAIN_FILE)  # shape (96,96), grid spacing = DX_TERRAIN meters

    with open(POINTS_FILE, "r") as f:
        text = f.read()

    m = re.search(r"\n(\d+)\n\(\n(.*?)\n\)\n", text, re.DOTALL)
    if not m:
        raise RuntimeError("points 리스트를 찾지 못했습니다")

    n_points = int(m.group(1))
    body = m.group(2)

    point_pattern = re.compile(r"\(([^()]+)\)")
    coords = [list(map(float, p.split())) for p in point_pattern.findall(body)]
    assert len(coords) == n_points

    coords_arr = np.array(coords)
    # 물리좌표(m) -> 지형 격자 인덱스(픽셀 단위)
    terrain_z = bilinear_sample(z, coords_arr[:, 0] / DX_TERRAIN, coords_arr[:, 1] / DX_TERRAIN)

    new_lines = []
    for (x, y, z0), tz in zip(coords, terrain_z):
        new_lines.append(f"({x:.6f} {y:.6f} {z0 + tz:.6f})")

    new_body = "\n".join(new_lines)
    new_block = f"\n{n_points}\n(\n{new_body}\n)\n"
    new_text = text[: m.start()] + new_block + text[m.end():]

    with open(POINTS_FILE, "w") as f:
        f.write(new_text)

    print(f"warped {n_points} points. terrain z range: {terrain_z.min():.3f} .. {terrain_z.max():.3f}")


if __name__ == "__main__":
    main()
