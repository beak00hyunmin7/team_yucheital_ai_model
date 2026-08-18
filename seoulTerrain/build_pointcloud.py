"""
서울시 등고선(폴리라인, 높이 속성) + 표고점(포인트, 높이 속성) shapefile을
하나의 (x, y, z) 산점 데이터(scattered point cloud)로 합쳐서 .npy로 캐싱한다.

좌표계: Korean 1985 / Modified Korea Central Belt (TM, 단위 m) - 그대로 사용
(투영 좌표계라 별도 변환 없이 x,y를 미터 단위 평면좌표로 쓸 수 있음)

입력:
  ../서울시 등고선/등고선 5000/N3L_F001.shp  (POLYLINE, 속성 HEIGHT)
  ../서울시 등고선/표고 5000/N3P_F002.shp    (POINT, 속성 HEIGHT... 아님 NUME)

출력:
  pointcloud.npy  -- shape (N, 3), columns = [x, y, z]
"""

import os
import numpy as np
import shapefile

BASE = os.path.join(os.path.dirname(__file__), "..", "서울시 등고선")
LINE_SHP = os.path.join(BASE, "등고선 5000", "N3L_F001.shp")
POINT_SHP = os.path.join(BASE, "표고 5000", "N3P_F002.shp")

OUT_FILE = os.path.join(os.path.dirname(__file__), "pointcloud.npy")


def load_lines():
    sf = shapefile.Reader(LINE_SHP, encoding="cp949")
    xs, ys, zs = [], [], []
    for i in range(len(sf)):
        shp = sf.shape(i)
        h = sf.record(i)["HEIGHT"]
        pts = shp.points
        xs.extend(p[0] for p in pts)
        ys.extend(p[1] for p in pts)
        zs.extend([h] * len(pts))
        if i % 2000 == 0:
            print(f"  lines {i}/{len(sf)}")
    return np.array(xs), np.array(ys), np.array(zs)


def load_points():
    sf = shapefile.Reader(POINT_SHP, encoding="cp949")
    xs, ys, zs = [], [], []
    for i in range(len(sf)):
        shp = sf.shape(i)
        rec = sf.record(i).as_dict()
        h = rec.get("HEIGHT", rec.get("NUME"))
        p = shp.points[0]
        xs.append(p[0])
        ys.append(p[1])
        zs.append(h)
    return np.array(xs), np.array(ys), np.array(zs)


def main():
    print("loading contour lines...")
    lx, ly, lz = load_lines()
    print(f"  -> {len(lx)} vertices")

    print("loading elevation points...")
    px, py, pz = load_points()
    print(f"  -> {len(px)} points")

    x = np.concatenate([lx, px])
    y = np.concatenate([ly, py])
    z = np.concatenate([lz, pz])

    cloud = np.column_stack([x, y, z]).astype(np.float32)
    np.save(OUT_FILE, cloud)
    print(f"saved {OUT_FILE}, shape={cloud.shape}")
    print("bbox x:", x.min(), x.max())
    print("bbox y:", y.min(), y.max())
    print("z range:", z.min(), z.max())


if __name__ == "__main__":
    main()
