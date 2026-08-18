"""
OpenFOAM interFoam 케이스(예: ../../hillTerrainCase)에서
- 지형 고도 그리드 (constant/polyMesh/points, k=0 층 = terrain 경계면)
- 수심 그리드 (해당 시간 디렉토리의 alpha.water 필드를 z방향으로 적분)
을 직접 파싱해서 numpy 배열로 뽑아낸다. WSL이나 ParaView 없이 순수 파이썬으로 동작한다.

전제: hillTerrainCase/warp_terrain.py, make_alpha.py 에서 쓴 것과 같은 규칙
(단일 hex 블록, k=0이 지형면/terrain 경계, cell/point index는 i가 가장 빠르게 변함)을
따르는 케이스만 지원한다.
"""

import glob
import os
import re

import numpy as np

_POINT_PATTERN = re.compile(r"\(([^()]+)\)")


def _parse_vector_field(text):
    """'N\n(\n(x y z)\n...\n)\n' 블록을 (N,3) 배열로 파싱."""
    m = re.search(r"\n(\d+)\n\(\n(.*?)\n\)\n", text, re.DOTALL)
    if not m:
        raise RuntimeError("포인트 리스트를 찾지 못했습니다.")
    n = int(m.group(1))
    coords = [list(map(float, p.split())) for p in _POINT_PATTERN.findall(m.group(2))]
    assert len(coords) == n, f"expected {n} points, parsed {len(coords)}"
    return np.array(coords, dtype=np.float64)


def _parse_scalar_field(text):
    """internalField의 'nonuniform List<scalar> N (...)' 또는 'uniform X' 형태를 파싱."""
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\n\((.*?)\n\)", text, re.DOTALL)
    if m:
        n = int(m.group(1))
        vals = np.array([float(x) for x in m.group(2).split()], dtype=np.float64)
        assert len(vals) == n, f"expected {n} values, parsed {len(vals)}"
        return vals

    m = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", text)
    if m:
        raise RuntimeError("internalField가 uniform 상수라 셀별 값을 알 수 없습니다.")

    raise RuntimeError("internalField를 파싱하지 못했습니다 (파일 포맷 확인 필요).")


def parse_block_dims(case_dir):
    """system/blockMeshDict에서 (nx, ny, nz)와 원통형 박스의 z방향 두께(box_height[m])를 읽는다."""
    with open(os.path.join(case_dir, "system", "blockMeshDict"), "r", encoding="utf-8") as f:
        text = f.read()

    m = re.search(r"hex\s*\([^()]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text)
    if not m:
        raise RuntimeError("blockMeshDict에서 hex block 크기를 찾지 못했습니다.")
    nx, ny, nz = (int(g) for g in m.groups())

    vm = re.search(r"vertices\s*\n\(\n(.*?)\n\)\s*;", text, re.DOTALL)
    if not vm:
        raise RuntimeError("blockMeshDict에서 vertices 블록을 찾지 못했습니다.")
    verts = [list(map(float, p.split())) for p in _POINT_PATTERN.findall(vm.group(1))]
    zs = [v[2] for v in verts]
    box_height = max(zs) - min(zs)

    return nx, ny, nz, box_height


def parse_terrain_grid(case_dir):
    """constant/polyMesh/points의 k=0 층(terrain 경계면)에서 (X, Y, Z) 격자를 뽑는다.

    반환: X, Y, Z 모두 shape (ny+1, nx+1) (Z가 고도[m]).
    """
    nx, ny, nz, _ = parse_block_dims(case_dir)

    with open(os.path.join(case_dir, "constant", "polyMesh", "points"), "r", encoding="utf-8") as f:
        text = f.read()
    pts = _parse_vector_field(text)  # (N, 3)

    n_layer = (nx + 1) * (ny + 1)
    layer0 = pts[:n_layer]  # k=0 층 = terrain 경계면

    X = layer0[:, 0].reshape(ny + 1, nx + 1)
    Y = layer0[:, 1].reshape(ny + 1, nx + 1)
    Z = layer0[:, 2].reshape(ny + 1, nx + 1)
    return X, Y, Z


def parse_depth_grid(case_dir, time):
    """<time>/alpha.water를 z방향으로 적분해 셀 중심 기준 수심(depth) 격자를 뽑는다.

    반환: depth, shape (ny, nx) [m]
    """
    nx, ny, nz, box_height = parse_block_dims(case_dir)
    dz = box_height / nz

    time_str = _format_time(time) if not isinstance(time, str) else time
    path = os.path.join(case_dir, time_str, "alpha.water")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 가 없습니다. list_available_times()로 존재하는 시간을 확인하세요.")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    vals = _parse_scalar_field(text)
    assert len(vals) == nx * ny * nz, f"cell count mismatch: {len(vals)} vs {nx*ny*nz}"

    alpha3d = vals.reshape(nz, ny, nx)  # k, j, i (i가 가장 빠르게 변함)
    depth = alpha3d.sum(axis=0) * dz    # (ny, nx)
    return depth


def list_available_times(case_dir):
    """alpha.water가 있는 시간 디렉토리 이름들을 시간순으로 반환."""
    entries = []
    for d in glob.glob(os.path.join(case_dir, "*")):
        name = os.path.basename(d)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "alpha.water")):
            try:
                t = float(name)
            except ValueError:
                continue
            entries.append((t, name))
    entries.sort(key=lambda x: x[0])
    return [name for _, name in entries]


def _format_time(t):
    return f"{t:g}"
