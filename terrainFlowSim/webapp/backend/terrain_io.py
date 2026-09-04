"""
사용자가 올린 '등고선 데이터'(고도 격자)를 numpy (H, W) float 배열로 파싱한다.

실무에서 지형/DEM 데이터는 포맷이 제각각이라, 확장자별로 대표적인 것들을 처리한다.

  .npy            numpy 배열 그대로
  .csv .txt .dat  숫자 격자 (콤마 또는 공백 구분). 헤더 줄이 있어도 자동 무시
  .asc .grd       ESRI ASCII Grid (ncols/nrows/cellsize/NODATA_value 헤더 6줄)
  .tif .tiff      단일 밴드 GeoTIFF/일반 TIFF (Pillow로 읽히는 경우)
  .png .jpg ...    회색조 높이맵(밝을수록 높음)으로 해석

반환: (terrain(np.float32 2D), meta(dict))
  meta = {"source_format", "grid_shape", "cellsize_m"(있으면), "nodata_filled"(개수)}

바꿔야 할 때: 팀에서 쓰는 실제 포맷이 정해지면 아래 _PARSERS에 함수 하나만 추가하면 된다.
GeoTIFF 좌표계/투영까지 제대로 다루려면 rasterio(gdal)를 넣고 _parse_tiff를 교체.
"""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image

MAX_GRID_DIM = 512          # 이보다 큰 격자는 등간격 샘플링으로 축소 (등고선 렌더 속도)
MIN_GRID_DIM = 8


class InvalidTerrainData(ValueError):
    """등고선 데이터로 해석할 수 없는 업로드."""


def load_terrain(raw: bytes, filename: str) -> tuple[np.ndarray, dict]:
    ext = _ext(filename)
    parser = _PARSERS.get(ext)
    if parser is None:
        raise InvalidTerrainData(
            f"지원하지 않는 확장자입니다: '{ext or '없음'}'. "
            "지원: .npy .csv .txt .dat .asc .grd .tif .tiff .png .jpg"
        )

    terrain, meta = parser(raw)
    terrain = np.asarray(terrain, dtype=np.float64)

    if terrain.ndim == 3:
        # (H, W, C) 로 들어온 경우 첫 밴드만 사용
        terrain = terrain[..., 0]
    if terrain.ndim != 2:
        raise InvalidTerrainData(f"2차원 격자가 아닙니다 (shape={terrain.shape}).")
    if min(terrain.shape) < MIN_GRID_DIM:
        raise InvalidTerrainData(f"격자가 너무 작습니다 (shape={terrain.shape}, 최소 {MIN_GRID_DIM}x{MIN_GRID_DIM}).")

    # NODATA / NaN / inf 정리
    nodata_filled = 0
    bad = ~np.isfinite(terrain)
    if bad.any():
        nodata_filled = int(bad.sum())
        terrain[bad] = np.nan
        finite_min = np.nanmin(terrain)
        terrain[np.isnan(terrain)] = finite_min
    meta["nodata_filled"] = meta.get("nodata_filled", 0) + nodata_filled

    if float(np.ptp(terrain)) < 1e-9:
        raise InvalidTerrainData("고도 값이 전부 동일합니다. 실제 지형 격자를 넣어주세요.")

    # 너무 큰 격자 축소
    terrain, cellsize = _downsample(terrain, meta.get("cellsize_m"))
    if cellsize is not None:
        meta["cellsize_m"] = cellsize

    meta["grid_shape"] = list(terrain.shape)
    return terrain.astype(np.float32), meta


# --------------------------------------------------------------------------- #
# 확장자별 파서
# --------------------------------------------------------------------------- #
def _parse_npy(raw: bytes) -> tuple[np.ndarray, dict]:
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise InvalidTerrainData(f".npy 를 읽지 못했습니다: {exc}") from exc
    return arr, {"source_format": "npy"}


def _parse_text_grid(raw: bytes) -> tuple[np.ndarray, dict]:
    text = raw.decode("utf-8-sig", errors="replace")
    delimiter = "," if text.count(",") > text.count("\t") and "," in text else None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[,\s]+", line) if delimiter is None else line.split(",")
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            # 헤더/주석 줄은 건너뜀
            continue
    if not rows:
        raise InvalidTerrainData("숫자 격자를 찾지 못했습니다.")
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]      # 폭이 안 맞는 줄 제거
    if len(rows) < MIN_GRID_DIM:
        raise InvalidTerrainData("일관된 숫자 행이 너무 적습니다.")
    return np.array(rows, dtype=np.float64), {"source_format": "text"}


def _parse_esri_ascii(raw: bytes) -> tuple[np.ndarray, dict]:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header = {}
    body_start = 0
    for i, line in enumerate(lines[:8]):
        m = re.match(r"^\s*([A-Za-z_]+)\s+(-?[\d.eE+]+)\s*$", line)
        if not m:
            body_start = i
            break
        header[m.group(1).lower()] = float(m.group(2))
        body_start = i + 1

    values: list[float] = []
    for line in lines[body_start:]:
        values.extend(float(v) for v in line.replace(",", " ").split())

    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    if ncols and nrows and len(values) >= ncols * nrows:
        grid = np.array(values[: ncols * nrows], dtype=np.float64).reshape(nrows, ncols)
    else:
        raise InvalidTerrainData("ESRI ASCII Grid 헤더(ncols/nrows)와 본문 크기가 맞지 않습니다.")

    meta = {"source_format": "esri_ascii"}
    nodata = header.get("nodata_value")
    if nodata is not None:
        grid[grid == nodata] = np.nan
    cellsize = header.get("cellsize")
    if cellsize:
        meta["cellsize_m"] = float(cellsize)
    return grid, meta


def _parse_tiff(raw: bytes) -> tuple[np.ndarray, dict]:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.array(im)
    except Exception as exc:  # noqa: BLE001
        raise InvalidTerrainData(
            f"TIFF 를 읽지 못했습니다({exc}). 좌표계가 있는 GeoTIFF라면 팀에 rasterio 설치를 요청하세요."
        ) from exc
    return arr.astype(np.float64), {"source_format": "tiff"}


def _parse_heightmap_image(raw: bytes) -> tuple[np.ndarray, dict]:
    with Image.open(io.BytesIO(raw)) as im:
        arr = np.asarray(im.convert("F"))
    return arr, {"source_format": "heightmap_image"}


_PARSERS = {
    "npy": _parse_npy,
    "csv": _parse_text_grid,
    "txt": _parse_text_grid,
    "dat": _parse_text_grid,
    "asc": _parse_esri_ascii,
    "grd": _parse_esri_ascii,
    "tif": _parse_tiff,
    "tiff": _parse_tiff,
    "png": _parse_heightmap_image,
    "jpg": _parse_heightmap_image,
    "jpeg": _parse_heightmap_image,
    "webp": _parse_heightmap_image,
}


# --------------------------------------------------------------------------- #
def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in (filename or "") else ""


def _downsample(terrain: np.ndarray, cellsize: float | None) -> tuple[np.ndarray, float | None]:
    step = max(1, int(np.ceil(max(terrain.shape) / MAX_GRID_DIM)))
    if step == 1:
        return terrain, cellsize
    return terrain[::step, ::step], (cellsize * step if cellsize else None)
