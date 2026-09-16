"""
예측 배수맵 웹 백엔드 (FastAPI).

프론트엔드에서 다음을 받는다:
  - mode: "image"  -> 등고선 이미지를 그대로 모델에 입력
          "terrain" -> 등고선 데이터(고도 격자)를 학습 때와 동일한 방식으로
                       등고선 이미지로 렌더링한 뒤 모델에 입력
  - rain_mm: 강우량(mm). FiLM 조건 체크포인트에서 사용
  - time_s : 강우 시작 후 경과 시간(초). cond_dim=2 체크포인트에서 사용
  - dx: (terrain 모드) 격자 셀 크기[m]. 등고선 이미지 모양에는 거의 영향 없음

응답(JSON):
  contour_png_base64     실제로 모델에 들어간 등고선 이미지
  prediction_png_base64  예측 배수맵
  overlay_png_base64     등고선 위에 예측을 겹친 이미지
  + 메타데이터

모델 연결:
  ai_model/src/inference_api.py 의 predict_from_image 를 그대로 재사용한다.
  기본 체크포인트는 checkpoints_p2_wet8/best.pt (젖은 픽셀 가중 L1, 128px/10.5M).
  구버전 checkpoints_film 대비: 강우+시점 조건(cond_dim=2), 최대수심 비율 0.49->0.62,
  IoU@5cm 0.378->0.426 (ai_model/IMPROVEMENT_GUIDE.md, eval_p2_wet8/summary.md 참조).
  학습 해상도는 체크포인트에서 자동 판별하므로 AI_MODEL_IMAGE_SIZE 를 줄 필요 없다.
  다른 체크포인트로 바꾸려면 환경변수 AI_MODEL_CHECKPOINT 설정 후 재시작.
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# --- ai_model 을 import 경로에 추가하고 기본 체크포인트를 FiLM 모델로 지정 --------
BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
AI_MODEL_DIR = WEBAPP_DIR.parent / "ai_model"
FRONTEND_DIR = WEBAPP_DIR / "frontend"

sys.path.insert(0, str(AI_MODEL_DIR))
os.environ.setdefault(
    "AI_MODEL_CHECKPOINT", str(AI_MODEL_DIR / "checkpoints_p2_wet8" / "best.pt")
)

from src.inference_api import (  # noqa: E402  (경로 설정 후 import)
    current_checkpoint,
    current_cond_dim,
    current_image_size,
    current_uses_film,
    predict_from_image,
)
from src.utils.render_fields import render_contour_rgb  # noqa: E402
from src.ground_truth import (  # noqa: E402
    GroundTruth,
    compare as gt_compare,
    depth_from_prediction_image,
    group_key,
    render_gt_png_bytes,
    split_of,
)

from terrain_io import InvalidTerrainData, load_terrain  # noqa: E402
from analysis_figure import intensity_from_prediction, render_analysis_figure  # noqa: E402
import auth  # noqa: E402  (회원가입/로그인 + MySQL)

RAIN_MIN_MM, RAIN_MAX_MM = 20.0, 80.0          # 학습 데이터가 다룬 강우 범위
TIME_MIN_S, TIME_MAX_S = 0.0, 120.0            # 학습 데이터가 다룬 경과 시간 범위

# 기본 조회 시점.
#
# 주의: 수심은 시간이 갈수록 "줄어든다". 학습 데이터(OpenFOAM)에서 비는 초반에 내리고
# 이후 물이 빠지기 때문에, val 정답의 평균 수심은 t0(18s) 0.039m -> t5(120s) 0.019m 로
# 단조 감소한다. 따라서 "이 지형 침수 예상"을 묻는 일반 조회에 time_s=120 을 쓰면
# 가장 덜 잠긴 상태를 보여주게 된다 (deploy_package/README.md 의 권장값은 이 점에서 틀렸다).
# 기본값은 가장 심한 시점 쪽인 18s 로 둔다.
DEFAULT_TIME_S = 18.0
DEFAULT_DX_M = 10.4166667                       # 1000m / 96격자
SAMPLES_DIR = AI_MODEL_DIR / "data" / "contours"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_HEAVY_LOCK = threading.Lock()                  # matplotlib/torch 직렬화 (데모 규모)
_GT = GroundTruth()                             # 검증 탭용 OpenFOAM 정답 조회

app = FastAPI(title="예측 배수맵 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(auth.router)
_DB_READY = auth.init_db()   # MySQL 없으면 False (앱은 뜨고 인증만 비활성)


# --------------------------------------------------------------------------- #
# 라우트
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    cond_dim = _safe_cond_dim()
    return {
        "status": "ok",
        "checkpoint": os.path.basename(str(current_checkpoint())),
        "rain_conditioned": cond_dim >= 1,
        "rain_range_mm": [RAIN_MIN_MM, RAIN_MAX_MM],
        "time_conditioned": cond_dim == 2,
        "time_range_s": [TIME_MIN_S, TIME_MAX_S],
        "default_time_s": DEFAULT_TIME_S,
        "image_size": _safe_image_size(),
        "device": _device(),
        "auth_enabled": _DB_READY,
        "db": auth.db_status() if _DB_READY else "down",
    }


_RV_ANY = re.compile(r"_rv\d+")


def _base_terrain_id(stem: str) -> str:
    """'seoul_sample_0003_rv1_f2_t3' -> 'seoul_sample_0003'.

    학습 때 train/val 을 가르는 것과 같은 함수를 쓴다. 예전의 자체 정규식은 파일명에
    `_t{idx}` 가 붙은 뒤로 증강 접미사를 못 떼서, 같은 지형의 회전본이 서로 다른
    지형으로 집계됐다(샘플 갤러리에 같은 지형이 여러 번 뜨는 원인).
    """
    return group_key(stem)


@app.get("/api/samples")
def samples(limit: int = 30) -> dict:
    """서로 다른 지형을 지역별로 고르게 섞어서 반환 (같은 지형의 회전본 나열 X)."""
    if not SAMPLES_DIR.is_dir():
        return {"samples": []}

    # 지형별 대표 파일 1개 (가능하면 회전 안 된 _r0)
    canonical: dict[str, str] = {}
    for p in SAMPLES_DIR.glob("*.png"):
        if _RV_ANY.search(p.stem):
            continue  # 강우 변형본은 제외
        base = _base_terrain_id(p.stem)
        # 대표본은 회전 안 된 원본(_r0_t0) 우선. 파일명이 _t{idx} 로 끝나므로
        # 예전처럼 endswith("_r0") 로 보면 절대 안 걸린다.
        if base not in canonical or "_r0_t" in p.stem:
            canonical[base] = p.name

    # 지역별로 모아 균등 간격 샘플링 후 라운드로빈
    by_region: dict[str, list[str]] = {}
    for base, fname in sorted(canonical.items()):
        by_region.setdefault(base.split("_", 1)[0], []).append(fname)

    picked: list[str] = []
    per_region = max(1, limit // max(len(by_region), 1))
    buckets = []
    for files in by_region.values():
        step = max(1, len(files) // per_region)
        buckets.append(files[::step][:per_region])
    for i in range(max((len(b) for b in buckets), default=0)):
        for b in buckets:
            if i < len(b):
                picked.append(b[i])

    return {"samples": picked[:limit]}


@app.get("/api/sample/{name}", include_in_schema=False)
def sample_image(name: str) -> FileResponse:
    path = (SAMPLES_DIR / name).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())) or not path.is_file():
        raise HTTPException(404, "샘플을 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/png")


_GT_MAX_CACHE: dict[str, float] = {}


def _gt_max_depth(name: str) -> float | None:
    """그 케이스의 정답 최대 수심[m]. 없으면 None."""
    if name in _GT_MAX_CACHE:
        return _GT_MAX_CACHE[name]
    g = _GT.get(name)
    v = float(g["depth_m"].max()) if g is not None else None
    if v is not None:
        _GT_MAX_CACHE[name] = v
    return v


@app.get("/api/testset")
def testset(split: str = "val", limit: int = 24, min_depth_m: float = 0.05) -> dict:
    """검증 탭용 테스트 케이스 목록.

    split="val" 이면 **학습에 쓰이지 않은 지형만** 고른다. 기본 샘플 갤러리(/api/samples)는
    이 구분을 하지 않아서 30개 중 25개가 학습 지형이었고, 그 위에서 잰 성능은 크게
    부풀려진다(MAE 0.0069 vs 0.0115, IoU@5cm 0.58 vs 0.43). 데모와 검증은 분리해야 한다.
    """
    if split not in {"val", "train", "any"}:
        raise HTTPException(400, "split 은 'val' | 'train' | 'any' 여야 합니다.")
    if not SAMPLES_DIR.is_dir():
        return {"cases": [], "split": split}

    # 지형당 1개만, 지역별로 고르게
    by_region: dict[str, dict[str, str]] = {}
    for p in sorted(SAMPLES_DIR.glob("*.png")):
        sp = split_of(p.name)
        if split != "any" and sp != split:
            continue
        terrain = _base_terrain_id(p.stem)
        region = terrain.split("_", 1)[0]
        slot = by_region.setdefault(region, {})
        if terrain not in slot:
            slot[terrain] = p.name

    # 지역 라운드로빈으로 후보를 먼저 늘어놓는다
    buckets = [list(v.values()) for v in by_region.values()]
    ordered: list[str] = []
    for i in range(max((len(b) for b in buckets), default=0)):
        for b in buckets:
            if i < len(b):
                ordered.append(b[i])

    # 정답에 물이 거의 없는 케이스는 배수 예측 검증에 쓸모가 없다(강원 val 은 대부분
    # 최대수심 1cm 미만). min_depth_m 이상인 케이스를 우선 채우고, 모자라면 나머지로
    # 채운 뒤 몇 개가 걸러졌는지 응답에 밝힌다 - 조용히 고르면 체리피킹이 된다.
    picked, spare, scanned = [], [], 0
    for n in ordered:
        if len(picked) >= limit or scanned >= 400:
            break
        scanned += 1
        d = _gt_max_depth(n)
        if d is None:
            continue
        (picked if d >= min_depth_m else spare).append(n)
    cases = picked[:limit]
    filtered_out = len(spare)
    if len(cases) < limit:
        cases += spare[:limit - len(cases)]

    return {"split": split, "count": len(cases), "min_depth_m": min_depth_m,
            "filtered_out": filtered_out, "scanned": scanned,
            "cases": [{"name": n, "split": split_of(n),
                       "gt_max_m": _gt_max_depth(n)} for n in cases]}


@app.get("/api/testcase")
def testcase(name: str) -> dict:
    """테스트 케이스 1건: 등고선 · OpenFOAM 정답 · 예측 + 정량 비교.

    조건(rain_mm/time_s)은 사용자가 고른 값이 아니라 **그 샘플의 실제 시뮬레이션 조건**을
    쓴다. 정답과 같은 조건이어야 비교가 성립하기 때문이다.
    """
    path = (SAMPLES_DIR / name).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())) or not path.is_file():
        raise HTTPException(404, "샘플을 찾을 수 없습니다.")

    gt = _GT.get(name)
    if gt is None:
        raise HTTPException(
            422, f"'{name}' 의 OpenFOAM 원본 정답을 찾지 못했습니다 "
                 f"(파일명 규칙이 다르거나 원본 데이터셋이 없습니다).")

    contour_img = Image.open(path).convert("RGB")
    cond_dim = _safe_cond_dim()
    cond_kwargs: dict = {}
    if cond_dim >= 1:
        cond_kwargs["rain_mm"] = gt["rain_mm"]
    if cond_dim == 2:
        cond_kwargs["time_s"] = gt["time_s"]

    with _HEAVY_LOCK:
        pred_img = predict_from_image(contour_img, **cond_kwargs)
        pred_depth = depth_from_prediction_image(pred_img, gt["depth_m"].shape)
        truth_png = render_gt_png_bytes(gt["depth_m"])

    metrics = gt_compare(pred_depth, gt["depth_m"])
    return {
        "name": name,
        "split": split_of(name),
        "region": gt["region"],
        "sample": gt["sample"],
        "rain_mm": gt["rain_mm"],
        "time_s": gt["time_s"],
        "grid": list(gt["depth_m"].shape),
        "contour_png_base64": _b64(contour_img),
        "truth_png_base64": base64.b64encode(truth_png).decode("ascii"),
        "prediction_png_base64": _b64(pred_img),
        "metrics": metrics,
        "checkpoint": os.path.basename(str(current_checkpoint())),
    }


def _auth_gate(authorization: str = Header(default="")) -> str | None:
    """인증이 켜져 있으면(=MySQL 연결됨) 토큰 필수, 아니면 통과.

    _DB_READY 는 기동 시점 한 번만 판정된다. 그런데 이 환경의 MariaDB 는 WSL2 안에 있고
    Windows -> 127.0.0.1:3306 포워딩이 수시로 끊긴다(같은 분 안에 20/20 성공 -> 20/20 실패를
    관측). 기동 때 살아 있었으면 인증이 켜진 채 고정되는데, 그 뒤 DB 가 끊기면 **로그인
    자체가 불가능**해져서 토큰을 받을 방법이 없고 예측이 전부 401 로 떨어진다
    ("예측 실패: 세션이 만료되었습니다").

    설계 의도는 "DB 가 없으면 인증을 생략한다" 이므로, 토큰 검증에 실패했을 때 DB 가
    실제로 끊긴 상태인지 확인해서 그렇다면 통과시킨다. DB 조회는 실패 경로에서만 하므로
    정상 상황의 예측 성능에는 영향이 없다.
    """
    if not _DB_READY:
        return None
    try:
        return auth.require_user(authorization)
    except HTTPException:
        if auth.db_status() != "ok":
            return None
        raise


@app.post("/api/predict")
async def predict(
    mode: str = Form(...),
    rain_mm: float = Form(50.0),
    time_s: float = Form(DEFAULT_TIME_S),
    dx: float = Form(DEFAULT_DX_M),
    file: UploadFile = File(...),
    _user: str | None = Depends(_auth_gate),
) -> dict:
    if mode not in {"image", "terrain"}:
        raise HTTPException(400, "mode 는 'image' 또는 'terrain' 이어야 합니다.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not raw:
        raise HTTPException(400, "빈 파일입니다.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"파일이 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 를 초과합니다.")

    rain_mm = float(np.clip(rain_mm, 0.0, 500.0))
    time_s = float(np.clip(time_s, TIME_MIN_S, TIME_MAX_S))
    meta: dict = {"mode": mode, "filename": file.filename}

    try:
        if mode == "image":
            try:
                contour_img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(422, f"이미지를 열 수 없습니다: {exc}") from exc
            meta["source_format"] = "contour_image"
        else:
            terrain, tmeta = load_terrain(raw, file.filename or "upload")
            meta.update(tmeta)
            effective_dx = float(tmeta.get("cellsize_m") or dx)
            meta["dx_m"] = effective_dx
            contour_img = _render_contour(terrain, effective_dx)

        # 체크포인트가 요구하는 조건만 넘긴다 (0=없음, 1=rain, 2=rain+time).
        cond_dim = _safe_cond_dim()
        cond_kwargs: dict = {}
        if cond_dim >= 1:
            cond_kwargs["rain_mm"] = rain_mm
        if cond_dim == 2:
            cond_kwargs["time_s"] = time_s
        meta["rain_conditioned"] = cond_dim >= 1
        meta["rain_mm"] = rain_mm if cond_dim >= 1 else None
        meta["time_conditioned"] = cond_dim == 2
        meta["time_s"] = time_s if cond_dim == 2 else None

        with _HEAVY_LOCK:
            pred_img = predict_from_image(contour_img, **cond_kwargs)
        overlay_img = _make_overlay(contour_img, pred_img)

        # 등고선 데이터 모드에서만: 실제 등고선(m) + 음영기복 + 축척 + 배수구 후보 지도
        analysis_b64 = None
        drain = None
        if mode == "terrain":
            with _HEAVY_LOCK:
                # render_contour_rgb 가 등고선 이미지를 상하반전(matplotlib Y축 위쪽)해서
                # 만들기 때문에, 그 이미지를 통과한 예측도 terrain 배열 기준으로는 상하반전
                # 상태다. terrain(등고선 선)과 겹쳐 그리려면 예측을 다시 뒤집어 맞춘다.
                inten = np.flipud(intensity_from_prediction(pred_img, terrain.shape))
                r, c = np.unravel_index(int(np.argmax(inten)), inten.shape)
                png = render_analysis_figure(
                    terrain, inten, effective_dx,
                    subtitle=(f"지형 고저차 {float(np.ptp(terrain)):.1f}m · dx {effective_dx:.1f}m"
                              + (f" · 강우 {rain_mm:.0f}mm · 경과 {time_s:.0f}s"
                                 if cond_dim == 2 else "")
                              + " · 회색 음영 = 실제 지형 굴곡"),
                    mark_idx=(int(r), int(c)),
                )
            analysis_b64 = base64.b64encode(png).decode("ascii")
            h, w = terrain.shape
            drain = {
                "pixel": {"x": int(c), "y": int(r)},
                "normalized": {"x": round(c / max(w - 1, 1), 4), "y": round(r / max(h - 1, 1), 4)},
            }

    except InvalidTerrainData as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return {
        "contour_png_base64": _b64(contour_img),
        "prediction_png_base64": _b64(pred_img),
        "overlay_png_base64": _b64(overlay_img),
        "analysis_png_base64": analysis_b64,
        "drain_candidate": drain,
        "checkpoint": os.path.basename(str(current_checkpoint())),
        "meta": meta,
        "notes": [
            "예측 배수맵은 색이 진한(파란) 곳일수록 지표 유출수가 모이는 침수·배수 취약 지점입니다.",
            "모델은 학습 데이터와 같은 스타일의 등고선(채색 20단계 + 검은 등고선)에서 가장 정확합니다.",
            "경과 시간이 짧을수록 물이 많습니다 - 비가 초반에 내리고 이후 빠지는 시뮬레이션이라, 가장 심한 상태는 20초 부근입니다.",
        ],
    }


# --------------------------------------------------------------------------- #
# 헬퍼
# --------------------------------------------------------------------------- #
def _render_contour(terrain: np.ndarray, dx: float) -> Image.Image:
    ny, nx = terrain.shape
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    X, Y = np.meshgrid(xs, ys)
    with _HEAVY_LOCK:
        rgb = render_contour_rgb(X, Y, terrain)
    return Image.fromarray(rgb)


def _make_overlay(contour_img: Image.Image, pred_img: Image.Image) -> Image.Image:
    size = (256, 256)
    # 배경은 등고선을 회색조로 (terrain 컬러맵의 파란 저지대가 '물'처럼 보여 혼동되므로).
    gray = np.asarray(contour_img.convert("L").resize(size)).astype(np.float64)
    base = np.stack([gray] * 3, axis=-1) * 0.35 + 255.0 * 0.65   # 옅게 깔기
    pred = np.asarray(pred_img.convert("RGB").resize(size)).astype(np.float64)

    # 예측 배수맵은 흰 배경 + 파란 유출. (255 - 밝기평균) 이 곧 유출 강도.
    # 절대값 기준 고정 게인 -> 유출 거의 없는 예측은 옅게, 강한 곳만 진하게.
    intensity = np.clip(255.0 - pred.mean(axis=2), 0.0, 255.0) / 255.0
    alpha = np.clip(intensity * 1.8, 0.0, 0.9)[..., None]
    highlight = np.array([12.0, 90.0, 235.0])
    blended = base * (1.0 - alpha) + highlight * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_uses_film() -> bool:
    try:
        return bool(current_uses_film())
    except Exception:  # noqa: BLE001  (체크포인트 로드 전/문제 시)
        return False


def _safe_cond_dim() -> int:
    """체크포인트가 요구하는 조건 개수. 0=없음, 1=rain_mm, 2=rain_mm+time_s."""
    try:
        return int(current_cond_dim())
    except Exception:  # noqa: BLE001
        return 0


def _safe_image_size() -> int | None:
    try:
        return int(current_image_size())
    except Exception:  # noqa: BLE001
        return None


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"
