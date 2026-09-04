"""
예측 배수맵 웹 백엔드 (FastAPI).

프론트엔드에서 다음을 받는다:
  - mode: "image"  -> 등고선 이미지를 그대로 모델에 입력
          "terrain" -> 등고선 데이터(고도 격자)를 학습 때와 동일한 방식으로
                       등고선 이미지로 렌더링한 뒤 모델에 입력
  - rain_mm: 강우량(mm). FiLM 강우 조건 체크포인트에서 사용
  - dx: (terrain 모드) 격자 셀 크기[m]. 등고선 이미지 모양에는 거의 영향 없음

응답(JSON):
  contour_png_base64     실제로 모델에 들어간 등고선 이미지
  prediction_png_base64  예측 배수맵
  overlay_png_base64     등고선 위에 예측을 겹친 이미지
  + 메타데이터

모델 연결:
  ai_model/src/inference_api.py 의 predict_from_image 를 그대로 재사용한다.
  기본 체크포인트는 강우 조건(FiLM) 모델(checkpoints_film/best.pt).
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
    "AI_MODEL_CHECKPOINT", str(AI_MODEL_DIR / "checkpoints_film" / "best.pt")
)

from src.inference_api import (  # noqa: E402  (경로 설정 후 import)
    current_checkpoint,
    current_uses_film,
    predict_from_image,
)
from src.utils.render_fields import render_contour_rgb  # noqa: E402

from terrain_io import InvalidTerrainData, load_terrain  # noqa: E402
from analysis_figure import intensity_from_prediction, render_analysis_figure  # noqa: E402
import auth  # noqa: E402  (회원가입/로그인 + MySQL)

RAIN_MIN_MM, RAIN_MAX_MM = 20.0, 80.0          # 학습 데이터가 다룬 강우 범위
DEFAULT_DX_M = 10.4166667                       # 1000m / 96격자
SAMPLES_DIR = AI_MODEL_DIR / "data" / "contours"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_HEAVY_LOCK = threading.Lock()                  # matplotlib/torch 직렬화 (데모 규모)

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
    uses_film = _safe_uses_film()
    return {
        "status": "ok",
        "checkpoint": os.path.basename(str(current_checkpoint())),
        "rain_conditioned": uses_film,
        "rain_range_mm": [RAIN_MIN_MM, RAIN_MAX_MM],
        "device": _device(),
        "auth_enabled": _DB_READY,
        "db": auth.db_status() if _DB_READY else "down",
    }


_AUG_RE = re.compile(r"_(?:r[0-3]|f[0-3])$")
_RV_ANY = re.compile(r"_rv\d+")


def _base_terrain_id(stem: str) -> str:
    """'seoul_sample_0003_rv1_f2' -> 'seoul_sample_0003' (증강/강우변형 접미사 제거)."""
    return _RV_ANY.sub("", _AUG_RE.sub("", stem))


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
        if base not in canonical or p.stem.endswith("_r0"):
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


def _auth_gate(authorization: str = Header(default="")) -> str | None:
    """인증이 켜져 있으면(=MySQL 연결됨) 토큰 필수, 아니면 통과."""
    if not _DB_READY:
        return None
    return auth.require_user(authorization)


@app.post("/api/predict")
async def predict(
    mode: str = Form(...),
    rain_mm: float = Form(50.0),
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

        uses_film = _safe_uses_film()
        meta["rain_conditioned"] = uses_film
        meta["rain_mm"] = rain_mm if uses_film else None

        with _HEAVY_LOCK:
            pred_img = predict_from_image(
                contour_img, rain_mm=rain_mm if uses_film else None
            )
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
                              " · 회색 음영 = 실제 지형 굴곡"),
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


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"
