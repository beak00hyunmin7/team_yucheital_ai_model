"""
B안 추론 API: 고도격자(numpy) + 강우량 -> 예측 수심격자(numpy, m) / 시각화 PNG.

    from src.inference_scalar import predict_depth, render_depth_png

    depth_m = predict_depth(terrain_array, rain_mm=45.0)   # (H, W) meters
    img = render_depth_png(depth_m)                        # PIL.Image (Blues)

체크포인트 경로는 환경변수 SCALAR_MODEL_CHECKPOINT 또는 checkpoints_scalar/best.pt.
체크포인트 안에 image_size / ngf / stats 가 들어있어 별도 설정 없이 로드된다.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.scalar_flow_dataset import y_to_depth
from src.models.unet_generator import UnetGenerator
from src.utils.flow_accum import d8_flow_accumulation

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints_scalar", "best.pt")
CKPT_PATH = os.environ.get("SCALAR_MODEL_CHECKPOINT", DEFAULT_CKPT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_loaded_path = None
_meta: dict = {}
_lock = threading.Lock()


def _load(path: str | None = None):
    global _model, _loaded_path, _meta
    path = path or _loaded_path or CKPT_PATH
    with _lock:
        if _model is not None and _loaded_path == path:
            return _model
        st = torch.load(path, map_location=DEVICE)
        img = st.get("image_size", 128)
        ngf = st.get("ngf", 32)
        in_ch = st.get("in_channels", 1)
        net = UnetGenerator(in_channels=in_ch, out_channels=1, image_size=img, ngf=ngf,
                            use_film=st.get("use_film", True), cond_dim=1).to(DEVICE)
        net.load_state_dict(st["netG"])
        net.eval()
        _model = net
        _loaded_path = path
        _meta = {"image_size": img, "ngf": ngf, "in_channels": in_ch,
                 "use_flowacc": bool(st.get("use_flowacc", in_ch >= 2)), **st.get("stats", {})}
        return _model


def reload_model(path: str):
    global _model
    with _lock:
        _model = None
    return _load(path)


def current_checkpoint() -> str:
    return _loaded_path or CKPT_PATH


def model_info() -> dict:
    _load()
    return dict(_meta)


@torch.no_grad()
def predict_depth(terrain: np.ndarray, rain_mm: float) -> np.ndarray:
    """(H, W) 고도격자[m] + 강우량[mm] -> (H, W) 예측 수심[m] (입력과 같은 해상도)."""
    net = _load()
    img = _meta["image_size"]
    ref, log_max = _meta["depth_ref_m"], _meta["depth_log_max"]
    r_lo, r_hi = _meta["rain_min_mm"], _meta["rain_max_mm"]

    terrain = np.asarray(terrain, dtype=np.float32)
    h0, w0 = terrain.shape
    lo, hi = float(terrain.min()), float(terrain.max())
    tn = (terrain - lo) / (hi - lo + 1e-6) * 2.0 - 1.0

    chans = [tn]
    if _meta.get("use_flowacc"):
        fa_log_max = _meta.get("flowacc_log_max") or 8.5
        acc = d8_flow_accumulation(terrain)
        chans.append(np.clip(np.log1p(acc) / fa_log_max, 0.0, 1.0).astype(np.float32) * 2.0 - 1.0)

    x = torch.from_numpy(np.stack(chans))[None].to(DEVICE)   # (1, C, H, W)
    x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False)
    cond = torch.tensor([[float(np.clip((rain_mm - r_lo) / (r_hi - r_lo), 0.0, 1.0))]],
                        dtype=torch.float32, device=DEVICE)

    y = net(x, cond)
    y = F.interpolate(y, size=(h0, w0), mode="bilinear", align_corners=False)
    return y_to_depth(y[0, 0].cpu().numpy(), ref, log_max)


def render_depth_png(depth_m: np.ndarray, vmax: float | None = None, upscale: int = 4):
    """예측 수심격자 -> Blues 컬러맵 PIL 이미지 (webapp 표시용)."""
    from PIL import Image

    d = np.maximum(np.asarray(depth_m, dtype=np.float64), 0.0)
    if vmax is None:
        vmax = max(float(np.percentile(d, 99.5)), 1e-6)
    a = np.clip(d / vmax, 0.0, 1.0)
    rgb = np.stack([1.0 - 0.85 * a, 1.0 - 0.55 * a, np.ones_like(a)], axis=-1)
    im = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    if upscale > 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.NEAREST)
    return im
