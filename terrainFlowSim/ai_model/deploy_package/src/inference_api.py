"""
백엔드 연동용 경량 추론 API.

등고선 이미지(또는 지형 원시 데이터) + 강우량 + 경과 시점을 넣으면 예측된
배수(침수) 결과를 돌려준다. 두 가지 형태로 받을 수 있다:

  1) 이미지  : predict_from_image(...) / predict_from_terrain(...)        -> PIL.Image
  2) 실제 수심: predict_depth_from_image(...) / predict_depth_from_terrain(...) -> np.ndarray [m]

사용법:
    from src.inference_api import predict_from_image, predict_depth_from_image

    img = predict_from_image(uploaded_bytes, rain_mm=45.0, time_s=120.0)
    img.save("predicted.png")

    depth_m = predict_depth_from_image(uploaded_bytes, rain_mm=45.0, time_s=120.0)
    print(depth_m.max(), depth_m.mean())   # 실제 미터 단위

조건 인자 (현재 배포 체크포인트 v5 기준, 둘 다 필수):
  - rain_mm : 강우량 [mm]. 학습 범위 20~80mm (범위 밖 값은 자동으로 잘림).
  - time_s  : 강우 시작 후 경과 시간 [s]. 학습 범위 0~120s.
              120이 "비가 그친 뒤 최종 상태"에 해당한다.

체크포인트가 무엇을 요구하는지는 로드 시 가중치에서 자동 판별한다:
  - cond_dim=2 (v5, 현재 배포본): rain_mm + time_s 둘 다 필요
  - cond_dim=1 (구버전 FiLM)     : rain_mm만 필요
  - FiLM 없음  (구버전 U-Net/GAN): 조건 인자 없이 호출
잘못 호출하면 어떤 인자가 필요한지 한국어 에러 메시지로 알려준다.

수심 환산에 대해:
  v5부터 학습 타깃을 전역 기준(DEPTH_VMAX_M=0.6m) + sqrt 압축으로 정규화했기 때문에
  예측 이미지를 실제 수심[m]으로 되돌릴 수 있다(predict_depth_*).
  v4 이전 체크포인트는 샘플별 자체 정규화로 학습되어 절대 수심 정보가 없으므로
  predict_depth_*를 호출하면 명시적으로 에러를 낸다.

모델 교체:
  - 환경변수 AI_MODEL_CHECKPOINT를 새 .pt 경로로 지정 후 프로세스 재시작, 또는
  - reload_model("path/to/best.pt")로 무중단 교체.
  in_channels / FiLM 사용 여부 / cond_dim / ngf 모두 가중치에서 자동 판별한다.
"""

import io
import os
import threading

import numpy as np
import torch
import matplotlib
from PIL import Image
from torchvision import transforms

from src.models.unet_generator import UnetGenerator
from src.utils.image_utils import tensor_to_uint8
from src.utils.render_fields import render_contour_rgb, render_rain_channel, norm_to_depth

# 학습 시 조건 정규화 기준 (src/datasets/contour_flow_dataset.py와 동일 값.
# deploy_package는 학습 전용 모듈에 의존하지 않도록 값만 복제해 둔다.)
RAIN_MIN_MM, RAIN_MAX_MM = 20.0, 80.0
TIME_MIN_S, TIME_MAX_S = 0.0, 120.0


def normalize_rain(rain_mm):
    return max(0.0, min(1.0, (rain_mm - RAIN_MIN_MM) / (RAIN_MAX_MM - RAIN_MIN_MM)))


def normalize_time(t_s):
    return max(0.0, min(1.0, (t_s - TIME_MIN_S) / (TIME_MAX_S - TIME_MIN_S)))


DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pt")
CHECKPOINT_PATH = os.environ.get("AI_MODEL_CHECKPOINT", DEFAULT_CHECKPOINT)
IMAGE_SIZE = int(os.environ.get("AI_MODEL_IMAGE_SIZE", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_loaded_checkpoint_path = None
_loaded_in_channels = 3
_loaded_use_film = False
_loaded_cond_dim = 0          # 0 = 조건 없음, 1 = rain만, 2 = rain+time
_model_lock = threading.Lock()

_contour_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
_rain_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# 예측 RGB -> [0,1] 정규화값 복원용 Blues 컬러맵 역 LUT
_LUT_N = 256
_LUT = (np.array([matplotlib.colormaps["Blues"](i / (_LUT_N - 1))[:3]
                   for i in range(_LUT_N)]) * 255.0).astype(np.float32)


def _load_model(checkpoint_path=None):
    global _model, _loaded_checkpoint_path, _loaded_in_channels, _loaded_use_film, _loaded_cond_dim
    path = checkpoint_path or _loaded_checkpoint_path or CHECKPOINT_PATH
    with _model_lock:
        if _model is not None and _loaded_checkpoint_path == path:
            return _model
        state = torch.load(path, map_location=DEVICE)
        g = state["netG"]
        if "model.model.0.weight" in g:
            outer = "model.model.0.weight"
        elif "model.down.0.weight" in g:
            outer = "model.down.0.weight"
        else:
            raise RuntimeError(f"'{path}'에서 가장 바깥 conv 레이어를 찾지 못했습니다 (알 수 없는 구조).")
        in_channels = g[outer].shape[1]
        ngf = g[outer].shape[0]
        film_keys = [k for k in g if k.endswith("film.net.0.weight")]
        use_film = bool(film_keys)
        cond_dim = g[film_keys[0]].shape[1] if film_keys else 0

        net = UnetGenerator(in_channels=in_channels, out_channels=3, image_size=IMAGE_SIZE,
                             ngf=ngf, use_film=use_film, cond_dim=max(cond_dim, 1)).to(DEVICE)
        net.load_state_dict(g)
        net.eval()
        _model = net
        _loaded_checkpoint_path = path
        _loaded_in_channels = in_channels
        _loaded_use_film = use_film
        _loaded_cond_dim = cond_dim
        return _model


def current_checkpoint():
    """지금 서빙 중인 체크포인트 경로 (헬스체크/버전 응답용)."""
    return _loaded_checkpoint_path or CHECKPOINT_PATH


def current_in_channels():
    _load_model()
    return _loaded_in_channels


def current_uses_film():
    _load_model()
    return _loaded_use_film


def current_cond_dim():
    """이 모델이 요구하는 조건 개수: 0=없음, 1=rain_mm, 2=rain_mm+time_s."""
    _load_model()
    return _loaded_cond_dim


def supports_depth_output():
    """predict_depth_* 사용 가능 여부 (v5 이후 모델만 True)."""
    _load_model()
    return _loaded_cond_dim == 2


def reload_model(checkpoint_path):
    """서빙 도중 다른 체크포인트로 즉시 교체 (프로세스 재시작 불필요)."""
    global _model
    with _model_lock:
        _model = None
    return _load_model(checkpoint_path)


def _build_cond(rain_mm, time_s):
    """체크포인트가 요구하는 조건 벡터 생성. 인자가 안 맞으면 안내 메시지와 함께 에러."""
    ck = current_checkpoint()
    if _loaded_cond_dim == 0:
        if rain_mm is not None or time_s is not None:
            raise ValueError(f"현재 체크포인트({ck})는 조건을 지원하지 않습니다. "
                             f"rain_mm/time_s 없이 호출하세요.")
        return None
    if _loaded_cond_dim == 1:
        if rain_mm is None:
            raise ValueError(f"현재 체크포인트({ck})는 rain_mm(mm)이 필요합니다. "
                             f"예: predict_from_image(img, rain_mm=45.0)")
        return torch.tensor([[normalize_rain(rain_mm)]], dtype=torch.float32).to(DEVICE)
    if rain_mm is None or time_s is None:
        raise ValueError(
            f"현재 체크포인트({ck})는 rain_mm(mm)과 time_s(초) 둘 다 필요합니다. "
            f"예: predict_from_image(img, rain_mm=45.0, time_s=120.0) "
            f"(time_s=120이 비가 그친 뒤 최종 상태)")
    return torch.tensor([[normalize_rain(rain_mm), normalize_time(time_s)]],
                         dtype=torch.float32).to(DEVICE)


@torch.no_grad()
def _run(contour_img: Image.Image, rain_mm=None, time_s=None) -> Image.Image:
    model = _load_model()
    input_t = _contour_transform(contour_img.convert("RGB"))

    if _loaded_in_channels == 4:
        # 구버전 채널 결합 방식 (권장하지 않음, 하위호환용)
        if rain_mm is None:
            raise ValueError(f"현재 체크포인트({current_checkpoint()})는 rain_mm이 필요합니다.")
        rain_t = _rain_transform(Image.fromarray(render_rain_channel(rain_mm), mode="L"))
        input_t = torch.cat([input_t, rain_t], dim=0)
        cond_t = None
    else:
        cond_t = _build_cond(rain_mm, time_s)

    pred_t = model(input_t.unsqueeze(0).to(DEVICE), cond_t)
    return Image.fromarray(tensor_to_uint8(pred_t[0]))


def _image_to_depth(img: Image.Image) -> np.ndarray:
    """예측 이미지 -> 실제 수심[m]. Blues 컬러맵 역 LUT + norm_to_depth."""
    rgb = np.array(img.convert("RGB")).astype(np.float32)
    flat = rgb.reshape(-1, 3)
    idx = ((flat[:, None, :] - _LUT[None, :, :]) ** 2).sum(-1).argmin(1)
    norm = (idx / (_LUT_N - 1)).reshape(rgb.shape[:2])
    return norm_to_depth(norm)


def predict_from_image(image, rain_mm=None, time_s=None) -> Image.Image:
    """등고선 이미지(경로 1)로 배수 예측 -> PIL.Image.

    image: PIL.Image / 파일 경로(str) / raw bytes(업로드 파일) 모두 허용.
    """
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)
    return _run(image, rain_mm=rain_mm, time_s=time_s)


def predict_from_terrain(terrain: np.ndarray, dx: float = 10.416666666666666,
                          rain_mm=None, time_s=None) -> Image.Image:
    """지형 원시 데이터(경로 2)로 배수 예측 -> PIL.Image.

    terrain: (H, W) 고도 배열 [m], dx: 셀 크기 [m]
    학습 때와 동일한 렌더링으로 등고선 이미지를 만든 뒤 추론한다.
    """
    ny, nx = terrain.shape
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
    return _run(Image.fromarray(render_contour_rgb(X, Y, terrain)),
                 rain_mm=rain_mm, time_s=time_s)


def predict_depth_from_image(image, rain_mm=None, time_s=None, out_shape=None) -> np.ndarray:
    """등고선 이미지로 예측한 뒤 실제 수심[m] 배열로 반환.

    out_shape: (H, W)를 주면 그 크기로 리사이즈해서 반환 (기본 256x256).
    반환값은 0 ~ 0.6m 범위 (학습 데이터 최대 수심 기준).
    """
    if not supports_depth_output():
        raise ValueError(
            f"현재 체크포인트({current_checkpoint()})는 절대 수심 정보를 담고 있지 않습니다"
            f"(v4 이전은 샘플별 자체 정규화로 학습됨). v5 이후 체크포인트를 사용하세요.")
    img = predict_from_image(image, rain_mm=rain_mm, time_s=time_s)
    if out_shape is not None:
        img = img.resize((out_shape[1], out_shape[0]), Image.BICUBIC)
    return _image_to_depth(img)


def predict_depth_from_terrain(terrain: np.ndarray, dx: float = 10.416666666666666,
                                rain_mm=None, time_s=None) -> np.ndarray:
    """지형 원시 데이터로 예측한 뒤 실제 수심[m] 배열(입력 지형과 같은 크기)로 반환."""
    if not supports_depth_output():
        raise ValueError(
            f"현재 체크포인트({current_checkpoint()})는 절대 수심 정보를 담고 있지 않습니다"
            f"(v4 이전은 샘플별 자체 정규화로 학습됨). v5 이후 체크포인트를 사용하세요.")
    img = predict_from_terrain(terrain, dx=dx, rain_mm=rain_mm, time_s=time_s)
    img = img.resize((terrain.shape[1], terrain.shape[0]), Image.BICUBIC)
    return _image_to_depth(img)
