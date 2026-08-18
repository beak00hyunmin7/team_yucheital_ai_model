"""
백엔드 연동용 경량 추론 API.

학습에 쓰인 TranslationModel(옵티마이저 + 판별자까지 포함하는 래퍼) 대신
UnetGenerator만 직접 불러와서 추론만 수행한다. mode: "gan"으로 학습한
체크포인트도 생성자 구조(UnetGenerator)는 mode: "unet"과 동일하므로
코드 변경 없이 그대로 로드된다 (판별자는 추론에 쓰이지 않음).

사용법:
    from src.inference_api import predict_from_image, predict_from_terrain

    # 경로 1: 이미 렌더링된 등고선 이미지 (PIL.Image / 파일경로 / bytes)
    result_img = predict_from_image(uploaded_file_bytes)

    # 경로 2: 지형 원시 데이터 (고도 격자, numpy (H, W)[m])
    result_img = predict_from_terrain(terrain_array, dx=10.416666666666666)

    result_img.save("predicted.png")   # 또는 io.BytesIO()에 저장해 그대로 응답

모델 교체 (성능이 더 좋은 체크포인트가 나왔을 때):
    - 가장 간단한 방법: 환경변수 AI_MODEL_CHECKPOINT를 새 .pt 경로로 바꾸고 백엔드 프로세스만 재시작.
      코드 수정 불필요.
    - 무중단으로 바꾸고 싶다면: reload_model("checkpoints_gan/best.pt") 처럼 직접 호출하면
      실행 중에도 즉시 다음 요청부터 새 가중치로 교체됨 (재시작 불필요).
    - 어느 방법이든 아키텍처(UnetGenerator)가 같은 체크포인트끼리는 100% 호환.
      입력/출력 채널 수(3, 3)나 image_size(256)를 바꾼 새 모델이면 이 파일의
      IMAGE_SIZE 쪽도 같이 맞춰줘야 함.
"""

import io
import os
import threading

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.unet_generator import UnetGenerator
from src.utils.image_utils import tensor_to_uint8
from src.utils.render_fields import render_contour_rgb

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pt")
CHECKPOINT_PATH = os.environ.get("AI_MODEL_CHECKPOINT", DEFAULT_CHECKPOINT)
IMAGE_SIZE = int(os.environ.get("AI_MODEL_IMAGE_SIZE", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_loaded_checkpoint_path = None
_model_lock = threading.Lock()

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def _load_model(checkpoint_path=None):
    global _model, _loaded_checkpoint_path
    path = checkpoint_path or CHECKPOINT_PATH
    with _model_lock:
        if _model is not None and _loaded_checkpoint_path == path:
            return _model
        net = UnetGenerator(in_channels=3, out_channels=3, image_size=IMAGE_SIZE).to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        net.load_state_dict(state["netG"])
        net.eval()
        _model = net
        _loaded_checkpoint_path = path
        return _model


def reload_model(checkpoint_path):
    """서빙 도중 다른 체크포인트로 즉시 교체한다 (프로세스 재시작 불필요).

    예: reload_model("checkpoints_gan/best.pt")
    """
    global _model
    with _model_lock:
        _model = None
    return _load_model(checkpoint_path)


def current_checkpoint():
    """지금 서빙 중인 체크포인트 경로 (헬스체크/버전 응답용)."""
    return _loaded_checkpoint_path or CHECKPOINT_PATH


@torch.no_grad()
def _run(contour_img: Image.Image) -> Image.Image:
    model = _load_model()
    input_t = _transform(contour_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    pred_t = model(input_t)
    return Image.fromarray(tensor_to_uint8(pred_t[0]))


def predict_from_image(image) -> Image.Image:
    """등고선 이미지(경로 1)로 배수 예측.

    image: PIL.Image, 파일 경로(str), 또는 raw bytes(업로드 파일 등) 모두 허용.
    """
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)
    return _run(image)


def predict_from_terrain(terrain: np.ndarray, dx: float = 10.416666666666666) -> Image.Image:
    """지형 원시 데이터(경로 2)로 배수 예측.

    terrain: (H, W) 고도 배열 [m]
    dx: 셀 크기 [m] (학습 데이터는 1000m 도메인 / 96 격자 = 약 10.4167m 사용)

    학습 때와 동일한 렌더링(render_contour_rgb)으로 먼저 등고선 이미지를 만든 뒤 추론한다.
    """
    ny, nx = terrain.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)
    contour_rgb = render_contour_rgb(X, Y, terrain)
    return _run(Image.fromarray(contour_rgb))
