# contour2flow 추론 패키지 (백엔드 연동용)

등고선 이미지(또는 지형 원시 데이터)를 넣으면 예측된 배수 이미지를 반환하는
추론 전용 패키지입니다. 학습 코드는 포함되어 있지 않습니다.

## 설치

```bash
pip install -r requirements.txt
```

GPU가 있으면 자동으로 사용하고(cuda), 없으면 CPU로 동작합니다.

## 사용법

이 폴더(`deploy_package/`)를 루트로 실행해야 합니다 (`src/`가 바로 아래 있어야 함).

```python
from src.inference_api import predict_from_image, predict_from_terrain

# 경로 1: 이미 렌더링된 등고선 이미지 (PIL.Image / 파일경로 / bytes)
result_img = predict_from_image(uploaded_file_bytes)
result_img.save("predicted.png")

# 경로 2: 지형 원시 데이터 (고도 격자, numpy (H, W)[m])
import numpy as np
terrain = np.load("terrain.npy")
result_img = predict_from_terrain(terrain, dx=10.416666666666666)
```

두 함수 모두 `PIL.Image`를 반환합니다. FastAPI라면 `io.BytesIO()`에 담아
`StreamingResponse(media_type="image/png")`로 바로 응답하면 됩니다.

### FastAPI 연동 예시

```python
import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from src.inference_api import predict_from_image

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    result_img = predict_from_image(image_bytes)

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
```

`uvicorn main:app`으로 띄우면 `/predict`에 이미지를 POST하는 것만으로 바로 동작합니다.

## 모델 교체

새 체크포인트(`.pt`)가 나오면 코드 수정 없이 아래 중 하나로 교체:

- 환경변수로 지정 후 프로세스 재시작:
  ```bash
  export AI_MODEL_CHECKPOINT=/path/to/new_best.pt
  ```
- 무중단 교체(실행 중에 바로 반영):
  ```python
  from src.inference_api import reload_model
  reload_model("/path/to/new_best.pt")
  ```

`mode: unet`이든 `mode: gan`이든 생성자 구조(`UnetGenerator`)가 동일해서
어떤 체크포인트든 그대로 로드됩니다. 단, 입력/출력 채널 수나 이미지 해상도
(현재 256x256)가 다른 모델로 바뀌면 `src/inference_api.py`의 `IMAGE_SIZE`도
같이 맞춰야 합니다.

## 폴더 구조

```
deploy_package/
├── checkpoints/
│   └── best.pt              # 학습된 모델 가중치
├── src/
│   ├── inference_api.py     # predict_from_image / predict_from_terrain
│   ├── models/
│   │   └── unet_generator.py
│   └── utils/
│       ├── image_utils.py
│       └── render_fields.py # 지형 원시 데이터 -> 등고선 이미지 렌더링 (경로 2용)
└── requirements.txt
```
