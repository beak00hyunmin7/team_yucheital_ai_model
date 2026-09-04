# 등고선 → 배수 예측 AI 모델 (백엔드/프론트엔드 전달 문서)

지형 등고선(또는 원시 고도 데이터)과 **강우량·경과시간**을 입력하면, OpenFOAM
시뮬레이션 없이 **예측된 침수 수심(m)**을 즉시 반환하는 추론 전용 패키지입니다.
이 폴더(`deploy_package/`)만으로 독립 실행됩니다.

> **⚠️ v5부터 인터페이스가 바뀌었습니다.** 이전 버전을 쓰고 계셨다면 아래
> "v4 → v5 마이그레이션"을 먼저 봐주세요.

---

## 1. 이 모델이 하는 일

- **학습 데이터**: 서울/부산/강원 실제 지형(국가기본도 등고선) 위에 OpenFOAM
  (`interFoam`, VOF)으로 강우 시뮬레이션을 돌린 결과. 지형 586개 × 강우조건 ×
  6개 시점 = 학습 이미지 48,528쌍.
- **모델**: pix2pix 스타일 U-Net 생성자 + PatchGAN 판별자, 강우량·경과시간을
  FiLM으로 병목에 주입(cond_dim=2).
- **입력**: 등고선 이미지(또는 고도 격자) + `rain_mm` + `time_s`
- **출력**: 침수 이미지(PIL) 또는 **실제 수심 배열(m)**

### 성능 (학습에 쓰이지 않은 val 지형 기준)

| 지표 | 값 |
|---|---|
| 평균 수심 MAE | 0.0049 m (정답 평균수심 0.0198 m 대비) |
| 예측↔정답 상관계수 | r = 0.919 |
| 강우량 방향성 정확도 | 70/70 쌍 = 100% (비 많이 오면 더 깊게 예측) |

---

## 2. 설치 및 사용법

```bash
pip install -r requirements.txt
```

GPU가 있으면 자동으로 사용하고(cuda), 없으면 CPU로 동작합니다.
이 폴더를 루트로 실행해야 합니다 (`src/`가 바로 아래 있어야 함).

```python
from src.inference_api import predict_from_image, predict_depth_from_image

# (A) 이미지로 받기 - 화면에 그대로 띄울 때
img = predict_from_image(uploaded_bytes, rain_mm=45.0, time_s=120.0)
img.save("predicted.png")

# (B) 실제 수심[m] 배열로 받기 - 수치가 필요할 때
depth_m = predict_depth_from_image(uploaded_bytes, rain_mm=45.0, time_s=120.0)
print(f"최대 침수 {depth_m.max():.3f}m, 평균 {depth_m.mean():.3f}m")
```

지형 원시 데이터(고도 격자)를 직접 넣는 경로도 있습니다:

```python
import numpy as np
from src.inference_api import predict_from_terrain, predict_depth_from_terrain

terrain = np.load("terrain.npy")          # (H, W) 고도 배열 [m]
img     = predict_from_terrain(terrain, dx=10.4167, rain_mm=45.0, time_s=120.0)
depth_m = predict_depth_from_terrain(terrain, dx=10.4167, rain_mm=45.0, time_s=120.0)
# depth_m은 입력 지형과 같은 (H, W) 크기
```

### 조건 인자

| 인자 | 의미 | 학습 범위 | 비고 |
|---|---|---|---|
| `rain_mm` | 강우량 [mm] | 20 ~ 80 | 범위 밖 값은 자동으로 잘림 |
| `time_s` | 강우 시작 후 경과 시간 [s] | 0 ~ 120 | **120 = 비 그친 뒤 최종 상태** |

일반적인 "이 지형 침수 예상" 조회라면 `time_s=120.0`을 쓰면 됩니다.
`time_s`를 20/40/…/120으로 훑으면 물이 차올랐다 빠지는 **시계열 애니메이션**을
만들 수 있습니다.

인자를 빠뜨리면 무엇이 필요한지 한국어 에러 메시지로 알려줍니다.

### FastAPI 연동 예시

```python
import io
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from src.inference_api import predict_from_image, predict_depth_from_image

app = FastAPI()

@app.post("/predict")           # 이미지 응답
async def predict(file: UploadFile, rain_mm: float = Form(...), time_s: float = Form(120.0)):
    img = predict_from_image(await file.read(), rain_mm=rain_mm, time_s=time_s)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/predict/depth")     # 수치 응답
async def predict_depth(file: UploadFile, rain_mm: float = Form(...), time_s: float = Form(120.0)):
    d = predict_depth_from_image(await file.read(), rain_mm=rain_mm, time_s=time_s)
    return JSONResponse({
        "max_depth_m": float(d.max()),
        "mean_depth_m": float(d.mean()),
        "flooded_area_ratio": float((d > 0.05).mean()),   # 5cm 이상 잠긴 면적 비율
    })
```

---

## 3. v4 → v5 마이그레이션 (⚠️ 중요)

| | v4 이전 | **v5 (현재)** |
|---|---|---|
| 필수 인자 | 없음 | **`rain_mm`, `time_s`** |
| 출력 | 렌더링 이미지만 | 이미지 + **실제 수심(m)** |
| 수심 수치 표시 | 불가 | **가능** |
| 체크포인트 크기 | 686 MB | 218 MB (추론에 불필요한 옵티마이저·판별자 제거) |

**왜 바뀌었나**: v4까지는 학습 타깃을 만들 때 각 샘플을 자기 최댓값으로 정규화해서
절대 수심 정보가 사라져 있었습니다. 그 결과 "비가 많이 오면 더 침수된다"는 관계가
학습 데이터에 57.5%(사실상 랜덤)만 담겨 있었고, 강우량을 바꿔도 예측이 거의
변하지 않았습니다. v5에서는 전역 기준(0.6m) + sqrt 압축으로 바꿔 99.3%로 올렸고,
그 덕분에 강우 조건이 실제로 작동하고 절대 수심 환산도 가능해졌습니다.

기존 v4 체크포인트는 `checkpoints/best_v4_gan_legacy.pt`로 보관돼 있습니다.
그걸 로드하면 `rain_mm`/`time_s` 없이 호출해야 하고 `predict_depth_*`는 막힙니다
(절대 수심 정보가 없으므로 의도적으로 에러를 냅니다).

---

## 4. 알려진 한계

- **최대 수심 과소예측**: 국소적으로 가장 깊은 지점은 실제보다 낮게 예측하는
  경향이 있습니다(예: 정답 0.120m → 예측 0.080m). 평균 수심은 오차 10% 내로
  잘 맞습니다. "최악의 한 점" 수치보다 **평균·면적 기반 지표**를 쓰는 걸 권장합니다.
- **경계 디테일**: 침수 영역의 대략적 위치와 크기는 맞지만, 실제 시뮬레이션이
  보여주는 날카로운 경계선까지는 재현하지 못합니다(다소 뭉개짐).
- **지형 스케일**: 학습 데이터는 1km × 1km, 96×96 격자(`dx ≈ 10.42m`) 기준입니다.
  크게 다른 스케일의 지형은 신뢰도가 떨어집니다.
- **강우 모델**: 균일 강우를 가정합니다(공간적으로 다른 강우 분포는 미지원).

---

## 5. 모델 교체

```bash
export AI_MODEL_CHECKPOINT=/path/to/new_best.pt   # 후 프로세스 재시작
```
```python
from src.inference_api import reload_model
reload_model("/path/to/new_best.pt")               # 무중단 교체
```

`in_channels` / FiLM 사용 여부 / `cond_dim` / `ngf`는 모두 가중치에서 자동
판별하므로 코드 수정이 필요 없습니다. 상태 확인용 함수도 있습니다:

```python
from src.inference_api import current_checkpoint, current_cond_dim, supports_depth_output
current_checkpoint()      # 지금 서빙 중인 .pt 경로
current_cond_dim()        # 0=조건없음, 1=rain만, 2=rain+time
supports_depth_output()   # predict_depth_* 사용 가능 여부
```

---

## 폴더 구조

```
deploy_package/
├── checkpoints/
│   ├── best.pt                    # v5 (218MB, 추론 전용 슬림)
│   └── best_v4_gan_legacy.pt      # 구버전 보관용
├── src/
│   ├── inference_api.py           # predict_* / predict_depth_*
│   ├── models/unet_generator.py   # U-Net + FiLM
│   └── utils/
│       ├── image_utils.py
│       └── render_fields.py       # 등고선 렌더링, 수심 정규화/역변환
└── requirements.txt
```
