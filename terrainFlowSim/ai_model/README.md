# contour2flow — 등고선 → 침수 수심맵 모델

등고선 이미지와 강우 조건을 입력받아 침수 수심맵을 출력하는 image-to-image 모델입니다.
pix2pix 계열 U-Net 생성자에 강우량·경과시간을 **FiLM**으로 병목에 주입합니다.

저장소 전체 개요는 [루트 README](../../README.md), 개선 경과와 실험 근거는
[`IMPROVEMENT_GUIDE.md`](IMPROVEMENT_GUIDE.md) 를 보세요.

```
고도 격자 ──render_contour_rgb──> 등고선 RGB(3ch)
                                      │  + cond = [강우량, 경과시간]
                                      ▼
                              U-Net (FiLM 조건 주입)
                                      ▼
                               수심 RGB (Blues)
                                      │  역LUT + norm_to_depth
                                      ▼
                                 수심 배열 [m]
```

---

## 현행 모델

| | |
|---|---|
| 체크포인트 | `checkpoints_p2_wet8/best.pt` |
| 설정 | `configs/config_p2_wet8.yaml` |
| 크기 | 생성자 10.5M, 128×128, ngf 32, GAN 없음 |
| 조건 | `cond_dim=2` — 강우량(20~80 mm) + 경과시간(0~120 s) |
| 손실 | 젖은 픽셀 가중 L1 (`wet_weight: 8`) |

**val 성능** (학습에 쓰지 않은 지형 87종 / 911장, 정답은 OpenFOAM 원본 `depth.npy`):

| MAE | r | IoU@1cm | IoU@5cm | 최대수심 비율 | 최심점오차 | 강우 단조성 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0115 m | 0.682 | 0.688 | 0.426 | 0.619 | 300 m | 98.8% |

전부 `scripts/evaluate.py` 로 재현됩니다 (`eval_p2_wet8/summary.md`).

### 체크포인트 계보

| 이름 | 무엇 | 비고 |
|---|---|---|
| `checkpoints_film` | 강우 FiLM (`cond_dim=1`), 256px, 54.5M | **타깃 정규화 버그 있음.** 쓰지 말 것 |
| `checkpoints_v5` | 강우+시점 (`cond_dim=2`), 256px + GAN, 54.5M | 타깃 버그 수정본. 686 MB |
| **`checkpoints_p2_wet8`** | v5와 같은 조건, 128px/10.5M, 젖은 픽셀 가중 L1 | **현행.** v5보다 모든 물리 지표에서 우위 |
| `checkpoints_p2_{base,wet16,wet32}` | `wet_weight` 스윕 (1/16/32) | w=8에서 포화함을 보인 대조군 |
| `checkpoints_scale{75,150,300,full}` | 학습 지형 수만 바꾼 스케일링 실험 | |

v5(54.5M, 256px, GAN, 10시간 학습)보다 wet8(10.5M, 128px, 21분 학습)이 낫습니다 —
IoU@5cm 0.426 vs 0.378, 최심점오차 300 vs 324 m. **큰 모델·GAN·고해상도가 값을 못 하고 있습니다.**

---

## 설치 (처음 한 번만)

### 1. Python 3.11

Windows에서 `python` 이 마이크로소프트 스토어 더미로 연결돼 있는 경우가 많습니다.

```powershell
winget install -e --id Python.Python.3.11 --source winget
```

설치 후 **새 PowerShell 창**을 열어야 PATH가 반영됩니다.

```powershell
python --version   # Python 3.11.x (스토어 더미면 버전 없이 "Python"만 출력됨)
```

### 2. 가상환경 + 라이브러리

```powershell
cd ai_model
.\scripts\setup_env.ps1
```

기본값은 CUDA 12.8 빌드(cu128)입니다. RTX 50xx(Blackwell)는 이 버전 이상이 필요합니다.
다른 세대 GPU나 CPU만 쓴다면 `scripts/setup_env.ps1` 의 `--index-url` 을 바꾸세요
(https://pytorch.org/get-started/locally/). Linux는 `bash scripts/setup_env.sh`.

### 3. 확인

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 데이터셋 준비

OpenFOAM 결과(`terrain.npy` + `depth_series.npy` + `meta.json`)를 학습용 이미지 쌍으로 변환합니다.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_openfoam_dataset.py
```

만들어지는 것:

```
data/
├── contours/           등고선 RGB (입력)
├── flow/               수심 RGB (타깃)
├── rain_values.json    {파일명: 강우량 mm}
└── time_values.json    {파일명: 경과시간 s}
```

파일명 규칙은 `{지역}_{sample}_{증강}_t{시점}.png` 입니다
(예: `seoul_sample_0012_rv1_r0_t3.png`). 증강은 D4군 8종(`r0`~`r3`, `f0`~`f3`),
`rv1`/`rv2` 는 같은 지형에 강우량만 바꿔 다시 돌린 시뮬레이션입니다.

현재 규모: 고유 지형 586종, 학습 쌍 48,528장.

> **타깃 정규화 주의**: 수심은 전역 기준 `DEPTH_VMAX_M=0.6 m` + sqrt 압축으로 정규화한 뒤
> 렌더링합니다. 예전처럼 샘플마다 자기 최댓값으로 정규화하면 절대 수심 정보가 사라져서
> **어떤 아키텍처로도 강우 조건을 학습할 수 없습니다** (실제로 v1~v4가 여기서 막혔습니다).
> `src/utils/render_fields.py` 주석 참고.

---

## 학습

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\config_p2_wet8.yaml
```

주요 설정 키:

| 키 | 의미 |
|---|---|
| `mode` | `unet`(L1만) 또는 `gan`(+ PatchGAN) |
| `image_size`, `ngf` | 해상도와 채널 폭 |
| `rain_values_json`, `time_values_json` | 주면 FiLM 조건으로 사용 (둘 다 주면 `cond_dim=2`) |
| `wet_weight` | 젖은 픽셀 가중 L1. 1이면 균등 L1과 동일 |
| `max_train_groups` | 학습에 쓸 **고유 지형 수** 제한 (스케일링 실험용) |

train/val 분리는 **지형 단위**입니다. 같은 지형의 회전본·강우변형·다른 시점이 train과 val로
나뉘지 않습니다 (`contour_flow_dataset.py: _group_key`).

`sample_epoch_freq` 에폭마다 `outputs_*/sample_epoch_XXXX.png` 에
[입력 | 예측 | 정답]이 나란히 저장됩니다.

---

## 평가

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
    --checkpoint checkpoints_p2_wet8/best.pt --config configs/config_p2_wet8.yaml --tag p2_wet8
```

`eval_p2_wet8/` 에 `summary.md`(표), `per_sample.csv`, `worst_N.png`(최악 사례)가 나옵니다.

설계상 중요한 두 가지:

1. **배포 경로 그대로** 잽니다 (`등고선 → UNet → 예측 RGB → 역LUT → 수심`).
   모델 출력 텐서를 직접 쓰면 사용자가 받는 값이 아니라 중간값을 재게 됩니다.
2. **정답은 학습 타깃 PNG가 아니라 OpenFOAM 원본 `depth.npy`(m)** 입니다.
   타깃 PNG와 비교하면 렌더링 손실이 정답 쪽에도 들어가서 오차가 실제보다 작게 나옵니다.

유용한 옵션:

| 옵션 | 용도 |
|---|---|
| `--split train` | 학습에 쓴 지형에서 평가 → 과적합/과소적합 판별 |
| `--limit-terrains N` | 빠른 점검 (지역 균형 유지) |
| `--equivariance N` | D4 등변성 측정 (같은 지형 8방향 예측의 흔들림) |

### 함께 쓰는 도구

```powershell
# 여러 모델 한 표로 비교 (최대수심 비율 포함)
.\.venv\Scripts\python.exe scripts\compare_evals.py v5 p2_base p2_wet8

# 표현 방식(렌더→역LUT) 자체의 정보손실 상한
.\.venv\Scripts\python.exe scripts\diagnose_pipeline.py

# 데이터 스케일링 곡선
.\.venv\Scripts\python.exe scripts\scaling_study.py --base configs/config_p2_wet8.yaml
```

---

## 추론

```python
from src.inference_api import predict_from_image, predict_depth_from_terrain

img     = predict_from_image(uploaded_bytes, rain_mm=60.0, time_s=18.0)   # PIL 이미지
depth_m = predict_depth_from_terrain(terrain, dx=10.4167, rain_mm=60.0, time_s=18.0)
```

체크포인트에서 **입력 채널 수·FiLM 사용 여부·`cond_dim`·`ngf`·학습 해상도를 모두 자동 판별**하므로,
체크포인트 경로만 바꾸면 됩니다 (`AI_MODEL_CHECKPOINT` 환경변수). 필요한 조건을 빼먹으면
한국어 메시지로 무엇이 필요한지 알려줍니다.

> **경과 시간 주의**: 수심은 시간이 갈수록 **줄어듭니다** (비가 초반에 내리고 빠지는 시뮬레이션).
> 정답 평균 수심이 t=18 s 0.039 m → t=120 s 0.019 m. "가장 심한 상태"를 보려면 18초 부근을 쓰세요.

> **방향 주의**: `predict_depth_*` 반환 배열은 입력 지형 대비 **상하반전**입니다.
> `terrain` 과 겹쳐 그릴 때는 `np.flipud` 로 되돌려야 합니다.

---

## 폴더 구조

```
ai_model/
├── configs/                학습 설정 (config_p2_wet8.yaml 이 현행)
├── data/                   학습 이미지 쌍 (git 제외)
├── src/
│   ├── datasets/           이미지 쌍 로딩, 지형 단위 train/val 분리
│   ├── models/             U-Net(FiLM) 생성자, PatchGAN 판별자, 학습 래퍼
│   ├── utils/              렌더링, 이미지 변환, D8 흐름누적
│   ├── inference_api.py    추론 진입점 (배포용)
│   ├── ground_truth.py     파일명 → OpenFOAM 정답 조회 (웹앱과 공용)
│   └── train.py
├── scripts/                데이터 준비 + 평가 도구
├── deploy_package/         백엔드 전달용 추론 전용 패키지
├── checkpoints*/           학습 가중치 (git 제외)
├── eval_*/                 평가 결과
├── EXPERIMENTS.md          경량화·스칼라 회귀 실험 기록
└── IMPROVEMENT_GUIDE.md    검증·개선 경과 (Phase 0~5)
```

---

## 다음에 해볼 것

`IMPROVEMENT_GUIDE.md` 에 근거와 함께 정리돼 있습니다. 요약하면:

- **해상도 정합** — 원본은 96×96인데 128/256으로 올렸다 내립니다. 얇은 배수채널이 그 과정에서
  뭉개지므로, 최대수심 비율 0.62를 더 끌어올릴 다음 후보입니다
- **스칼라 회귀 재시도** — 렌더/역LUT를 거치지 않고 고도 격자 → 수심 격자로 직접 회귀.
  접었던 판단은 학습 쌍이 4,568장이던 시절 것이고 지금은 10배가 됐습니다
- **데이터** — 지형 2배당 r +0.028의 로그선형. 듣긴 하지만 OpenFOAM 계산 시간이 크게 듭니다
