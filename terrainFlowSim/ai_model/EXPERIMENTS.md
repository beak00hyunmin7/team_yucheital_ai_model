# 경량화 / 성능 실험 (A · B)

현행 FiLM 모델: `checkpoints_film/best.pt`, netG **54.5M** params, 256×256, val L1 0.081.
원본 지형은 96×96 (OpenFOAM) → 256은 과한 해상도. 데이터는 지형 491종(seoul 263 / busan 277 / gangwon 31), D4 증강 후 4568쌍.

## A안 — 같은 파이프라인, 크기만 축소

- `configs/config_film_light.yaml`: `image_size 256→128`, `ngf 64→32`
- netG **54.5M → 10.5M** (약 5×), 추론 4×+ 가속, 작은 데이터셋 과적합 완화
- 코드 변경: `TranslationModel`/`train.py` 에 `ngf` 인자 추가, `inference_api.py` 가 체크포인트에서 `ngf`·`in_channels` 자동 판별 (기존 256/64 체크포인트도 그대로 로드됨)

실행:
```powershell
.venv\Scripts\python.exe -m src.train --config configs\config_film_light.yaml
```
결과: `checkpoints_film_light/best.pt`, 미리보기 `outputs_film_light/`

webapp 에서 쓰려면:
```powershell
$env:AI_MODEL_CHECKPOINT = "...\ai_model\checkpoints_film_light\best.pt"
$env:AI_MODEL_IMAGE_SIZE = "128"
```

## B안 — 이미지 대신 스칼라 격자 회귀 (권장)

현행은 `지형 → matplotlib 등고선 RGB(256) → UNet → 수심 RGB(256) → 강도 역해석`.
렌더/역해석에서 정밀도가 날아가고, 모델이 컬러맵 디코딩까지 배워야 함.
수심이 sub-cm(≈0.1~40mm)라 RGB 타깃이 거의 흰색 → 예측이 뿌옇게 나오는 원인.

B안: **1채널 고도격자 → 1채널 수심격자**, 강우량은 FiLM 조건.
- `scripts/build_scalar_dataset.py` — OpenFOAM 원본 npy 를 `data_scalar/` 로 수집 (렌더링 X)
- 수심은 `y = log1p(depth/0.1mm) / log_max` 로 정규화 → 치우친 분포에서도 학습됨
- 손실: **젖은 픽셀 가중 L1** (`wet_weight=4`) → 배경 필름이 아니라 유출 경로에 페널티 집중
- `src/datasets/scalar_flow_dataset.py`, `src/train_scalar.py`, `src/inference_scalar.py`, `configs/config_scalar.yaml`

실행:
```powershell
.venv\Scripts\python.exe scripts\build_scalar_dataset.py          # 1회 (데이터 갱신 시 재실행)
.venv\Scripts\python.exe -m src.train_scalar --config configs\config_scalar.yaml
```
결과: `checkpoints_scalar/best.pt` (netG 10.5M, image_size·ngf·정규화상수 내장), 미리보기 `outputs_scalar/`
검증 로그에 `val L1(depth) … mm` 로 실제 수심 오차(mm)까지 표시.

추론:
```python
from src.inference_scalar import predict_depth, render_depth_png
depth_m = predict_depth(terrain_96x96, rain_mm=45.0)   # (H,W) meters
render_depth_png(depth_m).save("pred.png")
```

## 스칼라 회귀 레버 실험 결과 (2026-08-28~29)

val 5샘플 평균 MAE (실제 수심 mm, GT 최대 40~144mm):

| 실험 | 변경 | val MAE | 비고 |
|---|---|---:|---|
| v1 | 스칼라 1ch→1ch, L1, ref 0.1mm, wet 4 | 4.84 mm | 기준 |
| **B1** | ref 0.1→1mm, wet 4→8 | **4.57 mm** | 소폭 개선 (최선) |
| B2 | B1 + PatchGAN (LSGAN, gan_w 0.5) | 5.49 mm | **악화**. 체커보드 아티팩트, 판별자 붕괴(d≈0.01) |
| B3 | B1 + Sobel gradient L1 (grad_w 3) | 4.78 mm | ~중립. 블러 경계만 살짝 |

**결론: 손실 함수 튜닝으로는 안 됨.** 네 변형 모두 "대충 젖는 구역"(상대오차 5~8%)은
맞추지만 정답의 가는 수지상 배수망은 하나도 못 살림. 488개 지형 / 128px / 순수
elevation→depth UNet 의 한계.

### 최종 정량 비교 (val 83샘플 전체)

| 모델 | MAE | 상대오차(peak대비) | netG | 결론 |
|---|---:|---:|---:|---|
| 원본 FiLM 256 | — | — | 54.5M | 기준(이미지) |
| **A** 이미지 128/ngf32 | val L1 0.079 | — | 10.5M | 이미지 파이프라인 경량화, 품질 유지 |
| v1 스칼라 | 8.69mm | 5.4% | 10.5M | |
| B1 (ref1mm+wet8) | 8.73mm | 5.6% | 10.5M | |
| B2 (+GAN) | — | — | +0.7M | ❌ 악화, 아티팩트 |
| B3 (+grad) | 8.57mm | 5.3% | 10.5M | 미세하게 최선 |
| B4 (+흐름누적 채널) | 9.12mm | 6.2% | 10.5M | 채널 구조 살짝, 수치는 오히려 하락 |

**결론: 아키텍처/손실/입력 튜닝은 여기서 천장.** v1~B4 전부 상대오차 5~6% 대에서
노이즈 수준 차이. 모델이 "어디가 젖나"는 잡지만(5%대) 정답의 날카로운 배수 채널은 못 냄.
**병목은 데이터 양** (지형 488종). 진행 중인 OpenFOAM 으로 지형 수 2~3배 늘린 뒤
B1/B3 재학습이 맞는 다음 스텝. 웹 데모용으로는 현재 정확도(peak 대비 5%)로 충분.

### B4 — 흐름누적 입력 채널 (완료, 위 표 참조)

- `src/utils/flow_accum.py`: priority-flood 함몰지 메움 + D8 유향/흐름누적 (~30ms / 96²)
- `build_scalar_dataset.py --add-flowacc`: 571개 npz 에 flowacc 채널 추가 (D8 은 D4 equivariant 라
  증강 시 terrain 과 같이 회전/반전)
- 입력 `[정규화 고도, log(흐름누적)/8.27]` 2채널, 나머지는 B1 과 동일
- config: `configs/config_scalar_b4.yaml`, 실행: `python -m src.train_scalar --config configs/config_scalar_b4.yaml`
- 사전 확인: D8 채널이 OpenFOAM 수심의 수지상 구조와 위치가 잘 맞음 (특히 seoul/gangwon).
  픽셀 상관은 낮지만(큰 평지 ponding 때문) 채널 지오메트리는 일치.

### 그 다음

- **데이터 확보**: 지형 수 늘리기 (진행 중인 OpenFOAM). gangwon 31종은 특히 부족.
- flowacc 를 MFD/D-infinity 로 부드럽게 (D8 은 줄무늬 아티팩트 있음)
- 고해상/고용량은 데이터 제한 때문에 위험.

## 진행 순서

1. **A 먼저** 돌려서 baseline(0.081) 대비 val L1 + 육안 품질 비교 → "128/ngf32 로 낮춰도 되나?" 확인
2. 유효하면 **B** 로 전환 (여기서 "가벼우면서 성능 좋게" 가 실제로 나옴)
3. 데이터가 늘면 A는 `data/` 재생성(`prepare_openfoam_dataset.py`), B는 `build_scalar_dataset.py` 재실행 후 재학습

## OpenFOAM 데이터 확보 우선순위

- 진행 중인 강우변형 배치는 그대로 완료 (샷당 10~20분, 중단 손해)
- 이후엔 **같은 지형에 강우값 추가(rv3, rv4…)보다 새 지형 확보가 우선** — 일반화엔 지형 다양성이 더 중요
  - gangwon 31종 + 강우변형 0 → 여기가 가장 빈약
  - 지형당 강우 2~3점이면 FiLM 조건 학습엔 충분
- 모델이 96~128만 쓰므로 OpenFOAM 메시를 더 조밀하게 할 필요 없음 (현재 96 격자로 충분)
