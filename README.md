# 유체이탈 — 등고선에서 배수 취약 지점을 예측하는 AI

지형 등고선과 강우 조건을 넣으면, OpenFOAM 유체 시뮬레이션을 돌리지 않고 **예측 침수 수심맵**을
즉시 내놓는 모델과 웹앱입니다. 시뮬레이션 한 건에 10~20분 걸리던 것을 수백 밀리초로 줄이는 것이 목표입니다.

```
등고선 이미지 + 강우량(mm) + 경과시간(s)  →  U-Net  →  침수 수심맵(m) + 배수구 후보 지점
```

---

## 지금 상태

**배포 모델**: `checkpoints_p2_wet8/best.pt` — U-Net 생성자 10.5M, 128×128,
강우량·경과시간을 FiLM으로 병목에 주입(`cond_dim=2`).

**성능** (학습에 쓰지 않은 지형 87종 / 911장, 정답은 OpenFOAM 원본 `depth.npy`):

| 지표 | 값 | 읽는 법 |
|---|---:|---|
| 평균 수심 오차 (MAE) | 0.0115 m | 정답 평균 수심 0.028 m 대비 |
| 상관계수 r | 0.682 | 공간 패턴 일치도 |
| 젖은영역 IoU @1cm | 0.688 | "발이 젖는 구역"을 맞춘 정도 |
| 젖은영역 IoU @5cm | 0.426 | "차량 통행 지장" 구역 |
| 최대수심 비율 | 0.619 | 1.0이 정답과 같은 깊이 (파이프라인 상한 0.93) |
| 강우 단조성 | 98.8% | 비가 많을 때 더 깊게 예측한 비율 (정답 상한 99.3%) |

전부 `scripts/evaluate.py` 로 재현됩니다. **이 수치를 쓰세요** —
`deploy_package/README.md` 에 적혀 있던 예전 값(MAE 0.0049 / r 0.919)은 재현되지 않습니다.

### 잘하는 것과 못하는 것

- **잘하는 것**: 물이 어디에 모이는지 (IoU@1cm 0.69), 비가 많아지면 더 잠긴다는 관계 (98.8%)
- **못하는 것**: 정답의 **가늘고 날카로운 배수 채널**. 최대 수심을 정답의 62%로만 냅니다.
  최심점 위치 오차가 300 m(1 km 격자 기준)라 "가장 깊은 한 점"을 찍는 용도로는 아직 약합니다
- **지역 편차**: 부산 r=0.77 / 서울 r=0.62 / **강원 r=0.30**. 강원은 학습에 쓴 지형에서도
  r=0.29라 데이터가 아니라 모델·타깃 표현의 한계 쪽입니다 (강원 지형은 수심이 매우 얕음)

한계와 개선 경과는 [`terrainFlowSim/ai_model/IMPROVEMENT_GUIDE.md`](terrainFlowSim/ai_model/IMPROVEMENT_GUIDE.md) 에 단계별로 정리돼 있습니다.

---

## 빠르게 돌려보기

### 웹앱

```powershell
& ".\terrainFlowSim\webapp\run_windows.bat"
```

`http://127.0.0.1:8000` 접속. 두 개의 탭이 있습니다.

- **예측** — 등고선 이미지나 고도 격자를 올리고 강우량·경과시간을 정해 예측.
  `시점 애니메이션`을 누르면 0~120초를 7프레임으로 훑어 물이 차올랐다 빠지는 과정을 재생합니다
- **검증** — 예측을 OpenFOAM 정답과 나란히 놓고 수치까지 냅니다.
  `학습 안 쓴 지형(val)` / `학습에 쓴 지형(train)` 토글로 데이터 누출 영향을 직접 대조할 수 있습니다

> **경과 시간 주의**: 학습 데이터는 비가 초반에 내리고 이후 빠지는 시뮬레이션이라
> **시간이 짧을수록 물이 많습니다**. 정답 평균 수심이 t=18초 0.039 m → t=120초 0.019 m 로
> 줄어듭니다. "이 지형이 얼마나 잠기나"를 묻는 조회의 기본값은 18초입니다.

테스트용 지형 파일은 이 명령으로 만듭니다.

```powershell
& ".\terrainFlowSim\ai_model\.venv\Scripts\python.exe" ".\terrainFlowSim\webapp\make_test_input.py"
```

### 파이썬에서 직접

```python
from src.inference_api import predict_depth_from_terrain

depth_m = predict_depth_from_terrain(terrain, dx=10.4167, rain_mm=60.0, time_s=18.0)
print(f"최대 침수 {depth_m.max():.3f} m")
```

학습 범위는 강우 20~80 mm, 경과시간 0~120초입니다. 범위 밖 값은 경계값으로 잘립니다.
반환 배열은 **입력 지형 대비 위아래가 뒤집혀 있습니다** (아래 "알려진 문제" 참고).

---

## 저장소 구조

```
├── {seoul,busan,gangwon}OpenFOAMDataset/   지형 추출 + OpenFOAM 배치 실행 스크립트
│   └── dataset/                            시뮬레이션 결과 (용량이 커서 git 제외)
├── {seoul,busan,gangwon}Terrain/           국가기본도 등고선 → 고도 격자 추출
└── terrainFlowSim/
    ├── ai_model/                           모델 학습·평가
    │   ├── src/                            데이터셋·모델·학습·추론
    │   ├── scripts/                        데이터 준비 + 평가 도구
    │   ├── configs/                        학습 설정 (YAML)
    │   ├── deploy_package/                 백엔드 전달용 추론 전용 패키지
    │   └── IMPROVEMENT_GUIDE.md            검증·개선 경과 (읽을 가치 있음)
    └── webapp/                             FastAPI 백엔드 + 단일 페이지 프런트
```

학습 데이터·체크포인트·로그는 `.gitignore` 로 제외됩니다 (수십 GB).

---

## 데이터

실제 지형(국가기본도 등고선)에 OpenFOAM `interFoam`(VOF) 강우 시뮬레이션을 돌린 결과입니다.

| | |
|---|---|
| 고유 지형 | **586종** (부산 284 / 서울 223 / 강원 79) |
| 시뮬레이션 | 1,086건 (지형당 평균 강우조건 1.7개) |
| 학습 이미지 쌍 | **48,528** (= 1,011 시뮬 × 6 시점 × D4 8증강) |
| 격자 | 96×96, dx 10.4167 m (1 km × 1 km) |
| 강우 범위 | 20 ~ 80 mm |

train/val 분리는 **지형 단위**입니다 (`_group_key`). 같은 지형의 회전본·강우변형·다른 시점이
train과 val에 나뉘어 들어가지 않습니다.

### 데이터를 더 모아야 할까?

학습 지형 수만 75/150/300/499로 바꿔 측정한 결과입니다 (val 셋 고정).

| 학습 지형 수 | r | IoU@5cm |
|---|---:|---:|
| 75 | 0.604 | 0.342 |
| 150 | 0.635 | 0.373 |
| 300 | 0.660 | 0.401 |
| 499 (전체) | 0.682 | 0.426 |

곡선은 아직 평평해지지 않았지만 **로그선형**입니다 — 지형을 2배로 늘릴 때마다 r이 +0.028
(적합도 R²=0.999). OpenFOAM 샷당 15분으로 환산하면 r을 0.75까지 올리는 데 지형 2,167종 추가,
연속 계산 23일이 필요합니다. **데이터는 듣지만 비쌉니다.**

---

## 주요 도구

| 명령 | 하는 일 |
|---|---|
| `python -m src.train --config configs/config_p2_wet8.yaml` | 학습 |
| `python scripts/evaluate.py --checkpoint ... --config ...` | val/train 전체 평가 (표·CSV·최악사례 그림) |
| `python scripts/compare_evals.py v5 p2_base p2_wet8` | 여러 모델 한 표로 비교 |
| `python scripts/diagnose_pipeline.py` | 표현 방식 자체의 정보손실 상한 측정 |
| `python scripts/scaling_study.py` | 데이터 스케일링 곡선 |
| `python scripts/prepare_openfoam_dataset.py` | OpenFOAM 결과 → 학습용 이미지 쌍 |

평가는 **배포되는 경로 그대로**(역LUT 양자화 포함) 재고, 정답은 학습 타깃 PNG가 아니라
**OpenFOAM 원본 `depth.npy`(m)** 를 씁니다. 타깃 PNG와 비교하면 렌더링 손실이 정답 쪽에도
똑같이 들어가서 오차가 실제보다 작게 나옵니다.

---

## 알려진 문제

- **`predict_depth_*` 반환 배열이 입력 지형 대비 상하반전입니다.** matplotlib의
  `contourf`(Y축이 위로 증가)와 `imshow(origin="lower")`를 거치면서 생깁니다. 학습에는 입력과
  타깃이 같이 뒤집혀 지장이 없지만, API를 직접 쓰면 틀린 방향의 지도를 받습니다.
  현재 `webapp/backend/app.py` 와 `scripts/visualize_inference.py` 가 각자 `np.flipud` 로
  보정하고 있습니다. API 자체를 고치려면 이 두 곳을 같이 손봐야 합니다
- **최대 수심 과소예측** (정답의 62%). 균등 L1이 얇은 배수채널을 뭉개는 쪽을 선호하기 때문이며,
  젖은 픽셀 가중 L1으로 0.49 → 0.62까지 회수했지만 `wet_weight=8`에서 포화합니다
- **학습 데이터와 다른 스타일의 등고선 이미지를 넣으면 정확도가 급락합니다.**
  학습 렌더링은 matplotlib `contourf(cmap="terrain", levels=20)` + 검은 등고선입니다.
  고도 격자를 직접 넣는 경로(`predict_*_from_terrain`)를 쓰면 이 문제가 없습니다
- **체크포인트에 옵티마이저·판별자가 같이 들어 있습니다** (wet8 125 MB). 추론용 `netG`만
  추출하면 ~40 MB로 줄어듭니다

---

## 관련 저장소

| | |
|---|---|
| AI 모델 (이 저장소) | https://github.com/beak00hyunmin7/team_yucheital_ai_model |
| 백엔드 | https://github.com/beak00hyunmin7/team_yucheital_backend |
| 프론트엔드 | https://github.com/beak00hyunmin7/team_yucheital_frontend |

백엔드의 `app/ai/unet_infer/` 는 이 저장소 `deploy_package/src` 의 사본입니다.
모델을 갱신하면 그쪽도 같이 동기화해야 합니다.
