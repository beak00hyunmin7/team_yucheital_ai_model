# 데이터가 늘었을 때 모델 갱신 절차 (C안)

현재 결정: **모델은 그대로 두고 웹은 현행(`checkpoints_film/best.pt`) 유지.**
OpenFOAM 데이터가 충분히 쌓이면 아래대로 재학습해서 비교한다.

## 1. OpenFOAM 데이터 확보 우선순위

현황 (2026-08-29): 고유 지형 **491종** (seoul 223 / busan 237 / gangwon 31).

| 우선순위 | 할 일 | 이유 |
|---|---|---|
| 1 | **gangwon staged 미실행 ~49개 돌리기** (`gangwonOpenFOAMDataset` 에서 `run_batch.py`) | 31종뿐, 산간 실지형이라 가장 부족·중요 |
| 2 | **busan staged 미실행 ~47개** | 이미 staging 준비됨, 바로 실행 가능 |
| 3 | 새 지형 staging (`prepare_samples.py --n ...`) 후 실행 | 600종+ 목표 |
| 4 | 강우변형(rv3+) 추가 | **가장 후순위** — 지형당 2~3점이면 FiLM 학습 충분 |

WSL 실행법은 각 `*OpenFOAMDataset/README.md` 참고.

## 2. 스칼라 데이터셋 재생성

```powershell
cd project\terrainFlowSim\ai_model
.venv\Scripts\python.exe scripts\build_scalar_dataset.py                       # 전체 재수집
.venv\Scripts\python.exe scripts\build_scalar_dataset.py --add-flowacc          # 흐름누적 채널
.venv\Scripts\python.exe scripts\build_scalar_dataset.py --stats-only --depth-ref-mm 1.0
```

## 3. 재학습 (B3 = 지금까지 최선)

```powershell
.venv\Scripts\python.exe -m src.train_scalar --config configs\config_scalar_b3.yaml
```

또는 이미지 경량 모델(A):
```powershell
.venv\Scripts\python.exe -m src.train --config configs\config_film_light.yaml
```

## 4. 비교

`EXPERIMENTS.md` 의 비교 스크립트 패턴으로 val MAE / 상대오차 측정.
현재 기준선: **상대오차 ~5.4% (peak 대비), MAE ~8.7mm**.
유의미하게(예: 상대오차 <4%) 좋아지면 웹에 연결.

## 5. 웹 연결 (좋아졌을 때만)

이미지 모델이면 환경변수만:
```powershell
$env:AI_MODEL_CHECKPOINT = "...\checkpoints_film_light\best.pt"
$env:AI_MODEL_IMAGE_SIZE = "128"
```

스칼라 모델(B3 등)이면 `webapp/backend/app.py` 에 스칼라 경로 추가 필요
(`src.inference_scalar.predict_depth` 사용, "등고선 데이터" 모드에서 분기).
