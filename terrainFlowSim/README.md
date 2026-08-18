# 지형 → 유체 이동맵 학습 데이터 생성 (1차 근사 시뮬레이터)

`drainCase`/`drainPipe`가 배관 **내부** 유동(OpenFOAM interFoam)이었다면, 여기는
AI 서로게이트 모델이 배울 **지형 표면 유출(강우 → 침수/유출 맵)** 데이터를 만드는 부분입니다.

## 왜 OpenFOAM으로 바로 안 했는가

`drainCase`(배관 하나) 실행에도 WSL에서 수 시간이 걸린 기록이 남아있습니다. AI 학습에는
지형 100~500개가 필요한데, 매번 3D CFD(snappyHexMesh + interFoam)를 돌리면 비현실적으로
오래 걸립니다. 그래서:

1. **1차(지금)**: `overland_flow_sim.py`의 셀룰러 오토마타 기반 확산파(diffusive wave)
   근사 모델로 물리적으로 타당한 (지형, 수심맵) 쌍을 빠르게(샘플당 약 3~4초) 대량 생성합니다.
2. **2차(추후, 선택)**: 이 근사 모델로 만든 지형 중 대표 케이스 몇 개만 골라 실제
   OpenFOAM(예: `interFoam` + 지형 STL, 또는 얕은물방정식 solver)으로 돌려서 결과를 비교하고,
   필요하면 근사 모델의 파라미터(`k_rate` 등)를 보정합니다. AI 모델 자체는 1차 데이터로
   먼저 학습/파이프라인 검증을 끝내고, 여유가 되면 고정밀 데이터로 교체/보강하면 됩니다.

## 파일 구성

- `generate_terrain.py` — 가우시안 언덕 1~3개로 이루어진 지형(DEM) 생성
- `overland_flow_sim.py` — 강우 → 지표 유출 시뮬레이터 (질량 보존 보장, 경계에서 자연 배수)
- `generate_dataset.py` — 위 둘을 조합해 (지형, 강우조건, 최종 수심맵) 샘플을 대량 생성
- `dataset/sample_XXXX/`
  - `terrain.npy` — (size, size) 고도 [m] — **AI 모델의 입력**
  - `depth.npy` — (size, size) 최종 수심 [m] — **AI 모델이 예측할 타깃**
  - `meta.json` — 강우강도/지속시간, 질량 수지(투입/잔류/유출량) 기록

## 사용법

```bash
python generate_dataset.py --n 100 --size 96 --out dataset
```

- `--n`: 생성할 샘플 개수
- `--size`: 격자 한 변 크기 (96이면 96x96, 셀 크기 dx=1m 기준 96m x 96m 도메인)
- `--out`: 저장 폴더

샘플 하나를 직접 눈으로 보고 싶으면:

```bash
python overland_flow_sim.py   # terrain_preview.png, flow_preview.png 생성
```

## 다음 단계 (U-Net 학습으로 넘어갈 때)

- `terrain.npy`를 정규화(예: 0~1 스케일)해서 U-Net 입력 채널로 사용
- 강우조건(`rain_rate_mmhr`, `rain_duration_s`)도 지형과 같은 크기의 상수맵으로 만들어
  추가 입력 채널로 쌓으면(멀티채널 입력), "같은 지형이라도 강우 조건이 바뀌면 결과가
  어떻게 달라지는지"까지 모델이 배울 수 있음 (배수구 배치 최적화 단계에서 필요)
- `depth.npy`를 타깃으로 L1 + SSIM 손실로 학습 시작
