# OpenFOAM 배치 전 확인용 등고선 샘플 (1개)

`seoulOpenFOAMDataset/prepare_samples.py` 와 동일한 규격으로 서울 등고선
(`seoulTerrain/pointcloud.npy`, 국가기본도 1:5000)에서 패치 1장을 뽑은 것.

| 파일 | 용도 |
|---|---|
| `terrain_raw.npy` | **OpenFOAM staging 입력 포맷** (96×96 float32, 고도 m). `staging/sample_XXXX/` 에 `params.json` 과 함께 넣으면 `run_batch.py` 가 그대로 돌린다 |
| `params.json` | 강우/시뮬레이션 조건. `rain_mm` 은 학습범위(20~80) 중앙값 50 으로 고정 |
| `terrain.csv` | 웹앱 "등고선 데이터" 모드 업로드용 (`terrain_io.py` 가 파싱) |
| `contour.png` | 학습 데이터와 같은 렌더 스타일(`contourf(cmap="terrain", levels=20)` + 검은 등고선). 웹앱 "등고선 이미지" 모드 업로드용 |
| `contour_map.png` | 사람이 눈으로 확인하는 축척/고도 레전드 포함 지도 |

제원: 1000 m × 1000 m 도메인, dx = 10.4167 m, 고도 27.5~110.0 m (기복 82.5 m),
max_slope 0.347 (메쉬 왜곡 한계 0.35 아래), 중심 TM (203663, 460067).

## 배치에 넣는 법

```bash
cp -r sample_check <...>/seoulOpenFOAMDataset/staging/sample_9000
# params.json 의 "sample" 값을 폴더명과 맞춘 뒤
python3 run_batch.py 9000
```

## 주의

- 행 index 0 = 남쪽(y 작은 쪽). `contour*.png` 도 같은 방향으로 그려서 `.npy` 와 상하가 일치한다.
  (학습 파이프라인의 `render_contour_rgb` → `predict_depth_*` 경로는 상하반전이 있으니 그쪽은 `np.flipud` 보정 필요.)
- `terrain_raw.npy` 는 급경사 완화(`smooth_steep_regions`)가 이미 적용된 상태 — 원본 보간값 그대로가 아니다.

## OpenFOAM 시험 실행 결과 (2026-11-01, 1건)

`openfoam_result/` — WSL Ubuntu-22.04 / OpenFOAM 2212 `interFoam` 8-way MPI, **874초**(기존 샘플 612초보다 김).
프로젝트 `staging/`(223건)은 건드리지 않고 `~/openfoam_work/sampleCheck/` 에 1건만 격리해서 돌렸다.

| 파일 | 내용 |
|---|---|
| `depth.npy` | 최종 t=120s 수심 [m] — **거의 비어 있음** (max 3.9 mm, 투입량의 0.1%만 잔류) |
| `depth_series.npy` | (60, 96, 96) 2초 간격 전 시점 수심 — 실제로 쓸 건 이쪽 |
| `times.npy` | 2.0 ~ 120.0 s, 60스텝 |
| `result_check.png` | t=12/18/30s 수심을 지형 등고선 위에 겹친 검증 그림 |

**물리 검증 통과**: 투입 50,000 m³(=50 mm) → t=12s 71% 잔류 → t=60s 3.5% → t=120s 0.1%.
질량이 경계로 빠져나가며 단조 감소하고, 물이 모이는 위치가 등고선 골짜기와 일치한다.

**저지대 집중도는 t=24~30s 에서 최대** (저지대/고지대 평균수심 비 9.8~12.4배, corr(고도,수심) −0.25).
t=120s 는 이미 다 빠진 뒤라 상관이 **+0.275로 뒤집힌다** — 남은 물이 골짜기가 아니라 경사면 잔류분이다.
즉 `depth.npy`(최종 스냅샷)만 타깃으로 쓰면 배수 신호가 거의 없다. 학습엔 `depth_series.npy` 의 이른 시점을 써야 한다.
