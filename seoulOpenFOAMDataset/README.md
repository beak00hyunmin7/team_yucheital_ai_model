# 서울시 실제 지형 기반 OpenFOAM 배수 데이터셋

서울시 등고선(`seoulTerrain/pointcloud.npy`)에서 1000m x 1000m 패치를 96x96 격자
(dx ≈ 10.4m)로 잘라내고, 각 패치에 대해 실제 OpenFOAM(`interFoam`, VOF 기반
자유표면 solver)로 강우->배수 시뮬레이션을 돌려서 만든 학습 데이터셋이다.

## 폴더 구조

```
seoulOpenFOAMDataset/
├── caseTemplate/          # OpenFOAM 케이스 템플릿 (blockMeshDict, fvSchemes, ...)
├── staging/                # 준비된 패치(terrain_raw.npy + params.json), 총 30개
├── dataset/                # 완료된 결과 (terrain.npy + depth.npy + meta.json), 30/30
├── viz/                     # visualize_dataset.py가 생성하는 시각화 출력 (git 추적 X)
├── estimate_duration.py    # 유역 크기/경사로 시뮬레이션 총 시간을 동적으로 추정
├── prepare_samples.py      # pointcloud.npy에서 패치를 뽑아 staging/에 준비
├── run_batch.py            # staging/ 각 샘플에 대해 실제 OpenFOAM 실행 (WSL 안에서)
└── visualize_dataset.py    # 결과 시각화 (아래 참고)
```

## 데이터셋 상태

- `staging/` 30개 패치 전부 준비됨
- `dataset/` **30/30 완료** (모두 실제 `interFoam` 산출물, 실패 샘플 없음)
- 각 `dataset/sample_XXXX/meta.json`의 `wall_clock_s`, `actual_end_time` 필드가
  실제 OpenFOAM 실행 증거 (파이썬 간이 시뮬레이터로 만든
  `seoulTerrain/dataset`에는 이 필드가 없음)

## 배치 실행 방법 (재실행/추가 샘플이 필요할 때)

OpenFOAM은 Windows에 직접 설치하지 않고 **WSL(Ubuntu) 안에서** 돌린다
(경로에 공백/한글이 있으면 OpenFOAM이 깨지기 때문에, 케이스 파일을 WSL
홈 디렉토리로 복사해서 작업한다).

```powershell
# 1) WSL 안에 OpenFOAM 없으면 설치 (한 번만)
wsl -d Ubuntu-22.04 -u root -- bash -c "wget -O - https://dl.openfoam.com/add-debian-repo.sh | bash && apt-get install -y openfoam2212-default python3-numpy"

# 2) 케이스 파일을 WSL 홈으로 복사 (공백/한글 경로 회피)
wsl -d Ubuntu-22.04 -u root -- bash -c "mkdir -p ~/openfoam_work && cp -r '/mnt/c/Users/<사용자>/.../seoulOpenFOAMDataset' ~/openfoam_work/"

# 3) 배치 실행 (start_idx부터 끝까지)
wsl -d Ubuntu-22.04 -u root -- bash -c "source /usr/lib/openfoam/openfoam2212/etc/bashrc && cd ~/openfoam_work/seoulOpenFOAMDataset && python3 run_batch.py 0"

# 4) 완료 후 결과를 Windows 쪽 dataset/로 복사
wsl -d Ubuntu-22.04 -u root -- bash -c "cp -r ~/openfoam_work/seoulOpenFOAMDataset/dataset/. '/mnt/c/Users/<사용자>/.../seoulOpenFOAMDataset/dataset/'"
```

**주의**: `run_batch.py`의 `mpirun` 호출에는 `--allow-run-as-root`가 필요하다
(WSL 기본 사용자가 root라서, 이 플래그 없으면 OpenMPI가 즉시 실행을 거부하고
`interFoam`이 1~2초 만에 실패한다 — 실제로 처음 배치에서 이 문제로 14개 샘플이
전부 실패했었음. 지금 스크립트에는 이미 반영되어 있음).

장시간(샘플당 10~15분, 30개 기준 몇 시간) 걸리므로 세션이 끊겨도 계속 돌게
`setsid nohup ... & disown`으로 완전히 분리해서 실행하는 걸 권장:

```bash
cd ~/openfoam_work/seoulOpenFOAMDataset
setsid nohup python3 run_batch.py 0 > batch_run.log 2>&1 < /dev/null & disown
```

## 시각화 방법

`visualize_dataset.py`가 지형(terrain) 등고선을 침수 깊이(depth) 맵 위에
오버레이해서 그려준다. `terrainFlowSim/ai_model/.venv`(numpy, matplotlib 이미
설치되어 있음)의 파이썬으로 실행하면 된다.

```powershell
# ai_model 가상환경의 python 사용 (matplotlib, numpy 포함)
$py = "..\terrainFlowSim\ai_model\.venv\Scripts\python.exe"

# 특정 샘플 1개
& $py visualize_dataset.py --sample sample_0021

# max_depth 상위 N개를 개별 이미지로
& $py visualize_dataset.py --top 6

# 전체 30개를 격자 1장으로 요약
& $py visualize_dataset.py --all
```

출력은 `viz/` 폴더에 저장된다:
- `--sample` / `--top`: `viz/<sample>_contour_overlay.png` (지형 등고선 + 침수 깊이, 컬러바 포함)
- `--all`: `viz/all_contour_overlay_grid.png` (6열 격자, 샘플당 등고선 + 침수 깊이 축소판)

`plot_contour_overlay()` 함수를 임포트해서 다른 스크립트(예: 학습 중 검증 샘플
미리보기)에서도 재사용할 수 있다.
