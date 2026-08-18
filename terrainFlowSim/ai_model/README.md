# 등고선 -> 배수 시뮬레이션 이미지 변환 모델 (contour2flow)

등고선 이미지를 입력받아 배수(침수/유출) 시뮬레이션 결과 이미지(첨부하신 파란 정사각형 스타일)를
출력하는 이미지-투-이미지 모델입니다. 지금은 U-Net 기반 오토인코더로 시작하고, 필요하면
`mode: gan`으로 바꿔 pix2pix(U-Net + PatchGAN)로 확장할 수 있게 만들었습니다.

`terrainFlowSim/dataset`(고도/수심 `.npy`)와는 **연결되어 있지 않습니다.** 이 모듈은
순수하게 이미지 쌍(`data/contours/*.png` <-> `data/flow/*.png`)만으로 학습합니다.
데이터셋은 나중에 준비하실 예정이라 `data/` 폴더는 비어 있습니다.

## 폴더 구조

```
ai_model/
├── configs/
│   └── config.yaml          # 학습 설정 (경로, 하이퍼파라미터, mode: unet|gan)
├── data/
│   ├── contours/             # 입력 등고선 이미지 (나중에 채워 넣기)
│   ├── flow/                 # 타깃 배수 이미지 (같은 파일명으로 대응)
│   └── README.md
├── src/
│   ├── datasets/
│   │   └── contour_flow_dataset.py  # 이미지 쌍 로딩/전처리/augmentation
│   ├── models/
│   │   ├── unet_generator.py        # U-Net(오토인코더) 생성자
│   │   ├── discriminator.py         # PatchGAN 판별자 (mode: gan 전용)
│   │   └── translation_model.py     # 학습 루프용 모델 래퍼 (unet/gan 공용)
│   ├── utils/
│   │   ├── image_utils.py           # 텐서<->PNG 변환, 미리보기 저장
│   │   └── video_utils.py           # 프레임 시퀀스 -> mp4 (추후 영상 출력용)
│   ├── train.py
│   └── inference.py
├── scripts/
│   ├── setup_env.ps1         # Windows 가상환경 설치 스크립트
│   └── setup_env.sh          # Linux 가상환경 설치 스크립트
├── checkpoints/               # 학습 중 저장되는 모델 가중치
├── outputs/                   # 검증 샘플 미리보기, 추론 결과
└── requirements.txt
```

## 설치 (처음 한 번만)

### 1. Python 설치

Windows에 `python` 명령이 마이크로소프트 스토어 더미로 연결되어 있는 경우가 많습니다.
아래처럼 winget으로 실제 Python 3.11을 설치하세요.

```powershell
winget install -e --id Python.Python.3.11 --source winget
```

설치 후 **새 PowerShell 창**을 열어야 PATH가 반영됩니다. 확인:

```powershell
python --version   # Python 3.11.x 가 나와야 정상 (스토어 더미면 버전 없이 "Python"만 출력됨)
```

### 2. 가상환경 생성 + 라이브러리 설치

`ai_model` 디렉터리에서:

```powershell
cd ai_model
.\scripts\setup_env.ps1        # venv 생성 + torch(CUDA) + requirements.txt 설치
```

(Linux 워크스테이션이면 `bash scripts/setup_env.sh` — 이 경우 스크립트 안의 CUDA `pip install torch ...`
줄 주석을 풀고 워크스테이션 CUDA 버전에 맞는 index-url로 바꿔서 먼저 설치하세요.)

`setup_env.ps1`은 기본값으로 CUDA 12.8 빌드(cu128)를 설치합니다. RTX 50xx(Blackwell) GPU는
이 버전 이상이 필요합니다. RTX 30xx/40xx 등 다른 세대 GPU를 쓰거나 GPU가 없다면
`scripts/setup_env.ps1` 안의 `--index-url` 값을 워크스테이션에 맞게 바꾸세요
(https://pytorch.org/get-started/locally/ 에서 확인, GPU 없으면 이 줄 자체를 지우고
`requirements.txt`의 torch가 CPU 빌드로 설치되게 두면 됩니다).

### 3. 설치 확인

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True`와 GPU 이름이 출력되면 정상입니다. 이후에는 매번 아래처럼 가상환경만 활성화하고 작업하면 됩니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 데이터셋 준비 (나중에)

`data/README.md` 참고. 요약하면 `data/contours/`와 `data/flow/`에 같은 파일명으로
입력-정답 이미지 쌍을 넣으면 됩니다. `src/datasets/contour_flow_dataset.py`가 파일명이
일치하는 쌍만 자동으로 찾아 로딩합니다.

## 학습

```bash
python -m src.train --config configs/config.yaml
```

- `configs/config.yaml`의 `mode`를 `unet`(순수 오토인코더, L1 손실만) 또는
  `gan`(pix2pix: U-Net 생성자 + PatchGAN 판별자, adversarial + L1 손실)으로 선택합니다.
- 학습 중 `sample_epoch_freq` 에폭마다 `outputs/sample_epoch_XXXX.png`에
  [입력 등고선 | 예측 배수 이미지 | 정답 배수 이미지] 가 나란히 저장되어 눈으로 진행 상황을
  확인할 수 있습니다.
- `checkpoint_dir`(기본 `checkpoints/`)에 `save_epoch_freq` 에폭마다 가중치가 저장됩니다.

## 추론

```bash
python -m src.inference --checkpoint checkpoints/epoch_0200.pt \
    --input path/to/contour.png --output outputs/predicted.png --mode unet
```

새 등고선 이미지 한 장을 넣으면 예측된 배수 이미지를 저장합니다.

## 앞으로 확장할 부분

- **영상 출력**: 지금은 정지 이미지 1장만 예측합니다. 시간에 따른 배수 흐름 영상을 만들려면
  (1) 시뮬레이션에서 시간별 프레임 시퀀스를 얻고, (2) 그 프레임들을 예측하는 시계열 모델
  (ConvLSTM, 시간축을 추가한 U-Net 등)을 추가로 학습한 뒤, (3) `src/utils/video_utils.py`의
  `frames_to_video()`로 그 프레임들을 mp4로 인코딩하면 됩니다. 지금 구조를 그대로 재사용할 수
  있도록 `translation_model.py`/`video_utils.py`를 분리해 두었습니다.
- **mode: gan 전환**: 데이터가 어느 정도 쌓이고 U-Net만으로 결과가 흐릿하다면(L1 loss의
  전형적인 특징) `config.yaml`의 `mode: gan`으로 바꿔서 더 선명한 결과를 노려볼 수 있습니다.
