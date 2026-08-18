#!/usr/bin/env bash
# Linux 워크스테이션용 가상환경 설정 스크립트
# ai_model 디렉터리에서 실행: bash scripts/setup_env.sh
set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# GPU(CUDA) 워크스테이션이면 아래처럼 CUDA 빌드를 먼저 설치하는 것을 권장합니다.
# (CUDA 버전은 워크스테이션 GPU 드라이버에 맞춰 https://pytorch.org/get-started/locally/ 에서 확인)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

echo "설치 완료. 다음부터는 'source .venv/bin/activate'로 가상환경만 활성화하면 됩니다."
