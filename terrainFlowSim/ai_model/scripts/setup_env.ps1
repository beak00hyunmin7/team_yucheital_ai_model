# Windows 워크스테이션용 가상환경 설정 스크립트
# ai_model 디렉터리에서 실행: .\scripts\setup_env.ps1
#
# 사전 준비: Python 3.11 설치 필요 (winget install -e --id Python.Python.3.11 --source winget)
# 설치 후에는 새 PowerShell 창을 열어야 PATH가 반영됩니다.

python -m venv .venv

# RTX 50xx(Blackwell, sm_120) 계열 GPU는 CUDA 12.8 빌드(cu128) 이상이 필요합니다.
# 다른 GPU(예: RTX 30xx/40xx)를 쓴다면 cu121/cu124 등으로 바꿔도 됩니다.
# (워크스테이션에 맞는 인덱스는 https://pytorch.org/get-started/locally/ 에서 확인)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "설치 완료. 확인: .\.venv\Scripts\python.exe -c `"import torch; print(torch.cuda.is_available())`""
Write-Host "다음부터는 '.\.venv\Scripts\Activate.ps1'로 가상환경만 활성화하면 됩니다."
