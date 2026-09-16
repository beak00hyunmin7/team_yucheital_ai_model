@echo off
REM 예측 배수맵 웹 실행 (Windows)
REM   - ai_model\.venv 재사용
REM   - 회원가입/로그인용 MySQL(MariaDB)은 WSL Ubuntu 안에서 구동

setlocal
cd /d "%~dp0"

set VENV_PY=..\ai_model\.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
  echo [오류] ..\ai_model\.venv 를 찾을 수 없습니다.
  pause
  exit /b 1
)

echo [1/3] WSL MariaDB 시작...
wsl -d Ubuntu-22.04 -u root -- service mariadb start 2>nul
if errorlevel 1 (
  echo   (WSL/MariaDB 없음 - 인증 없이 예측만 동작합니다)
) else (
  REM MariaDB 는 기동에 몇 초 걸린다. 바로 서버를 띄우면 init_db 가 실패해서
  REM 인증이 꺼진 채 뜨거나, 켜진 채 로그인이 안 되는 어정쩡한 상태가 된다.
  REM 응답할 때까지 최대 20초 기다린다.
  echo   MariaDB 준비 대기 중...
  for /L %%i in (1,1,20) do (
    wsl -d Ubuntu-22.04 -u root -- mysqladmin --silent ping >nul 2>&1 && goto :db_ready
    timeout /t 1 /nobreak >nul
  )
  echo   (경고: MariaDB 가 20초 안에 응답하지 않았습니다 - 로그인이 안 될 수 있습니다)
  :db_ready
)

echo [2/3] 웹 서버 의존성 확인...
"%VENV_PY%" -m pip install -q -r backend\requirements.txt

echo [3/3] 서버 시작: http://127.0.0.1:8000
"%VENV_PY%" -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000

endlocal
