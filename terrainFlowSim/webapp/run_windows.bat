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

echo [1/4] WSL MariaDB 시작...
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

REM WSL2 VM 은 아무도 쓰지 않으면 스스로 종료되고, 그때 VM 안의 MariaDB 도 같이 죽는다.
REM (MariaDB 는 systemd 로 enabled 되어 있어서 VM 이 다시 뜰 때 자동 기동된다 - 그래서
REM  "wsl 명령을 쓴 직후에는 되고 조금 두면 안 되는" 모습으로 보인다.)
REM 실측: wsl 명령 직후 5/5 연결 성공, 1~2분 방치 후 20/20 연결거부(WinError 10061),
REM 그 상태에서 wsl 로 들어가 보면 uptime 이 "up 0 minutes" - VM 이 새로 뜬 것이다.
REM 서버가 도는 동안 VM 을 붙잡아 둘 닻 프로세스를 띄운다. 이게 없으면 몇 분 뒤
REM 회원가입/로그인이 조용히 죽는다.
echo [2/4] WSL 유지용 프로세스 시작...
start "yuche-wsl-keepalive" /min wsl -d Ubuntu-22.04 -u root -- sleep infinity

echo [3/4] 웹 서버 의존성 확인...
"%VENV_PY%" -m pip install -q -r backend\requirements.txt

echo [4/4] 서버 시작: http://127.0.0.1:8000
"%VENV_PY%" -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000

REM 서버가 멈추면 닻도 내린다 (안 내리면 WSL VM 이 계속 메모리를 잡고 있는다).
echo 종료 - WSL 유지용 프로세스 정리...
wsl -d Ubuntu-22.04 -u root -- pkill -f "sleep infinity" >nul 2>&1

endlocal
