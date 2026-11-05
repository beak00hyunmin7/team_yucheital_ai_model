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
REM 서버가 도는 동안 VM 을 계속 깨워 둘 유지용 루프를 띄운다. 이게 없으면 몇 분 뒤
REM 회원가입/로그인이 조용히 죽는다.
echo [2/4] WSL 유지용 프로세스 시작...
rem   WSL 안쪽에 sleep 프로세스를 띄워두는 것만으로는 부족하다. VM 이 살아 있어도
rem   Windows -> WSL 의 127.0.0.1:3306 포워딩이 따로 끊기기 때문에, Windows 쪽에서
rem   주기적으로 wsl 명령을 날려 VM 유지와 포워딩 갱신을 동시에 한다.
start "yuche-wsl-keepalive" /min cmd /c "for /l %%k in (1,0,2) do (wsl -d Ubuntu-22.04 -u root -- true & timeout /t 45 /nobreak >nul)"

echo [3/4] 웹 서버 의존성 확인...
"%VENV_PY%" -m pip install -q -r backend\requirements.txt

REM 같은 공유기에 물린 다른 PC/폰에서도 들어올 수 있게 모든 인터페이스에 바인딩한다.
REM 이 PC 에서만 쓰려면 아래 BIND 를 127.0.0.1 로 바꾸면 된다.
REM 방화벽은 최초 1회만 열어주면 된다 (관리자 PowerShell):
REM   New-NetFirewallRule -DisplayName "yuche webapp 8000" -Direction Inbound ^
REM     -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
set BIND=0.0.0.0

for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object { $_.PrefixOrigin -eq 'Dhcp' -and $_.IPAddress -notlike '169.254.*' } ^| Select-Object -First 1).IPAddress"`) do set LANIP=%%i

echo [4/4] 서버 시작
echo       이 PC        : http://127.0.0.1:8000
if defined LANIP echo       같은 공유기  : http://%LANIP%:8000
if not defined LANIP echo       (랜 주소를 못 찾았습니다 - ipconfig 로 확인하세요)
echo.
if "%BIND%"=="0.0.0.0" echo       * 같은 네트워크에 있는 사람은 누구나 들어올 수 있습니다.
echo.
"%VENV_PY%" -m uvicorn app:app --app-dir backend --host %BIND% --port 8000

REM 서버가 멈추면 유지용 루프도 닫는다 (안 닫으면 WSL VM 이 계속 메모리를 잡고 있는다).
echo 종료 - WSL 유지용 프로세스 정리...
taskkill /fi "WINDOWTITLE eq yuche-wsl-keepalive" /f >nul 2>&1

endlocal
