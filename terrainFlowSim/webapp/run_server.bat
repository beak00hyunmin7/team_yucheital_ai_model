@echo off
REM yuche server launcher (this PC = server).
REM   - DB      : MySQL80 on this PC, database yuche_ai_db (accounts + analysis history)
REM   - files   : webapp\data\history (history images), webapp\.yuche_secret (login key)
REM   - screen  : serves the team frontend (vtkjs.html) at http://<this PC>:8000
setlocal
cd /d "%~dp0"

set "VENV_PY=..\ai_model\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] ..\ai_model\.venv not found.
  pause
  exit /b 1
)

set "YUCHE_DB_URL=mysql+pymysql://root:1234@127.0.0.1:3306/yuche_ai_db?charset=utf8mb4"
REM frontend folder (team_yucheital_frontend\project_yucheital\yuche_frontend). Set YUCHE_SITE_DIR before running to override.
if not defined YUCHE_SITE_DIR set "YUCHE_SITE_DIR=%~dp0..\..\..\sejong_yucheital_project\project_yucheital\yuche_frontend"
if not exist "%YUCHE_SITE_DIR%\vtkjs.html" echo [WARN] frontend folder not found: %YUCHE_SITE_DIR%

echo [1/3] MySQL80 database...
"%VENV_PY%" -c "import pymysql; c=pymysql.connect(host='127.0.0.1',user='root',password='1234'); c.cursor().execute('CREATE DATABASE IF NOT EXISTS yuche_ai_db CHARACTER SET utf8mb4'); c.commit(); print('  DB ready: yuche_ai_db')"
if errorlevel 1 (
  echo [ERROR] Cannot connect to MySQL. Check that the MySQL80 service is running.
  pause
  exit /b 1
)

echo [2/3] packages...
"%VENV_PY%" -m pip install -q -r backend\requirements.txt


echo [3/3] server start
echo.
echo       this PC    : http://127.0.0.1:8000
"%VENV_PY%" backend\lan_ip.py
echo.
echo       (Ctrl+C to stop)
echo.
"%VENV_PY%" -m uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000

endlocal
pause
