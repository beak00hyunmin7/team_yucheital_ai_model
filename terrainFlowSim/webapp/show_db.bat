@echo off
chcp 65001 >nul
cd /d "%~dp0"
..\ai_model\.venv\Scripts\python.exe backend\show_db.py
start "" "%~dp0data\db_export"
pause
