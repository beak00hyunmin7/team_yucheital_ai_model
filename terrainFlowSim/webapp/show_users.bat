@echo off
cd /d "%~dp0"
..\ai_model\.venv\Scripts\python.exe backend\show_users.py
pause
