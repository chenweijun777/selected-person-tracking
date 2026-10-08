@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tools\run_tracking_gui.py
) else (
  python tools\run_tracking_gui.py
)
if errorlevel 1 pause
