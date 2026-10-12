@echo off
cd /d "%~dp0"
python tools\run_tracking_gui.py
if errorlevel 1 pause
