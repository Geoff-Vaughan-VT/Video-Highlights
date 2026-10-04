@echo off
REM Native Windows runner (venv + CUDA torch, API + worker). Args pass through, e.g. -Inline
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_native_windows.ps1" %*
