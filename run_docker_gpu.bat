@echo off
REM NVIDIA stack: API + GPU worker (Docker Desktop with WSL2 GPU support).
setlocal
cd /d "%~dp0"
if not exist .env (copy .env.example .env >nul & echo created .env from .env.example - edit VH_MEDIA_DIR for example D:/Videos)
if not exist media mkdir media
REM The CPU worker must not compete for the same queue.
docker compose stop worker >nul 2>&1
docker compose --profile gpu up --build -d api worker-gpu
if errorlevel 1 exit /b 1
echo Studio: http://localhost:8000   GPU status: http://localhost:8000/v1/health/gpu
docker compose --profile gpu exec -T worker-gpu nvidia-smi -L
