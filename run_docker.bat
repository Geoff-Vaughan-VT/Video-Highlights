@echo off
REM CPU stack: API + Studio UI + queue worker.
setlocal
cd /d "%~dp0"
if not exist .env (copy .env.example .env >nul & echo created .env from .env.example - edit VH_MEDIA_DIR for example D:/Videos)
if not exist media mkdir media
docker compose up --build -d api worker
if errorlevel 1 exit /b 1
echo Studio: http://localhost:8000   API docs: http://localhost:8000/docs
echo Change the host port with VH_API_PORT in .env
