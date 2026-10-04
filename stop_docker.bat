@echo off
REM Stop every service (all profiles). Data volumes are kept; add -v to wipe them.
cd /d "%~dp0"
docker compose --profile gpu --profile cloud down %*
