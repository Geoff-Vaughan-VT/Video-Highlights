#!/usr/bin/env bash
# Stop every service (all profiles). Data volumes are kept; add -v to wipe them.
set -euo pipefail
cd "$(dirname "$0")"
docker compose --profile gpu --profile cloud down "$@"
