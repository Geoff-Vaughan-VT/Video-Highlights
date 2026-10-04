#!/usr/bin/env python3
"""Container HEALTHCHECK: GET http://127.0.0.1:$VH_API_PORT/v1/health -> exit 0/1.

Worker-only containers have no HTTP server; compose disables the check for
them (``healthcheck: disable: true``). ``VH_HEALTHCHECK_URL`` overrides the URL.
"""

import os
import sys
import urllib.request

url = os.getenv("VH_HEALTHCHECK_URL") or f"http://127.0.0.1:{os.getenv('VH_API_PORT', '8000')}/v1/health"
try:
    with urllib.request.urlopen(url, timeout=4) as response:  # noqa: S310 - fixed local URL
        sys.exit(0 if response.status == 200 else 1)
except Exception as exc:  # pragma: no cover - exercised by docker only
    print(f"healthcheck failed: {exc}", file=sys.stderr)
    sys.exit(1)
