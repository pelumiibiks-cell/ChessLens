"""Tiny shared fetch helper for the ingest scripts - stdlib only, no need
to pull httpx into a build-time-only code path.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

USER_AGENT = "chesslens-corpus-builder (https://github.com/, contact via project owner)"


def fetch_text(url: str, timeout: int = 30, max_retries: int = 4) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else delay)
                delay *= 2
                continue
            raise
    raise RuntimeError(f"unreachable: retries exhausted for {url}")
