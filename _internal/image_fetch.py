"""
Low-level image download helpers (HTTP + PIL) with retry, backoff, and hard timeout.
"""

from __future__ import annotations

import io
import random
import threading
import time
from typing import Optional, Tuple

import requests
from PIL import Image

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_FETCH_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/jpeg,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.flickr.com/",
}

# Pillow decompression-bomb guard — CLIP resizes anyway, so a generous cap is fine.
_PIL_MAX_PIXELS = 300_000_000
_pil_limit_applied = False


def fetch_pil_rgb(
    url: str,
    *,
    timeout: float = 60.0,
    max_retries: int = 5,
    base_backoff: float = 2.0,
) -> Tuple[Optional[object], Optional[str]]:
    """
    Download *url* and return (PIL.Image RGB, None) on success or
    (None, error_string) on permanent failure.
    Retries on HTTP 429 / 503 with exponential back-off + jitter.
    """
    global _pil_limit_applied
    if not _pil_limit_applied:
        Image.MAX_IMAGE_PIXELS = _PIL_MAX_PIXELS
        _pil_limit_applied = True

    last_err = "unknown error"
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=_FETCH_HEADERS)
            if r.status_code in (429, 503):
                retry_after = float(r.headers.get("Retry-After", 0) or 0)
                backoff = max(retry_after, base_backoff * (2 ** attempt)) + random.uniform(0, 1)
                last_err = f"HTTP {r.status_code}"
                if attempt < max_retries:
                    time.sleep(backoff)
                continue
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            return img.convert("RGB"), None
        except requests.HTTPError as exc:
            last_err = f"HTTP {exc.response.status_code if exc.response is not None else '?'}"
            break
        except requests.Timeout:
            last_err = "timeout"
            if attempt < max_retries:
                time.sleep(base_backoff * (2 ** attempt))
            continue
        except Exception as exc:
            last_err = str(exc)[:120]
            break

    return None, last_err


def fetch_with_deadline(
    url: str,
    *,
    timeout: float,
    max_retries: int,
    base_backoff: float,
    hard_timeout: float,
) -> Tuple[Optional[object], Optional[str]]:
    """
    Run fetch_pil_rgb in a daemon thread with a hard wall-clock deadline.
    Guards against servers that trickle data or PIL hanging on corrupt images.
    """
    _result: list = [None, f"hard timeout after {hard_timeout:.0f}s (network/PIL stall)"]

    def _target() -> None:
        _result[0], _result[1] = fetch_pil_rgb(
            url, timeout=timeout, max_retries=max_retries, base_backoff=base_backoff
        )

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=hard_timeout)
    return _result[0], _result[1]
