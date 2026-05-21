"""
Low-level image download helper — single attempt, fail fast.

Rate-limit resume is handled at the DB level: failed rows stay NULL and are
retried on the next manual run. No sleeping, no retries, no blocking.
"""

from __future__ import annotations

import io
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
    timeout: float = 30.0,
) -> Tuple[Optional[object], Optional[str]]:
    """
    Download *url* and return (PIL.Image RGB, None) on success or
    (None, error_string) on any failure.

    Single attempt, no retries, no sleeping. If Flickr rate-limits (429/503)
    or the download fails for any reason, returns None immediately. The caller
    leaves the row NULL in the DB so it is picked up on the next manual run.
    """
    global _pil_limit_applied
    if not _pil_limit_applied:
        Image.MAX_IMAGE_PIXELS = _PIL_MAX_PIXELS
        _pil_limit_applied = True

    try:
        r = requests.get(url, timeout=timeout, headers=_FETCH_HEADERS)
        if r.status_code in (429, 503):
            return None, f"HTTP {r.status_code} (rate limited — retry next run)"
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        return img.convert("RGB"), None
    except requests.HTTPError as exc:
        return None, f"HTTP {exc.response.status_code if exc.response is not None else '?'}"
    except requests.Timeout:
        return None, "timeout"
    except Exception as exc:
        return None, str(exc)[:120]
