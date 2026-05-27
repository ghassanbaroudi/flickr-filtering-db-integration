"""
OpenCLIP zero-shot building vs non-building scoring (batched on GPU when available).
"""

from __future__ import annotations

import io
import random
import threading
import time
from typing import List, Optional, Sequence, Tuple

import requests

# Lazy imports for torch/open_clip in ClipVisionRuntime.__init__

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

# Pillow refuses images above a default pixel count (decompression-bomb guard).  High-res
# Flickr JPEGs often exceed it; CLIP resizes anyway — use a generous but finite cap.
_PIL_MAX_PIXELS = 300_000_000  # ~300 MP
_pil_bomb_limit_applied = False


# Two prompts: index 0 = building-positive, index 1 = non-building / wrong subject
CLIP_TEXT_PROMPTS = (
        "a photograph taken outdoors showing the real exterior facade of a built structure with walls windows or roof visible in the scene",
    # 1 — documents (already there)
    "a close-up photograph of paper pages documents blueprints floor plans maps letters newspapers books or archival sheets with printed text or diagrams",
    # 2 — screen/poster (already there)
    "a photograph of a computer screen tablet projector slide framed poster or museum label displaying an image or drawing of a building",
    # 3 — NEW: people dominating the scene
    "a photograph of people or a crowd where faces or bodies are the main subject and any building in the background is minor or unclear",
    # 4 — NEW: indoor interior
    "a photograph taken indoors showing the interior of a room hallway lobby or space with furniture floors or ceilings visible",
    # 5 — NEW: boats and water
    "a photograph of a boat ship vessel or watercraft on water with no building as the main subject",
    # 6 — NEW: nature / landscape with no building
    "a photograph of nature trees fields mountains water or sky with no man-made structure visible as the main subject",
)


def fetch_pil_rgb(
    url: str,
    timeout: float = 60.0,
    max_retries: int = 5,
    base_backoff: float = 2.0,
) -> "tuple[object, str | None]":
    """
    Return (PIL Image RGB, None) on success, or (None, error_string) on permanent failure.
    Retries on 429 / 503 with exponential back-off + jitter (up to max_retries attempts).
    """
    global _pil_bomb_limit_applied
    from PIL import Image

    if not _pil_bomb_limit_applied:
        Image.MAX_IMAGE_PIXELS = _PIL_MAX_PIXELS
        _pil_bomb_limit_applied = True

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


def _fetch_with_deadline(
    url: str,
    timeout: float,
    max_retries: int,
    base_backoff: float,
    hard_timeout: float,
) -> "tuple[object, str | None]":
    """
    Run fetch_pil_rgb in a daemon thread with a hard wall-clock deadline.

    This guards against cases where the socket timeout does not fire — e.g. a
    server that trickles data slowly or PIL hanging on a malformed image.  If
    the thread has not finished within *hard_timeout* seconds the call returns
    an error immediately; the daemon thread is left to die on its own.
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


class ClipVisionRuntime:
    """Loads OpenCLIP once; precomputes text features; classifies images in batches."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        *,
        device: Optional[str] = None,
        text_prompts: Optional[Sequence[str]] = None,
    ) -> None:
        import torch
        import open_clip

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self._model_name = model_name
        self._tokenizer = open_clip.get_tokenizer(model_name)
        if text_prompts is None:
            prompts: Tuple[str, ...] = CLIP_TEXT_PROMPTS
        else:
            cleaned = [str(p).strip() for p in text_prompts if str(p).strip()]
            if len(cleaned) < 2:
                raise ValueError("text_prompts must contain at least two non-empty strings (softmax needs 2+ classes).")
            prompts = tuple(cleaned)
        self.text_prompts: Tuple[str, ...] = prompts
        text_tokens = self._tokenizer(list(self.text_prompts)).to(self.device)
        with torch.no_grad():
            tf = self.model.encode_text(text_tokens)
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)
        self._logit_scale = self.model.logit_scale.exp()

    def image_softmax_probs_batch(self, images: List[object]) -> List[Tuple[float, ...]]:
        """Per image: full softmax over ``self.text_prompts`` (index 0 = building-positive)."""
        import torch

        if not images:
            return []
        tensors = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(tensors)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * (img_feat @ self.text_features.T)
            probs = logits.softmax(dim=-1)
        rows = probs.detach().cpu().numpy().tolist()
        return [tuple(float(x) for x in r) for r in rows]

    def score_text_metadata_batch(self, texts: List[str]) -> List[float]:
        """
        For each metadata string, compute softmax P(class 0 = building-positive) vs the same
        CLIP_TEXT_PROMPTS used for images (N-way softmax; p_building = probs[0]).
        """
        import torch

        if not texts:
            return []
        # Avoid empty strings breaking tokenizers; single space is effectively "no signal".
        cleaned = [(t or "").strip() if (t or "").strip() else " " for t in texts]
        tokens = self._tokenizer(cleaned).to(self.device)
        with torch.no_grad():
            tf = self.model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * (tf @ self.text_features.T)
            probs = logits.softmax(dim=-1)
            p0 = probs[:, 0].detach().cpu().numpy().tolist()
        return [float(x) for x in p0]

    def classify_pil_batch(self, images: List[object]) -> List[Tuple[bool, float]]:
        """
        images: list of PIL RGB images (same length as batch).
        Returns (is_building, p_building) per image.
        is_building = True when argmax == 0 (building-positive prompt).
        p_building  = softmax probability for class 0.
        """
        prob_rows = self.image_softmax_probs_batch(images)
        out: List[Tuple[bool, float]] = []
        for probs in prob_rows:
            pred = max(range(len(probs)), key=lambda k: probs[k])
            p_building = float(probs[0])
            out.append((pred == 0, p_building))
        return out


def run_batched_clip(
    urls: Sequence[str],
    cache,
    runtime: ClipVisionRuntime,
    *,
    batch_size: int,
    timeout: float,
    max_retries: int = 5,
    base_backoff: float = 2.0,
    progress: bool = True,
    stop_after_errors: int = 0,
    on_result: "Optional[callable]" = None,
) -> Tuple[List[Tuple[Optional[bool], Optional[float]]], bool]:
    """
    Score each URL with OpenCLIP.

    Returns ``(results, stopped_early)``.

    Each entry in ``results`` is ``(is_building, p_building)``:
      - On success : ``(True/False, float)``  — is_building = argmax==0, p_building in [0,1]
      - On failure : ``(None, None)``          — download or decode failed; row stays unprocessed

    Failed rows are NOT written to cache so they will be retried on the next run.

    ``stopped_early`` is True when the run was aborted after ``stop_after_errors``
    consecutive persistent download failures. ``stop_after_errors=0`` disables this.

    If *on_result* is provided it is called immediately after each image is scored as
    ``on_result(global_idx, is_building, p_building)`` where is_building/p_building
    are None on failure. Use this for incremental cache writes.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    # Hard per-URL wall-clock deadline — kills requests that hang indefinitely
    # because the socket timeout never fires (trickle servers, corrupt PIL decode).
    hard_timeout_per_url = timeout + 20.0

    results: List[Tuple[Optional[bool], Optional[float]]] = []
    n = len(urls)
    bs = max(1, int(batch_size))
    consecutive_errors = 0
    stopped_early = False
    recent_errors: List[str] = []

    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(total=n, desc="OpenCLIP", unit="img", mininterval=0.3)

    batch_num = 0
    for start in range(0, n, bs):
        batch_num += 1
        chunk_urls = urls[start : start + bs]
        pil_list: List[object] = []
        ok_mask: List[bool] = []
        dl_errors: List[Optional[str]] = []

        for url in chunk_urls:
            u = (url or "").strip()
            if not u:
                pil_list.append(None)
                ok_mask.append(False)
                dl_errors.append("missing image_url")
                continue
            # img, err = _fetch_with_deadline(
            #     u,
            #     timeout=timeout,
            #     max_retries=max_retries,
            #     base_backoff=base_backoff,
            #     hard_timeout=hard_timeout_per_url,
            # )
            img = cache.get(url)
            if img is None:
                pil_list.append(None)
                ok_mask.append(False)
                dl_errors.append(err or "download or decode failed")
            else:
                pil_list.append(img)
                ok_mask.append(True)
                dl_errors.append(None)

        good_imgs = [pil_list[i] for i in range(len(pil_list)) if ok_mask[i]]
        classified = runtime.classify_pil_batch(good_imgs) if good_imgs else []

        gi = 0
        for i in range(len(chunk_urls)):
            global_idx = start + i
            if not ok_mask[i]:
                err_msg = dl_errors[i] or "download or decode failed"
                results.append((None, None))
                consecutive_errors += 1
                recent_errors.append(err_msg)
                if len(recent_errors) > stop_after_errors + 2:
                    recent_errors.pop(0)
                if on_result is not None:
                    on_result(global_idx, None, None)
            else:
                is_building, p_building = classified[gi]
                gi += 1
                results.append((is_building, p_building))
                consecutive_errors = 0
                recent_errors.clear()
                if on_result is not None:
                    on_result(global_idx, is_building, p_building)

        if pbar is not None:
            pbar.update(len(chunk_urls))
        elif progress and tqdm is None and n > 0:
            done = min(start + len(chunk_urls), n)
            if batch_num % 25 == 0 or done >= n:
                print(f"[INFO] OpenCLIP progress: {done:,} / {n:,} ({100.0*done/n:.1f}%)", flush=True)

        if stop_after_errors > 0 and consecutive_errors >= stop_after_errors:
            err_detail = "; ".join(dict.fromkeys(recent_errors))
            print(
                f"\n[STOP] {consecutive_errors} consecutive download error(s) after all retries.\n"
                f"       Last error(s): {err_detail}\n"
                "       Saving progress and stopping. Re-run after a cooldown to resume.",
                flush=True,
            )
            stopped_early = True
            break

    if pbar is not None:
        pbar.close()

    return results, stopped_early
