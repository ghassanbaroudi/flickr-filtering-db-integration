"""
embedding.py — image embedding functions for the flickr-filtering pipeline.

Public API
----------
clip(df, ...) -> df
    Takes a DataFrame with at minimum an ``url_o`` column (original Flickr image URL),
    downloads each image, runs OpenCLIP ViT-B-32, and returns the DataFrame with a
    new ``clip_vect_224`` column containing the L2-normalised 512-d embedding as a
    numpy float32 array (or None on download / decode failure).

    The output column name ``clip_vect_224`` reflects the 224×224 input resolution
    used by ViT-B-32, consistent with Nathanael's ``sig_lip_vect_n`` naming convention.

Column requirements
-------------------
Input  : df must contain  ``url_o``  (and ideally ``id``, ``owner_nsid`` for tracing)
Output : same rows; adds   ``clip_vect_224`` (numpy array of shape (512,), or None)
                           ``clip_error``     (str describing the failure, or "")
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ._internal.clip_runtime import ClipRuntime, DEFAULT_TEXT_PROMPTS
from ._internal.image_fetch import fetch_with_deadline

# ---------------------------------------------------------------------------
# Default model settings — override via environment variables or function args
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 5


def clip(
    df: pd.DataFrame,
    *,
    model_name: str = "",
    pretrained: str = "",
    device: Optional[str] = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    timeout: float = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    url_column: str = "url_o",
    text_prompts: Optional[Sequence[str]] = None,
    show_progress: bool = True,
    on_batch: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Compute OpenCLIP image embeddings for every row in *df*.

    Parameters
    ----------
    df           : Input DataFrame. Must contain *url_column* (default ``url_o``).
                   Should also carry ``id`` and ``owner_nsid`` for DB write-back.
    model_name   : OpenCLIP architecture (default: env OPENCLIP_MODEL or "ViT-B-32").
    pretrained   : Checkpoint tag (default: env OPENCLIP_PRETRAINED or "laion2b_s34b_b79k").
    device       : "cuda" / "cpu" / None for auto.
    batch_size   : Images per GPU batch.
    timeout      : Per-image download timeout in seconds.
    max_retries  : Download retries before giving up.
    url_column   : Column containing image URLs (default "url_o" = original resolution).
    text_prompts : Not used for embedding (no text interaction), kept for API symmetry.
    show_progress: Print a tqdm progress bar.
    on_batch     : Optional callback for incremental DB write-back (crash-safe resume).

                   Called after every batch with a DataFrame slice containing the rows
                   just embedded, with ``clip_vect_224`` and ``clip_error`` already set.

                   Nathanael should use this to write embeddings immediately:

                       def on_batch(batch_df: pd.DataFrame) -> None:
                           update_ml_photo(batch_df, 'clip_vect_224')

                   Resume works automatically: pass only rows WHERE clip_vect_224 IS NULL
                   as the input *df* (already-embedded rows are not included).

    Returns
    -------
    DataFrame identical to *df* plus one new column:
      ``clip_vect_224`` — numpy float32 array of shape (512,), or None on failure.
                          None rows are not passed to on_batch and stay NULL in the DB,
                          so they are retried automatically on the next run.

    Notes for Nathanael's DB
    ------------------------
    ``clip_vect_224`` maps to ``clip_vect_224 VECTOR(512)`` on ``machine_learning_photo``.
    The column must first be added:

        ALTER TABLE machine_learning_photo ADD COLUMN clip_vect_224 VECTOR(512);
    """
    if url_column not in df.columns:
        raise ValueError(
            f"Column '{url_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    _model = (model_name or os.getenv("OPENCLIP_MODEL", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
    _pretrained = (pretrained or os.getenv("OPENCLIP_PRETRAINED", _DEFAULT_PRETRAINED)).strip() or _DEFAULT_PRETRAINED

    print(f"[embedding.clip] model={_model!r} pretrained={_pretrained!r} "
          f"rows={len(df):,} url_column={url_column!r}")

    runtime = ClipRuntime(
        model_name=_model,
        pretrained=_pretrained,
        device=device,
        text_prompts=text_prompts or DEFAULT_TEXT_PROMPTS,
    )
    print(f"[embedding.clip] model loaded on {runtime.device}, embed_dim={runtime.embed_dim}")

    hard_timeout = timeout + 20.0
    urls: List[str] = df[url_column].fillna("").astype(str).tolist()
    n = len(urls)
    bs = max(1, int(batch_size))

    vectors: List[Optional[np.ndarray]] = [None] * n

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n, desc="OpenCLIP embed", unit="img") if show_progress else None
    except ImportError:
        pbar = None

    for start in range(0, n, bs):
        chunk_urls = urls[start: start + bs]
        pil_images: List[Optional[object]] = []
        ok_flags: List[bool] = []

        for url in chunk_urls:
            u = url.strip()
            if not u:
                pil_images.append(None)
                ok_flags.append(False)
                continue
            try:
                img, err = fetch_with_deadline(
                    u,
                    timeout=timeout,
                    max_retries=max_retries,
                    base_backoff=2.0,
                    hard_timeout=hard_timeout,
                )
                if img is None:
                    raise RuntimeError(err or "download failed")
                pil_images.append(img)
                ok_flags.append(True)
            except Exception:
                pil_images.append(None)
                ok_flags.append(False)

        good_images = [im for im, ok in zip(pil_images, ok_flags) if ok]
        try:
            embeddings = runtime.embed_images(good_images) if good_images else []
        except Exception as exc:
            print(f"[embedding.clip] batch inference failed: {exc} — skipping batch.", flush=True)
            embeddings = []
            ok_flags = [False] * len(ok_flags)

        ei = 0
        successful_indices: List[int] = []
        for local_i, ok in enumerate(ok_flags):
            global_i = start + local_i
            if ok and ei < len(embeddings):
                vectors[global_i] = embeddings[ei]
                ei += 1
                successful_indices.append(global_i)

        # Only pass successfully embedded rows to on_batch.
        # Failed rows stay None in `vectors` — they remain NULL in the DB and are retried next run.
        if on_batch is not None and successful_indices:
            batch_slice = df.iloc[successful_indices].copy()
            batch_slice["clip_vect_224"] = [vectors[i] for i in successful_indices]
            batch_slice["clip_error"] = [""] * len(successful_indices)
            on_batch(batch_slice)

        if pbar is not None:
            pbar.update(len(chunk_urls))

    if pbar is not None:
        pbar.close()

    n_ok = sum(1 for v in vectors if v is not None)
    n_pending = n - n_ok
    print(f"[embedding.clip] done: {n_ok:,} embedded, {n_pending:,} failed (will retry next run).")

    out = df.copy()
    out["clip_vect_224"] = vectors  # None for failed rows — stays NULL in DB
    return out
