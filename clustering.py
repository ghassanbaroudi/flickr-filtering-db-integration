"""
clustering.py — filtering, geospatial clustering, and building labeling.

Public API
----------
filter(df, ...) -> df
    Geo + keyword pre-filter. Keeps rows with valid lat/lon and at least one
    building keyword in title/description/tags.

cluster(df, ...) -> (photos_df, clusters_df)
    DBSCAN geospatial clustering only. Fast, no network I/O.
    Adds ``geo_cluster_id`` to photos_df.
    Returns clusters_df shaped for Nathanael's ``geo_cluster`` table.

label_buildings(df, ...) -> df
    OpenCLIP vision scoring only. Downloads and classifies each image.
    Adds ``is_building`` (bool/None) and ``p_building`` (float/None).
    Failed rows stay None — retried automatically next run via IS NULL query.

vision_and_keywords(df, ...) -> (photos_df, clusters_df)
    Convenience wrapper that calls cluster() then label_buildings() in sequence.
    Kept for backwards compatibility.

Column requirements
-------------------
filter()
  Input  : df[id, owner_nsid, latitude, longitude, title, description, tags, context?, ...]
  Output : same schema, filtered rows + ``keyword_hits`` debug column

cluster()
  Input  : df[id, owner_nsid, latitude, longitude, ...]
  Output : same columns + ``geo_cluster_id`` (-1 = noise)
           + clusters_df[id, min/avg/max latitude, min/avg/max longitude]

label_buildings()
  Input  : df[id, owner_nsid, url_o, ...]
  Output : same columns + ``is_building`` (True/False/None), ``p_building`` (float/None)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Set

import pandas as pd

from _internal.keywords import load_keywords, apply_keyword_filter, apply_geo_filter, apply_context_filter
from _internal.dbscan import run_dbscan, cluster_summary_df, ClusterSummary
from _internal.clip_runtime import ClipRuntime, DEFAULT_TEXT_PROMPTS
from _internal.image_fetch import fetch_with_deadline

# ---------------------------------------------------------------------------
# Default settings — can be overridden via kwargs or environment variables
# ---------------------------------------------------------------------------
_DEFAULT_EPS_METERS = 50.0
_DEFAULT_MIN_SAMPLES = 5
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# filter()
# ---------------------------------------------------------------------------

def filter(
    df: pd.DataFrame,
    *,
    keywords: Optional[List[str]] = None,
    keywords_file: Optional[Path] = None,
    contexts: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Keep the rows that are interesting for building-cluster analysis.

    Parameters
    ----------
    df            : Full photo DataFrame (e.g. read from ``photo`` table).
                    Required columns: ``latitude``, ``longitude``,
                    ``title`` / ``description`` / ``tags`` (at least one).
    keywords      : Override default building keyword list (substring match).
    keywords_file : Path to a text file with one keyword per line (takes priority
                    over *keywords*).
    contexts      : Flickr context values to keep, e.g. {"0", "2"}.
                    None or empty set = keep all contexts (default).

    Returns
    -------
    Filtered DataFrame with only candidate building-related, geo-valid rows.
    Adds ``keyword_hits`` column (';'-separated matched keywords).
    """
    kws = load_keywords(keywords, keywords_file)

    df_geo = apply_geo_filter(df)
    df_ctx = apply_context_filter(df_geo, contexts or set())
    df_kw = apply_keyword_filter(df_ctx, kws)

    if df_kw.empty:
        print("[filter] Warning: no rows survived filtering.")
    else:
        print(f"[filter] Final: {len(df_kw):,} rows after geo + keyword + context filter.")
    return df_kw


# ---------------------------------------------------------------------------
# cluster()
# ---------------------------------------------------------------------------

def cluster(
    df: pd.DataFrame,
    *,
    eps_meters: float = _DEFAULT_EPS_METERS,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DBSCAN geospatial clustering on (latitude, longitude). Fast — no network I/O.

    Parameters
    ----------
    df          : DataFrame with ``latitude`` and ``longitude`` columns.
                  Should also carry ``id`` and ``owner_nsid`` for DB write-back.
    eps_meters  : Neighbourhood radius in metres (default 50 m).
    min_samples : Minimum points to form a cluster (default 5).

    Returns
    -------
    (photos_df, clusters_df)

    photos_df   — *df* with added column:
        ``geo_cluster_id``  INT  (≥ 0 for clustered rows, -1 = noise)

    clusters_df — one row per cluster, shaped for Nathanael's ``geo_cluster`` table:
        id, min_latitude, avg_latitude, max_latitude,
        min_longitude, avg_longitude, max_longitude

    DB write-back (Nathanael)
    --------------------------
    photos_out, clusters_out = clustering.cluster(df)

    save_clusters(clusters_out)   # insert into geo_cluster first (FK constraint)
    update_ml_photo(photos_out[['owner_nsid', 'id', 'geo_cluster_id']], 'geo_cluster_id')
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    print(f"[cluster] DBSCAN (eps={eps_meters}m, min_samples={min_samples}, rows={len(df):,})")
    clustered_df, summary = run_dbscan(df, eps_meters=eps_meters, min_samples=min_samples)
    clusters_df = cluster_summary_df(summary)
    return clustered_df, clusters_df


# ---------------------------------------------------------------------------
# label_buildings()
# ---------------------------------------------------------------------------

def label_buildings(
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
    OpenCLIP vision scoring — classifies each image as building or not.

    Parameters
    ----------
    df           : DataFrame with image URLs. Should carry ``id`` and ``owner_nsid``
                   for DB write-back. Does NOT need to be clustered first.
    model_name   : OpenCLIP architecture (default: env OPENCLIP_MODEL or "ViT-B-32").
    pretrained   : Checkpoint (default: env OPENCLIP_PRETRAINED or "laion2b_s34b_b79k").
    device       : "cuda" / "cpu" / None for auto-detect.
    batch_size   : Images per GPU batch (default 32).
    timeout      : Per-image download timeout in seconds.
    max_retries  : Retries on 429/503 before giving up.
    url_column   : Column containing image URLs (default "url_o").
    text_prompts : Custom (class0_positive, *negatives) prompts. Defaults to built-in.
    show_progress: Show tqdm progress bar.
    on_batch     : Optional callback called after each successful batch for incremental
                   DB write-back. Only successfully scored rows are passed — failed
                   rows stay None and are never sent to the callback.

                   Nathanael should use this to commit results immediately:

                       def on_batch(batch_df: pd.DataFrame) -> None:
                           update_ml_photo(batch_df, 'is_building')
                           update_ml_photo(batch_df, 'p_building')

                   Resume: query WHERE is_building IS NULL before calling this function.
                   Failed rows stay NULL in the DB and are retried automatically.

    Returns
    -------
    DataFrame identical to *df* plus two new columns:
      ``is_building`` — True (building) / False (not a building) / None (download failed)
      ``p_building``  — softmax probability for class 0 (building), or None on failure
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found. Available: {list(df.columns)}")

    _model = (model_name or os.getenv("OPENCLIP_MODEL", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
    _pretrained = (pretrained or os.getenv("OPENCLIP_PRETRAINED", _DEFAULT_PRETRAINED)).strip() or _DEFAULT_PRETRAINED

    print(f"[label_buildings] model={_model!r} pretrained={_pretrained!r} rows={len(df):,}")

    runtime = ClipRuntime(
        model_name=_model,
        pretrained=_pretrained,
        device=device,
        text_prompts=text_prompts or DEFAULT_TEXT_PROMPTS,
    )
    print(f"[label_buildings] model ready on {runtime.device}")

    hard_timeout = timeout + 20.0
    urls: List[str] = df[url_column].fillna("").astype(str).tolist()
    n = len(urls)
    bs = max(1, int(batch_size))

    # None = not yet scored / download failed — stays NULL in DB, retried next run.
    is_buildings: List[Optional[bool]] = [None] * n
    p_buildings: List[Optional[float]] = [None] * n

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n, desc="OpenCLIP label", unit="img") if show_progress else None
    except ImportError:
        pbar = None

    for start in range(0, n, bs):
        chunk_urls = urls[start: start + bs]
        pil_images = []
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
            scored = runtime.score_images(good_images) if good_images else []
        except Exception as exc:
            print(f"[label_buildings] batch inference failed: {exc} — skipping batch.", flush=True)
            scored = []
            ok_flags = [False] * len(ok_flags)

        gi = 0
        successful_indices: List[int] = []
        for local_i, ok in enumerate(ok_flags):
            global_i = start + local_i
            if ok and gi < len(scored):
                is_building, p0, _probs = scored[gi]
                gi += 1
                is_buildings[global_i] = is_building
                p_buildings[global_i] = p0
                successful_indices.append(global_i)

        # Only pass successfully scored rows to on_batch.
        # Failed rows stay None — remain NULL in DB and are retried next run.
        if on_batch is not None and successful_indices:
            batch_slice = df.iloc[successful_indices].copy()
            batch_slice["is_building"] = [is_buildings[i] for i in successful_indices]
            batch_slice["p_building"] = [p_buildings[i] for i in successful_indices]
            on_batch(batch_slice)

        if pbar is not None:
            pbar.update(len(chunk_urls))

    if pbar is not None:
        pbar.close()

    n_yes = sum(1 for v in is_buildings if v is True)
    n_no = sum(1 for v in is_buildings if v is False)
    n_pending = sum(1 for v in is_buildings if v is None)
    print(f"[label_buildings] building={n_yes}, not_building={n_no}, pending(retry)={n_pending}")

    out = df.copy()
    out["is_building"] = is_buildings
    out["p_building"] = p_buildings
    return out


# ---------------------------------------------------------------------------
# vision_and_keywords() — backwards-compatible wrapper
# ---------------------------------------------------------------------------

def vision_and_keywords(
    df: pd.DataFrame,
    *,
    eps_meters: float = _DEFAULT_EPS_METERS,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: calls cluster() then label_buildings() in sequence.

    Returns (photos_df, clusters_df) — same as cluster(), but photos_df also
    has ``is_building`` and ``p_building`` from label_buildings().

    See cluster() and label_buildings() for full parameter documentation.
    """
    clustered_df, clusters_df = cluster(df, eps_meters=eps_meters, min_samples=min_samples)
    labeled_df = label_buildings(
        clustered_df,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
        timeout=timeout,
        max_retries=max_retries,
        url_column=url_column,
        text_prompts=text_prompts,
        show_progress=show_progress,
        on_batch=on_batch,
    )
    return labeled_df, clusters_df


# ---------------------------------------------------------------------------
# Convenience re-export so callers can import cluster_summary_df from here
# ---------------------------------------------------------------------------
__all__ = ["filter", "cluster", "label_buildings", "vision_and_keywords", "cluster_summary_df"]
