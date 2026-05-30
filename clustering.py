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

from ._internal.keywords import load_keywords, apply_keyword_filter, apply_geo_filter, apply_context_filter
from ._internal.dbscan import run_dbscan, cluster_summary_df, ClusterSummary
from ._internal.clip_runtime import DEFAULT_TEXT_PROMPTS
from ._internal.clip_vision import run_batched_clip, ClipVisionRuntime

_DEFAULT_EPS_METERS = 50.0
_DEFAULT_MIN_SAMPLES = 5
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_TIMEOUT = 30.0


def _filter(
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


def cluster(df:pd.DataFrame) -> pd.DataFrame:
    photos, clusters = _cluster(_filter(df))
    return photos


def _cluster(
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


def label_buildings(
    df: pd.DataFrame,
    cache,
    *,
    model_name: str = "",
    pretrained: str = "",
    device: Optional[str] = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    timeout: float = _DEFAULT_TIMEOUT,
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
    timeout      : Per-image download timeout in seconds (single attempt, no retries).
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

    _runtime = ClipVisionRuntime(
        model_name=_model,
        pretrained=_pretrained,
        device=device,
        text_prompts=text_prompts,
    )
    print(f"[label_buildings] model ready on {_runtime.device}")

    urls: List[str] = df[url_column].fillna("").astype(str).tolist()
    n = len(urls)

    is_buildings: List[Optional[bool]] = [None] * n
    p_buildings: List[Optional[float]] = [None] * n

    def _on_result(global_idx: int, is_building: Optional[bool], p_building: Optional[float]) -> None:
        is_buildings[global_idx] = is_building
        p_buildings[global_idx] = p_building
        if on_batch is not None and is_building is not None:
            row_slice = df.iloc[[global_idx]].copy()
            row_slice["is_building"] = [is_building]
            row_slice["p_building"] = [p_building]
            on_batch(row_slice)

    run_batched_clip(
        urls,
        cache,
        _runtime,
        batch_size=batch_size,
        timeout=timeout,
        progress=show_progress,
        on_result=_on_result,
    )

    n_yes = sum(1 for v in is_buildings if v is True)
    n_no = sum(1 for v in is_buildings if v is False)
    n_pending = sum(1 for v in is_buildings if v is None)
    print(f"[label_buildings] building={n_yes}, not_building={n_no}, pending(retry)={n_pending}")

    out = df.copy()
    out["is_building"] = is_buildings
    out["p_building"] = p_buildings
    return out


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
        url_column=url_column,
        text_prompts=text_prompts,
        show_progress=show_progress,
        on_batch=on_batch,
    )
    return labeled_df, clusters_df


__all__ = ["filter", "cluster", "label_buildings", "vision_and_keywords", "cluster_summary_df"]
