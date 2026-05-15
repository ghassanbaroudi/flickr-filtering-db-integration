"""
clustering.py — filtering and geospatial clustering for the flickr-filtering pipeline.

Public API
----------
filter(df, ...) -> df
    Takes the full ~1 million row photo DataFrame, keeps only rows that:
      1. have valid, non-zero latitude/longitude,
      2. match at least one building keyword in title/description/tags,
      3. optionally: belong to the requested Flickr geo context (0/1/2).

vision_and_keywords(df, ...) -> df
    Takes the filtered DataFrame (which must already contain ``clip_vect_224`` from
    embedding.clip()), runs DBSCAN geospatial clustering, then uses OpenCLIP to
    score each image (YES/NO building) and returns the enriched DataFrame.

Column requirements
-------------------
filter()
  Input  : df[id, owner_nsid, latitude, longitude, title, description, tags, context?, ...]
  Output : same schema, filtered rows + ``keyword_hits`` debug column

vision_and_keywords()
  Input  : df[id, owner_nsid, latitude, longitude, url_o, clip_vect_224, ...]
  Output : same columns + ``geo_cluster_id``, ``vision_label``, ``p_building``,
           ``vision_reason``

DB change requirements for Nathanael
-------------------------------------
Two new columns are needed in ``machine_learning_photo``:

    ALTER TABLE machine_learning_photo
        ADD COLUMN clip_vect_224 VECTOR(512),       -- from embedding.clip()
        ADD COLUMN vision_label  TEXT,              -- "YES" / "NO" / "ERROR"
        ADD COLUMN p_building    DOUBLE PRECISION;  -- softmax prob of class-0 (building)

The ``geo_cluster_id`` column already exists; DBSCAN assigns it here.
The ``geo_cluster`` table already exists; use cluster_summary_df() to get the rows
to insert, then call ``save_clusters()`` from Nathanael's db.py.
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
# vision_and_keywords()
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
    Cluster and vision-score a filtered building-candidate DataFrame.

    Step 1 — DBSCAN geospatial clustering on (latitude, longitude).
    Step 2 — OpenCLIP image classification (YES / NO) per image, with incremental
              write-back after each batch so progress survives crashes / rate-limits.

    Parameters
    ----------
    df           : Filtered DataFrame from filter().
                   Required columns: ``latitude``, ``longitude``, ``url_o`` (or *url_column*),
                   and ideally ``id`` / ``owner_nsid`` for DB write-back.
    eps_meters   : DBSCAN neighbourhood radius in metres (default 50 m).
    min_samples  : DBSCAN minimum cluster size (default 5).
    model_name   : OpenCLIP architecture (default: env OPENCLIP_MODEL or "ViT-B-32").
    pretrained   : Checkpoint (default: env OPENCLIP_PRETRAINED or "laion2b_s34b_b79k").
    device       : "cuda" / "cpu" / None for auto.
    batch_size   : Images per GPU batch.
    timeout      : Per-image download timeout.
    max_retries  : Download retries.
    url_column   : Column with image URLs (default "url_o").
    text_prompts : Custom (class0_positive, *negatives) prompts. Defaults to built-in prompts.
    show_progress: Show tqdm bar during vision scoring.
    on_batch     : Optional callback for incremental DB write-back (crash-safe resume).

                   Called after every batch with a DataFrame slice containing the rows
                   that were just scored. The slice has the same columns as the full
                   output plus ``vision_label``, ``p_building``, ``vision_reason``.

                   Nathanael should use this to write vision results immediately:

                       def on_batch(batch_df: pd.DataFrame) -> None:
                           update_ml_photo(batch_df, 'vision_label')
                           update_ml_photo(batch_df, 'p_building')

                   Resume works automatically: pass only rows WHERE vision_label IS NULL
                   as the input *df* (rows already scored are simply not included).

    Returns
    -------
    (photos_df, clusters_df)

    photos_df   — *df* enriched with:
        ``geo_cluster_id``  INT  (-1 = noise / no cluster)
        ``vision_label``    TEXT ("YES" / "NO" / "ERROR")
        ``p_building``      FLOAT  (softmax probability for class 0 = building)
        ``vision_reason``   TEXT  (brief classifier note)

    clusters_df — one row per cluster, shaped for Nathanael's ``geo_cluster`` table:
        id, min_latitude, avg_latitude, max_latitude,
        min_longitude, avg_longitude, max_longitude

    DB write-back pattern (Nathanael)
    -----------------------------------
    def on_batch(batch_df):
        update_ml_photo(batch_df, 'vision_label')
        update_ml_photo(batch_df, 'p_building')

    photos_out, clusters_out = clustering.vision_and_keywords(df, on_batch=on_batch)

    # After the full run, write cluster rows and geo_cluster_id (safe to do at the end
    # since DBSCAN is fast and runs entirely before any network I/O):
    save_clusters(clusters_out)
    update_ml_photo(photos_out[['owner_nsid','id','geo_cluster_id']], 'geo_cluster_id')
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty — run filter() first.")
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found. Available: {list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 1 — DBSCAN clustering
    # ------------------------------------------------------------------
    print(f"[vision_and_keywords] Step 1/2 — DBSCAN (eps={eps_meters}m, min_samples={min_samples})")
    clustered_df, summary = run_dbscan(df, eps_meters=eps_meters, min_samples=min_samples)

    # ------------------------------------------------------------------
    # Step 2 — OpenCLIP vision scoring
    # ------------------------------------------------------------------
    _model = (model_name or os.getenv("OPENCLIP_MODEL", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
    _pretrained = (pretrained or os.getenv("OPENCLIP_PRETRAINED", _DEFAULT_PRETRAINED)).strip() or _DEFAULT_PRETRAINED

    print(f"[vision_and_keywords] Step 2/2 — OpenCLIP scoring "
          f"(model={_model!r}, {len(clustered_df):,} images)")

    runtime = ClipRuntime(
        model_name=_model,
        pretrained=_pretrained,
        device=device,
        text_prompts=text_prompts or DEFAULT_TEXT_PROMPTS,
    )
    print(f"[vision_and_keywords] model ready on {runtime.device}")

    hard_timeout = timeout + 20.0
    urls: List[str] = clustered_df[url_column].fillna("").astype(str).tolist()
    n = len(urls)
    bs = max(1, int(batch_size))

    vision_labels: List[str] = ["ERROR"] * n
    p_buildings: List[float] = [float("nan")] * n
    vision_reasons: List[str] = [""] * n

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n, desc="OpenCLIP score", unit="img") if show_progress else None
    except ImportError:
        pbar = None

    for start in range(0, n, bs):
        chunk_urls = urls[start: start + bs]
        pil_images = []
        ok_flags: List[bool] = []
        err_msgs: List[str] = []

        for url in chunk_urls:
            u = url.strip()
            if not u:
                pil_images.append(None)
                ok_flags.append(False)
                err_msgs.append("missing url")
                continue
            img, err = fetch_with_deadline(
                u,
                timeout=timeout,
                max_retries=max_retries,
                base_backoff=2.0,
                hard_timeout=hard_timeout,
            )
            pil_images.append(img)
            ok_flags.append(img is not None)
            err_msgs.append(err or "")

        good_images = [im for im, ok in zip(pil_images, ok_flags) if ok]
        scored = runtime.score_images(good_images) if good_images else []

        gi = 0
        batch_indices: List[int] = []
        for local_i, (ok, err) in enumerate(zip(ok_flags, err_msgs)):
            global_i = start + local_i
            batch_indices.append(global_i)
            if ok:
                label, p0, _probs = scored[gi]
                gi += 1
                vision_labels[global_i] = label
                p_buildings[global_i] = p0
                vision_reasons[global_i] = f"openclip p(building)={p0:.3f}"
            else:
                vision_labels[global_i] = "ERROR"
                vision_reasons[global_i] = err

        if on_batch is not None:
            batch_slice = clustered_df.iloc[batch_indices].copy()
            batch_slice["vision_label"] = [vision_labels[i] for i in batch_indices]
            batch_slice["p_building"] = [p_buildings[i] for i in batch_indices]
            batch_slice["vision_reason"] = [vision_reasons[i] for i in batch_indices]
            on_batch(batch_slice)

        if pbar is not None:
            pbar.update(len(chunk_urls))

    if pbar is not None:
        pbar.close()

    n_yes = vision_labels.count("YES")
    n_no = vision_labels.count("NO")
    n_err = vision_labels.count("ERROR")
    print(f"[vision_and_keywords] vision results: YES={n_yes}, NO={n_no}, ERROR={n_err}")

    clustered_df = clustered_df.copy()
    clustered_df["vision_label"] = vision_labels
    clustered_df["p_building"] = p_buildings
    clustered_df["vision_reason"] = vision_reasons

    clusters_df = cluster_summary_df(summary)
    return clustered_df, clusters_df


# ---------------------------------------------------------------------------
# Convenience re-export so callers can import cluster_summary_df from here
# ---------------------------------------------------------------------------
__all__ = ["filter", "vision_and_keywords", "cluster_summary_df"]
