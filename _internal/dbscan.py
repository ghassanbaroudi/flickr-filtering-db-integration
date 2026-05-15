"""
Haversine DBSCAN clustering helper — pure DataFrame in / DataFrame out.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6_371_000.0


def _circular_mean_degrees(longitudes: pd.Series) -> float:
    """Circular mean of longitudes, avoiding antimeridian wrap artefacts."""
    lon = pd.to_numeric(longitudes, errors="coerce").dropna().to_numpy(dtype=float)
    if lon.size == 0:
        return float("nan")
    ang = np.radians(lon)
    s, c = float(np.sin(ang).mean()), float(np.cos(ang).mean())
    if s == 0.0 and c == 0.0:
        return float(np.mean(lon))
    out = float(np.degrees(np.arctan2(s, c)))
    if out >= 180.0:
        out -= 360.0
    if out < -180.0:
        out += 360.0
    return out


ClusterSummary = Dict[int, Dict[str, float]]


def run_dbscan(
    df: pd.DataFrame,
    *,
    eps_meters: float = 50.0,
    min_samples: int = 5,
) -> Tuple[pd.DataFrame, ClusterSummary]:
    """
    Run haversine DBSCAN on (latitude, longitude).

    Adds a ``geo_cluster_id`` column (≥ 0 for clustered rows, -1 for noise).
    Also returns a summary dict keyed by cluster_id.
    """
    if df.empty:
        raise ValueError("Cannot cluster an empty DataFrame.")

    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(
        eps=eps_meters / EARTH_RADIUS_M,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    ).fit_predict(coords_rad).astype(int)

    out = df.copy()
    out["geo_cluster_id"] = labels

    summary: ClusterSummary = {}
    for cid, g in out[out["geo_cluster_id"] >= 0].groupby("geo_cluster_id"):
        summary[int(cid)] = {
            "geo_cluster_id": int(cid),
            "count": int(len(g)),
            "min_latitude": float(g["latitude"].min()),
            "avg_latitude": float(g["latitude"].mean()),
            "max_latitude": float(g["latitude"].max()),
            "min_longitude": float(g["longitude"].min()),
            "avg_longitude": float(_circular_mean_degrees(g["longitude"])),
            "max_longitude": float(g["longitude"].max()),
        }

    n_clusters = len(summary)
    n_noise = int((labels == -1).sum())
    print(f"[dbscan] {n_clusters} clusters, {n_noise} noise points "
          f"(eps={eps_meters}m, min_samples={min_samples}).")
    return out, summary


def cluster_summary_df(summary: ClusterSummary) -> pd.DataFrame:
    """
    Convert the summary dict from run_dbscan into a DataFrame shaped for
    Nathanael's ``geo_cluster`` table:
      (id, min_latitude, avg_latitude, max_latitude,
           min_longitude, avg_longitude, max_longitude)
    """
    if not summary:
        return pd.DataFrame(columns=[
            "id", "min_latitude", "avg_latitude", "max_latitude",
            "min_longitude", "avg_longitude", "max_longitude",
        ])
    rows = []
    for info in summary.values():
        rows.append({
            "id": info["geo_cluster_id"],
            "min_latitude": info["min_latitude"],
            "avg_latitude": info["avg_latitude"],
            "max_latitude": info["max_latitude"],
            "min_longitude": info["min_longitude"],
            "avg_longitude": info["avg_longitude"],
            "max_longitude": info["max_longitude"],
        })
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
