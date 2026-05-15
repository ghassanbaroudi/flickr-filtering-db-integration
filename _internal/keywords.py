"""
Keyword and geo-filtering helpers — pure DataFrame in / DataFrame out, no I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Default keyword list (substring match, case-insensitive).
# Can be overridden by passing a custom list to filter().
# ---------------------------------------------------------------------------
DEFAULT_BUILDING_KEYWORDS: List[str] = [
    "building",
    "architecture",
    "architectural",
    "facade",
    "cathedral",
    "church",
    "chapel",
    "tower",
    "castle",
    "palace",
    "museum",
    "monument",
    "dome",
    "city hall",
    "town hall",
    "theatre",
    "theater",
    "college",
    "landmark",
    "skyscraper",
    "structure",
]


def load_keywords(
    keywords: Optional[List[str]] = None,
    keywords_file: Optional[Path] = None,
) -> List[str]:
    """
    Return the keyword list to use, in priority order:
      1. keywords_file (one per line, # comments allowed)
      2. keywords list argument
      3. DEFAULT_BUILDING_KEYWORDS
    """
    if keywords_file is not None:
        if not keywords_file.exists():
            raise FileNotFoundError(f"Keywords file not found: {keywords_file}")
        result: List[str] = []
        for line in keywords_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                result.append(line)
        if not result:
            raise ValueError(f"No keywords found in {keywords_file}")
        return result

    if keywords:
        kws = [k.strip() for k in keywords if k.strip()]
        if not kws:
            raise ValueError("Keyword list is empty after stripping.")
        return kws

    return list(DEFAULT_BUILDING_KEYWORDS)


def _row_text(row: pd.Series) -> str:
    parts = []
    for col in ("title", "description", "tags"):
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts).lower()


def apply_keyword_filter(
    df: pd.DataFrame,
    keywords: List[str],
) -> pd.DataFrame:
    """
    Keep rows where at least one keyword appears as a substring in
    title + description + tags (case-insensitive).
    Adds ``keyword_hits`` column (';'-separated matched terms).
    """
    kws = [k.lower().strip() for k in keywords if k.strip()]
    if not kws:
        raise ValueError("No non-empty keywords.")

    hits_col: List[str] = []
    passes: List[bool] = []
    for _, row in df.iterrows():
        text = _row_text(row)
        matched = [kw for kw in kws if kw in text]
        passes.append(bool(matched))
        hits_col.append(";".join(matched))

    out = df.copy()
    out["keyword_hits"] = hits_col
    out = out.loc[passes].reset_index(drop=True)
    print(f"[filter] keyword match: {len(out):,} / {len(df):,} rows kept.")
    return out


def apply_geo_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows with valid, non-zero latitude/longitude within standard bounds.
    Coerces both columns to numeric in-place.
    """
    required = {"latitude", "longitude"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    mask = (
        out["latitude"].notna()
        & out["longitude"].notna()
        & (out["latitude"] != 0)
        & (out["longitude"] != 0)
        & out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    )
    out = out.loc[mask].reset_index(drop=True)
    print(f"[filter] valid geo: {len(out):,} rows kept.")
    return out


def apply_context_filter(df: pd.DataFrame, contexts: Set[str]) -> pd.DataFrame:
    """
    Restrict to specific Flickr geo contexts (0=unknown, 1=indoors, 2=outdoors).
    Pass contexts=None or an empty set to skip.
    """
    if not contexts:
        return df
    if "context" not in df.columns:
        raise KeyError("`context` column is required for context filtering.")
    out = df[df["context"].astype(str).str.strip().isin(contexts)].reset_index(drop=True)
    print(f"[filter] context {sorted(contexts)}: {len(out):,} rows kept.")
    return out
