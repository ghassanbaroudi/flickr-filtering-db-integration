# Integration requirements for `flico-nathanael`

This document lists every change needed in `flico-nathanael` to support the
`flickr-filtering` pipeline. Nothing here changes Ghass's code — all items below
are additions or fixes on your side.

---

## 1. Database schema — `tables.sql`

### 1a. New columns on `machine_learning_photo`

```sql
ALTER TABLE machine_learning_photo
    ADD COLUMN clip_vect_224  VECTOR(512),      -- OpenCLIP ViT-B-32 L2-normalised embedding
    ADD COLUMN vision_label   TEXT,             -- "YES" / "NO" / "ERROR"
    ADD COLUMN p_building     DOUBLE PRECISION; -- softmax P(class-0 = building), range [0.0, 1.0]
```

> `geo_cluster_id` already exists — no change needed there.  
> `geo_cluster` table already exists — no change needed there.

### 1b. Why these types

| Column | Type | Reason |
|---|---|---|
| `clip_vect_224` | `VECTOR(512)` | ViT-B-32 outputs 512-dimensional embeddings. Named `_224` to reflect the 224×224 input resolution, consistent with your `sig_lip_vect_n` (`_n` = 320px) convention. |
| `vision_label` | `TEXT` | Three possible values: `"YES"` (building), `"NO"` (not a building), `"ERROR"` (download failed — will be retried). |
| `p_building` | `DOUBLE PRECISION` | Softmax probability for the building-positive class. Useful for ranking or threshold tuning later. |

---

## 2. SQLAlchemy model — `src/core/model.py`

Add the three new columns to `ml_photo_table`:

```python
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Float  # add Float to existing imports

ml_photo_table = Table(
    "machine_learning_photo",
    metadata,
    # ... existing columns unchanged ...

    Column("clip_vect_224", VECTOR(512)),   # ADD
    Column("vision_label",  Text),          # ADD
    Column("p_building",    Float),         # ADD

    # ... existing ForeignKeyConstraints unchanged ...
)
```

This is needed so `update_ml_photo()` can write these columns via SQLAlchemy.

---

## 3. Bug fix in `src/trainer/db.py` — `save_clusters`

The existing `save_clusters` function has a typo (`_psql_insert_signore` → `_psql_insert_ignore`).
Fix it before using it:

```python
# BEFORE (broken):
def save_clusters(df: pd.DataFrame):
    df.to_sql(
        name='geo_cluster',
        con=get_engine('trainer'),
        if_exists='append',
        index=False,
        method=_psql_insert_signore,   # ← typo, will crash at runtime
        chunksize=1000
    )

# AFTER (fixed):
def save_clusters(df: pd.DataFrame):
    df.to_sql(
        name='geo_cluster',
        con=get_engine('trainer'),
        if_exists='append',
        index=False,
        method=_psql_insert_ignore,    # ← correct
        chunksize=1000
    )
```

---

## 4. New query functions in `src/trainer/db.py`

Add these three functions alongside the existing ones (`flickr_mlphoto_to_embed`, etc.).

### 4a. `flickr_photo_to_clip_embed()`

Returns photos that still need an OpenCLIP embedding.
Mirrors your existing `flickr_mlphoto_to_embed()` pattern.

```python
def flickr_photo_to_clip_embed() -> pd.DataFrame:
    """Photos that still need a clip_vect_224 embedding."""
    query = text("""--sql
        SELECT * FROM photo AS P
        JOIN machine_learning_photo AS MLP
        ON P.owner_nsid = MLP.owner_nsid AND P.id = MLP.id
        WHERE MLP.clip_vect_224 IS NULL
    """)
    df = pd.read_sql_query(query, get_engine("trainer"))
    return df.loc[:, ~df.columns.duplicated()]
```

### 4b. `flickr_photo_to_vision_score(skip_errors=False)`

Returns photos that still need vision scoring.

- `NULL` rows = never attempted → always included.
- `'ERROR'` rows = download failed last time → **retried by default** (same behaviour as Ghass's
  existing `vision_cache.jsonl` pipeline).
- Pass `skip_errors=True` to leave ERROR rows alone (useful if a URL is permanently broken).

```python
def flickr_photo_to_vision_score(skip_errors: bool = False) -> pd.DataFrame:
    """
    Photos that need vision scoring (NULL = never scored, ERROR = failed last run).
    Only returns rows that already have a geo_cluster_id (DBSCAN must run first).
    Pass skip_errors=True to leave permanently-failed rows untouched.
    """
    if skip_errors:
        condition = "MLP.vision_label IS NULL"
    else:
        condition = "(MLP.vision_label IS NULL OR MLP.vision_label = 'ERROR')"

    query = text(f"""--sql
        SELECT * FROM photo AS P
        JOIN machine_learning_photo AS MLP
        ON P.owner_nsid = MLP.owner_nsid AND P.id = MLP.id
        WHERE MLP.geo_cluster_id IS NOT NULL
        AND {condition}
    """)
    df = pd.read_sql_query(query, get_engine("trainer"))
    return df.loc[:, ~df.columns.duplicated()]
```

### 4c. `flickr_photo_to_dbscan()`

Returns photos that need DBSCAN clustering (geo-valid, not yet assigned a cluster).
DBSCAN must receive **all unprocessed rows at once** — not in chunks — because it needs
the full point cloud to produce consistent cluster IDs.

```python
def flickr_photo_to_dbscan() -> pd.DataFrame:
    """
    Photos that need geospatial clustering.
    Must be called once for the full unprocessed set — DBSCAN is not resumable.
    """
    query = text("""--sql
        SELECT * FROM photo AS P
        JOIN machine_learning_photo AS MLP
        ON P.owner_nsid = MLP.owner_nsid AND P.id = MLP.id
        WHERE MLP.geo_cluster_id IS NULL
        AND P.latitude IS NOT NULL
        AND P.longitude IS NOT NULL
        AND P.latitude  != 0
        AND P.longitude != 0
    """)
    df = pd.read_sql_query(query, get_engine("trainer"))
    return df.loc[:, ~df.columns.duplicated()]
```

---

## 5. How to call the pipeline (orchestration example)

Place this in your trainer script or notebook. The `on_batch` callbacks ensure that
results are committed to the DB after every batch, so a crash or Flickr rate-limit
loses **at most one batch** (default 32 images).

```python
import sys
sys.path.insert(0, "/path/to/flickr-filtering")  # adjust to actual path

import embedding
import clustering
from src.trainer.db import (
    flickr_photo_to_clip_embed,
    flickr_photo_to_dbscan,
    flickr_photo_to_vision_score,
    save_clusters,
    update_ml_photo,
)

# ── Step 1: OpenCLIP embeddings ──────────────────────────────────────────────
# Re-run freely; only rows with clip_vect_224 IS NULL are fetched.

def _on_clip_batch(batch_df):
    update_ml_photo(batch_df, 'clip_vect_224')

df_to_embed = flickr_photo_to_clip_embed()
if not df_to_embed.empty:
    embedding.clip(df_to_embed, on_batch=_on_clip_batch)

# ── Step 2: DBSCAN clustering ────────────────────────────────────────────────
# Run once on all unprocessed rows. geo_cluster_id + cluster rows written at the end.
# DBSCAN is fast (no network I/O) so no on_batch needed here.

df_to_cluster = flickr_photo_to_dbscan()
if not df_to_cluster.empty:
    photos_clustered, clusters_df = clustering.vision_and_keywords(
        df_to_cluster,
        on_batch=None,   # vision scoring handled in step 3 below
    )
    # Write clusters first (FK constraint: geo_cluster_id references geo_cluster.id)
    save_clusters(clusters_df)
    update_ml_photo(
        photos_clustered[['owner_nsid', 'id', 'geo_cluster_id']],
        'geo_cluster_id',
    )

# ── Step 3: Vision scoring ───────────────────────────────────────────────────
# Re-run freely; ERROR rows are automatically retried (skip_errors=False).
# Each batch is committed immediately via on_batch — safe to interrupt.

def _on_vision_batch(batch_df):
    update_ml_photo(batch_df, 'vision_label')
    update_ml_photo(batch_df, 'p_building')

df_to_score = flickr_photo_to_vision_score(skip_errors=False)
if not df_to_score.empty:
    clustering.vision_and_keywords(df_to_score, on_batch=_on_vision_batch)
```

> **Note:** Steps 2 and 3 both call `vision_and_keywords()` but for different purposes.
> Step 2 uses it only for DBSCAN (pass the full unprocessed set, ignore vision output for now).
> Step 3 uses it only for vision scoring (pass only unscored rows via `flickr_photo_to_vision_score`).
> Alternatively, Ghass can split `vision_and_keywords` into two separate functions on request.

---

## 6. Summary of all changes

| File | Change type | Description |
|---|---|---|
| `tables.sql` | Schema addition | 3 new columns on `machine_learning_photo` |
| `src/core/model.py` | Code addition | 3 new `Column(...)` entries in `ml_photo_table` |
| `src/trainer/db.py` | Bug fix | Typo `_psql_insert_signore` → `_psql_insert_ignore` in `save_clusters` |
| `src/trainer/db.py` | Code addition | New function `flickr_photo_to_clip_embed()` |
| `src/trainer/db.py` | Code addition | New function `flickr_photo_to_vision_score(skip_errors)` |
| `src/trainer/db.py` | Code addition | New function `flickr_photo_to_dbscan()` |
