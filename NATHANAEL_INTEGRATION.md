# Integration requirements for `flico-nathanael`

This document lists every change needed in `flico-nathanael` to support the
`flickr-filtering` pipeline. Nothing here changes my code — all items below
are additions or fixes on your side.

---

## 0. Install the package

Clone / copy the `flickr-filtering` folder somewhere on your machine, then install it as an editable package **into your project's virtual environment**:

```bash
pip install -e /path/to/flickr-filtering
```

After that, imports work from anywhere on any machine without touching `sys.path`:

```python
import flickr_filtering.embedding as embedding
import flickr_filtering.clustering as clustering
```

---

## 1. Database schema — `tables.sql`

### 1a. New columns on `machine_learning_photo`

```sql
ALTER TABLE machine_learning_photo
    ADD COLUMN clip_vect_224  VECTOR(512),      -- OpenCLIP ViT-B-32 L2-normalised embedding
    ADD COLUMN is_building    BOOLEAN,          -- True = building, False = not a building, NULL = not yet scored (retry)
    ADD COLUMN p_building     DOUBLE PRECISION; -- softmax P(class-0 = building), range [0.0, 1.0]
```

> `geo_cluster_id` already exists — no change needed there.  
> `geo_cluster` table already exists — no change needed there.

### 1b. Why these types

| Column | Type | Reason |
|---|---|---|
| `clip_vect_224` | `VECTOR(512)` | ViT-B-32 outputs 512-dimensional embeddings. Named `_224` to reflect the 224×224 input resolution, consistent with your `sig_lip_vect_n` (`_n` = 320px) convention. |
| `is_building` | `BOOLEAN` | `True` = building, `False` = not a building, `NULL` = download failed or not yet processed. NULL rows are always retried on the next run — no separate error state needed. |
| `p_building` | `DOUBLE PRECISION` | Softmax probability for the building-positive class. Useful for ranking or threshold tuning later. |

---

## 2. SQLAlchemy model — `src/core/model.py`

Add the three new columns to `ml_photo_table`:

```python
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Boolean, Float  # add Boolean, Float to existing imports

ml_photo_table = Table(
    "machine_learning_photo",
    metadata,
    # ... existing columns unchanged ...

    Column("clip_vect_224", VECTOR(512)),   # ADD
    Column("is_building",   Boolean),       # ADD
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

### 4b. `flickr_photo_to_vision_score()`

Returns photos that still need vision scoring. `NULL` means either never attempted or the
download failed last run — both cases are retried automatically. There is no separate error
state: failed rows simply stay `NULL` and are picked up on the next run.

```python
def flickr_photo_to_vision_score() -> pd.DataFrame:
    """
    Photos that need vision scoring.
    NULL = never scored OR download failed last run (retried automatically).
    Only returns rows that already have a geo_cluster_id (DBSCAN must run first).
    """
    query = text("""--sql
        SELECT * FROM photo AS P
        JOIN machine_learning_photo AS MLP
        ON P.owner_nsid = MLP.owner_nsid AND P.id = MLP.id
        WHERE MLP.geo_cluster_id IS NOT NULL
        AND MLP.is_building IS NULL
    """)
    df = pd.read_sql_query(query, get_engine("trainer"))
    return df.loc[:, ~df.columns.duplicated()]
```

### 4c. `flickr_photo_to_cluster()`

Returns confirmed building photos that have not yet been assigned a cluster.
**Run only after `label_buildings` is complete** — clustering groups buildings that
are geographically close, so it only makes sense on rows where `is_building = TRUE`.
DBSCAN must receive all unprocessed rows at once (not resumable).

```python
def flickr_photo_to_cluster() -> pd.DataFrame:
    """
    Confirmed building photos that still need a geo_cluster_id.
    Only call once label_buildings is complete for the full dataset.
    """
    query = text("""--sql
        SELECT * FROM photo AS P
        JOIN machine_learning_photo AS MLP
        ON P.owner_nsid = MLP.owner_nsid AND P.id = MLP.id
        WHERE MLP.is_building = TRUE
        AND MLP.geo_cluster_id IS NULL
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

**Order matters:** label first (slow, resumable), cluster second (fast, one-shot on buildings only).

```python
import flickr_filtering.embedding as embedding
import flickr_filtering.clustering as clustering
from src.trainer.db import (
    flickr_photo_to_clip_embed,
    flickr_photo_to_vision_score,
    flickr_photo_to_cluster,
    save_clusters,
    update_ml_photo,
)

# ── Step 1: OpenCLIP embeddings ──────────────────────────────────────────────
# Re-run freely. Each successfully embedded image is committed immediately.
# Rate-limited / failed rows stay NULL and are retried on next run.

def _on_clip_batch(batch_df):
    update_ml_photo(batch_df, 'clip_vect_224')

df_to_embed = flickr_photo_to_clip_embed()
if not df_to_embed.empty:
    embedding.clip(df_to_embed, on_batch=_on_clip_batch)

# ── Step 2: Keyword + geo pre-filter ─────────────────────────────────────────
# CRITICAL: must run before label_buildings. This is what makes our results match.
# Keeps only rows with valid geo AND at least one building keyword in title/description/tags.
# Without this step, non-building photos enter the vision scorer and pollute the clusters.
#
# The url_column must match whatever column name holds the original image URL in your DB.
# In our pipeline it is "image_url" — check your photo table and pass the right name.

df_to_score = flickr_photo_to_vision_score()  # WHERE is_building IS NULL
if not df_to_score.empty:
    df_to_score = clustering._filter(df_to_score)  # geo + keyword filter

# ── Step 3: Building labeling ─────────────────────────────────────────────────
# Re-run freely until complete. Each successfully scored image is committed immediately.
# On 429 rate-limit, the download returns None instantly (no sleeping/blocking).
# Re-run after a cooldown — only NULL rows are fetched each time.
# ⚠ url_column: change "url_o" to whatever your photo table calls the image URL column.

def _on_vision_batch(batch_df):
    update_ml_photo(batch_df, 'is_building')
    update_ml_photo(batch_df, 'p_building')

if not df_to_score.empty:
    clustering.label_buildings(df_to_score, url_column="url_o", on_batch=_on_vision_batch)

# ── Step 3: DBSCAN clustering ─────────────────────────────────────────────────
# Run ONCE after labeling is complete, on confirmed buildings only (is_building = TRUE).
# Fast — no network I/O. Write cluster rows first (FK constraint).

df_to_cluster = flickr_photo_to_cluster()
if not df_to_cluster.empty:
    photos_clustered, clusters_df = clustering.cluster(df_to_cluster)
    save_clusters(clusters_df)
    update_ml_photo(
        photos_clustered[['owner_nsid', 'id', 'geo_cluster_id']],
        'geo_cluster_id',
    )
```

> `cluster()` and `label_buildings()` are fully independent and can be called separately.
> `vision_and_keywords()` is kept as a convenience wrapper that calls both in sequence.

---

## 6. Summary of all changes

| File | Change type | Description |
|---|---|---|
| `tables.sql` | Schema addition | 3 new columns on `machine_learning_photo` |
| `src/core/model.py` | Code addition | 3 new `Column(...)` entries in `ml_photo_table` |
| `src/trainer/db.py` | Bug fix | Typo `_psql_insert_signore` → `_psql_insert_ignore` in `save_clusters` |
| `src/trainer/db.py` | Code addition | New function `flickr_photo_to_clip_embed()` |
| `src/trainer/db.py` | Code addition | New function `flickr_photo_to_vision_score()` |
| `src/trainer/db.py` | Code addition | New function `flickr_photo_to_cluster()` |
