# flickr-filtering

DataFrame-oriented building-image filtering pipeline, designed to slot into
Nathanael's DB-backed `flickr-commons-metadata` project.

For step-by-step DB schema, SQLAlchemy, and orchestration code, see
[NATHANAEL_INTEGRATION.md](NATHANAEL_INTEGRATION.md).

## Structure

```
flickr-filtering/
├── embedding.py       ← public API: clip(df, cache) → df
├── clustering.py      ← public API: label_buildings(df, cache) → df
│                                    cluster(df) → photos_df
│                                    _cluster(df) → (photos_df, clusters_df)
│                                    _filter(df) → df   (geo + keyword pre-filter)
│                                    vision_and_keywords(df) → wrapper (legacy order)
├── requirements.txt
└── _internal/         ← implementation details (not part of public API)
    ├── clip_runtime.py
    ├── clip_vision.py
    ├── dbscan.py
    ├── image_fetch.py
    └── keywords.py
```

Install as an editable package:

```bash
pip install -e /path/to/flickr-filtering
# or, from inside this folder:
pip install -e .
```

```python
import flickr_filtering.embedding as embedding
import flickr_filtering.clustering as clustering
```

## Pipeline order

**Recommended order (matches production use):**

1. **Embed** — `embedding.clip()` adds `clip_vect_224` (slow, resumable)
2. **Pre-filter** — `clustering._filter()` keeps rows with valid geo + building keywords
3. **Label** — `clustering.label_buildings()` adds `is_building` / `p_building` (slow, resumable)
4. **Cluster** — `clustering._cluster()` on confirmed buildings only (`is_building = True`); fast, no network

Label **before** cluster. DBSCAN groups geographically close **buildings**; running cluster before vision would include non-building photos.

`vision_and_keywords()` is a legacy convenience wrapper (cluster then label). Prefer the explicit order above for Nathanael's pipeline.

## Rate limits and resume

Image downloads hit Flickr directly. On **429 / 503** or other failures, the row is left unscored (`is_building = NULL`, `clip_vect_224 = NULL`) — there is no text label column and no separate error state.

**Resume model (DB-backed runs):**

- Pass only rows where the target column is `NULL`.
- Use `on_batch` to commit each successful row immediately.
- Re-run the same step after a cooldown; failed rows are retried automatically.

No sleeping or blocking inside the library — progress across rate limits is **manual reruns** on Nathanael's side.

## Usage example (DataFrames)

You must provide a **cache** object with a `.get(url)` method that returns a PIL RGB image or `None` (Nathanael's project supplies this for proxy / download handling).

```python
import flickr_filtering.embedding as embedding
import flickr_filtering.clustering as clustering

# df = load from DB ...

# 1. OpenCLIP embeddings (resumable: pass rows WHERE clip_vect_224 IS NULL)
embedded = embedding.clip(df, cache, on_batch=_on_clip_batch)

# 2. Geo + keyword pre-filter (required before vision scoring)
candidates = clustering._filter(embedded)

# 3. Building labeling (resumable: pass rows WHERE is_building IS NULL)
labeled = clustering.label_buildings(
    candidates,
    cache,
    url_column="url_o",
    on_batch=_on_vision_batch,
)

# 4. DBSCAN on confirmed buildings only (run once when labeling is complete)
buildings = labeled[labeled["is_building"] == True]
photos_out, clusters_out = clustering._cluster(buildings)

# photos_out  → geo_cluster_id per photo
# clusters_out → rows for geo_cluster table:
#   id, min_latitude, avg_latitude, max_latitude,
#   min_longitude, avg_longitude, max_longitude
```

**Vision output columns:**

| Column | Type | Meaning |
|--------|------|---------|
| `is_building` | `bool` or `None` | `True` = building, `False` = not, `None` = not yet scored / download failed |
| `p_building` | `float` or `None` | Softmax P(class 0 = building), range [0, 1] |

There is no `vision_label` text field — only these booleans and probabilities.

## Required DB schema additions

```sql
ALTER TABLE machine_learning_photo
    ADD COLUMN clip_vect_224  VECTOR(512),      -- L2-normalised OpenCLIP ViT-B-32 embedding
    ADD COLUMN is_building    BOOLEAN,          -- True / False / NULL (NULL = retry)
    ADD COLUMN p_building     DOUBLE PRECISION; -- softmax P(class 0 = building)
```

`geo_cluster_id` and the `geo_cluster` table already exist — no changes needed there.

## Configuration via environment variables

| Variable              | Default                | Description           |
|-----------------------|------------------------|-----------------------|
| `OPENCLIP_MODEL`      | `ViT-B-32`             | OpenCLIP architecture |
| `OPENCLIP_PRETRAINED` | `laion2b_s34b_b79k`    | Checkpoint tag        |
