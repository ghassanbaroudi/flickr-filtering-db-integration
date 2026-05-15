# flickr-filtering

DataFrame-oriented building-image filtering pipeline, designed to slot into
the `flickr-commons-metadata` project alongside Nathanael's DB-backed pipeline.

## Structure

```
flickr-filtering/
├── embedding.py       ← public API: clip(df) → df
├── clustering.py      ← public API: filter(df) → df
│                                    vision_and_keywords(df) → (photos_df, clusters_df)
├── requirements.txt
└── _internal/         ← implementation details (not part of public API)
    ├── clip_runtime.py
    ├── dbscan.py
    ├── image_fetch.py
    └── keywords.py
```

## Quick start

```bash
pip install -r requirements.txt
```

## Usage example

```python
import pandas as pd
import embedding
import clustering

# 1. Load photos from DB (Nathanael's db.py)
# df = flickr_photo()                   # full ~1M row table from `photo`

# 2. Keep only building-related rows with valid geo
filtered = clustering.filter(df)

# 3. Compute OpenCLIP embeddings (adds clip_vect_224 column)
embedded = embedding.clip(filtered)

# 4. DBSCAN clustering + vision scoring
photos_out, clusters_out = clustering.vision_and_keywords(embedded)

# photos_out columns added:
#   geo_cluster_id, vision_label, p_building, vision_reason
#
# clusters_out columns (matches geo_cluster table):
#   id, min_latitude, avg_latitude, max_latitude,
#   min_longitude, avg_longitude, max_longitude
```

## Required DB schema additions

```sql
-- machine_learning_photo table
ALTER TABLE machine_learning_photo
    ADD COLUMN clip_vect_224  VECTOR(512),      -- L2-normalised OpenCLIP ViT-B-32 embedding
    ADD COLUMN vision_label   TEXT,             -- "YES" / "NO" / "ERROR"
    ADD COLUMN p_building     DOUBLE PRECISION; -- softmax P(class0=building), range [0, 1]
```

`geo_cluster_id` and the `geo_cluster` table already exist — no changes needed there.

## Configuration via environment variables

| Variable              | Default                | Description                      |
|-----------------------|------------------------|----------------------------------|
| `OPENCLIP_MODEL`      | `ViT-B-32`             | OpenCLIP architecture            |
| `OPENCLIP_PRETRAINED` | `laion2b_s34b_b79k`    | Checkpoint tag                   |
