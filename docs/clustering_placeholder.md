# Clustering-Based Bucketing Placeholder

Current MVP uses manual `bucket_id` values (e.g., `family`, `plans`, `decline_polite`).  
Future versions can auto-cluster memory chunks and phrase exemplars.

## Proposed approach
1. Generate embeddings for every LTM/STM/PB chunk.
2. Cluster vectors with KMeans (fixed K) or HDBSCAN (adaptive).
3. Extract representative phrases and top n-grams to name each cluster.
4. Store `cluster_id` next to existing manual `bucket_id`.
5. Update retrieval planner to prioritize `cluster_id` + recency/weight.

## Why this helps
- Scales beyond manually maintained tags.
- Adapts to personal speech drift over time.
- Improves fallback retrieval when explicit bucket tags are sparse.

## Code placeholder
See `home/aac/memory/clustering_placeholder.py`.
