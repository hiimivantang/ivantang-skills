---
name: milvus-sizing
description: Estimate Milvus resource requirements (memory, disk, node count) for a given vector dataset using the official sizing formulas from https://milvus.io/tools/sizing. Use this skill whenever the user asks about Milvus capacity planning, sizing, how much RAM/disk/nodes they need, or how many vectors their cluster can hold. Trigger on phrases like "how much memory for Milvus", "size a Milvus cluster", "Milvus resources for N vectors", "how many nodes", "Milvus capacity". Default to HNSW index and distributed mode unless the user specifies otherwise.
---

# Milvus Sizing

Computes Milvus resource requirements using the official index memory formulas from
https://milvus.io/tools/sizing. Always defaults to **HNSW** index and **distributed** mode.

## Index Memory Formulas

All formulas use:
- `raw_data_size = num_vectors × dim × 4`  (float32, 4 bytes/element)
- `row_size = dim × 4`

| Index        | Memory formula                                              | Disk formula                        |
|--------------|-------------------------------------------------------------|-------------------------------------|
| FLAT         | `raw_data_size`                                             | —                                   |
| HNSW (default) | `(1 + 2×M/dim) × raw_data_size`  (M default 30)          | —                                   |
| IVF_FLAT     | `raw_data_size + nlist × row_size`                          | —                                   |
| IVF_SQ8      | `raw_data_size/4 + nlist × row_size`                        | —                                   |
| IVF_PQ       | `raw_data_size / (dsub×32/nbits) + nlist × row_size`        | —                                   |
| IVF_RABITQ   | `raw × (1+N)/32 + nlist × row_size` (N=8 for SQ8)          | —                                   |
| SCANN        | `(1/8) × raw_data_size` (or `(1/8+1)×raw` with raw data)   | —                                   |
| DISKANN      | `raw_data_size / 4`                                         | `(1 + max_degree/dim) × raw_data_size` |

## Workflow

### 1. Collect Parameters

Ask the user for the required inputs (get both in a single question if not already provided):

- **Number of vectors** — e.g. 10 million, 1B
- **Vector dimension** — e.g. 1536, 768, 3072

Optional (show defaults, ask only if the user seems to care):
- Index type (default: **HNSW**)
- HNSW M parameter (default: 30)
- Replicas (default: 1)
- RAM per query node in GB (default: 64)

### 2. Run the Script

The bundled script handles all formulas and generates a full distributed-mode report.

```bash
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SKILL_DIR/scripts/milvus_sizing.py"

python3 "$SCRIPT" \
  --vectors <num_vectors> \
  --dim <dim> \
  [--index hnsw|flat|ivf_flat|ivf_sq8|ivf_pq|scann|diskann|ivf_rabitq] \
  [--hnsw-m 30] \
  [--replicas 1] \
  [--node-memory-gb 64]
```

No external dependencies — uses only the Python standard library.

### 3. Present Results

Show the full script output to the user. Then add a short plain-English summary:

- Index memory total (with formula breakdown)
- Recommended query node count and per-node RAM
- Object storage estimate (MinIO)
- Link to https://milvus.io/tools/sizing for interactive cross-check

### 4. Sanity-Check Against Formulas

After showing the output, verify the key number manually inline:

```
raw_data_size = num_vectors × dim × 4 bytes
HNSW memory   = (1 + 2×M/dim) × raw_data_size
```

If the user's numbers seem off (e.g. they expect only a few GB for billions of vectors),
flag it and walk through the formula step-by-step.

## Notes

- **Milvus sizing tool has no public API** — https://milvus.io/tools/sizing is a JS app.
  The script implements the same formulas so results should match closely.
- **HNSW is fully in-memory** — no mmap fallback for HNSW. The entire graph must fit in RAM.
- **DiskANN** is the right choice when index memory exceeds available RAM; it uses ~25 % of
  raw size in RAM and stores the full graph on local NVMe.
- **Replicas** multiply query-node memory requirements; 1 replica (default) means the index
  is sharded across query nodes.
- **Growing segments**: the script adds a 20 % headroom on top of the index size for
  streaming data buffered in memory before being sealed and offloaded to object storage.
- For datasets larger than 700 M vectors or multiple vector fields, contact the Milvus team
  directly — the sizing tool (and this skill) are designed for single-field scenarios.
