---
name: milvus-sizing
description: Estimate Milvus resource requirements (memory, disk, node count) for a given vector dataset using the official sizing formulas from https://milvus.io/tools/sizing. Use this skill whenever the user asks about Milvus capacity planning, sizing, how much RAM/disk/nodes they need, or how many vectors their cluster can hold. Trigger on phrases like "how much memory for Milvus", "size a Milvus cluster", "Milvus resources for N vectors", "how many nodes", "Milvus capacity". Default to HNSW index and distributed mode unless the user specifies otherwise.
---

# Milvus Sizing

Computes Milvus resource requirements using the official index memory formulas from
https://milvus.io/tools/sizing. Always defaults to **HNSW** index and **distributed** mode.

## Index Memory Formulas

Sourced directly from `sizingTool.ts` in [milvus-io/milvus.io](https://github.com/milvus-io/milvus.io).

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
| SCANN        | `(9/8) × raw_data_size` or `(1/8) × raw_data_size`          | —                                   |
| DISKANN      | `raw_data_size / 4`                                         | `(1 + max_degree/dim) × raw_data_size` |

### Loading memory (applied after index memory)

```
loading_memory = (index_memory + segment_size × 2) × 1.15   # non-DiskANN
loading_memory = index_memory × 1.15                         # DiskANN
```

Default segment size: 512 MiB. This matches `vectorLoadingMemory` in `sizingTool.ts`.

### Distributed node tiers (`clusterNodesConfigCalculator`)

| Loading memory | Query Nodes | per QN spec | Data Nodes |
|----------------|-------------|-------------|------------|
| ≤ 8 GiB        | 1           | 2 vCPU / 8 GiB  | 1 |
| ≤ 16 GiB       | 1           | 4 vCPU / 16 GiB | 1 |
| ≤ 32 GiB       | 2           | 4 vCPU / 16 GiB | 2 |
| ≤ 64 GiB       | 4           | 4 vCPU / 16 GiB | 2 |
| ≤ 96 GiB       | 6           | 4 vCPU / 16 GiB | 4 |
| ≤ 512 GiB      | ⌈GiB/32⌉   | 8 vCPU / 32 GiB | max(2, QN//4) |
| ≤ 2048 GiB     | ⌈GiB/64⌉   | 16 vCPU / 64 GiB | max(2, QN//4) |
| > 2048 GiB     | ⌈GiB/128⌉  | 32 vCPU / 128 GiB | max(2, QN//4) |

### Dependency sizing

| Component | Formula |
|-----------|---------|
| MinIO PVC | `max(⌈(raw + loading) GiB⌉, 30 GiB)` |
| Pulsar Ledgers | `max(⌈raw GiB⌉, 20 GiB)` |
| Pulsar Journal | `min(⌈raw GiB⌉ × 0.5, 50 GiB)` |
| etcd (×3 HA) | 8 GiB SSD per node |

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
  [--segment-size-mb 512|1024|2048]
```

Add `--json` to get machine-readable output. No external dependencies.

### 2b. Generate a Visual HTML Screenshot (optional)

To produce a visual report matching the Milvus sizing tool layout:

```bash
python3 "$SKILL_DIR/scripts/generate_sizing_screenshot.py" \
  --vectors <num_vectors> \
  --dim <dim> \
  --out /tmp/milvus_sizing.png
```

Requires Playwright + Chromium (available in this environment at `/opt/pw-browsers`).
Example output screenshots are in `examples/`.

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

- **Formulas are verified against source** — `sizingTool.ts` from `milvus-io/milvus.io` was
  read directly. Results match `https://milvus.io/tools/sizing` (which has no public API).
- **HNSW is fully in-memory** — no mmap fallback. The entire graph must fit in RAM across
  query nodes.
- **DiskANN** is the right choice when index memory exceeds available RAM; it uses ~25 % of
  raw size in RAM and stores the full graph on local NVMe.
- **Loading overhead** = `(index_memory + 2×segment_size) × 1.15` — the extra `2×segment`
  accounts for growing (unsealed) segments buffered in memory during ingestion.
- **MinIO minimum is 30 GiB** — the tool enforces this floor even for tiny datasets.
- **Pulsar Journal caps at 50 GiB** — regardless of data size.
- For datasets larger than 700 M vectors or multiple vector fields, contact the Milvus team
  directly — the sizing tool (and this skill) are designed for single-field scenarios.
