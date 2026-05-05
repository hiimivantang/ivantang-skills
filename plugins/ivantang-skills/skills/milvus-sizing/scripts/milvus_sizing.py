#!/usr/bin/env python3
"""
Milvus sizing calculator — bundled script for the milvus-sizing skill.

Formulas sourced from https://milvus.io/tools/sizing (official Milvus sizing tool).
Reference: Vector Index Size Estimation (internal Milvus doc, Feb 2025).

Usage:
  python milvus_sizing.py --vectors 1000000 --dim 1536
  python milvus_sizing.py --vectors 1e9 --dim 768 --hnsw-m 16 --node-memory-gb 128
  python milvus_sizing.py --vectors 5e6 --dim 1536 --index diskann
"""

import argparse
import math
import sys

BYTES_PER_FLOAT32 = 4
BYTES_PER_GB = 1024 ** 3


def fmt(byte_count):
    """Human-readable byte count."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(byte_count) < 1024.0:
            return f"{byte_count:.2f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.2f} EB"


def raw_data_size(num_vectors, dim):
    return num_vectors * dim * BYTES_PER_FLOAT32


# ---------------------------------------------------------------------------
# Index memory formulas (Disable mmap / fully in-memory column)
# Source: Vector Index Size Estimation table
# ---------------------------------------------------------------------------

def hnsw_memory(num_vectors, dim, M=30):
    # (1 + 2*M/dim) * raw_data_size
    return (1 + 2 * M / dim) * raw_data_size(num_vectors, dim)


def flat_memory(num_vectors, dim):
    # raw_data_size
    return raw_data_size(num_vectors, dim)


def ivf_flat_memory(num_vectors, dim, nlist=128):
    # raw_data_size + nlist * row_size  (row_size = dim * sizeof(float32))
    row_size = dim * BYTES_PER_FLOAT32
    return raw_data_size(num_vectors, dim) + nlist * row_size


def ivf_sq8_memory(num_vectors, dim, nlist=128):
    # raw_data_size/4 + nlist * row_size
    row_size = dim * BYTES_PER_FLOAT32
    return raw_data_size(num_vectors, dim) / 4 + nlist * row_size


def ivf_pq_memory(num_vectors, dim, nlist=128, m=None, nbits=8):
    # raw_data_size / (dsub * 32 / nbits) + nlist * row_size
    # dsub = dim / m  (m is the number of sub-quantizers, default dim/8 or dim/4)
    if m is None:
        m = max(1, dim // 8)
    dsub = dim / m
    row_size = dim * BYTES_PER_FLOAT32
    return raw_data_size(num_vectors, dim) / (dsub * 32 / nbits) + nlist * row_size


def scann_memory(num_vectors, dim, with_raw_data=False):
    # with_raw_data=False: (1/8) * raw_data_size
    # with_raw_data=True:  (1/8 + 1) * raw_data_size
    raw = raw_data_size(num_vectors, dim)
    return (1 / 8 + (1 if with_raw_data else 0)) * raw


def diskann_sizing(num_vectors, dim, max_degree=56):
    # Memory: raw_data_size / 4
    # Disk:   (1 + max_degree/dim) * raw_data_size
    raw = raw_data_size(num_vectors, dim)
    return raw / 4, (1 + max_degree / dim) * raw


def ivf_rabitq_memory(num_vectors, dim, nlist=128, refine_type="SQ8"):
    # raw_data_size * (1/32 + N/32) + nlist * row_size
    # N: SQ6→6, SQ8→8, FP16/BF16→16, FP32→32
    N_map = {"SQ6": 6, "SQ8": 8, "FP16": 16, "BF16": 16, "FP32": 32}
    N = N_map.get(refine_type.upper(), 8)
    raw = raw_data_size(num_vectors, dim)
    row_size = dim * BYTES_PER_FLOAT32
    return raw * (1 + N) / 32 + nlist * row_size


# ---------------------------------------------------------------------------
# Distributed mode node recommendations
# ---------------------------------------------------------------------------

def _next_power_of_two(n):
    return 2 ** math.ceil(math.log2(max(1, n)))


def distributed_recommendations(index_memory_bytes, raw_bytes, index_disk_bytes=0, node_memory_gb=64, replicas=1):
    """
    Estimate Milvus distributed mode resource requirements.

    Query nodes must hold the full index (× replicas) in memory.
    A 20 % headroom is added for growing segments and OS/process overhead.
    """
    node_mem = node_memory_gb * BYTES_PER_GB

    # Query nodes: replicated index + 20 % overhead
    effective = index_memory_bytes * replicas * 1.2
    num_qn = max(2, math.ceil(effective / node_mem))
    per_qn_gb = math.ceil(effective / num_qn / BYTES_PER_GB)
    per_qn_gb = _next_power_of_two(max(8, per_qn_gb))

    # Data nodes: handle ingestion (growing segments, WAL replay)
    # Typical: 2 nodes, 16 GB each covers most workloads up to ~100 M vectors/day
    num_dn = 2
    dn_mem_gb = 16

    # Index nodes: CPU-intensive index building
    num_in = 1
    in_mem_gb = max(32, per_qn_gb)
    in_cpu = 16

    # Object storage (MinIO / S3):
    #   sealed segment raw data stored in columnar parquet ≈ 60 % of raw float32
    #   index files are also persisted (same size as in-memory index)
    minio_bytes = raw_bytes * 0.6 + index_memory_bytes

    # etcd: metadata (segments, collections, partitions)
    # Very small compared to data; 10 GB minimum, grows with segment count
    num_segments = max(1, math.ceil(raw_bytes / (512 * 1024 * 1024)))
    etcd_gb = max(10, math.ceil(num_segments * 0.001))  # ~1 MB per 1 000 segments

    return {
        "num_query_nodes": num_qn,
        "per_query_node_memory_gb": per_qn_gb,
        "query_node_cpu": 16,
        "num_data_nodes": num_dn,
        "data_node_memory_gb": dn_mem_gb,
        "data_node_cpu": 8,
        "num_index_nodes": num_in,
        "index_node_memory_gb": in_mem_gb,
        "index_node_cpu": in_cpu,
        "coordinator_memory_gb": 8,
        "coordinator_cpu": 4,
        "minio_bytes": minio_bytes,
        "etcd_gb": etcd_gb,
        "index_disk_bytes": index_disk_bytes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Milvus resource sizing calculator (distributed mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vectors", type=float, required=True,
                        help="Number of vectors, e.g. 1000000 or 1e6")
    parser.add_argument("--dim", type=int, required=True,
                        help="Vector dimension, e.g. 1536")
    parser.add_argument("--index", type=str, default="hnsw",
                        choices=["hnsw", "flat", "ivf_flat", "ivf_sq8", "ivf_pq",
                                 "scann", "diskann", "ivf_rabitq"],
                        help="Index type (default: hnsw)")
    parser.add_argument("--hnsw-m", type=int, default=30,
                        help="HNSW M parameter, range [2, 2048] (default: 30)")
    parser.add_argument("--nlist", type=int, default=128,
                        help="IVF nlist, range [1, 65536] (default: 128)")
    parser.add_argument("--ivfpq-m", type=int, default=None,
                        help="IVF_PQ number of sub-quantizers m (default: dim/8)")
    parser.add_argument("--ivfpq-nbits", type=int, default=8,
                        help="IVF_PQ nbits per sub-quantizer (default: 8)")
    parser.add_argument("--diskann-max-degree", type=int, default=56,
                        help="DiskANN max_degree, range [1, 2048] (default: 56)")
    parser.add_argument("--scann-with-raw-data", action="store_true",
                        help="SCANN with_raw_data=true (default: false)")
    parser.add_argument("--rabitq-refine-type", type=str, default="SQ8",
                        choices=["SQ6", "SQ8", "FP16", "BF16", "FP32"],
                        help="IVF_RABITQ refine_type (default: SQ8)")
    parser.add_argument("--replicas", type=int, default=1,
                        help="Number of data replicas (default: 1)")
    parser.add_argument("--node-memory-gb", type=int, default=64,
                        help="RAM per query node in GB (default: 64)")
    args = parser.parse_args()

    num_vectors = int(args.vectors)
    dim = args.dim
    idx = args.index.lower()

    raw = raw_data_size(num_vectors, dim)
    index_mem = 0
    index_disk = 0
    formula_str = ""
    params_str = ""

    if idx == "hnsw":
        M = args.hnsw_m
        index_mem = hnsw_memory(num_vectors, dim, M)
        formula_str = f"(1 + 2×{M}/{dim}) × raw_data_size"
        params_str = f"M={M}"
    elif idx == "flat":
        index_mem = flat_memory(num_vectors, dim)
        formula_str = "raw_data_size"
    elif idx == "ivf_flat":
        nlist = args.nlist
        index_mem = ivf_flat_memory(num_vectors, dim, nlist)
        formula_str = "raw_data_size + nlist × row_size"
        params_str = f"nlist={nlist}"
    elif idx == "ivf_sq8":
        nlist = args.nlist
        index_mem = ivf_sq8_memory(num_vectors, dim, nlist)
        formula_str = "raw_data_size/4 + nlist × row_size"
        params_str = f"nlist={nlist}"
    elif idx == "ivf_pq":
        nlist = args.nlist
        m = args.ivfpq_m or max(1, dim // 8)
        nbits = args.ivfpq_nbits
        index_mem = ivf_pq_memory(num_vectors, dim, nlist, m, nbits)
        dsub = dim / m
        formula_str = f"raw_data_size / ({dsub:.1f} × 32/{nbits}) + nlist × row_size"
        params_str = f"nlist={nlist}, m={m}, nbits={nbits}"
    elif idx == "scann":
        wr = args.scann_with_raw_data
        index_mem = scann_memory(num_vectors, dim, wr)
        formula_str = "(1/8 + 1) × raw_data_size" if wr else "(1/8) × raw_data_size"
        params_str = f"with_raw_data={str(wr).lower()}"
    elif idx == "diskann":
        md = args.diskann_max_degree
        index_mem, index_disk = diskann_sizing(num_vectors, dim, md)
        formula_str = "mem: raw/4  |  disk: (1 + max_degree/dim) × raw"
        params_str = f"max_degree={md}"
    elif idx == "ivf_rabitq":
        nlist = args.nlist
        rt = args.rabitq_refine_type
        index_mem = ivf_rabitq_memory(num_vectors, dim, nlist, rt)
        N_map = {"SQ6": 6, "SQ8": 8, "FP16": 16, "BF16": 16, "FP32": 32}
        N = N_map[rt.upper()]
        formula_str = f"raw × (1+{N})/32 + nlist × row_size"
        params_str = f"nlist={nlist}, refine_type={rt}"

    recs = distributed_recommendations(
        index_mem, raw, index_disk, args.node_memory_gb, args.replicas
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    SEP = "=" * 62

    print(f"\n{SEP}")
    print(f"  MILVUS SIZING — DISTRIBUTED MODE")
    print(SEP)
    print(f"  Vectors      : {num_vectors:>15,}")
    print(f"  Dimensions   : {dim:>15,}")
    print(f"  Index        : {idx.upper()}" + (f"  ({params_str})" if params_str else ""))
    print(f"  Replicas     : {args.replicas}")
    print(f"  Node memory  : {args.node_memory_gb} GB (query nodes)")
    print(SEP)

    print(f"\n  RAW DATA")
    print(f"    Size  : {fmt(raw)}")
    print(f"    ({num_vectors:,} vectors × {dim} dims × {BYTES_PER_FLOAT32} bytes/float32)")

    print(f"\n  INDEX MEMORY")
    print(f"    Formula : {formula_str}")
    print(f"    Size    : {fmt(index_mem)}", end="")
    if args.replicas > 1:
        print(f"  ×{args.replicas} replicas = {fmt(index_mem * args.replicas)}", end="")
    print()
    if index_disk:
        print(f"    Disk    : {fmt(index_disk)}")

    print(f"\n  DISTRIBUTED COMPONENTS")
    print(f"\n    Query Nodes  ×{recs['num_query_nodes']}")
    print(f"      RAM : {recs['per_query_node_memory_gb']} GB each")
    print(f"      CPU : {recs['query_node_cpu']} vCPU each")

    print(f"\n    Data Nodes   ×{recs['num_data_nodes']}")
    print(f"      RAM : {recs['data_node_memory_gb']} GB each")
    print(f"      CPU : {recs['data_node_cpu']} vCPU each")

    print(f"\n    Index Nodes  ×{recs['num_index_nodes']}")
    print(f"      RAM : {recs['index_node_memory_gb']} GB each")
    print(f"      CPU : {recs['index_node_cpu']} vCPU each")

    print(f"\n    Coordinators ×1  (×3 for HA)")
    print(f"      RAM : {recs['coordinator_memory_gb']} GB")
    print(f"      CPU : {recs['coordinator_cpu']} vCPU")

    if recs["index_disk_bytes"]:
        per_qn = recs["index_disk_bytes"] / recs["num_query_nodes"]
        print(f"\n    Local Disk (query nodes, DiskANN)")
        print(f"      {fmt(per_qn)} per node")

    print(f"\n  DEPENDENCIES")
    print(f"\n    MinIO / Object Storage")
    print(f"      Total : {fmt(recs['minio_bytes'])}")
    print(f"      (sealed segment parquet ~60 % compressed + index files)")

    print(f"\n    etcd  (3 nodes for HA)")
    print(f"      Disk per node : {recs['etcd_gb']} GB SSD")

    total_qn_ram = recs["num_query_nodes"] * recs["per_query_node_memory_gb"]
    total_dn_ram = recs["num_data_nodes"] * recs["data_node_memory_gb"]
    total_in_ram = recs["num_index_nodes"] * recs["index_node_memory_gb"]
    total_ram_gb = total_qn_ram + total_dn_ram + total_in_ram + recs["coordinator_memory_gb"]
    total_nodes = recs["num_query_nodes"] + recs["num_data_nodes"] + recs["num_index_nodes"] + 1

    print(f"\n  TOTALS")
    print(f"    Milvus nodes      : {total_nodes}  (+ 3 etcd + MinIO cluster)")
    print(f"    Total RAM         : ~{total_ram_gb} GB  (Milvus nodes)")
    print(f"    Object storage    : {fmt(recs['minio_bytes'])}")

    print(f"\n  VERIFY")
    print(f"    https://milvus.io/tools/sizing")
    print(f"    (Select: HNSW, Distributed, {num_vectors:,} vectors, {dim} dims)")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
