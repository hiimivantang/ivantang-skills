#!/usr/bin/env python3
"""
Milvus sizing calculator — bundled script for the milvus-sizing skill.

Formulas match the official Milvus sizing tool source code:
https://github.com/milvus-io/milvus.io/blob/master/src/utils/sizingTool.ts

Usage:
  python milvus_sizing.py --vectors 1000000 --dim 1536
  python milvus_sizing.py --vectors 1e9 --dim 768 --hnsw-m 16
  python milvus_sizing.py --vectors 5e6 --dim 1536 --index diskann
  python milvus_sizing.py --vectors 1e6 --dim 1536 --json   # machine-readable
"""

import argparse
import json
import math

BYTES_PER_FLOAT32 = 4
GiB = 1024 ** 3
DEFAULT_SEGMENT_GiB = 0.5  # 512 MiB default segment size


def fmt(byte_count):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(byte_count) < 1024.0:
            return f"{byte_count:.2f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.2f} EiB"


# ---------------------------------------------------------------------------
# Raw data size
# ---------------------------------------------------------------------------

def raw_data_size(num_vectors, dim):
    """vectorRawDataSize = num * d * 4  (float32)"""
    return num_vectors * dim * BYTES_PER_FLOAT32


# ---------------------------------------------------------------------------
# Index memory formulas
# Source: memoryAndDiskCalculator() in sizingTool.ts
# ---------------------------------------------------------------------------

def hnsw_memory(num_vectors, dim, M=30):
    # (1 + 2*M/d) * rawDataSize
    return (1 + 2 * M / dim) * raw_data_size(num_vectors, dim)


def flat_memory(num_vectors, dim):
    return raw_data_size(num_vectors, dim)


def ivf_flat_memory(num_vectors, dim, nlist=128):
    row_size = dim * BYTES_PER_FLOAT32
    return raw_data_size(num_vectors, dim) + nlist * row_size


def ivf_sq8_memory(num_vectors, dim, nlist=128):
    row_size = dim * BYTES_PER_FLOAT32
    return raw_data_size(num_vectors, dim) / 4 + nlist * row_size


def ivf_pq_memory(num_vectors, dim, nlist=128, m=None, nbits=8):
    if m is None:
        m = max(1, dim // 8)
    dsub = dim / m
    row_size = dim * BYTES_PER_FLOAT32
    return raw_data_size(num_vectors, dim) / (dsub * 32 / nbits) + nlist * row_size


def scann_memory(num_vectors, dim, with_raw_data=False):
    raw = raw_data_size(num_vectors, dim)
    return (9 / 8 if with_raw_data else 1 / 8) * raw


def diskann_sizing(num_vectors, dim, max_degree=56):
    raw = raw_data_size(num_vectors, dim)
    memory = raw / 4
    disk = (1 + max_degree / dim) * raw
    return memory, disk


def ivf_rabitq_memory(num_vectors, dim, nlist=128, refine_type="SQ8"):
    N_map = {"SQ6": 6, "SQ8": 8, "FP16": 16, "BF16": 16, "FP32": 32}
    N = N_map.get(refine_type.upper(), 8)
    raw = raw_data_size(num_vectors, dim)
    row_size = dim * BYTES_PER_FLOAT32
    return raw * (1 + N) / 32 + nlist * row_size


# ---------------------------------------------------------------------------
# Loading memory
# Source: sizingTool.ts — vectorLoadingMemory calculation
# Formula: (indexMemory + segmentSize * 2) * 1.15
# DiskANN:  indexMemory * 1.15  (no segment buffer needed)
# ---------------------------------------------------------------------------

def loading_memory(index_memory_bytes, segment_gib=DEFAULT_SEGMENT_GiB, is_diskann=False):
    if is_diskann:
        return index_memory_bytes * 1.15
    segment_bytes = segment_gib * GiB
    return (index_memory_bytes + segment_bytes * 2) * 1.15


# ---------------------------------------------------------------------------
# Cluster node configuration
# Source: clusterNodesConfigCalculator() in sizingTool.ts
# Key tiers (loading_memory_GiB → node specs):
#   ≤8     : 1 QN (2c/8G),  1 DN (2c/8G),  coord 1c/4G,  proxy 1c/4G
#   ≤16    : 1 QN (4c/16G), 1 DN (4c/16G), coord 2c/8G,  proxy 2c/8G
#   ≤32    : 2 QN (4c/16G), 2 DN (4c/16G), coord 2c/8G,  proxy 2c/8G
#   ≤48    : 3 QN (4c/16G), 2 DN (4c/16G)
#   ≤64    : 4 QN (4c/16G), 2 DN (4c/16G)
#   ≤80    : 5 QN (4c/16G), 4 DN (4c/16G)
#   ≤96    : 6 QN (4c/16G), 4 DN (4c/16G)
#   ≤512   : ceil(GB/32) QN (8c/32G)
#   ≤2048  : ceil(GB/64) QN (16c/64G)
#   >2048  : ceil(GB/128) QN (32c/128G)
# ---------------------------------------------------------------------------

_SMALL_TIERS = [
    # (max_gib, qn, qn_cpu, qn_mem, dn, dn_cpu, dn_mem, coord_cpu, coord_mem, proxy_cpu, proxy_mem)
    (8,   1, 2,  8,  1, 2,  8,  1, 4, 1, 4),
    (16,  1, 4, 16,  1, 4, 16,  2, 8, 2, 8),
    (32,  2, 4, 16,  2, 4, 16,  2, 8, 2, 8),
    (48,  3, 4, 16,  2, 4, 16,  2, 8, 2, 8),
    (64,  4, 4, 16,  2, 4, 16,  2, 8, 2, 8),
    (80,  5, 4, 16,  4, 4, 16,  2, 8, 2, 8),
    (96,  6, 4, 16,  4, 4, 16,  2, 8, 2, 8),
]


def cluster_node_config(loading_mem_bytes):
    gib = loading_mem_bytes / GiB
    for max_gib, qn, qn_cpu, qn_mem, dn, dn_cpu, dn_mem, coord_cpu, coord_mem, proxy_cpu, proxy_mem in _SMALL_TIERS:
        if gib <= max_gib:
            return dict(qn=qn, qn_cpu=qn_cpu, qn_mem=qn_mem,
                        dn=dn, dn_cpu=dn_cpu, dn_mem=dn_mem,
                        coord_cpu=coord_cpu, coord_mem=coord_mem,
                        proxy_cpu=proxy_cpu, proxy_mem=proxy_mem)

    if gib <= 512:
        n = math.ceil(gib / 32)
        qn_cpu, qn_mem = 8, 32
    elif gib <= 2048:
        n = math.ceil(gib / 64)
        qn_cpu, qn_mem = 16, 64
    else:
        n = math.ceil(gib / 128)
        qn_cpu, qn_mem = 32, 128

    dn = max(2, n // 4)
    dn_cpu, dn_mem = qn_cpu // 2, qn_mem // 2
    coord_cpu, coord_mem = min(8, qn_cpu), min(32, qn_mem)
    proxy_cpu, proxy_mem = min(8, qn_cpu), min(32, qn_mem)
    return dict(qn=n, qn_cpu=qn_cpu, qn_mem=qn_mem,
                dn=dn, dn_cpu=dn_cpu, dn_mem=dn_mem,
                coord_cpu=coord_cpu, coord_mem=coord_mem,
                proxy_cpu=proxy_cpu, proxy_mem=proxy_mem)


# ---------------------------------------------------------------------------
# Dependency sizing
# Source: dependencyCalculator() in sizingTool.ts
# ---------------------------------------------------------------------------

def dependency_sizing(raw_bytes, loading_mem_bytes):
    raw_gib = raw_bytes / GiB
    # MinIO PVC: max(ceil(rawDataSize + loadingMemory in GiB), 30)
    minio_gib = max(math.ceil((raw_bytes + loading_mem_bytes) / GiB), 30)
    # Pulsar ledgers: max(ceil(rawDataSize in GiB), 20)
    pulsar_ledgers_gib = max(math.ceil(raw_gib), 20)
    # Pulsar journal: min(ceil(rawDataSize in GiB) * 0.5, 50)
    pulsar_journal_gib = min(math.ceil(raw_gib) * 0.5, 50)
    # etcd: small metadata store — 8 GiB recommended
    etcd_gib = 8
    return dict(
        minio_gib=minio_gib,
        pulsar_ledgers_gib=pulsar_ledgers_gib,
        pulsar_journal_gib=pulsar_journal_gib,
        etcd_gib=etcd_gib,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute(args):
    num_vectors = int(args.vectors)
    dim = args.dim
    idx = args.index.lower()
    segment_gib = args.segment_size_mb / 1024

    raw = raw_data_size(num_vectors, dim)
    index_mem = 0
    index_disk = 0
    formula_str = ""
    params_str = ""
    is_diskann = idx == "diskann"

    if idx == "hnsw":
        M = args.hnsw_m
        index_mem = hnsw_memory(num_vectors, dim, M)
        formula_str = f"(1 + 2×{M}/{dim}) × raw"
        params_str = f"M={M}"
    elif idx == "flat":
        index_mem = flat_memory(num_vectors, dim)
        formula_str = "raw"
    elif idx == "ivf_flat":
        index_mem = ivf_flat_memory(num_vectors, dim, args.nlist)
        formula_str = "raw + nlist×row_size"
        params_str = f"nlist={args.nlist}"
    elif idx == "ivf_sq8":
        index_mem = ivf_sq8_memory(num_vectors, dim, args.nlist)
        formula_str = "raw/4 + nlist×row_size"
        params_str = f"nlist={args.nlist}"
    elif idx == "ivf_pq":
        m = args.ivfpq_m or max(1, dim // 8)
        index_mem = ivf_pq_memory(num_vectors, dim, args.nlist, m, args.ivfpq_nbits)
        dsub = dim / m
        formula_str = f"raw/({dsub:.1f}×32/{args.ivfpq_nbits}) + nlist×row_size"
        params_str = f"nlist={args.nlist}, m={m}, nbits={args.ivfpq_nbits}"
    elif idx == "scann":
        index_mem = scann_memory(num_vectors, dim, args.scann_with_raw_data)
        formula_str = "(9/8)×raw" if args.scann_with_raw_data else "(1/8)×raw"
        params_str = f"with_raw_data={str(args.scann_with_raw_data).lower()}"
    elif idx == "diskann":
        index_mem, index_disk = diskann_sizing(num_vectors, dim, args.diskann_max_degree)
        formula_str = "mem=raw/4, disk=(1+max_degree/dim)×raw"
        params_str = f"max_degree={args.diskann_max_degree}"
    elif idx == "ivf_rabitq":
        index_mem = ivf_rabitq_memory(num_vectors, dim, args.nlist, args.rabitq_refine_type)
        N_map = {"SQ6": 6, "SQ8": 8, "FP16": 16, "BF16": 16, "FP32": 32}
        N = N_map[args.rabitq_refine_type.upper()]
        formula_str = f"raw×(1+{N})/32 + nlist×row_size"
        params_str = f"nlist={args.nlist}, refine_type={args.rabitq_refine_type}"

    load_mem = loading_memory(index_mem, segment_gib, is_diskann)
    nodes = cluster_node_config(load_mem)
    deps = dependency_sizing(raw, load_mem)

    return dict(
        num_vectors=num_vectors, dim=dim, idx=idx, params_str=params_str,
        formula_str=formula_str, segment_gib=segment_gib,
        raw=raw, index_mem=index_mem, index_disk=index_disk,
        load_mem=load_mem, nodes=nodes, deps=deps,
    )


def print_report(r):
    SEP = "=" * 64
    num_vectors = r["num_vectors"]
    dim = r["dim"]
    idx = r["idx"]

    print(f"\n{SEP}")
    print(f"  MILVUS SIZING — DISTRIBUTED MODE")
    print(SEP)
    print(f"  Vectors      : {num_vectors:>18,}")
    print(f"  Dimensions   : {dim:>18,}")
    idx_label = idx.upper() + (f"  ({r['params_str']})" if r["params_str"] else "")
    print(f"  Index        : {idx_label}")
    print(f"  Segment size : {r['segment_gib'] * 1024:.0f} MiB")
    print(SEP)

    print(f"\n  RAW DATA")
    print(f"    {fmt(r['raw'])}")
    print(f"    ({num_vectors:,} × {dim} × {BYTES_PER_FLOAT32} bytes/float32)")

    print(f"\n  INDEX MEMORY")
    print(f"    Formula : {r['formula_str']}")
    print(f"    Size    : {fmt(r['index_mem'])}")
    if r["index_disk"]:
        print(f"    Disk    : {fmt(r['index_disk'])}")

    print(f"\n  LOADING MEMORY  (index + 2×segment buffer, ×1.15 overhead)")
    print(f"    {fmt(r['load_mem'])}")

    n = r["nodes"]
    print(f"\n  DISTRIBUTED COMPONENTS")
    print(f"\n    Query Nodes    ×{n['qn']}")
    print(f"      CPU : {n['qn_cpu']} vCPU each")
    print(f"      RAM : {n['qn_mem']} GiB each")
    print(f"\n    Data Nodes     ×{n['dn']}")
    print(f"      CPU : {n['dn_cpu']} vCPU each")
    print(f"      RAM : {n['dn_mem']} GiB each")
    print(f"\n    mixCoord       ×1")
    print(f"      CPU : {n['coord_cpu']} vCPU")
    print(f"      RAM : {n['coord_mem']} GiB")
    print(f"\n    Proxy          ×1")
    print(f"      CPU : {n['proxy_cpu']} vCPU")
    print(f"      RAM : {n['proxy_mem']} GiB")

    d = r["deps"]
    print(f"\n  DEPENDENCIES")
    print(f"\n    MinIO  (object storage)")
    print(f"      PVC : {d['minio_gib']} GiB")
    print(f"\n    Pulsar  (message broker)")
    print(f"      Ledgers : {d['pulsar_ledgers_gib']} GiB")
    print(f"      Journal : {d['pulsar_journal_gib']} GiB")
    print(f"\n    etcd   (3 nodes for HA)")
    print(f"      Disk per node : {d['etcd_gib']} GiB SSD")

    qn_total = n["qn"] * n["qn_mem"]
    dn_total = n["dn"] * n["dn_mem"]
    total_ram = qn_total + dn_total + n["coord_mem"] + n["proxy_mem"]
    total_nodes = n["qn"] + n["dn"] + 2  # +mixCoord +proxy
    print(f"\n  TOTALS")
    print(f"    Milvus nodes : {total_nodes}  (+ 3 etcd + MinIO + Pulsar)")
    print(f"    Total RAM    : ~{total_ram} GiB  (Milvus nodes)")
    print(f"    Object store : {d['minio_gib']} GiB")

    print(f"\n  VERIFY")
    print(f"    https://milvus.io/tools/sizing")
    print(f"    (HNSW · Distributed · {num_vectors:,} vectors · {dim} dims)")
    print(f"\n{SEP}\n")


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
                                 "scann", "diskann", "ivf_rabitq"])
    parser.add_argument("--hnsw-m", type=int, default=30)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--ivfpq-m", type=int, default=None)
    parser.add_argument("--ivfpq-nbits", type=int, default=8)
    parser.add_argument("--diskann-max-degree", type=int, default=56)
    parser.add_argument("--scann-with-raw-data", action="store_true")
    parser.add_argument("--rabitq-refine-type", type=str, default="SQ8",
                        choices=["SQ6", "SQ8", "FP16", "BF16", "FP32"])
    parser.add_argument("--segment-size-mb", type=int, default=512,
                        choices=[512, 1024, 2048])
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    args = parser.parse_args()

    result = compute(args)

    if args.json:
        out = {k: v for k, v in result.items()
               if k not in ("nodes", "deps")}
        out["nodes"] = result["nodes"]
        out["deps"] = result["deps"]
        print(json.dumps(out, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
