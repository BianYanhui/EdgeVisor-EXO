#!/usr/bin/env python3
"""Generate EdgeVisor pipeline-parallel ratios from device memory.

EdgeVisor's runtime already accepts manual pipeline plans such as
``1@10*1@9*1@9``.  This helper implements the EXO-style policy layer:
read device memory, allocate transformer layers proportionally, and emit a
pipeline-only ratio string.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class DeviceMemory:
    index: int
    total_mib: int
    free_mib: int
    used_mib: int
    name: str

    def weight(self, field: str) -> int:
        if field == "total":
            return self.total_mib
        if field == "free":
            return self.free_mib
        raise ValueError(f"unsupported memory field: {field}")


def parse_gpu_indices(value: str) -> List[int]:
    indices: List[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw, 10)
        if idx < 0:
            raise ValueError("GPU indices must be non-negative")
        indices.append(idx)
    if not indices:
        raise ValueError("at least one GPU index is required")
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicated GPU index in {value!r}")
    return indices


def query_nvidia_smi() -> List[DeviceMemory]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,memory.used,name",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    rows: List[DeviceMemory] = []
    for fields in csv.reader(proc.stdout.splitlines()):
        if len(fields) < 5:
            continue
        rows.append(
            DeviceMemory(
                index=int(fields[0].strip()),
                total_mib=int(fields[1].strip()),
                free_mib=int(fields[2].strip()),
                used_mib=int(fields[3].strip()),
                name=",".join(fields[4:]).strip(),
            )
        )
    return rows


def allocate_layers(total_layers: int, weights: Sequence[int], min_layers: int = 1) -> List[int]:
    """Allocate integer layers with largest-remainder rounding.

    Each device gets at least ``min_layers`` when possible.  If the model has
    fewer layers than devices, the largest-memory devices receive one layer
    first and the rest receive zero.
    """

    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    n = len(weights)
    if sum(weights) <= 0:
        weights = [1 for _ in weights]

    if total_layers < n * min_layers:
        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        out = [0] * n
        for i in order[:total_layers]:
            out[i] = 1
        return out

    reserved = [min_layers] * n
    remaining = total_layers - sum(reserved)
    if remaining == 0:
        return reserved

    weight_sum = float(sum(weights))
    raw = [w / weight_sum * remaining for w in weights]
    floors = [int(v) for v in raw]
    rem = remaining - sum(floors)
    frac_order = sorted(range(n), key=lambda i: (raw[i] - floors[i], weights[i]), reverse=True)

    out = [reserved[i] + floors[i] for i in range(n)]
    for i in frac_order[:rem]:
        out[i] += 1
    return out


def ratios_from_layers(layers: Iterable[int]) -> str:
    return "*".join(f"1@{layer_count}" for layer_count in layers)


def build_plan(gpu_indices: Sequence[int], total_layers: int, memory_field: str, min_layers: int) -> dict:
    all_devices = {dev.index: dev for dev in query_nvidia_smi()}
    missing = [idx for idx in gpu_indices if idx not in all_devices]
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPU indices: {missing}")

    devices = [all_devices[idx] for idx in gpu_indices]
    weights = [dev.weight(memory_field) for dev in devices]
    layers = allocate_layers(total_layers=total_layers, weights=weights, min_layers=min_layers)
    return {
        "policy": "memory_proportional_pipeline",
        "memory_field": memory_field,
        "total_layers": total_layers,
        "gpu_indices": list(gpu_indices),
        "devices": [asdict(dev) for dev in devices],
        "weights_mib": weights,
        "layers": layers,
        "ratios": ratios_from_layers(layers),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate EdgeVisor-EXO pipeline ratios from GPU memory")
    parser.add_argument("--gpu-indices", default="0,1,2", help="Comma-separated physical GPU indices")
    parser.add_argument("--total-layers", type=int, default=28, help="Transformer layer count to split")
    parser.add_argument(
        "--memory-field",
        choices=("total", "free"),
        default="total",
        help="Use total capacity or currently free memory as the proportional weight",
    )
    parser.add_argument("--min-layers", type=int, default=1, help="Minimum layers per device when possible")
    parser.add_argument("--json", action="store_true", help="Print the full plan as JSON")
    args = parser.parse_args(argv)

    try:
        gpu_indices = parse_gpu_indices(args.gpu_indices)
        plan = build_plan(gpu_indices, args.total_layers, args.memory_field, args.min_layers)
    except Exception as exc:
        print(f"edgevisor-exo-plan: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(plan["ratios"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
