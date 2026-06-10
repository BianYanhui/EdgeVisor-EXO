# EdgeVisor-EXO

EdgeVisor-EXO is an EdgeVisor variant that adds an EXO-style placement layer for
pipeline-parallel inference.  It does not use tensor parallelism.

The runtime still uses EdgeVisor's existing `dllama` engine and `--ratios`
interface.  The added EXO layer performs one step before launch:

1. Read the selected GPU memory with `nvidia-smi`.
2. Allocate transformer layers in proportion to device memory.
3. Emit a pipeline-only ratio string such as `1@10*1@9*1@9`.
4. Start the root and worker processes with that generated ratio.

For example, if three devices have memory capacity `16GB, 8GB, 8GB`, a 32-layer
model is assigned as `16, 8, 8` layers, matching the `2:1:1` memory ratio.

## Quick Start

Build the engine:

```bash
./scripts/build.sh
```

Run the default 3-GPU static pipeline test on GPU0/1/2:

```bash
./run_gpu_exo_static.sh
```

Useful overrides:

```bash
EDGEVISOR_EXO_GPUS=0,1,2 \
EDGEVISOR_EXO_TOTAL_LAYERS=28 \
EDGEVISOR_EXO_MEMORY_FIELD=total \
EDGEVISOR_EXO_STEPS=16 \
./run_gpu_exo_static.sh
```

`EDGEVISOR_EXO_MEMORY_FIELD=total` uses device capacity.  `free` can be used for
a load-aware split, but the default follows the EXO memory-capacity policy.

## Policy Helper

Generate only the ratio string:

```bash
python3 EdgeVisor/src/edgevisor_exo_plan.py --gpu-indices 0,1,2 --total-layers 28
```

Print the full placement plan:

```bash
python3 EdgeVisor/src/edgevisor_exo_plan.py --gpu-indices 0,1,2 --total-layers 28 --json
```

## Notes

- EdgeVisor-EXO only generates pipeline-parallel layer splits.
- It does not change EdgeVisor's C++ runtime scheduling core.
- GPU3 is not used unless explicitly included in `EDGEVISOR_EXO_GPUS`.
