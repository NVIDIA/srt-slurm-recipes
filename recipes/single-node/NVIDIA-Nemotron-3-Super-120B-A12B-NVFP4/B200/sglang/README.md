# Nemotron-3-Super-120B-A12B (NVFP4) — Aggregated, single-node B200 · SGLang

NVIDIA-verified aggregated serving recipes for
**nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** (MoE, NVFP4) on **B200**
(x86_64, 8 GPU per node), served through the **dynamo** frontend with the **SGLang**
backend. All recipes run on a single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13
  dynamo 1.1.1 · sglang 0.5.10.post1
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (revision `4f0cf9da…caf6`) is
pulled via the `hf:` handle:

```yaml
model:
  path: "hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  precision: "fp4"
```

## Recipes

| file | ISL / OSL | parallelism sweep | context length |
|---|---|---|---:|
| `1k1k.yaml` | 1k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | 2176 |
| `8k1k.yaml` | 8k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | 9216 |

Both recipes `sweep` over `tensor/pipeline/data/expert-parallel-size`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `mem-fraction-static: 0.85`, `cuda-graph-max-bs: 1024`, `max-running-requests: 1024`.
- `disable-piecewise-cuda-graph: true`.
- `trust-remote-code: true`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/B200/sglang/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/B200/sglang/1k1k.yaml"
```

## References

- [Nemotron-3-Super-120B-A12B-NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
