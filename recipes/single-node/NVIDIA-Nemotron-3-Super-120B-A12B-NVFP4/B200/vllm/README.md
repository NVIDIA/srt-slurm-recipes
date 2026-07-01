# Nemotron-3-Super-120B-A12B (NVFP4) — Aggregated, single-node B200 · vLLM

NVIDIA-verified aggregated serving recipes for
**nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** (MoE, NVFP4) on **B200**
(x86_64, 8 GPU per node), served through the **dynamo** frontend with the **vLLM**
backend. All recipes run on a single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3
  dynamo 1.2.0 · vllm 0.20.1
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

| file | ISL / OSL | parallelism sweep | max-model-len |
|---|---|---|---:|
| `1k1k.yaml` | 1k / 1k | TP · PP · DP ∈ {1, 2, 4, 8}, EP ∈ {on, off} | 2176 |
| `8k1k.yaml` | 1k / 1k* | TP · PP · DP ∈ {1, 2, 4, 8}, EP ∈ {on, off} | 2176 |

\* **Note:** `8k1k.yaml` currently sets `isl: 1024 / osl: 1024` with
`max-model-len: 2176` — identical to `1k1k.yaml`. Update the `benchmark` block
(and bump `max-model-len`) if an 8k-input sweep is intended.

Both recipes `sweep` over `tensor/pipeline/data-parallel-size` and toggle
`enable-expert-parallel`. Benchmark concurrency sweeps `1 → 1024`,
`random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `quantization: modelopt` — ModelOpt NVFP4 path.
- `kv-cache-dtype: "fp8_e4m3"`, `attention-backend: "FLASHINFER"`.
- `async-scheduling: true`, `no-enable-prefix-caching: true`.
- `max-num-seqs: 1024`, `gpu-memory-utilization: 0.9`, `max-cudagraph-capture-size: 1024`.
- `trust-remote-code: true`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/B200/vllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/B200/vllm/1k1k.yaml"
```

## References

- [Nemotron-3-Super-120B-A12B-NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
