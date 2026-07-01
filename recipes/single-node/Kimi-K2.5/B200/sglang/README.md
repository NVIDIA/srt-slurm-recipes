# Kimi-K2.5 (NVFP4) — Aggregated, single-node B200 · SGLang

NVIDIA-verified aggregated serving recipes for **nvidia/Kimi-K2.5-NVFP4** (MoE,
NVFP4 weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **SGLang** backend. All recipes run on a single node
(full 8-GPU node per worker).

## Container

```text
nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13
  dynamo 1.1.1 · sglang 0.5.10.post1
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`nvidia/Kimi-K2.5-NVFP4` (revision `0fd0a5e6…ae3d`) is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:nvidia/Kimi-K2.5-NVFP4"
  precision: "fp4"
```

## Recipes

| file | ISL / OSL | parallelism sweep | context length | target |
|---|---|---|---:|---|
| `1k1k.yaml` | 1k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | 2304 | 1k/1k throughput sweep |
| `8k1k.yaml` | 8k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | 9472 | 8k/1k throughput sweep |

Both recipes `sweep` over `tensor/pipeline/data/expert-parallel-size`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `quantization`-free NVFP4 weights with `kv-cache-dtype: "fp8_e4m3"`.
- `mem-fraction-static: 0.85`, `cuda-graph-max-bs: 1024`, `max-running-requests: 1024`.
- `disable-radix-cache`, `disable-flashinfer-autotune`,
  `disable-piecewise-cuda-graph` — synthetic-benchmark best practice.
- `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2: "1"` (single-node only), plus the
  `SGLANG_OPT_USE_JIT_NORM` / `JIT_INDEXER_METADATA` / `TOPK_V2` JIT optimizations.
- `TORCH_CUDA_ARCH_LIST: "10.0"` — target Blackwell `sm_100`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/Kimi-K2.5/B200/sglang/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/Kimi-K2.5/B200/sglang/1k1k.yaml"
```

## References

- [InferenceX: `kimik2.5_fp4_b200.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/kimik2.5_fp4_b200.sh)
- [Kimi-K2.5-NVFP4 model card](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4)
