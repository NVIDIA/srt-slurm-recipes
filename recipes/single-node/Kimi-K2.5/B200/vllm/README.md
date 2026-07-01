# Kimi-K2.5 (NVFP4) — Aggregated, single-node B200 · vLLM

NVIDIA-verified aggregated serving recipes for **nvidia/Kimi-K2.5-NVFP4** (MoE,
NVFP4 weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **vLLM** backend. All recipes run on a single node
(full 8-GPU node per worker).

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
  dynamo 1.1.1 · vllm 0.19.0
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

| file | ISL / OSL | parallelism sweep | max-model-len | target |
|---|---|---|---:|---|
| `1k1k.yaml` | 1k / 1k | TP · PP ∈ {1, 2, 4, 8} | 2304 | 1k/1k throughput sweep |
| `8k1k.yaml` | 8k / 1k | TP · PP ∈ {1, 2, 4, 8} | 9472 | 8k/1k throughput sweep |

Both recipes `sweep` over `tensor/pipeline-parallel-size` (the upstream InferenceX
recipe does not pass DP/EP flags for Kimi). The 1k/1k recipe uses
`max-num-batched-tokens: 2048`; the 8k/1k recipe `16384`. Benchmark concurrency
sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `kv-cache-dtype: "fp8"`, `gpu-memory-utilization: 0.9`, `max-num-seqs: 1024`.
- `reasoning-parser: "kimi_k2"` — Kimi K2 reasoning format.
- `compilation-config`: `fuse_allreduce_rms`; `no-enable-prefix-caching: true`.
- `stream-interval: 20`.
- `TORCH_CUDA_ARCH_LIST: "10.0"` (Blackwell `sm_100`), `NCCL_CUMEM_ENABLE: "1"`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/Kimi-K2.5/B200/vllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/Kimi-K2.5/B200/vllm/1k1k.yaml"
```

## References

- [InferenceX: `kimik2.5_fp4_b200.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/kimik2.5_fp4_b200.sh)
- [Kimi-K2.5-NVFP4 model card](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4)
