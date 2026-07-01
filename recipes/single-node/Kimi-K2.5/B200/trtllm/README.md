# Kimi-K2.5 (NVFP4) — Aggregated, single-node B200 · TensorRT-LLM

NVIDIA-verified aggregated serving recipes for **nvidia/Kimi-K2.5-NVFP4** (MoE,
NVFP4 weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **TensorRT-LLM** PyTorch backend. All recipes run on a
single node (full 8-GPU node per worker).

## Container

```text
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1
  dynamo 1.1.1 · tensorrt_llm 1.3.0rc13
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

| file | ISL / OSL | parallelism sweep | max seq len | target |
|---|---|---|---:|---|
| `1k1k.yaml` | 1k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} | 2304 | 1k/1k throughput sweep |
| `8k1k.yaml` | 8k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} | 9472 | 8k/1k throughput sweep |

Both recipes `sweep` over `tensor/pipeline/moe-expert-parallel-size`. The 1k/1k
recipe uses `max_num_tokens: 2048`; the 8k/1k recipe `16384`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `backend: pytorch`, `moe_config.backend: "TRTLLM"`.
- `nvfp4_gemm_config.allowed_backends`: `cutlass`, `cublaslt`, `cutedsl`, `cuda_core`.
- `kv_cache_config.dtype: fp8`, `free_gpu_memory_fraction: 0.8`, no block reuse.
- `cuda_graph_config`: padding on, `max_batch_size: 256`.
- `TRTLLM_ENABLE_PDL: "1"`, GC disabled on server/worker, `stream_interval: 20`,
  `num_postprocess_workers: 4`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/Kimi-K2.5/B200/trtllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/Kimi-K2.5/B200/trtllm/1k1k.yaml"
```

## References

- [InferenceX: `kimik2.5_fp4_b200.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/kimik2.5_fp4_b200.sh)
- [Kimi-K2.5-NVFP4 model card](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4)
