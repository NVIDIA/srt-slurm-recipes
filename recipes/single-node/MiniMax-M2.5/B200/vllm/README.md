# MiniMax-M2.5 (FP8) — Aggregated, single-node B200 · vLLM

NVIDIA-verified aggregated serving recipes for **MiniMaxAI/MiniMax-M2.5** (MoE, FP8)
on **B200** (x86_64, 8 GPU per node), served through the **dynamo** frontend with
the **vLLM** backend. All recipes run on a single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
  dynamo 1.1.1 · vllm 0.19.0
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`MiniMaxAI/MiniMax-M2.5` (revision `f710177d…f21f`) is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:MiniMaxAI/MiniMax-M2.5"
  precision: "fp8"
```

## Recipes

| file | ISL / OSL | parallelism sweep | max-model-len | target |
|---|---|---|---:|---|
| `1k1k.yaml` | 1k / 1k | TP ∈ {2, 4, 8} + expert-parallel | 2068  | 1k/1k throughput sweep |
| `8k1k.yaml` | 8k / 1k | TP ∈ {2, 4, 8} + expert-parallel | 10240 | 8k/1k throughput sweep |

Both recipes `sweep` over `tensor-parallel-size` with `enable-expert-parallel: true`.
The 1k/1k recipe uses `max-num-batched-tokens: 2048`; the 8k/1k recipe `16384`.
Benchmark concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `enable-expert-parallel: true`, `kv-cache-dtype: "fp8"`, `block-size: 32`.
- `gpu-memory-utilization: 0.90`, `max-cudagraph-capture-size: 2048`, `stream-interval: 20`.
- `no-enable-prefix-caching: true` — synthetic-benchmark best practice.
- `TORCH_CUDA_ARCH_LIST: "10.0"` (Blackwell `sm_100`),
  `VLLM_FLOAT32_MATMUL_PRECISION: "high"`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/MiniMax-M2.5/B200/vllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/MiniMax-M2.5/B200/vllm/1k1k.yaml"
```

## References

- [InferenceX: `minimaxm2.5_fp8_b200.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/minimaxm2.5_fp8_b200.sh)
- [MiniMax-M2.5 model card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
