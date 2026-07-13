# Qwen3.5-397B-A17B (FP8) — Aggregated, single-node B200 · vLLM

NVIDIA-verified aggregated serving recipes for **Qwen/Qwen3.5-397B-A17B-FP8**
(hybrid Mamba/GDN + attention MoE, FP8) on **B200** (x86_64, 8 GPU per node),
served through the **dynamo** frontend with the **vLLM** backend. All recipes run
on a single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
  dynamo 1.1.1 · vllm 0.19.0
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`Qwen/Qwen3.5-397B-A17B-FP8` (revision `ea5b4f81…781d`) is pulled via the `hf:`
handle:

```yaml
model:
  path: "hf:Qwen/Qwen3.5-397B-A17B-FP8"
  precision: "fp8"
```

## Recipes

These recipes use a **fixed** TP=1 · DP=8 · expert-parallel topology (no `sweep`
block); only the sequence-length / batching knobs differ.

| file | ISL / OSL | topology | max-model-len | max-num-batched-tokens |
|---|---|---|---:|---:|
| `1k1k.yaml` | 1k / 1k | TP=1 · DP=8 · EP on | 2068 | 8192 |
| `8k1k.yaml` | 8k / 1k | TP=1 · DP=8 · EP on | 9236 | 16384 |

Benchmark concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `language-model-only: true`, `reasoning-parser: "qwen3"`, `enable-prefix-caching: true`.
- `no-disable-hybrid-kv-cache-manager: true` — Qwen3.5 mixes full-attention and
  GDN/Mamba layers; this Dynamo build auto-disables the Hybrid KV Cache Manager,
  after which `unify_hybrid_kv_cache_specs` crashes on the mixed
  `FullAttentionSpec` + `MambaSpec`. Forcing HMA back on is required
  (trackers: vllm-project/vllm#41860, ai-dynamo/dynamo#8988).
- `kv-cache-dtype: "fp8"`, `block-size: 32`, `gpu-memory-utilization: 0.90`.
- `TORCH_CUDA_ARCH_LIST: "10.0"`, `VLLM_FLOAT32_MATMUL_PRECISION: "high"`.

> **Perf note.** Throughput is below what a 17B-active MoE on 8× B200 would
> suggest. Qwen3.5's Mamba/GDN layers don't yet have fused, prefix-cache-friendly
> kernels in vLLM, so every step pays an extra state-cache copy and falls off the
> CUDA-graph fast path. This per-token overhead is paid regardless of model size,
> matching the gap reported upstream on the smaller Qwen3 hybrid model
> (tracker: vllm-project/vllm#36627).

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/Qwen3.5/B200/vllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/Qwen3.5/B200/vllm/1k1k.yaml"
```

## References

- [Qwen3.5-397B-A17B-FP8 model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
- [vLLM Qwen3.5 serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
