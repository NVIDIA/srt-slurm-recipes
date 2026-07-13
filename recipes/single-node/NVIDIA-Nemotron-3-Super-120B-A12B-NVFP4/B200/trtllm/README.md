# Nemotron-3-Super-120B-A12B (NVFP4) — Aggregated, single-node B200 · TensorRT-LLM

NVIDIA-verified aggregated serving recipes for
**nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** (MoE, NVFP4) on **B200**
(x86_64, 8 GPU per node), served through the **dynamo** frontend with the
**TensorRT-LLM** PyTorch backend. All recipes run on a single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1
  dynamo 1.1.1 · tensorrt_llm 1.3.0rc13
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

| file | ISL / OSL | parallelism sweep | max_seq_len |
|---|---|---|---:|
| `1k1k.yaml` | 1k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} | 2176 |
| `8k1k.yaml` | 8k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} | 9416 |

Both recipes `sweep` over `tensor/pipeline/moe_expert_parallel_size` with
`max_batch_size: 1024` and `max_num_tokens: 2176`. Benchmark concurrency sweeps
`1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `backend: pytorch` — TensorRT-LLM PyTorch runtime.
- `TRTLLM_ENABLE_PDL: "1"` — programmatic dependent launch.
- `TRTLLM_SERVER_DISABLE_GC` / `TRTLLM_WORKER_DISABLE_GC: "1"` — disable Python GC
  on the server/worker hot path.
- `trust_remote_code: true`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/B200/trtllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/B200/trtllm/1k1k.yaml"
```

## References

- [Nemotron-3-Super-120B-A12B-NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
