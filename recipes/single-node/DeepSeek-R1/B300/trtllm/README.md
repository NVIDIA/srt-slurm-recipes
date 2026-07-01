# DeepSeek-R1 (NVFP4) — Aggregated, single-node B300 · TensorRT-LLM

NVIDIA-verified aggregated serving recipe for **nvidia/DeepSeek-R1-0528-FP4-v2**
(MoE, NVFP4 weights + FP8 KV) on **B300** (x86_64, 8 GPU per node), served through
the **dynamo** frontend with the **TensorRT-LLM** PyTorch backend. Runs on a single
node with MTP speculative decoding.

## Container

```text
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1
  dynamo 1.1.1 · tensorrt_llm 1.3.0rc13
```

The image is referenced directly in `model.container`; no local mounts are declared.

## Model checkpoint

`nvidia/DeepSeek-R1-0528-FP4-v2` is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:nvidia/DeepSeek-R1-0528-FP4-v2"
  precision: "fp4"
```

## Recipes

| file | ISL / OSL | parallelism sweep | spec decode | max seq len |
|---|---|---|---|---:|
| `1k1k-mtp.yaml` | 1k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} | MTP (3 nextn layers) | 2176 |

The recipe `sweep`s over `tensor/pipeline/moe-expert-parallel-size`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `backend: pytorch`, `moe_config.backend: "TRTLLM"`.
- `kv_cache_config.dtype: fp8`, `free_gpu_memory_fraction: 0.8`, no block reuse.
- `cuda_graph_config`: padding on, `max_batch_size: 1024`; `max_num_tokens: 5248`.
- `TRTLLM_ENABLE_PDL: "1"`, GC disabled on server/worker, `stream_interval: 10`.
- MTP: `decoding_type: "MTP"`, `num_nextn_predict_layers: 3`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/trtllm/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/trtllm/1k1k-mtp.yaml"
```

## References

- [InferenceX: `dsr1_fp4_b200_trt.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/dsr1_fp4_b200_trt.sh)
- [DeepSeek-R1-0528-FP4-v2 model card](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2)
