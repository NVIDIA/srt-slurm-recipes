# DeepSeek-V4-Pro (NVFP4) — Aggregated, single-node B200 · TensorRT-LLM

NVIDIA-verified aggregated serving recipes for **deepseek-ai/DeepSeek-V4-Pro** (MoE,
NVFP4 weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **TensorRT-LLM** PyTorch backend. All recipes run on a
single node with MTP speculative decoding and attention data-parallelism.

## Container

```text
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.1
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`deepseek-ai/DeepSeek-V4-Pro` is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:deepseek-ai/DeepSeek-V4-Pro"
  precision: "fp4"
```

These recipes use a custom benchmark tokenizer
(`sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer`).

## Recipes

| file | ISL / OSL | parallelism sweep | spec decode | max seq len |
|---|---|---|---|---:|
| `1k1k-mtp.yaml` | 1k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} + attention-DP | MTP (2 nextn layers) | 8192 |
| `8k1k-mtp.yaml` | 8k / 1k | TP · PP · EP ∈ {1, 2, 4, 8} + attention-DP | MTP (2 nextn layers) | 9280 |

Both recipes `sweep` over `tensor/pipeline/moe-expert-parallel-size`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `backend: pytorch`, `moe_config.backend: "TRTLLM"`, `enable_attention_dp: true`
  with balanced `attention_dp_config`.
- `kv_cache_config`: `dtype: fp8`, `tokens_per_block: 128`,
  `free_gpu_memory_fraction: 0.50`, no block reuse.
- `cuda_graph_config`: padding on, `max_batch_size: 1024`.
- `TRTLLM_ENABLE_PDL: "1"`, GC disabled, plus the DSv4 MPI launch env
  (`TRTLLM_DSV4_USE_MPIRUN`, `TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV`,
  `TRTLLM_DSV4_MTP_NUM_NEXTN_LAYERS: "2"`, `NCCL_NVLS_ENABLE: "0"`).
- MTP: `decoding_type: "MTP"`, `num_nextn_predict_layers: 2`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-V4-Pro/B200/trtllm/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-V4-Pro/B200/trtllm/1k1k-mtp.yaml"
```

## References

- [InferenceX: `dsv4_fp4_b200_trt_mtp.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/dsv4_fp4_b200_trt_mtp.sh)
- [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
