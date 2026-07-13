# DeepSeek-V4-Pro (NVFP4) — Aggregated, single-node B200 · vLLM

NVIDIA-verified aggregated serving recipes for **deepseek-ai/DeepSeek-V4-Pro** (MoE,
NVFP4 weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **vLLM** backend. All recipes run on a single node with
MTP speculative decoding.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3
  dynamo 1.2.0 · vllm 0.20.1
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
(`sa_bench_tokenizers.vllm_deepseek_v4.VLLMDeepseekV4Tokenizer`).

## Recipes

| file | ISL / OSL | parallelism sweep | spec decode | max-model-len |
|---|---|---|---|---:|
| `1k1k-mtp.yaml` | 1k / 1k | TP · PP ∈ {1, 2, 4, 8}, EP ∈ {on, off} | MTP (2 tokens) | 2176 |
| `8k1k-mtp.yaml` | 8k / 1k | TP · PP ∈ {1, 2, 4, 8}, EP ∈ {on, off} | MTP (2 tokens) | 9280 |

Both recipes `sweep` over `tensor/pipeline-parallel-size` and toggle
`enable-expert-parallel`. Benchmark concurrency sweeps `1 → 1024`,
`random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `tokenizer-mode: deepseek_v4`, `reasoning-parser: deepseek_v4`.
- `attention-config: {"use_fp4_indexer_cache": true}`.
- `kv-cache-dtype: "fp8"`, `block-size: 256`, `gpu-memory-utilization: 0.95`.
- `compilation-config: {"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]}`.
- `speculative-config: {"method": "mtp", "num_speculative_tokens": 2}`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-V4-Pro/B200/vllm/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-V4-Pro/B200/vllm/1k1k-mtp.yaml"
```

## References

- [InferenceX: `dsv4_fp4_b200_vllm_mtp.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/dsv4_fp4_b200_vllm_mtp.sh)
- [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
