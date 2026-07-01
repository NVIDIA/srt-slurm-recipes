# DeepSeek-V4-Pro (NVFP4) — Aggregated, single-node B200 · SGLang

NVIDIA-verified aggregated serving recipes for **deepseek-ai/DeepSeek-V4-Pro** (MoE,
NVFP4 weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **SGLang** backend. All recipes run on a single node
with MTP (EAGLE) speculative decoding and the DeepGEMM mega-MoE path.

## Container

```text
nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1
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

| file | ISL / OSL | parallelism sweep | spec decode | context length |
|---|---|---|---|---:|
| `1k1k-mtp.yaml` | 1k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | EAGLE (1 step, 2 draft tokens) | 2176 |
| `8k1k-mtp.yaml` | 8k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | EAGLE (1 step, 2 draft tokens) | 9280 |

Both recipes `sweep` over `tensor/pipeline/data/expert-parallel-size`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `enable-dp-attention: true`, `moe-a2a-backend: "deepep"` with
  `deepep-config num_sms: 96`.
- `mem-fraction-static: 0.92`, `chunked-prefill-size: 32768`,
  `max-running-requests: 256`, `disable-radix-cache: true`.
- DeepGEMM mega-MoE via the `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` / `*_MEGA_MOE` env
  set, plus SWA optimizations and `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` (single-node).
- EAGLE: `speculative-num-steps: 1`, `speculative-num-draft-tokens: 2`,
  `speculative-eagle-topk: 1`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-V4-Pro/B200/sglang/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-V4-Pro/B200/sglang/1k1k-mtp.yaml"
```

## References

- [InferenceX: `dsv4_fp4_b200.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/dsv4_fp4_b200.sh)
- [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
