# DeepSeek-R1 (NVFP4) — Aggregated, single-node B300 · SGLang

NVIDIA-verified aggregated serving recipe for **nvidia/DeepSeek-R1-0528-FP4-v2**
(MoE, NVFP4 weights + FP8 KV) on **B300** (x86_64, 8 GPU per node), served through
the **dynamo** frontend with the **SGLang** backend. Runs on a single node with MTP
(EAGLE) speculative decoding.

## Container

```text
nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13
  dynamo 1.1.1 · sglang 0.5.10.post1
```

The image is referenced directly in `model.container`; no local mounts are declared.

## Model checkpoint

`nvidia/DeepSeek-R1-0528-FP4-v2` is pulled via the `hf:` handle (served as
`deepseek-ai/DeepSeek-R1-0528`):

```yaml
model:
  path: "hf:nvidia/DeepSeek-R1-0528-FP4-v2"
  precision: "fp4"
```

## Recipes

| file | ISL / OSL | parallelism sweep | spec decode | context length |
|---|---|---|---|---:|
| `1k1k-mtp.yaml` | 1k / 1k | TP · PP · DP · EP ∈ {1, 2, 4, 8} | EAGLE (2 steps, 3 draft tokens) | 2176 |

The recipe `sweep`s over `tensor/pipeline/data/expert-parallel-size`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `attention-backend: "trtllm_mla"`, `moe-runner-backend: "flashinfer_trtllm"`.
- `kv-cache-dtype: "fp8_e4m3"`, `mem-fraction-static: 0.82`, `disable-radix-cache: true`.
- `enable-flashinfer-allreduce-fusion`, `enable-symm-mem`.
- `SGLANG_ENABLE_SPEC_V2: "1"`; EAGLE: `speculative-num-steps: 2`,
  `speculative-num-draft-tokens: 3`, `speculative-eagle-topk: 1`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/sglang/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/sglang/1k1k-mtp.yaml"
```

## References

- [InferenceX: `dsr1_fp4_b300.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/dsr1_fp4_b300.sh)
- [DeepSeek-R1-0528-FP4-v2 model card](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2)
