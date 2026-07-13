# GPT-OSS-120B (FP4) — Aggregated, single-node B200 · vLLM

NVIDIA-verified aggregated serving recipes for **openai/gpt-oss-120b** (MoE, MXFP4
MoE weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **vLLM** backend. All recipes run on a single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
  dynamo 1.1.1 · vllm 0.19.0
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`openai/gpt-oss-120b` (revision `b5c939de…70f8a`) is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:openai/gpt-oss-120b"
  precision: "fp4"
```

EAGLE3 recipes additionally pull `nvidia/gpt-oss-120b-Eagle3-{short,long}-context`.

## Recipes

| file | ISL / OSL | TP sweep | spec decode | max-model-len | target |
|---|---|---|---|---:|---|
| `1k1k.yaml`       | 1k / 1k | {1, 2, 4, 8} | —                                  | 2068 | 1k/1k throughput sweep |
| `1k1k-eagle.yaml` | 1k / 1k | {1, 2, 4, 8} | EAGLE3, 3 draft tokens (short-ctx) | 2068 | 1k/1k low-latency / spec decode |
| `8k1k.yaml`       | 8k / 1k | {1, 2, 4, 8} | —                                  | 9472 | 8k/1k throughput sweep |
| `8k1k-eagle.yaml` | 8k / 1k | {1, 2, 4}    | EAGLE3, 3 draft tokens (long-ctx)  | 9472 | 8k/1k low-latency / spec decode |

Each recipe `sweep`s over `tensor-parallel-size` (the `8k1k-eagle` recipe stops at
TP=4). Benchmark concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`,
`sa-bench`.

## Key flags

- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: "1"` — FlashInfer MXFP4 MoE kernels (Blackwell).
- `TORCH_CUDA_ARCH_LIST: "10.0"` — target Blackwell `sm_100`.
- `kv-cache-dtype: "fp8"`, `gpu-memory-utilization: 0.9`, `max-num-seqs: 512`.
- `max-cudagraph-capture-size: 2048`, `max-num-batched-tokens: 8192`.
- `no-enable-prefix-caching: true` — synthetic-benchmark best practice.
- `compilation-config`: `fuse_allreduce_rms` + `eliminate_noops`; EAGLE adds
  `cudagraph_mode: FULL_DECODE_ONLY`.
- EAGLE3 via `speculative-config` (`method: eagle3`, `num_speculative_tokens: 3`).

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/gpt-oss-120b/B200/vllm/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/gpt-oss-120b/B200/vllm/1k1k.yaml"
```

## References

- [InferenceX: `gptoss_fp4_b200.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/gptoss_fp4_b200.sh)
- [vLLM GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- [gpt-oss-120b model card](https://huggingface.co/openai/gpt-oss-120b)
