# GPT-OSS-120B (FP4) — Aggregated, single-node B200 · TensorRT-LLM

NVIDIA-verified aggregated serving recipes for **openai/gpt-oss-120b** (MoE, NVFP4
weights + FP8 KV) on **B200** (x86_64, 8 GPU per node), served through the
**dynamo** frontend with the **TensorRT-LLM** PyTorch backend. All recipes run on a
single node.

## Container

```text
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3
  dynamo 1.1.0 · tensorrt_llm 1.3.0rc11
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`openai/gpt-oss-120b` (revision `b5c939de…70f8a`) is pulled via the `hf:` handle in
`model.path` — no manual download step required:

```yaml
model:
  path: "hf:openai/gpt-oss-120b"
  precision: "fp4"
```

EAGLE3 recipes additionally pull the draft model
`nvidia/gpt-oss-120b-Eagle3-{short,long}-context`.

## Recipes

| file | ISL / OSL | parallelism | spec decode | max batch | target |
|---|---|---|---|---:|---|
| `1k1k-agg-tp1.yaml`   | 1k / 1k | TP=1        | —                                     |  512 | single-GPU 1k/1k baseline |
| `8k1k-agg-tp1.yaml`   | 8k / 1k | TP=1        | —                                     |  512 | single-GPU long-input |
| `1k8k-agg-tp1.yaml`   | 1k / 8k | TP=1        | —                                     |  512 | single-GPU long-output |
| `1k1k-agg-eagle.yaml` | 1k / 1k | TP ∈ {4, 8} | EAGLE3 one-model, draft=3 (short-ctx) | 1024 | low-latency / spec-decode 1k/1k |
| `8k1k-agg-eagle.yaml` | 8k / 1k | TP ∈ {4, 8} | EAGLE3 one-model, draft=3 (long-ctx)  | 1024 | low-latency / spec-decode 8k/1k |

The `-tp1` recipes pin a single GPU; the `-eagle` recipes `sweep` over `tp: [4, 8]`
(`gpus_per_agg = {tp}`). Benchmark concurrency sweeps `1 → 1024` (`-eagle`) or a
fixed `256` (`-tp1`), at the `isl/osl` shown, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `backend: pytorch` — TRT-LLM PyTorch flow.
- `TRTLLM_ENABLE_PDL: "1"` — programmatic dependent launch on Blackwell.
- `TRTLLM_SERVER_DISABLE_GC` / `TRTLLM_WORKER_DISABLE_GC: "1"` — steadier tail latency.
- `max_num_tokens: 20000`, `max_seq_len: 2248` (1k/1k) or `9416` (8k/1k, 1k/8k).
- `precision: fp4` — NVFP4 weights.
- EAGLE3: `eagle3_one_model: true`, `max_draft_len: 3`, draft model
  `nvidia/gpt-oss-120b-Eagle3-{short,long}-context`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/gpt-oss-120b/B200/trtllm/1k1k-agg-eagle.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/gpt-oss-120b/B200/trtllm/1k1k-agg-eagle.yaml"
```

## References

- [InferenceX: `gptoss_fp4_b200_trt.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/gptoss_fp4_b200_trt.sh)
- [TensorRT-LLM GPT-OSS + Eagle3 tech blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md)
- [gpt-oss-120b model card](https://huggingface.co/openai/gpt-oss-120b)
