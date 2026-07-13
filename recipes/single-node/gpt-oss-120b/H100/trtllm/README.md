# GPT-OSS-120B (FP4) — Aggregated, single-node H100 · TensorRT-LLM

NVIDIA-verified aggregated serving recipes for **openai/gpt-oss-120b** (MoE, FP4
weights) on **H100** (x86_64, 8 GPU per node), served through the **dynamo**
frontend with the **TensorRT-LLM** PyTorch backend. All recipes run on a single
node and use EAGLE3 speculative decoding.

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

Both recipes also pull the EAGLE3 draft model
`nvidia/gpt-oss-120b-Eagle3-{short,long}-context`.

## Recipes

| file | ISL / OSL | parallelism | spec decode | max batch | target |
|---|---|---|---|---:|---|
| `1k1k-agg-eagle.yaml` | 1k / 1k | TP ∈ {4, 8} | EAGLE3 one-model, draft=3 (short-ctx) | 1024 | low-latency / spec-decode 1k/1k |
| `8k1k-agg-eagle.yaml` | 8k / 1k | TP ∈ {4, 8} | EAGLE3 one-model, draft=3 (long-ctx)  | 1024 | low-latency / spec-decode 8k/1k |

Unlike the B200 set, the H100 folder ships only the EAGLE3 recipes (no TP-1
baseline). Both `sweep` over `tp: [4, 8]` (`gpus_per_agg = {tp}`). Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `backend: pytorch` — TRT-LLM PyTorch flow.
- `TRTLLM_ENABLE_PDL: "1"` — programmatic dependent launch.
- `TRTLLM_SERVER_DISABLE_GC` / `TRTLLM_WORKER_DISABLE_GC: "1"` — steadier tail latency.
- `max_num_tokens: 20000`, `max_seq_len: 2248` (1k/1k) or `9416` (8k/1k).
- `precision: fp4` — FP4 weights.
- EAGLE3: `eagle3_one_model: true`, `max_draft_len: 3`, draft model
  `nvidia/gpt-oss-120b-Eagle3-{short,long}-context`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/gpt-oss-120b/H100/trtllm/1k1k-agg-eagle.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/gpt-oss-120b/H100/trtllm/1k1k-agg-eagle.yaml"
```

## References

- [TensorRT-LLM GPT-OSS + Eagle3 tech blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md)
- [gpt-oss-120b model card](https://huggingface.co/openai/gpt-oss-120b)
