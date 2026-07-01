# DeepSeek-R1 (NVFP4) — Aggregated, single-node B300 · vLLM

NVIDIA-verified aggregated serving recipe for **nvidia/DeepSeek-R1-0528-FP4-v2**
(MoE, NVFP4 weights + FP8 KV) on **B300** (x86_64, 8 GPU per node), served through
the **dynamo** frontend with the **vLLM** backend. Runs on a single node with MTP
speculative decoding.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1-cuda13
  dynamo 1.1.1 · vllm 0.19.0+cu130
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

| file | ISL / OSL | parallelism sweep | spec decode | max-model-len |
|---|---|---|---|---:|
| `1k1k-mtp.yaml` | 1k / 1k | TP · PP ∈ {1, 2, 4, 8}, EP ∈ {on, off} | DeepSeek MTP (3 tokens) | 2176 |

The recipe `sweep`s over `tensor/pipeline-parallel-size` and toggles
`enable-expert-parallel`. Benchmark concurrency sweeps `1 → 1024`,
`random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `quantization: modelopt_fp4`, `kv-cache-dtype: "fp8_e4m3"`.
- `async-scheduling: true`, `gpu-memory-utilization: 0.9`, `no-enable-prefix-caching: true`.
- `compilation-config: {"max_cudagraph_capture_size": 1024}`.
- `speculative-config: {"method": "deepseek_mtp", "num_speculative_tokens": 3}`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/vllm/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/vllm/1k1k-mtp.yaml"
```

## References

- [DeepSeek-R1-0528-FP4-v2 model card](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2)
- [InferenceX benchmarks](https://github.com/SemiAnalysisAI/InferenceX/tree/main/benchmarks/single_node)
