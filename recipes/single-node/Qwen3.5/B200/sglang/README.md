# Qwen3.5-397B-A17B (FP8) — Aggregated, single-node B200 · SGLang

NVIDIA-verified aggregated serving recipes for **Qwen/Qwen3.5-397B-A17B-FP8**
(hybrid Mamba/GDN + attention MoE, FP8) on **B200** (x86_64, 8 GPU per node),
served by **native SGLang**. All recipes run on a single node.

## Container

```text
lmsysorg/sglang:v0.5.12-cu130
  sglang 0.5.12
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared.

## Model checkpoint

`Qwen/Qwen3.5-397B-A17B-FP8` (revision `ea5b4f81…781d`) is pulled via the `hf:`
handle:

```yaml
model:
  path: "hf:Qwen/Qwen3.5-397B-A17B-FP8"
  precision: "fp8"
```

## Recipes

| file | ISL / OSL | parallelism sweep | context length |
|---|---|---|---:|
| `1k1k.yaml` | 1k / 1k | TP ∈ {4, 8} (EP = TP, DP = 1) | 2068 |
| `8k1k.yaml` | 8k / 1k | TP ∈ {4, 8} (EP = TP, DP = 1) | 9236 |

Both recipes `sweep` over `tensor-parallel-size` (`expert-parallel-size` tracks
TP). Benchmark concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`,
`sa-bench`. The `8k1k.yaml` recipe additionally sets
`max-prefill-tokens` / `chunked-prefill-size: 16384`.

## Key flags

- `quantization: "fp8"`, `kv-cache-dtype: "fp8_e4m3"`, `mamba-ssm-dtype: "bfloat16"`.
- `attention-backend: "trtllm_mha"`, `moe-runner-backend: "flashinfer_trtllm"`.
- `cuda-graph-bs: [1, 32, 64, 128, 256, 512, 1024]` — trimmed to the 7 sweep sizes
  (vs SGLang's default ~52) so capture stays under ~30 s on a 397 GB FP8 cold load,
  avoiding the gateway's AddWorker retry budget at high concurrency.
- `enable-symm-mem: true`, `disable-radix-cache: true`, `mem-fraction-static: 0.8`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/Qwen3.5/B200/sglang/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/Qwen3.5/B200/sglang/1k1k.yaml"
```

## References

- [Qwen3.5-397B-A17B-FP8 model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
- Upstream benchmark reference: SemiAnalysisAI/InferenceX `benchmarks/single_node/qwen3.5_fp4_b200.sh`
