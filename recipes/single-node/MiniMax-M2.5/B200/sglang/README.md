# MiniMax-M2.5 (FP8) — Aggregated, single-node B200 · SGLang

NVIDIA-verified aggregated serving recipes for **MiniMaxAI/MiniMax-M2.5** (MoE, FP8)
on **B200** (x86_64, 8 GPU per node), served with the **native SGLang** frontend and
backend. All recipes run on a single node.

## Container

```text
lmsysorg/sglang:v0.5.12-cu130
  sglang 0.5.12
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared. (This recipe uses the upstream SGLang image and native SGLang
frontend rather than the dynamo runtime.)

## Model checkpoint

`MiniMaxAI/MiniMax-M2.5` (revision `f710177d…f21f`) is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:MiniMaxAI/MiniMax-M2.5"
  precision: "fp8"
```

## Recipes

| file | ISL / OSL | parallelism sweep | context length | target |
|---|---|---|---:|---|
| `1k1k.yaml` | 1k / 1k | TP = EP ∈ {2, 4, 8}, DP=1 | 2068 | 1k/1k throughput sweep |
| `8k1k.yaml` | 8k / 1k | TP = EP ∈ {2, 4, 8}, DP=1 | 9236 | 8k/1k throughput sweep |

Both recipes `sweep` over `tp` (with `expert-parallel-size = {tp}`). The 8k/1k
recipe adds `chunked-prefill-size` / `max-prefill-tokens: 16384`. Benchmark
concurrency sweeps `1 → 1024`, `random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `tool-call-parser: "minimax-m2"`, `reasoning-parser: "minimax-append-think"`.
- `mem-fraction-static: 0.85`, `disable-radix-cache: true`, `enable-symm-mem: true`.
- `stream-interval: 50`, `scheduler-recv-interval: 30`, `tokenizer-worker-num: 6`.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/MiniMax-M2.5/B200/sglang/1k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/MiniMax-M2.5/B200/sglang/1k1k.yaml"
```

## References

- [MiniMax-M2.5 SGLang deploy guide](https://github.com/MiniMax-AI/MiniMax-M2.5/blob/main/docs/sglang_deploy_guide.md)
- [MiniMax-M2.5 model card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
