# MiniMax-M2.5 (FP8) — Disaggregated, multi-node GB200 · vLLM · 1k/1k

NVIDIA-verified **disaggregated** (separate prefill + decode workers) serving
recipes for **MiniMaxAI/MiniMax-M2.5** (MoE, FP8) on **GB200** (ARM64 Grace +
Blackwell, 4 GPU per node), 1024 input / 1024 output. KV is transferred between
prefill and decode over **NIXL**; the **dynamo** frontend fronts the workers.

## Container

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0
  dynamo installed from wheel 1.2.0.dev20260526
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared. Dynamo is installed from a dev wheel at launch, and several
recipes run `setup_script: install-deps.sh` for additional setup.

## Model checkpoint

`MiniMaxAI/MiniMax-M2.5` is pulled via the `hf:` handle:

```yaml
model:
  path: "hf:MiniMaxAI/MiniMax-M2.5"
  precision: "fp8"
```

## Recipes

`XpYd` denotes **X prefill workers + Y decode workers**. Each GB200 node has 4
GPUs. Prefill workers are consistently TP1·DP2 with expert parallelism (2 GPU
each); the decode topology is what each recipe varies. The configs are
**rate-matched** — prefill:decode worker ratios are chosen so prefill throughput
feeds the decode tier without starving or backing up (see the per-file comments
for the `X/P` math).

| file | topology | nodes | decode parallelism |
|---|---|---:|---|
| `disagg-gb200-1p1d-tp4.yaml`    | 1P + 1D | 2 | decode TP=4 |
| `disagg-gb200-1p2d-tp4.yaml`    | 1P + 2D | 3 | decode TP=4 |
| `disagg-gb200-1p3d-tp4ep.yaml`  | 1P + 3D | 4 | decode TP=4 + expert-parallel |
| `disagg-gb200-1p4d-dep2.yaml`   | 1P + 4D | 3 | decode DEP=2 (TP1·DP2·EP) |
| `disagg-gb200-2p1d-dep8.yaml`   | 2P + 1D | 3 | decode DEP=8 (single 8-GPU worker) |
| `disagg-gb200-2p3d-dep4.yaml`   | 2P + 3D | 4 | decode DEP=4 (TP1·DP4·EP) |

Benchmark concurrency is a fixed, rate-matched value per topology (`sa-bench`,
`isl=osl=1024`, `random_range_ratio: 0.8`).

## Key flags

- `kv-transfer-config`: `NixlConnector`, `kv_role: kv_both` on both prefill and decode.
- `VLLM_FLASHINFER_ALLREDUCE_BACKEND: "mnnvl"` — MNNVL all-reduce across Grace-Blackwell.
- `kv-cache-dtype: "fp8"`, `enable-expert-parallel: true`, `no-enable-prefix-caching: true`.
- `safetensors-load-strategy: "prefetch"` — faster weight load.
- `VLLM_ENGINE_READY_TIMEOUT_S: "3600"` — long startup budget for multi-node bring-up.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/multi-node/MiniMax-M2.5/GB200/1k1k/disagg-gb200-1p1d-tp4.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/multi-node/MiniMax-M2.5/GB200/1k1k/disagg-gb200-1p1d-tp4.yaml"
```

## References

- [MiniMax-M2.5 model card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
