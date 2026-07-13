# MiniMax-M2.5 (FP8) — Disaggregated, multi-node GB200 · vLLM

NVIDIA-verified **disaggregated** serving recipes for **MiniMaxAI/MiniMax-M2.5**
(MoE, FP8) on **GB200** (4 GPU per node), served through the **dynamo** frontend
with the **vLLM** backend. Each recipe spans **multiple Slurm nodes** with separate
prefill and decode worker pools connected by **NixlConnector** KV transfer.

Layout names follow `{P}P{D}D` — prefill workers × decode workers — with suffixes
for decode parallelism mode (`tp4`, `tp4ep`, `dep2`, `dep4`, `dep8`).


## Containers

Upstream vLLM image + dynamo wheel**

```text
vllm/vllm-openai:v0.20.1
  dynamo wheel 1.2.0.dev20260526 (install-deps.sh)
```

Used by: `1p4d-dep2.yaml`, `2p3d-dep4.yaml`, and all `8k1k/` recipes.

## Model checkpoint

`MiniMaxAI/MiniMax-M2.5` is pulled via the `hf:` handle. Recipes mount a shared
HuggingFace cache and run with `HF_HUB_OFFLINE=1`:

```yaml
model:
  path: "hf:MiniMaxAI/MiniMax-M2.5"
  precision: "fp8"
```

Adjust `extra_mount` and `HF_HOME` paths for your cluster's model cache.

## Recipes — 1k / 1k

| file | layout | total GPUs | decode mode | conc sweep | target |
|---|---|---:|---|---|---|
| `1k1k/disagg-gb200-1p1d-tp4.yaml` | 1P1D | 6 | decode TP=4 | 1→64 | baseline disagg curve |
| `1k1k/disagg-gb200-1p2d-tp4.yaml` | 1P2D | 10 | decode TP=4 × 2 workers | 2→512 | scale decode replicas |
| `1k1k/disagg-gb200-1p3d-tp4ep.yaml` | 1P3D | 14 | decode TP=4 EP × 3 workers | 1024 | high-conc 1P3D |
| `1k1k/disagg-gb200-1p4d-dep2.yaml` | 1P4D | 10 | decode DP=2 × 4 workers | 4096 | rate-matched dep2 |
| `1k1k/disagg-gb200-2p1d-dep8.yaml` | 2P1D | 12 | decode DP=8 × 1 worker | 512, 1024 | dual prefill |
| `1k1k/disagg-gb200-2p3d-dep4.yaml` | 2P3D | 16 | decode DP=4 × 3 workers | 4096, 8192 | peak 1k/1k throughput |

All 1k/1k recipes use `isl: 1024`, `osl: 1024`, `sa-bench`, `random_range_ratio: 0.8`.

## Recipes — 8k / 1k

| file | layout | total GPUs | decode mode | conc sweep | target |
|---|---|---:|---|---|---|
| `8k1k/disagg-gb200-1p1d-tp4.yaml` | 1P1D | 6 | decode TP=4 | 1→128 | long-context baseline |
| `8k1k/disagg-gb200-1p1d-tp4ep.yaml` | 1P1D EP | 6 | decode TP=4 + EP | 256, 512 | EP variant @ high conc |
| `8k1k/disagg-gb200-3p2d-dep4.yaml` | 3P2D | 14 | decode DP=4 × 2 workers | 1024→4096 | peak 8k/1k throughput |

8k/1k recipes set `isl: 8192`, `osl: 1024`. Prefill uses
`max-num-batched-tokens: 16384` where long context is exercised.

## Parallelism conventions

Every recipe shares the same prefill worker template:

- `tensor-parallel-size: 1`, `data-parallel-size: 2`, `enable-expert-parallel: true`
- `gpus_per_prefill: 2` (one prefill worker = 2 GB200 GPUs)

Decode workers vary by layout:

| suffix | decode config | meaning |
|---|---|---|
| `tp4` | `tensor-parallel-size: 4` | single TP=4 decode replica per worker |
| `tp4ep` | TP=4 + `enable-expert-parallel: true` | expert-parallel decode |
| `dep2` | `data-parallel-size: 2` + EP | 2-way DP decode per worker |
| `dep4` | `data-parallel-size: 4` + EP | 4-way DP decode per worker |
| `dep8` | `data-parallel-size: 8` + EP | 8-way DP decode (wide single worker) |

`rate-matched` layouts (`dep2`, `dep4`) pick decode worker counts so prefill
output rate ≈ aggregate decode capacity. Comments in each YAML document the
target ratio.

## Key flags

- **KV transfer:** `NixlConnector` on both prefill and decode
  (`kv-transfer-config: '{"kv_connector": "NixlConnector", "kv_role": "kv_both"}'`).
- **Cache:** `kv-cache-dtype: "fp8"`, `no-enable-prefix-caching: true`.
- **Loading:** `safetensors-load-strategy: "prefetch"`, `trust-remote-code: true`.
- **GB200 comms:** `VLLM_FLASHINFER_ALLREDUCE_BACKEND: "mnnvl"`.
- **Startup:** `VLLM_ENGINE_READY_TIMEOUT_S: "3600"` (slow cold load on MoE).
- **Stream:** `stream-interval: 32` (8k/1k and baseline) or `128` (high-conc 1k/1k).

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes

# 1k/1k — full 1P1D concurrency curve
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/multi-node/MiniMax-M2.5/GB200/vllm/1k1k/disagg-gb200-1p1d-tp4.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/multi-node/MiniMax-M2.5/GB200/vllm/1k1k/disagg-gb200-1p1d-tp4.yaml"

# 1k/1k — peak throughput layout
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/multi-node/MiniMax-M2.5/GB200/vllm/1k1k/disagg-gb200-2p3d-dep4.yaml"

# 8k/1k — long-context sweep
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/multi-node/MiniMax-M2.5/GB200/vllm/8k1k/disagg-gb200-3p2d-dep4.yaml"
```

Outputs land under `<srt-slurm>/outputs/<jobid>/` (or a renamed result directory).
Visualize with the Pareto canvas kit — see [`scripts/README.md`](../../../../../scripts/README.md).


## References

- [InferenceX — MiniMax-M2.5 benchmarks](https://inferencex.semianalysis.com/inference)
- [MiniMax-M2.5 model card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) 

