# GLM-5 (NVFP4 + FP8) — Disaggregated, multi-node GB300 · SGLang

NVIDIA-verified **disaggregated** (separate prefill + decode workers) serving
recipes for **GLM-5** (MoE with native sparse attention) on **GB300** (ARM64
Grace + Blackwell, 4 GPU per node), in two quantizations:

- **NVFP4** — `nvidia/GLM-5-NVFP4` (`glm5-nvfp4-*`), 1k/1k and 8k/1k.
- **FP8** — `zai-org/GLM-5-FP8` (`glm5-fp8-*`), 1k/1k and 8k/1k.

KV is transferred between the prefill and decode tiers over **NIXL**, and the
**dynamo** frontend fronts the workers. Recipes cover **low-latency** and
**throughput** tuning profiles.

## Container

```text
lmsysorg/sglang:v0.5.11-cu130
  dynamo 1.1.0 (installed at launch: dynamo.install = true)
```

The image is referenced directly in each recipe's `model.container`; no local
mounts are declared. All recipes target the `gb300` SLURM partition with a 3h
time limit.

## Model checkpoint

Both checkpoints are pulled via the `hf:` handle:

```yaml
# NVFP4 recipes (glm5-nvfp4-*)
model:
  path: "hf:nvidia/GLM-5-NVFP4"     # revision dc54ff55…44c6
  precision: "fp4"

# FP8 recipes (glm5-fp8-*)
model:
  path: "hf:zai-org/GLM-5-FP8"      # revision 4f96cc5e…9516
  precision: "fp8"
```

## Recipes

Serving profiles, each tuned for a different point on the latency/throughput
curve:

- **`lowlat*`** — many small decode workers (one 4-GPU worker per node, TP=4),
  small running-batch, MNNVL all-reduce fusion. Minimizes per-user latency.
- **`maxtpt*`** (NVFP4) — a single wide decode worker spanning **8 nodes / 32 GPU**
  (TP=32 · EP=32 · DP=32) with **DeepEP** expert all-to-all and 32 redundant
  experts. Maximizes total throughput; prefill workers are scaled to keep the
  decode tier fed (more/longer inputs ⇒ more prefill workers).
- **`hightpt*`** (FP8) — same wide-DeepEP idea, but holds a **fixed 18-node /
  72-GPU budget** and sweeps the **prefill : decode split** (decode EP domain
  24 → 56 GPU) to find the best rate-match at high concurrency.

Prefill is consistently **TP=4 · DP=4** (DP-attention + DP-LM-head, one 4-GPU
worker per node). `XP + YD` below denotes **X prefill workers + Y decode
workers**.

### NVFP4 — 1k / 1k

| file | topology | nodes / GPUs | decode parallelism | target | concurrency |
|---|---|---|---|---|---|
| `glm5-nvfp4-1k1k-lowlat1-c32.yaml` | 1P + 17D | 18 / 72 | TP=4 · EP=1 · DP=1, batch 1 | min latency | 32 |
| `glm5-nvfp4-1k1k-lowlat0-c512-256-128-64.yaml` | 1P + 17D | 18 / 72 | TP=4 · EP=1 · DP=1, batch 32 | low-latency curve | 512 → 64 |
| `glm5-nvfp4-1k1k-maxtpt2-c2500-1024.yaml` | 1P + 1D | 9 / 36 | TP=32 · EP=32 · DP=32, DeepEP | throughput | 2500, 1024 |
| `glm5-nvfp4-1k1k-maxtpt1-c8300.yaml` | 2P + 1D | 10 / 40 | TP=32 · EP=32 · DP=32, DeepEP | max throughput | 8300 |

### NVFP4 — 8k / 1k

| file | topology | nodes / GPUs | decode parallelism | target | concurrency |
|---|---|---|---|---|---|
| `glm5-nvfp4-8k1k-lowlat2-c32.yaml` | 1P + 9D | 10 / 40 | TP=4 · EP=1 · DP=1, batch 32 | min latency | 32 |
| `glm5-nvfp4-8k1k-lowlat1-c64.yaml` | 1P + 5D | 6 / 24 | TP=4 · EP=1 · DP=1, batch 64 | low latency | 64 |
| `glm5-nvfp4-8k1k-maxtpt0-c2048.yaml` | 5P + 1D | 13 / 52 | TP=32 · EP=32 · DP=32, DeepEP | throughput | 2048 |
| `glm5-nvfp4-8k1k-maxtpt2-c4096.yaml` | 10P + 1D | 18 / 72 | TP=32 · EP=32 · DP=32, DeepEP | max throughput | 4096 |

### FP8 — 1k / 1k

All FP8 configs hold a fixed 18-node / 72-GPU budget; `hightpt*` sweeps the
prefill : decode split (decode is a single wide DeepEP worker).

| file | topology | nodes / GPUs | decode parallelism | target | concurrency |
|---|---|---|---|---|---|
| `glm5-fp8-1k1k-lowlat1-c32.yaml` | 1P + 17D | 18 / 72 | TP=4 · EP=1 · DP=1, batch 1 | min latency | 32 |
| `glm5-fp8-1k1k-lowlat0-c512-256-128-64.yaml` | 1P + 17D | 18 / 72 | TP=4 · EP=1 · DP=1, batch 32 | low-latency curve | 512 → 64 |
| `glm5-fp8-1k1k-hightpt4-c5700.yaml` | 4P + 14D | 18 / 72 | TP=56 · EP=56 · DP=56, DeepEP | throughput | 5700 |
| `glm5-fp8-1k1k-hightpt3-c6500.yaml` | 6P + 12D | 18 / 72 | TP=48 · EP=48 · DP=48, DeepEP | throughput | 6500 |
| `glm5-fp8-1k1k-hightpt2-c7300.yaml` | 8P + 10D | 18 / 72 | TP=40 · EP=40 · DP=40, DeepEP | throughput | 7300 |
| `glm5-fp8-1k1k-hightpt1-c7500.yaml` | 10P + 8D | 18 / 72 | TP=32 · EP=32 · DP=32, DeepEP | throughput | 7500 |
| `glm5-fp8-1k1k-hightpt0-c8192.yaml` | 12P + 6D | 18 / 72 | TP=24 · EP=24 · DP=24, DeepEP | max throughput | 8192 |

### FP8 — 8k / 1k

The `hightpt*` recipes hold a fixed 18-node / 72-GPU budget and sweep the
prefill : decode split for the longer 8k input; the `lowlat*` recipes scale out
small TP=4 decode workers for the best per-user latency.

| file | topology | nodes / GPUs | decode parallelism | target | concurrency |
|---|---|---|---|---|---|
| `glm5-fp8-8k1k-lowlat2-c24.yaml` | 1P + 17D | 18 / 72 | TP=4 · EP=1 · DP=1, batch 1 | min latency | 24 |
| `glm5-fp8-8k1k-lowlat1-c128-64-32.yaml` | 1P + 17D | 18 / 72 | TP=4 · EP=1 · DP=1, batch 8 | low-latency curve | 128 → 32 |
| `glm5-fp8-8k1k-lowlat0-c150.yaml` | 1P + 9D | 10 / 40 | TP=4 · EP=1 · DP=1, batch 15 | low latency | 150 |
| `glm5-fp8-8k1k-hightpt3-c900.yaml` | 8P + 10D | 18 / 72 | TP=40 · EP=40 · DP=40, DeepEP | throughput | 900 |
| `glm5-fp8-8k1k-hightpt2-c1300.yaml` | 10P + 8D | 18 / 72 | TP=32 · EP=32 · DP=32, DeepEP | throughput | 1300 |
| `glm5-fp8-8k1k-hightpt1-c1700.yaml` | 12P + 6D | 18 / 72 | TP=24 · EP=24 · DP=24, DeepEP | throughput | 1700 |
| `glm5-fp8-8k1k-hightpt0-c2800.yaml` | 14P + 4D | 18 / 72 | TP=16 · EP=16 · DP=16, DeepEP | max throughput | 2800 |

Decode workers in the `maxtpt`/`hightpt` recipes are a single wide worker
(`decode_workers: 1`) spanning all decode nodes; `lowlat` recipes run one
independent 4-GPU decode worker per node (`decode_workers == decode_nodes`).
Benchmark is `sa-bench`, `osl=1024`, `req_rate: inf`; concurrency is swept per
the `sweep.conc` list.

## Key flags

**Shared (prefill + decode)**

- `quantization`: `"modelopt_fp4"` (NVFP4 recipes) / `"fp8"` (FP8 recipes); `kv-cache-dtype: "fp8_e4m3"` for both.
- `disaggregation-mode` + `disaggregation-transfer-backend: "nixl"`.
- `nsa-decode-backend` / `nsa-prefill-backend: "trtllm"` — native sparse attention kernels.
- `fp4-gemm-backend: "flashinfer_cutlass"`, `disable-radix-cache: true`.
- `MC_FORCE_MNNVL: "1"` + `NCCL_MNNVL_ENABLE: "1"` — MNNVL fabric across Grace-Blackwell.

**Prefill**

- `tensor-parallel-size: 4`, `data-parallel-size: 4`, `enable-dp-attention` + `enable-dp-lm-head`.
- `moe-runner-backend: "flashinfer_trtllm"`, `enable-flashinfer-allreduce-fusion: true`.
- `chunked-prefill-size: 32768`, `max-prefill-tokens: 8192`, `load-balance-method: "total_tokens"`.

**Decode — `maxtpt` / `hightpt` (wide DeepEP)**

- `tensor-parallel-size` = `expert-parallel-size` = `data-parallel-size` = decode GPU count
  (NVFP4 `maxtpt`: 32; FP8 `hightpt`: 16–56), `moe-dense-tp-size: 1`.
- `moe-a2a-backend: "deepep"`, `deepep-mode: "low_latency"`, `deepep-config: "/configs/deepep_config.json"` (container-internal).
- `ep-num-redundant-experts: 24–32`, `ep-dispatch-algorithm: "static"`.
- NVFP4 `maxtpt` decode pins `moe-runner-backend: "flashinfer_cutedsl"`; FP8 `hightpt` leaves it at the SGLang default.

**Decode — `lowlat` (small TP4 workers)**

- `tensor-parallel-size: 4`, `expert-parallel-size: 1`, `data-parallel-size: 1`.
- `moe-runner-backend: "flashinfer_trtllm"`, `enable-flashinfer-allreduce-fusion: true`.
- Small `max-running-requests` / `cuda-graph-max-bs` (1–64) to hold TPOT down.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/multi-node/GLM-5/GB300/sglang/glm5-nvfp4-1k1k-maxtpt1-c8300.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/multi-node/GLM-5/GB300/sglang/glm5-nvfp4-1k1k-maxtpt1-c8300.yaml"
```

## References

- [GLM-5-NVFP4 model card](https://huggingface.co/nvidia/GLM-5-NVFP4)
- [GLM-5-FP8 model card](https://huggingface.co/zai-org/GLM-5-FP8)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
