# Qwen3.5-397B-A17B (NVFP4) — Aggregated, single-node RTX PRO 6000 · SGLang

NVIDIA-verified aggregated serving recipes for **nvidia/Qwen3.5-397B-A17B-NVFP4**
(hybrid Mamba/GDN + attention MoE, NVFP4) on **8x RTX PRO 6000 Blackwell**
(SM120, PCIe-only, 96 GiB/GPU), using 4 of the 8 GPUs per job, served through
the **SGLang** frontend/backend. All recipes run on a single node.

## Container

```text
lmsysorg/sglang:v0.5.16-cu130
```

v0.5.16 is the first tag whose auto backend resolution knows that trtllm-gen
MoE is SM100-only, and routes `modelopt_fp4` routed experts to FlashInfer
CUTLASS on SM120.

## Model checkpoint

`nvidia/Qwen3.5-397B-A17B-NVFP4` (revision `67eb9fda…d43f5`) is pulled via the
`hf:` handle:

```yaml
model:
  path: "hf:nvidia/Qwen3.5-397B-A17B-NVFP4"
  precision: "fp4"
```

## Recipes

| file | topology | speculative decoding |
|---|---|---|
| `agg-tp4-8k1k.yaml` | TP=4 · EP=1 | — |
| `agg-tp4-ep-8k1k.yaml` | TP=4 · EP=4 | — |
| `agg-tp4-mtp-8k1k.yaml` | TP=4 · EP=1 | MTP (EAGLE, 3 steps, top-k 1, 4 draft tokens) |
| `agg-tp4-ep-mtp-8k1k.yaml` | TP=4 · EP=4 | MTP (EAGLE, 3 steps, top-k 1, 4 draft tokens) |

All four sweep concurrency `1 x 4 x 16 x 64` at ISL/OSL `8192/1024`,
`random_range_ratio: 0.8`, `sa-bench`.

## Key flags

- `attention-backend: flashinfer`, `moe-runner-backend: flashinfer_cutlass`,
  `fp4-gemm-backend: flashinfer_cutlass` — SM120 has no trtllm-gen kernels, so
  attention, the NVFP4 GEMMs, and the routed experts all run on FlashInfer
  CUTLASS instead of the `trtllm_mha` / `flashinfer_trtllm` pair used on
  B300.
- `disable-custom-all-reduce: true` and `NCCL_IB_DISABLE: "1"` — the node is
  PCIe-only (no NVLink), so collectives use plain NCCL, and IB/RoCE probing
  is disabled to avoid a driver-level NCCL crash on this node's bnxt_re RDMA
  devices while preserving local CUDA P2P/SHM.
- `mem-fraction-static: 0.7` (non-MTP) / `0.80` (MTP) with a 2-request
  prefill chunk (`chunked-prefill-size` / `max-prefill-tokens: 16384`) —
  96 GiB/GPU leaves far less headroom than the 288 GiB/GPU B300 recipe
  assumes; 0.8 on the non-MTP arm OOM'd the first 8k prefill, and the MTP
  arm's 4.05 GiB/rank draft head needs the fraction raised, not lowered.
- `tokenizer-worker-num: 1` — the B300 recipe's six workers each pin a
  ~0.7 GiB CUDA context on GPU 0, which this SKU can't spare.
- `max-running-requests: 64` on the MTP arms — sized to the tested
  concurrency rather than a fixed 128, since an oversized Mamba state pool
  is what starves the draft head.

## Running

```bash
RECIPES_PATH=/path/to/srt-slurm-recipes
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/Qwen3.5/RTX-PRO-6000/sglang/agg-tp4-8k1k.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/Qwen3.5/RTX-PRO-6000/sglang/agg-tp4-8k1k.yaml"
```

## References

- [nvidia/Qwen3.5-397B-A17B-NVFP4 model card](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
- [Upstream InferenceX PR #2312](https://github.com/SemiAnalysisAI/InferenceX/pull/2312)
