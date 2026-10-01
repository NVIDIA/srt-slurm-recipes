# DeepSeek-V4-Pro AgentX recipes (draft)

Local port of SemiAnalysis InferenceX **agentic-coding** (`cc-traces-weka`)
serving configs into the official `recipes/<node-count>/<model>/<gpu>/<framework>/`
layout. **Not pushed** — review before opening an MR.

Workload: Hugging Face `semianalysisai/cc-traces-weka-062126` via
`WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126`. Throughput uses
SPEED-Bench golden synthetic MTP acceptance length; eval keeps real verification.

Support matrix: [README AgentX](../../../README.md#agentx). Per-recipe index:
[recipes/multi-node/AGENTX.md](../AGENTX.md). Each YAML header has the
InferenceX blob URL.

## How to run

These recipes use `benchmark.type: custom` and call InferenceX
`benchmarks/multi_node/agentic_srt.sh` at `/infmax-workspace`. They are not
`sa-bench` 8k/1k sweeps. You need the InferenceX tree mounted as in the
AgentX CI flow, then:

```bash
cd /path/to/srt-slurm
RECIPES_PATH=/path/to/srt-slurm-recipes

uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/multi-node/DeepSeek-V4-Pro/GB300/vllm/agentic/agg-gb300-tp8-mtp-agentic.yaml"
```

`model.path` is often the cluster alias `deepseek-v4-pro`. Official 8k1k
recipes in this repo use `hf:deepseek-ai/DeepSeek-V4-Pro` (or
`hf:nvidia/DeepSeek-V4-Pro-NVFP4` on some B300 files). Override at apply time
if your cluster does not define the alias.

## Single-node

The B200/B300 single-node campaigns are translated from their launchers into
`recipes/single-node/DeepSeek-V4-Pro/*/*/agentic/`; see
[recipes/single-node/AGENTX.md](../../single-node/AGENTX.md). The launchers size
`max-num-seqs`, CUDA graphs, HiCache ratio and the SimpleCPU byte budget from
`CONC` / `TP` at runtime, so each arm keeps every published concurrency as a
`zip_override_conc` variant instead of freezing one.

## Intentionally not converted yet

**Missing InferenceX YAML** (referenced by `dsv4-fp4-gb300-dynamo-vllm-agentic`
in `nvidia-master.yaml`, files not in the InferenceX tree):

- `recipes/vllm/deepseek-v4/agentic/disagg-gb300-1p6d-dep4-tp4-agentic.yaml`
- `recipes/vllm/deepseek-v4/agentic/disagg-gb300-4p1d-dep4-dep8-24-c4096-agentic.yaml`

Closest 8k1k cousins exist under `vllm/deepseek-v4/8k1k/` with `sa-bench`,
not AgentX custom bench.

AMD MI355X AgentX bash is out of scope for this NVIDIA recipes repo.
