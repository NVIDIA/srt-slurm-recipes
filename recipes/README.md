# Recipes

## Reproducing Recipes with `srt-slurm`

Use these recipes with [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) to reproduce official inference benchmark sweeps on SLURM clusters. Run the setup from a shared filesystem that is visible to all allocated nodes.

### Prerequisites

- Access to a SLURM cluster with the GPU platform required by the recipe, for example `B200`, `B300`, `GB200`, `GB300`, or `H100`.
- Container runtime access from compute nodes for the image listed in each recipe under `model.container`.
- Model weights available to the benchmark. Recipes commonly use `model.path: hf:<owner>/<repo>`; make sure compute nodes can authenticate to Hugging Face and use a shared cache, or replace the recipe's `model.path` with a checkpoint path on shared storage. It is highly recommended to download the model ahead of time onto fast storage visible to ALL compute nodes so that model shard loading will be quick.

### Environment Setup

Clone and install `srt-slurm`, then clone this recipes repository somewhere reachable from the `srt-slurm` checkout:

```bash
# Enter a directory on NFS or another filesystem shared by all nodes.
git clone https://github.com/NVIDIA/srt-slurm.git
cd srt-slurm

uv venv
uv pip install -e .

# One-time setup. Choose the architecture that matches your cluster.
make setup ARCH=aarch64  # or ARCH=x86_64

# Clone this repository next to the srt-slurm checkout.
cd ..
git clone https://github.com/NVIDIA/srt-slurm-recipes.git srt-slurm-recipes
```

The setup step creates `srtslurm.yaml` in the `srt-slurm` checkout and prompts for cluster settings such as account and partition. Edit that file for site-specific defaults, including any additional SLURM arguments, shared cache locations, or model path overrides required by your cluster. See the upstream [`srtslurm.yaml.example`](https://github.com/NVIDIA/srt-slurm/blob/main/srtslurm.yaml.example) for supported settings.

### Running a Recipe

Choose a YAML file that matches the target node count, model, GPU platform, framework, and traffic shape. Validate the recipe first, then submit it:

```bash
cd /path/to/srt-slurm
RECIPES_PATH=/path/to/srt-slurm-recipes

uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/vllm/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/single-node/DeepSeek-R1/B300/vllm/1k1k-mtp.yaml"
```

For multi-node recipes, use the same command with a `recipes/multi-node/...` path:

```bash
uv run srtctl dry-run -f "${RECIPES_PATH}/recipes/multi-node/DeepSeek-R1/B300/sglang/1k1k-mtp.yaml"
uv run srtctl apply   -f "${RECIPES_PATH}/recipes/multi-node/DeepSeek-R1/B300/sglang/1k1k-mtp.yaml"
```

Use `srtctl dry-run` before submitting jobs to confirm the final sweep expansion and cluster configuration. After submission, use the normal `srt-slurm` job logs and analysis workflow to inspect results.

### Supported Recipes

Please refer to the [support matrix](../README.md#support-matrix) to view all recipes that are supported. Clicking on the &#9989; icon will take you to the directory containing recipes.

## Layout

This tree mirrors the official sweep matrix:

```text
<node-count>/<model>/<gpu>/<framework>/
```

Use these directory names for consistency:

- Node count: `single-node`, `multi-node`
- Model: model name only, without the model owner, for example `DeepSeek-R1`
- GPU: concrete platform name, for example `B200`, `H100`, `GB200`, or `GB300`
- Framework: inference framework name, for example `vllm`, `sglang`, or `trtllm`

Add concrete `*.yaml` files at the framework leaf when a sweep becomes official. Keep filenames descriptive, for example `isl_osl.yaml`, `isl_osl_performance-domain.yaml`, etc.
