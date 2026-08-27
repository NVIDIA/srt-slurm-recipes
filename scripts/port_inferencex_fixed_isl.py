#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Port InferenceX's published fixed-ISL/OSL recipes into this repository.

InferenceX publishes srt-slurm recipes under
`benchmarks/multi_node/srt-slurm-recipes/<framework>/<model>/...`. They are
already valid srt-slurm configs, so porting is a copy plus two edits: a
provenance header, and a `model.path` rewrite (see MODEL_PATHS below).

The script is idempotent and skips any upstream recipe whose `name:` already
exists here, so it doubles as a sync check when InferenceX publishes more:

    python scripts/port_inferencex_fixed_isl.py --inferencex ../InferenceX --dry-run

AgentX recipes are out of scope; they carry per-concurrency engine parameters
and are handled by scripts/generate_agentx_single_node.py and by direct copies.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_ROOT = REPO_ROOT / "recipes"

UPSTREAM_SUBDIR = Path("benchmarks/multi_node/srt-slurm-recipes")
BLOB_BASE = "https://github.com/SemiAnalysisAI/InferenceX/blob"

# Directories holding AgentX recipes rather than fixed-ISL sweeps.
EXCLUDED_DIR_NAMES = {"agentic"}
EXCLUDED_PATH_TOKENS = ("agentx",)


@dataclass(frozen=True)
class Family:
    """How one upstream model family maps into this repository."""

    model_dir: str
    # Appended to the ISL/OSL tag. DeepSeek-V4-Pro's InferenceX-sourced sweeps
    # already live in `8k1k-sa` directories, so new ones join them there.
    scenario_suffix: str = ""
    # MiniMax-M3's FP4 and FP8 sweeps reuse filenames across upstream's
    # precision-scoped directories (b300-fp4/1k1k and b300-fp8/1k1k both hold
    # 1p1d-dep2-tep8-1k1k.yaml), so the precision goes into the filename.
    precision_in_filename: bool = False


# Keyed by the upstream model directory name, longest match first.
FAMILIES = {
    "deepseek-v4": Family(model_dir="DeepSeek-V4-Pro", scenario_suffix="-sa"),
    "minimax-m3": Family(model_dir="MiniMax-M3", precision_in_filename=True),
}

# InferenceX recipes name their weights with a launcher alias that its runners
# define at submit time (`model_paths:` in the generated srtslurm.yaml, pointing
# at weights pre-staged on their clusters). Those aliases do not resolve outside
# InferenceX, so rewrite them to the `hf:` form this repository documents.
#
# The MXFP4 DeepSeek alias maps to the base checkpoint on purpose: MXFP4 is
# applied at load time by the serving framework, which is also how the MXFP4
# recipes already in this repository are written.
MODEL_PATHS = {
    "deepseek-v4-pro": "hf:deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-v4-pro-mxfp4": "hf:deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Pro": "hf:deepseek-ai/DeepSeek-V4-Pro",
    "minimax-m3-mxfp8": "hf:MiniMaxAI/MiniMax-M3-MXFP8",
    "MiniMaxAI/MiniMax-M3-MXFP8": "hf:MiniMaxAI/MiniMax-M3-MXFP8",
    "minimax-m3-nvfp4": "hf:nvidia/MiniMax-M3-NVFP4",
    "nvidia/MiniMax-M3-NVFP4": "hf:nvidia/MiniMax-M3-NVFP4",
}

# `benchmark.tokenizer_mode: deepseek_v4` predates srt-slurm's `custom_tokenizer`,
# which takes the `module.path.ClassName` of an adapter shipped under
# benchmarks/scripts/sa-bench/sa_bench_tokenizers/. These are the adapters for
# DeepSeek-V4's chat template, and are what upstream's own recipes (and the
# sibling InferenceX recipes that were written later) already name.
DEEPSEEK_V4_TOKENIZERS = {
    "vllm": "sa_bench_tokenizers.vllm_deepseek_v4.VLLMDeepseekV4Tokenizer",
    "sglang": "sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer",
}


def load(text: str) -> dict[str, Any]:
    """Upstream mixes block and flow style (one benchmark block is inline), so
    read the document rather than pattern-matching lines."""
    document = yaml.safe_load(text)
    return document if isinstance(document, dict) else {}


def length_tag(tokens: int) -> str:
    return f"{tokens // 1024}k" if tokens and tokens % 1024 == 0 else str(tokens)


def node_count(config: dict[str, Any]) -> str:
    resources = config.get("resources") or {}
    total = int(resources.get("prefill_nodes") or 0) + int(
        resources.get("decode_nodes") or 0
    )
    # Colocated 1P1D recipes declare one prefill node and no decode node, which
    # is a single-node deployment.
    total = total or int(resources.get("nodes") or 0) or 1
    return "multi-node" if total > 1 else "single-node"


def family_for(path: Path) -> Family | None:
    """Match against the upstream model directory, e.g. `minimax-m3-gb300-fp8`."""
    model_dir = path.parts[1] if len(path.parts) > 1 else ""
    for key in sorted(FAMILIES, key=len, reverse=True):
        if model_dir.startswith(key):
            return FAMILIES[key]
    return None


def is_fixed_isl(path: Path) -> bool:
    if EXCLUDED_DIR_NAMES & set(path.parts):
        return False
    return not any(token in str(path) for token in EXCLUDED_PATH_TOKENS)


def existing_names() -> dict[str, Path]:
    names: dict[str, Path] = {}
    for path in RECIPES_ROOT.rglob("*.yaml"):
        # Sweep files use `base:`/`zip_override_*`, which safe_load handles, but
        # a malformed recipe should not abort the whole scan.
        try:
            name = load(path.read_text(encoding="utf-8")).get("name")
        except yaml.YAMLError:
            continue
        if isinstance(name, str):
            names.setdefault(name, path)
    return names


def split_leading_comments(text: str) -> tuple[list[str], str]:
    lines = text.splitlines()
    index = 0
    while index < len(lines) and (
        lines[index].startswith("#") or not lines[index].strip()
    ):
        index += 1
    comments = [line for line in lines[:index] if line.startswith("#")]
    return comments, "\n".join(lines[index:]).strip() + "\n"


def drop_top_level_block(body: str, key: str) -> tuple[str, bool]:
    """Remove a top-level mapping and its children, plus the blank line after."""
    lines = body.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"{key}:")), None
    )
    if start is None:
        return body, False

    def belongs(index: int) -> bool:
        if lines[index].startswith((" ", "\t")):
            return True
        # A blank line is part of the block only if indented content follows.
        if lines[index].strip():
            return False
        following = next(
            (line for line in lines[index + 1 :] if line.strip()),
            "",
        )
        return following.startswith((" ", "\t"))

    end = start + 1
    while end < len(lines) and belongs(end):
        end += 1
    if end < len(lines) and not lines[end].strip():
        end += 1

    del lines[start:end]
    return "\n".join(lines).rstrip() + "\n", True


def rewrite_tokenizer_mode(body: str, framework: str) -> tuple[str, bool]:
    adapter = DEEPSEEK_V4_TOKENIZERS.get(framework)
    if not adapter:
        return body, False
    body, count = re.subn(
        r'^(\s*)tokenizer_mode:\s*"?deepseek_v4"?\s*$',
        lambda match: f'{match.group(1)}custom_tokenizer: "{adapter}"',
        body,
        flags=re.M,
    )
    return body, bool(count)


def compat_notes(
    dropped_telemetry: bool, moved_tokenizer: bool, framework: str
) -> list[str]:
    """Explain each edit made to keep the recipe loadable on srt-slurm main."""
    lines: list[str] = []
    if dropped_telemetry:
        lines += [
            "#",
            "# The upstream telemetry block is dropped. It sets a `provider` field that",
            "# srt-slurm no longer accepts (DCGM power is now the only provider, so the",
            "# field went away), and it pairs `enabled: true` with `required: true`,",
            "# which aborts a run when no DCGM exporter answers — InferenceX-cluster",
            "# infrastructure that a recipe here cannot assume. No other recipe in this",
            "# repository configures telemetry.",
        ]
    if moved_tokenizer:
        lines += [
            "#",
            "# benchmark.tokenizer_mode became benchmark.custom_tokenizer, naming the",
            f"# {framework} DeepSeek-V4 chat-template adapter that srt-slurm ships. This",
            "# is the same value upstream's own recipes and the sibling InferenceX",
            "# recipes use.",
        ]
    return lines


def build_header(relpath: Path, revision: str, alias: str, target: str | None) -> list[str]:
    lines = [
        "# Ported from SemiAnalysis InferenceX (published fixed-ISL/OSL recipe):",
        f"#   {BLOB_BASE}/{revision}/{UPSTREAM_SUBDIR / relpath}",
    ]
    if target and target != alias:
        lines.append("#")
        if target == f"hf:{alias}":
            lines += [
                f'# model.path gains the "hf:" prefix: upstream writes "{alias}",',
                "# which srtctl would resolve as a cluster model_paths alias or a",
                "# filesystem path rather than a Hugging Face ID.",
            ]
        else:
            lines += [
                f'# model.path is rewritten from the launcher alias "{alias}", which',
                "# InferenceX resolves through srtslurm.yaml:model_paths against weights",
                "# pre-staged on its own clusters.",
            ]
    return lines


def port(source: Path, relpath: Path, revision: str) -> tuple[Path, str]:
    text = source.read_text(encoding="utf-8")
    config = load(text)
    family = family_for(relpath)
    assert family is not None

    gpu = str((config.get("resources") or {}).get("gpu_type") or "unknown").upper()
    benchmark = config.get("benchmark") or {}
    tag = f"{length_tag(int(benchmark.get('isl') or 0))}" + (
        f"{length_tag(int(benchmark.get('osl') or 0))}"
    )
    scenario = f"{tag}{family.scenario_suffix}"

    # Keep any grouping directory upstream nests below the ISL/OSL level, e.g.
    # the `mtp/` speculative-decoding variants.
    parts = list(relpath.parts[2:-1])
    extra = parts[parts.index(tag) + 1 :] if tag in parts else []

    stem = source.stem
    model = config.get("model") or {}
    precision = model.get("precision")
    if family.precision_in_filename and precision:
        stem = f"{stem}-{precision}"

    destination = (
        RECIPES_ROOT
        / node_count(config)
        / family.model_dir
        / gpu
        / str((config.get("backend") or {}).get("type") or "unknown")
        / scenario
        / Path(*extra)
        / f"{stem}.yaml"
    )

    comments, body = split_leading_comments(text)
    framework = str((config.get("backend") or {}).get("type") or "")
    alias = str(model.get("path") or "")
    target = MODEL_PATHS.get(alias)
    if target:
        body, count = re.subn(
            r'^(\s*path:\s*)"?[^"\n]+"?\s*$',
            lambda match: f'{match.group(1)}"{target}"',
            body,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise SystemExit(f"{relpath}: could not rewrite model.path in place")

    body, dropped_telemetry = drop_top_level_block(body, "telemetry")
    body, moved_tokenizer = rewrite_tokenizer_mode(body, framework)

    lines = build_header(relpath, revision, alias, target)
    notes = compat_notes(dropped_telemetry, moved_tokenizer, framework)
    if target and target != alias and not notes:
        lines.append("# The engine configuration below is upstream's, unchanged.")
    lines += notes
    if comments:
        lines += ["#", "# Upstream notes:"] + [
            f"#   {line.lstrip('#').strip()}" if line.strip("# ") else "#"
            for line in comments
        ]
    return destination, "\n".join(lines) + "\n\n" + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inferencex",
        type=Path,
        default=REPO_ROOT.parent / "InferenceX",
        help="path to an InferenceX checkout (default: ../InferenceX)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without touching the tree",
    )
    args = parser.parse_args()

    upstream = args.inferencex / UPSTREAM_SUBDIR
    if not upstream.is_dir():
        print(f"error: {upstream} not found", file=sys.stderr)
        return 1

    revision = revision_of(args.inferencex)
    known = existing_names()
    written = skipped = 0
    collisions: list[str] = []

    for source in sorted(upstream.rglob("*.yaml")):
        relpath = source.relative_to(upstream)
        if not is_fixed_isl(relpath) or family_for(relpath) is None:
            continue
        name = load(source.read_text(encoding="utf-8")).get("name")
        if not isinstance(name, str):
            continue
        if name in known:
            skipped += 1
            continue

        destination, content = port(source, relpath, revision)
        if destination.exists():
            collisions.append(str(destination.relative_to(REPO_ROOT)))
            continue
        print(f"  {relpath}\n    -> {destination.relative_to(REPO_ROOT)}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        written += 1

    verb = "would port" if args.dry_run else "ported"
    print(f"\n{verb} {written} recipes; {skipped} already present")
    if collisions:
        print(f"error: {len(collisions)} destination collisions:", file=sys.stderr)
        for item in collisions:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


def revision_of(checkout: Path) -> str:
    """Pin provenance links to a commit so they survive upstream moves."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "main"


if __name__ == "__main__":
    raise SystemExit(main())
