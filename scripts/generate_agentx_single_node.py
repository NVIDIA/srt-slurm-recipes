#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the single-node AgentX recipes from the published InferenceX results.

There is no published single-node AgentX YAML to copy the way there is for
multi-node, so the recipe set is driven by what InferenceX actually published:

  * The dashboard API (``/api/v1/benchmarks``) supplies the point set. Every
    ``agentic_traces`` row for a non-GLM NVIDIA single-node run is one measured
    point, carrying its serving configuration, concurrency, container image and
    run URL. ``scripts/agentx_published_points.json`` is the committed snapshot;
    refresh it with ``--refresh-points``.
  * ``configs/nvidia-master.yaml`` and ``benchmarks/single_node/agentic/*.sh``
    supply the engine configuration behind each point, since the API reports the
    topology but not the engine flags.
  * ``configs/runners.yaml`` supplies the fleet facts used to reproduce
    ``TOTAL_CPU_DRAM_GB`` exactly as ``generate_sweep_configs.py`` computes it.

Points that share a serving configuration are combined: one recipe per
configuration, with that configuration's published concurrencies as
``base`` + ``zip_override_conc`` variants. Concurrency is not merely a client
knob here — the launchers derive engine settings from it — so the variants carry
those per-concurrency values. Submit one point with
``srtctl apply -f <recipe>:zip_override_conc[<i>]`` or the whole curve with
``srtctl apply -f <recipe>``.

Usage: python scripts/generate_agentx_single_node.py [--inferencex PATH]
                                                     [--check] [--refresh-points]
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import shutil
import sys
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_ROOT = REPO_ROOT / "recipes" / "single-node"
DEFAULT_INFERENCEX = REPO_ROOT.parent / "InferenceX"
POINTS_SNAPSHOT = Path(__file__).resolve().parent / "agentx_published_points.json"

BLOB = "https://github.com/SemiAnalysisAI/InferenceX/blob/main"
AGENTIC_SRT = f"{BLOB}/benchmarks/multi_node/agentic_srt.sh"
AGENTIC_SRT_COMMAND = "bash /infmax-workspace/benchmarks/multi_node/agentic_srt.sh"
DASHBOARD_API = "https://inferencex.semianalysis.com/api/v1/benchmarks"
# Dashboard display name -> InferenceX model prefix. GLM is out of scope.
API_MODELS = {
    "DeepSeek-V4-Pro": "dsv4",
    "Qwen-3.5-397B-A17B": "qwen3.5",
    "MiniMax-M3": "minimaxm3",
    "Kimi-K3": "kimik3",
}
# NVIDIA single-node SKUs; AMD rows share the API but are out of scope.
NVIDIA_HARDWARE = ("b200", "b300", "h100", "h200")

# generate_sweep_configs.py: BYTES_PER_MIB, BYTES_PER_GB and the 3 TB cap.
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB = 2_861_022
# validation.py: DEFAULT_AGENTIC_DURATION_SECONDS
AGENTIC_DURATION_SECONDS = 3600

MODEL_DIRS = {
    "dsv4": "DeepSeek-V4-Pro",
    "qwen3.5": "Qwen3.5",
    "kimik3": "Kimi-K3",
    "minimaxm3": "MiniMax-M3",
}
GPU_DIRS = {
    "cluster:b200-dgxc": "B200",
    "cluster:b300-nv": "B300",
    "cluster:h100-dgxc": "H100",
    "cluster:h200-dgxc": "H200",
}
FRAMEWORK_DIRS = {"sglang": "sglang", "vllm": "vllm", "trt": "trtllm"}

# Campaign -> the launcher that runs it. Every campaign maps to exactly one
# script; two campaigns can share a script and differ only in image or arms.
LAUNCHERS = {
    "dsv4-fp4-b200-sglang-agentic-hicache-mtp": "dsv4_fp4_b200_sglang_mtp.sh",
    "dsv4-fp4-b300-sglang-agentic-hicache-mtp": "dsv4_fp4_b300_sglang_mtp.sh",
    "dsv4-fp4-b200-vllm-agentic-mtp": "dsv4_fp4_b200_vllm_mtp.sh",
    "dsv4-fp4-b300-vllm-agentic-mtp": "dsv4_fp4_b300_vllm_mtp.sh",
    "qwen3.5-fp8-b200-sglang-agentic-mtp": "qwen3.5_fp8_b200_sglang_mtp.sh",
    "qwen3.5-fp4-b200-sglang-agentic-mtp": "qwen3.5_fp4_b200_sglang_mtp.sh",
    "qwen3.5-fp8-b300-sglang-agentic-mtp": "qwen3.5_fp8_b300_sglang_mtp.sh",
    "qwen3.5-fp4-b300-sglang-agentic-mtp": "qwen3.5_fp4_b300_sglang_mtp.sh",
    "qwen3.5-fp8-b300-sglang-agentic-power-ab": "qwen3.5_fp8_b300_sglang_mtp.sh",
    "qwen3.5-fp4-b300-sglang-agentic-power-ab": "qwen3.5_fp4_b300_sglang_mtp.sh",
    "qwen3.5-fp8-h100-sglang-agentic-mtp": "qwen3.5_fp8_h100_mtp.sh",
    "qwen3.5-fp8-h200-sglang-agentic-mtp": "qwen3.5_fp8_h200_mtp.sh",
    "qwen3.5-fp8-h200-sglang-agentic-hicache-mtp": "qwen3.5_fp8_h200_mtp.sh",
    "kimik3-fp4-b300-vllm-agentic-dspark": "kimik3_fp4_b300_vllm_mtp.sh",
    "minimaxm3-fp8-h100-vllm-agentic-mtp": "minimaxm3_fp8_h100_mtp.sh",
    "minimaxm3-fp8-h200-vllm-agentic-mtp": "minimaxm3_fp8_h200_mtp.sh",
    "minimaxm3-fp4-b200-vllm-agentic-mtp": "minimaxm3_fp4_b200_mtp.sh",
    "minimaxm3-fp4-b300-vllm-agentic-mtp": "minimaxm3_fp4_b300_mtp.sh",
    "minimaxm3-fp4-b200-trtllm-agentic-mtp": "minimaxm3_fp4_b200_trt_mtp.sh",
    "minimaxm3-fp4-b300-trtllm-agentic-mtp": "minimaxm3_fp4_b300_trt_mtp.sh",
}

# Present in the master config but not runnable or published. The H100 launcher
# resolves the no-spec campaign to qwen3.5_fp8_h100.sh; that file does not
# exist, and the public AgentX API has no corresponding spec_method=none rows.
# Do not silently infer it from the MTP launcher.
UNPUBLISHED_CAMPAIGNS = {
    "qwen3.5-fp8-h100-sglang-agentic",
}

# Campaigns whose file names would otherwise collide with a sibling campaign
# that shares model/platform/framework/precision and arm topology.
NAME_DISCRIMINATORS = {
    "qwen3.5-fp8-h200-sglang-agentic-hicache-mtp": "nightly",
}

# The cc-traces corpus each recipe replays, matching InferenceX's own default:
# the 062126 generation, unfiltered for families whose native context is 1M
# tokens and 256k-capped for the shorter-context families that cannot replay it.
#
# Every entry names a dated corpus. The undated
# `semianalysis_cc_traces_weka_with_subagents` alias, which the Qwen Hopper
# launchers still pin, is not "latest" — it is a frozen alias whose target moves
# with the aiperf build (061526 in InferenceX's pinned aiperf, 052726 in older
# checkouts), so a recipe carrying it does not name a reproducible corpus.
WEKA_OVERRIDES = {
    "dsv4": "semianalysis_cc_traces_weka_062126",
    "kimik3": "semianalysis_cc_traces_weka_062126",
    "minimaxm3": "semianalysis_cc_traces_weka_062126",
    "qwen3.5": "semianalysis_cc_traces_weka_062126_256k",
}

# Launchers that pin the undated alias instead of a dated corpus. Their recipes
# get the dated 062126 corpus and a header note, because the published numbers
# were measured against whatever that alias resolved to at the time.
UNDATED_ALIAS_LAUNCHERS = {("qwen3.5", "H100"), ("qwen3.5", "H200")}

DRAFT_MINIMAX = "Inferact/MiniMax-M3-EAGLE3-GQA"
DRAFT_KIMI = "Inferact/Kimi-K3-DSpark"


def jdump(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


# --------------------------------------------------------------------------
# Published points
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One published AgentX benchmark point."""

    conc: int
    image: str
    date: str
    run_url: str


def serving_signature(
    prefix: str,
    hardware: str,
    framework: str,
    precision: str,
    tp: int,
    ep: int,
    dp_attn: bool,
    offloading_on: bool,
    spec: str,
) -> tuple:
    """Identify a serving configuration on both the API and master-config side."""
    return (
        prefix,
        hardware.lower(),
        framework,
        precision.lower(),
        int(tp),
        int(ep),
        bool(dp_attn),
        bool(offloading_on),
        spec or "none",
    )


def fetch_published_rows() -> list[dict]:
    rows: list[dict] = []
    for display_name, prefix in API_MODELS.items():
        query = urllib.parse.urlencode({"model": display_name})
        request = urllib.request.Request(
            f"{DASHBOARD_API}?{query}", headers={"Accept-Encoding": "gzip"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                payload = gzip.decompress(payload)
        for row in json.loads(payload):
            if row.get("benchmark_type") != "agentic_traces":
                continue
            if row.get("is_multinode") or row.get("disagg"):
                continue
            if (row.get("hardware") or "").lower() not in NVIDIA_HARDWARE:
                continue
            rows.append(
                {
                    "prefix": prefix,
                    "hardware": row["hardware"],
                    "framework": row["framework"],
                    "precision": row["precision"],
                    "tp": row["prefill_tp"],
                    "ep": row["prefill_ep"],
                    "dp_attn": bool(row["prefill_dp_attention"]),
                    "offloading_on": row["offload_mode"] != "off",
                    "spec": row["spec_method"],
                    "conc": row["conc"],
                    "image": row["image"],
                    "date": row["date"],
                    "run_url": row["run_url"],
                }
            )
    rows.sort(key=lambda r: (r["prefix"], r["hardware"], r["framework"], r["precision"],
                             r["tp"], r["ep"], r["dp_attn"], r["offloading_on"],
                             r["spec"], r["conc"], r["date"]))
    return rows


def group_published_points(rows: list[dict]) -> "OrderedDict[tuple, list[Point]]":
    """Collapse published rows into one entry per serving configuration.

    A configuration can be re-measured, so the same concurrency appears more than
    once; the most recent measurement wins.
    """
    latest: dict[tuple, dict] = {}
    for row in rows:
        signature = serving_signature(
            row["prefix"], row["hardware"], row["framework"], row["precision"],
            row["tp"], row["ep"], row["dp_attn"], row["offloading_on"], row["spec"],
        )
        key = (signature, row["conc"])
        if key not in latest or row["date"] > latest[key]["date"]:
            latest[key] = row

    grouped: dict[tuple, list[Point]] = {}
    for (signature, conc), row in latest.items():
        grouped.setdefault(signature, []).append(
            Point(conc, row["image"], row["date"], row["run_url"])
        )
    return OrderedDict(
        (signature, sorted(grouped[signature], key=lambda p: p.conc))
        for signature in sorted(grouped)
    )


def load_published_points(refresh: bool) -> "OrderedDict[tuple, list[Point]]":
    if refresh or not POINTS_SNAPSHOT.is_file():
        rows = fetch_published_rows()
        POINTS_SNAPSHOT.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        rows = json.loads(POINTS_SNAPSHOT.read_text(encoding="utf-8"))
    return group_published_points(rows)


# --------------------------------------------------------------------------
# Fleet facts
# --------------------------------------------------------------------------


def total_cpu_dram_gb(
    available_mib: int, gpus_per_node: int, utilization: float, gpu_count: int
) -> int:
    """Reproduce ``agentic_dram_offload_gb`` for a single-node arm."""
    available = min(available_mib, MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB)
    proportional = (
        Decimal(available)
        * BYTES_PER_MIB
        * Decimal(str(utilization))
        * gpu_count
        / gpus_per_node
    )
    return int(proportional / BYTES_PER_GB)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


class Arm:
    def __init__(self, campaign: str, job: dict, block: dict, arm: dict, hardware: dict):
        self.campaign = campaign
        self.job = job
        self.arm = arm
        self.prefix = job["model-prefix"]
        self.model = job["model"]
        self.image = job["image"]
        self.precision = job["precision"]
        self.framework = job["framework"]
        self.runner = job["runner"]
        self.tp = arm["tp"]
        self.ep = arm.get("ep", 1)
        self.dcp = arm.get("dcp-size", 1)
        self.dp_attn = bool(arm.get("dp-attn", False))
        self.spec = arm.get("spec-decoding", "none")
        self.offloading = arm.get("kv-offloading", "none")
        backend = arm.get("kv-offload-backend") or {}
        self.offload_backend = backend.get("name")
        self.router = (arm.get("router") or {}).get("name")
        self.conc_list = list(arm["conc-list"])
        self.points: list[Point] = []
        self.also_campaigns: list[str] = []
        self.offload_from_published = False
        self.utilization = block.get("dram-utilization")
        self.gpus_per_node = hardware["gpus-per-node"]
        self.available_cpu_dram_mib = hardware["available-cpu-dram-mib"]
        self.total_dram_gb = self._dram_budget()

    def _dram_budget(self) -> int:
        if self.offloading != "dram":
            return 0
        return total_cpu_dram_gb(
            self.available_cpu_dram_mib,
            self.gpus_per_node,
            self.utilization,
            self.tp,
        )

    @property
    def launcher(self) -> str:
        return LAUNCHERS[self.campaign]

    @property
    def signature(self) -> tuple:
        return serving_signature(
            self.prefix,
            self.gpu_dir,
            self.framework,
            self.precision,
            self.tp,
            self.ep,
            self.dp_attn,
            self.offloading == "dram",
            self.spec,
        )

    @property
    def topology(self) -> tuple:
        """The serving configuration minus the KV-offload decision."""
        signature = list(self.signature)
        signature[7] = None
        return tuple(signature)

    def as_published(self, points: list[Point]) -> "Arm":
        """Bind this arm to its published points and their container image."""
        bound = copy.copy(self)
        bound.points = list(points)
        bound.conc_list = [point.conc for point in points]
        bound.image = points[0].image
        return bound

    def with_offload(self, donor: "Arm | None") -> "Arm":
        """Return this arm with its KV-offload decision taken from ``donor``.

        The published history is wider than any single revision of the master
        config: a topology can have been measured both with and without DRAM
        offload while the config now declares only one of the two. The launchers
        branch on ``KV_OFFLOADING`` and accept exactly one DRAM backend, so the
        other side of that branch is recoverable rather than guesswork —
        ``donor`` is the arm on the same launcher that declares it.
        """
        flipped = copy.copy(self)
        flipped.offload_from_published = True
        if donor is None:
            flipped.offloading = "none"
            flipped.offload_backend = None
        else:
            flipped.offloading = donor.offloading
            flipped.offload_backend = donor.offload_backend
            flipped.utilization = donor.utilization
        flipped.total_dram_gb = flipped._dram_budget()
        return flipped

    @property
    def model_dir(self) -> str:
        return MODEL_DIRS[self.prefix]

    @property
    def gpu_dir(self) -> str:
        return GPU_DIRS[self.runner]

    @property
    def framework_dir(self) -> str:
        return FRAMEWORK_DIRS[self.framework]

    @property
    def offload_label(self) -> str:
        if self.offloading == "none":
            return "kvnone"
        return {"hicache": "hicache", "vllm-simple": "vllmsimple",
                "mooncake": "mooncake", "native": "trthost"}[self.offload_backend]

    @property
    def spec_label(self) -> str:
        return {"mtp": "mtp", "none": "nospec"}.get(self.spec, self.spec)

    @property
    def stem(self) -> str:
        parts = [f"agg-{self.gpu_dir.lower()}", self.precision, f"tp{self.tp}"]
        if self.dp_attn:
            parts.append(f"dep{self.ep}")
        elif self.ep > 1:
            parts.append(f"ep{self.ep}")
        if self.dcp > 1:
            parts.append(f"dcp{self.dcp}")
        parts.append(self.offload_label)
        parts.append(self.spec_label)
        discriminator = NAME_DISCRIMINATORS.get(self.campaign)
        if discriminator:
            parts.append(discriminator)
        parts.append("agentic")
        return "-".join(parts)

    @property
    def path(self) -> Path:
        return (
            RECIPES_ROOT
            / self.model_dir
            / self.gpu_dir
            / self.framework_dir
            / "agentic"
            / f"{self.stem}.yaml"
        )


def index_by_signature(arms: list[Arm]) -> "OrderedDict[tuple, list[Arm]]":
    """Group the master arms by serving configuration.

    Several campaigns can declare the same configuration — the power-measurement
    A/B campaigns re-run a subset of a main campaign's concurrencies on the same
    launcher and image — and the published results do not distinguish them. The
    arm with the widest declared conc-list leads; the rest are recorded as
    additional campaigns that measured the same configuration.
    """
    grouped: dict[tuple, list[Arm]] = {}
    for arm in arms:
        grouped.setdefault(arm.signature, []).append(arm)
    return OrderedDict(
        (
            signature,
            sorted(grouped[signature], key=lambda a: (-len(a.conc_list), a.campaign)),
        )
        for signature in sorted(grouped)
    )


def resolve_arm(
    signature: tuple, by_signature: "OrderedDict[tuple, list[Arm]]", arms: list[Arm]
) -> list[Arm] | None:
    """Find the master arms behind a published serving configuration."""
    exact = by_signature.get(signature)
    if exact:
        return exact

    topology = list(signature)
    topology[7] = None
    candidates = [arm for arm in arms if arm.topology == tuple(topology)]
    if not candidates:
        return None

    base = sorted(candidates, key=lambda a: (-len(a.conc_list), a.campaign))[0]
    if signature[7]:
        donor = next(
            (
                arm
                for arm in arms
                if arm.campaign == base.campaign and arm.offloading == "dram"
            ),
            None,
        )
        if donor is None:
            return None
    else:
        donor = None
    return [base.with_offload(donor)]


def load_arms(inferencex: Path) -> list[Arm]:
    master = yaml.safe_load((inferencex / "configs" / "nvidia-master.yaml").read_text())
    runners = yaml.safe_load((inferencex / "configs" / "runners.yaml").read_text())
    hardware = runners["hardware"]

    arms: list[Arm] = []
    for campaign, job in master.items():
        if not isinstance(job, dict):
            continue
        blocks = (job.get("scenarios") or {}).get("agentic-coding")
        if not blocks or job.get("multinode", False) or "glm" in campaign.lower():
            continue
        if campaign in UNPUBLISHED_CAMPAIGNS:
            continue
        if campaign not in LAUNCHERS:
            raise SystemExit(f"no launcher mapped for campaign {campaign}")
        for block in blocks:
            for arm in block["search-space"]:
                arms.append(Arm(campaign, job, block, arm, hardware[job["runner"]]))
    return arms


# --------------------------------------------------------------------------
# Backends: DeepSeek-V4-Pro
# --------------------------------------------------------------------------

DSV4_SGLANG_ENV = {
    "PYTHONNOUSERSITE": "1",
    "TORCH_CUDA_ARCH_LIST": "10.0",
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
    "SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
    "SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT": "1",
    "SGLANG_OPT_USE_JIT_NORM": "1",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "1",
    "SGLANG_OPT_USE_TOPK_V2": "1",
    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "1",
    "SGLANG_SIMULATE_ACC_LEN": "2.49",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}

DSV4_EAGLE = {
    "speculative-algorithm": "EAGLE",
    "speculative-num-steps": 3,
    "speculative-eagle-topk": 1,
    "speculative-num-draft-tokens": 4,
}


def dsv4_sglang(arm: Arm) -> dict:
    b300 = arm.gpu_dir == "B300"
    env = dict(DSV4_SGLANG_ENV)
    if arm.dp_attn and not b300:
        env.update(
            {
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": "1",
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": "1",
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
            }
        )

    cfg = OrderedDict(
        [
            ("served-model-name", arm.model),
            ("trust-remote-code", True),
            ("tp-size", arm.tp),
        ]
    )

    if b300:
        cfg["attention-backend"] = "compressed"
        cfg["page-size"] = 256
        cfg["disable-shared-experts-fusion"] = True
        cfg["swa-full-tokens-ratio"] = 0.1
        cfg["allow-auto-truncate"] = True
        cfg["moe-runner-backend"] = "flashinfer_mxfp4"
        cfg["disable-flashinfer-autotune"] = True
        cfg["chunked-prefill-size"] = 16384 if arm.dp_attn else 8192
        if arm.dp_attn:
            cfg["dp-size"] = arm.tp
            cfg["tokenizer-worker-num"] = arm.tp
            cfg["enable-dp-attention"] = True
            cfg["enable-dp-attention-local-control-broadcast"] = True
            cfg["incremental-streaming-output"] = True
            cfg["stream-interval"] = 20
            cfg["ep-size"] = arm.ep
    else:
        cfg["mem-fraction-static"] = 0.88
        cfg["swa-full-tokens-ratio"] = 0.02 if arm.dp_attn else 0.1
        cfg["chunked-prefill-size"] = 8192 * arm.tp if arm.dp_attn else 8192
        cfg["enable-deepseek-v4-fp4-indexer"] = True
        cfg["disable-flashinfer-autotune"] = True
        cfg["weight-loader-prefetch-checkpoints"] = True
        if arm.dp_attn:
            cfg["dp-size"] = arm.tp
            cfg["tokenizer-worker-num"] = arm.tp
            cfg["enable-prefill-delayer"] = True
            cfg["prefill-decode-interval"] = 10
            cfg["enable-dp-attention"] = True
            cfg["enable-dp-attention-local-control-broadcast"] = True
            cfg["incremental-streaming-output"] = True
            cfg["stream-interval"] = 20
            cfg["ep-size"] = arm.ep
            cfg["moe-a2a-backend"] = "megamoe"
            cfg["disable-shared-experts-fusion"] = True
        else:
            cfg["moe-runner-backend"] = "flashinfer_mxfp4"

    cfg["tool-call-parser"] = "deepseekv4"
    cfg["reasoning-parser"] = "deepseek-v4"
    cfg["watchdog-timeout"] = 1800
    cfg.update(DSV4_EAGLE)
    cfg["enable-metrics"] = True
    cfg["enable-cache-report"] = True

    if arm.offload_backend == "hicache":
        cfg["enable-hierarchical-cache"] = True
        cfg["hicache-ratio"] = (2 if arm.tp >= 8 else 8) if b300 else 8
        cfg["hicache-write-policy"] = "write_back" if b300 else "write_through"
        cfg["hicache-io-backend"] = "direct"
        cfg["hicache-mem-layout"] = "page_first_direct"
        if b300:
            cfg["skip-server-warmup"] = True

    def per_conc(conc: int) -> dict:
        values = {"max-running-requests": 2 * conc}
        if b300:
            values["cuda-graph-max-bs"] = min(conc, 64)
            if arm.dp_attn:
                values["mem-fraction-static"] = 0.94 if conc >= 512 else 0.95
            else:
                values["mem-fraction-static"] = 0.88
        else:
            values["cuda-graph-max-bs"] = 2 * conc
        return values

    return {"env": env, "config": cfg, "per_conc": per_conc}


def dsv4_vllm(arm: Arm) -> dict:
    b300 = arm.gpu_dir == "B300"
    dep8 = arm.dp_attn and arm.tp == 8
    env = {
        "PYTHONNOUSERSITE": "1",
        "TORCH_CUDA_ARCH_LIST": "10.0",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "32768",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_RUST_FRONTEND": "1",
        "VLLM_DSV4_MEGA_FP8_COMBINE": "1",
        "VLLM_FLOAT32_MATMUL_PRECISION": "high",
    }
    if b300:
        env["NCCL_NVLS_ENABLE"] = "1"
        if not arm.dp_attn:
            env["VLLM_ALLREDUCE_USE_FLASHINFER"] = "1"
            env["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] = "auto"
    else:
        env["VLLM_RPC_TIMEOUT"] = "600000"
    if arm.dp_attn:
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if arm.offload_backend == "vllm-simple":
        env["PYTHONHASHSEED"] = "42"
        env["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "1"

    cfg = OrderedDict([("served-model-name", arm.model)])
    if arm.dp_attn:
        cfg["tensor-parallel-size"] = 1
        cfg["data-parallel-size"] = arm.tp
    else:
        cfg["tensor-parallel-size"] = arm.tp
        cfg["data-parallel-size"] = 1

    cfg["trust-remote-code"] = True
    cfg["kv-cache-dtype"] = "fp8"
    cfg["block-size"] = 256
    cfg["max-model-len"] = 1048576
    if b300:
        cfg["gpu-memory-utilization"] = 0.92 if dep8 else 0.95
        if not arm.dp_attn:
            cfg["disable-custom-all-reduce"] = True
    else:
        cfg["gpu-memory-utilization"] = 0.90
        cfg["numa-bind"] = True
        cfg["enable-cumem-allocator"] = True
    cfg["no-enable-flashinfer-autotune"] = True
    cfg["tokenizer-mode"] = "deepseek_v4"
    cfg["tool-call-parser"] = "deepseek_v4"
    cfg["enable-auto-tool-choice"] = True
    cfg["reasoning-parser"] = "deepseek_v4"
    attention = {
        "backend": "FLASHINFER_MLA_SPARSE_DSV4",
        "use_prefill_query_quantization": True,
        "use_fp4_indexer_cache": True,
    }
    cfg["attention-config"] = jdump(attention)
    cfg["speculative-config"] = jdump(
        {
            "method": "mtp",
            "num_speculative_tokens": 3,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": 2.49,
        }
    )
    cfg["no-disable-hybrid-kv-cache-manager"] = True
    cfg["disable-uvicorn-access-log"] = True
    if arm.ep > 1:
        cfg["enable-expert-parallel"] = True
        cfg["enable-ep-weight-filter"] = True
        cfg["moe-backend"] = "deep_gemm_amxf4_mega_moe"
    if arm.dp_attn:
        cfg["prefill-schedule-interval"] = 8
        cfg["long-prefill-token-threshold"] = 512
        cfg["max-num-batched-tokens"] = (16384 if dep8 else 8192) if b300 else 8192

    if arm.offload_backend == "vllm-simple":
        per_rank = arm.total_dram_gb * 1_000_000_000 // arm.tp
        extra = {
            ("cpu_bytes_to_use" if b300 else "cpu_bytes_to_use_per_rank"): per_rank,
            "enable_cross_layers_blocks": "true",
            "lazy_offload": (not arm.dp_attn) if b300 else False,
        }
        cfg["kv-transfer-config"] = jdump(
            {
                "kv_connector": "SimpleCPUOffloadConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": extra,
            }
        )

    def per_conc(conc: int) -> dict:
        max_num_seqs = 2 * conc // arm.tp if arm.dp_attn else 2 * conc
        sizes = [n * 4 for n in range(1, max_num_seqs + 1)]
        if not b300 and not arm.dp_attn:
            sizes = sorted(set(sizes) | {100, 200, 300, 400, 500})
        if b300 or arm.dp_attn:
            compilation = {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": sizes,
                "mode": 0,
            }
        else:
            compilation = {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": sizes,
            }
        return {
            "max-num-seqs": max_num_seqs,
            "compilation-config": jdump(compilation),
        }

    return {"env": env, "config": cfg, "per_conc": per_conc}


# --------------------------------------------------------------------------
# Backends: Qwen3.5
# --------------------------------------------------------------------------


def qwen_sglang(arm: Arm) -> dict:
    hopper = arm.gpu_dir in ("H100", "H200")
    cfg = OrderedDict([("served-model-name", arm.model)])
    cfg["trust-remote-code"] = True

    if hopper:
        env = {
            "PYTHONNOUSERSITE": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_SIMULATE_ACC_LEN": "3.39",
            "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
        }
        cfg["tensor-parallel-size"] = arm.tp
        cfg["data-parallel-size"] = 1
        cfg["expert-parallel-size"] = arm.ep
        cfg["quantization"] = "fp8"
        cfg["kv-cache-dtype"] = "fp8_e4m3"
        cfg["mamba-ssm-dtype"] = "bfloat16"
        cfg["attention-backend"] = "flashinfer"
        cfg["enable-flashinfer-allreduce-fusion"] = True
        cfg["mem-fraction-static"] = 0.75 if arm.gpu_dir == "H100" else 0.8
        cfg["stream-interval"] = 50
        cfg["scheduler-recv-interval"] = 10
        cfg["tokenizer-worker-num"] = 6
        cfg["tokenizer-path"] = arm.model
        cfg["enable-metrics"] = True
        cfg["speculative-algorithm"] = "EAGLE"
        cfg["speculative-num-steps"] = 3
        cfg["speculative-eagle-topk"] = 1
        cfg["speculative-num-draft-tokens"] = 4
    else:
        env = {
            "PYTHONNOUSERSITE": "1",
            "TORCH_CUDA_ARCH_LIST": "10.0",
            "NCCL_NVLS_ENABLE": "1",
            "SGL_ENABLE_JIT_DEEPGEMM": "false",
            "SGLANG_ENABLE_FLASHINFER_GEMM": "true",
            "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
            "SGLANG_SIMULATE_ACC_LEN": "3.39",
            "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
        }
        cfg["tp-size"] = arm.tp
        cfg["dp-size"] = 1
        cfg["ep-size"] = arm.ep
        cfg["enable-symm-mem"] = True
        if arm.precision == "fp4":
            cfg["quantization"] = "modelopt_fp4"
            cfg["fp4-gemm-backend"] = "flashinfer_cutlass"
        else:
            cfg["quantization"] = "fp8"
        cfg["kv-cache-dtype"] = "fp8_e4m3"
        cfg["mamba-ssm-dtype"] = "bfloat16"
        cfg["attention-backend"] = "trtllm_mha"
        cfg["moe-runner-backend"] = "flashinfer_trtllm"
        cfg["max-prefill-tokens"] = 16384
        cfg["chunked-prefill-size"] = 16384
        cfg["mem-fraction-static"] = 0.80
        cfg["stream-interval"] = 50
        cfg["scheduler-recv-interval"] = 10
        # The FP8 B200 launcher always sets the tokenizer workers; the other
        # three Blackwell launchers gate it on TP >= 4.
        always_tokenizer = arm.campaign == "qwen3.5-fp8-b200-sglang-agentic-mtp"
        if always_tokenizer or arm.tp >= 4:
            cfg["tokenizer-worker-num"] = 6
        cfg["tokenizer-path"] = arm.model
        cfg["reasoning-parser"] = "qwen3"
        cfg["tool-call-parser"] = "qwen3_coder"
        cfg["speculative-algorithm"] = "NEXTN"
        cfg["speculative-num-steps"] = 3
        cfg["speculative-eagle-topk"] = 1
        cfg["speculative-num-draft-tokens"] = 4
        cfg["enable-metrics"] = True
        cfg["enable-cache-report"] = True

    if arm.offload_backend == "hicache":
        if hopper:
            hicache_size = arm.total_dram_gb // arm.tp // 2
        else:
            hicache_size = (arm.total_dram_gb - arm.tp) * 15 // arm.tp // 31
        cfg["page-size"] = 64
        cfg["enable-hierarchical-cache"] = True
        cfg["hicache-size"] = hicache_size
        cfg["hicache-io-backend"] = "kernel"
        cfg["hicache-mem-layout"] = "page_first"
        cfg["hicache-write-policy"] = "write_through_selective"

    if hopper:
        return {"env": env, "config": cfg, "per_conc": lambda conc: {}}

    def per_conc(conc: int) -> dict:
        return {
            "max-running-requests": 2 * conc,
            "cuda-graph-max-bs": min(conc, 64),
        }

    return {"env": env, "config": cfg, "per_conc": per_conc}


# --------------------------------------------------------------------------
# Backends: Kimi-K3
# --------------------------------------------------------------------------


def kimi_vllm(arm: Arm) -> dict:
    offload = arm.offloading == "dram"
    env = {
        "PYTHONNOUSERSITE": "1",
        "TORCH_CUDA_ARCH_LIST": "10.0",
        "PYTHONHASHSEED": "42",
        "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
        "VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_DIRECT_DCP_A2A": "1",
        "VLLM_USE_DIRECT_DCP_Q_GATHER": "1",
        "VLLM_USE_DIRECT_DCP_KV_GATHER": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_RPC_TIMEOUT": "600000",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "0",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "900",
    }
    if offload:
        env.update(
            {
                "VLLM_MOONCAKE_LOAD_RECV_THREADS": "4",
                "MC_GID_INDEX": "3",
                "MC_STORE_MEMCPY": "1",
                "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
                "MC_SLICE_SIZE": "1048576",
                "MC_WORKERS_PER_CTX": "4",
                "WITH_NVIDIA_PEERMEM": "0",
            }
        )

    cfg = OrderedDict(
        [
            ("served-model-name", arm.model),
            ("tensor-parallel-size", arm.tp),
            ("decode-context-parallel-size", arm.dcp),
            ("dcp-comm-backend", "a2a"),
            ("max-num-batched-tokens", 16384),
            ("trust-remote-code", True),
            ("language-model-only", True),
            ("load-format", "fastsafetensors"),
            ("moe-backend", "auto"),
            ("no-enable-flashinfer-autotune", True),
            ("enable-cumem-allocator", True),
            ("enable-prefix-caching", True),
            ("prefix-match-unit", 128),
            ("kv-cache-dtype", "fp8"),
            ("stream-interval", 10),
            ("attention-backend", "TOKENSPEED_MLA"),
            (
                "attention-config",
                jdump(
                    {
                        "mla_prefill_backend": "TRTLLM_RAGGED",
                        "use_prefill_query_quantization": True,
                    }
                ),
            ),
            ("disable-uvicorn-access-log", True),
        ]
    )
    if offload:
        cfg["kv-transfer-config"] = jdump(
            {
                "kv_connector": "MooncakeStoreConnector",
                "kv_role": "kv_both",
                "kv_load_failure_policy": "recompute",
                "kv_connector_extra_config": {
                    "load_async": True,
                    "lookup_async": True,
                    "enable_offload": False,
                },
            }
        )

    def per_conc(conc: int) -> dict:
        max_num_seqs = 2 * conc
        if conc <= 8:
            num_spec, accept = 7, 3.84
        elif conc <= 16:
            num_spec, accept = 3, 3.00
        else:
            num_spec, accept = 0, None

        values = {
            # The launcher keeps the draft-length regimes disjoint in concurrency,
            # so each one becomes its own recipe.
            "_label": f"spec{num_spec}" if num_spec else "nospec",
            "max-num-seqs": max_num_seqs,
            "gpu-memory-utilization": 0.90 if conc >= 56 else 0.92,
        }
        if num_spec:
            values["speculative-config"] = jdump(
                {
                    "method": "dspark",
                    "model": DRAFT_KIMI,
                    "num_speculative_tokens": num_spec,
                    "attention_backend": "TOKENSPEED_MLA",
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "synthetic",
                    "synthetic_acceptance_length": accept,
                }
            )

        step = 1 + num_spec
        dense = min(max_num_seqs, 128)
        sizes = [n * step for n in range(1, dense + 1)]
        dense_max = dense * step
        sizes += [t for t in (64, 128, 256, 512, 1024, 2048, 4096, 8192) if t > dense_max]
        values["compilation-config"] = jdump(
            {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": sizes}
        )
        return values

    backend = {"env": env, "config": cfg, "per_conc": per_conc}
    if offload:
        backend["mooncake"] = OrderedDict(
            [
                ("metadata_server", "P2PHANDSHAKE"),
                ("global_segment_size", f"{arm.total_dram_gb // arm.tp}GB"),
                ("local_buffer_size", "4GB"),
                ("protocol", "rdma"),
                # The published c8 artifact selected mlx5_0. This remains a
                # cluster-local choice; use an active rail on other fleets.
                ("device_name", "mlx5_0"),
                ("mode", "embedded"),
                ("enable_offload", False),
            ]
        )
    return backend


# --------------------------------------------------------------------------
# Backends: MiniMax-M3
# --------------------------------------------------------------------------


def minimax_vllm(arm: Arm) -> dict:
    hopper = arm.gpu_dir in ("H100", "H200")
    spec = jdump(
        {
            "method": "eagle3",
            "model": DRAFT_MINIMAX,
            "num_speculative_tokens": 3,
            "attention_backend": "FLASH_ATTN",
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": 2.78,
        }
    )
    cfg = OrderedDict([("served-model-name", arm.model)])

    if hopper:
        env = {
            "PYTHONNOUSERSITE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        }
        cfg["tensor-parallel-size"] = arm.tp
        cfg["data-parallel-size"] = 1
        cfg["gpu-memory-utilization"] = 0.90
        if arm.gpu_dir == "H100":
            cfg["cpu-offload-gb"] = 26
        cfg["kv-cache-dtype"] = "fp8"
        cfg["attention-backend"] = "TRITON_ATTN"
        cfg["block-size"] = 128
        cfg["language-model-only"] = True
        cfg["enable-prefix-caching"] = True
        cfg["enable-prompt-tokens-details"] = True
        cfg["default-chat-template-kwargs"] = jdump({"thinking_mode": "enabled"})
        cfg["speculative-config"] = spec
        cfg["tool-call-parser"] = "minimax_m3"
        cfg["reasoning-parser"] = "minimax_m3"
        cfg["enable-auto-tool-choice"] = True
        if arm.gpu_dir == "H100":
            cfg["safetensors-load-strategy"] = "lazy"
        cfg["trust-remote-code"] = True
        if arm.offload_backend == "mooncake":
            env.update(
                {
                    "PYTHONHASHSEED": "0",
                    "MC_SLICE_SIZE": "1048576",
                    "MC_WORKERS_PER_CTX": "4",
                    "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
                }
            )
            cfg["kv-transfer-config"] = jdump(
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"load_async": True},
                }
            )

        def per_conc(conc: int) -> dict:
            return {
                "max-num-seqs": 2 * conc,
                "max-cudagraph-capture-size": 8 * conc,
            }

        mooncake = None
        if arm.offload_backend == "mooncake":
            # The launcher converts the decimal-GB allocation to GiB, reserves
            # the checkpoint page cache and Mooncake local buffer, and on H100
            # also reserves the model's 26 GiB CPU-offload budget.
            total_gib = arm.total_dram_gb * 1_000_000_000 // (1024**3)
            per_rank_gib = (total_gib - 414) // arm.tp - 4
            if arm.gpu_dir == "H100":
                per_rank_gib -= 26
            mooncake = OrderedDict(
                [
                    ("metadata_server", "P2PHANDSHAKE"),
                    ("global_segment_size", f"{per_rank_gib}GB"),
                    ("local_buffer_size", "4GB"),
                    ("protocol", "rdma"),
                    ("device_name", ""),
                    ("mode", "embedded"),
                    ("enable_offload", False),
                ]
            )

        result = {"env": env, "config": cfg, "per_conc": per_conc}
        if mooncake is not None:
            result["mooncake"] = mooncake
        return result

    env = {
        "PYTHONNOUSERSITE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_FLOAT32_MATMUL_PRECISION": "high",
        "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm",
    }
    cfg["tensor-parallel-size"] = arm.tp
    cfg["gpu-memory-utilization"] = 0.9
    cfg["block-size"] = 128
    cfg["language-model-only"] = True
    cfg["enable-prefix-caching"] = True
    cfg["no-enable-flashinfer-autotune"] = True
    cfg["reasoning-parser"] = "minimax_m3"
    cfg["tool-call-parser"] = "minimax_m3"
    cfg["enable-auto-tool-choice"] = True
    cfg["default-chat-template-kwargs"] = jdump({"thinking_mode": "enabled"})
    cfg["attention-config"] = jdump(
        {
            "backend": "FLASHINFER",
            "use_trtllm_attention": True,
            "indexer_kv_dtype": "fp8",
        }
    )
    cfg["kv-cache-dtype"] = "fp8"
    cfg["max-cudagraph-capture-size"] = 512
    cfg["max-num-batched-tokens"] = 16384
    cfg["stream-interval"] = 20
    cfg["trust-remote-code"] = True
    cfg["speculative-config"] = spec
    if arm.offload_backend == "vllm-simple":
        env["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "1"
        cfg["kv-transfer-config"] = jdump(
            {
                "kv_connector": "SimpleCPUOffloadConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": arm.total_dram_gb * 1024 * 1024 * 1024,
                    "lazy_offload": True,
                },
            }
        )

    return {"env": env, "config": cfg, "per_conc": lambda conc: {}}


MINIMAX_TRT_ENV = {
    "PYTHONNOUSERSITE": "1",
    "TLLM_LOG_LEVEL": "INFO",
    "TRTLLM_SERVER_DISABLE_GC": "1",
    "TRTLLM_WORKER_DISABLE_GC": "1",
    "TLLM_PROFILE_LOG_RANKS": "all",
    "TRTLLM_ENABLE_PDL": "1",
    # The launcher exports `yes`. srtctl resolves override variants by dumping with
    # ruamel (YAML 1.2, so `yes` is emitted unquoted) and re-reading with PyYAML
    # (YAML 1.1, where bare `yes` is a boolean), which fails env validation. Enroot
    # treats any non-empty value as enabled, so `y` is the same setting.
    "ENROOT_ALLOW_DEV": "y",
    "NCCL_GRAPH_MIXING_SUPPORT": "0",
    "MIMALLOC_PURGE_DELAY": "0",
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TRTLLM_SERVE_ENABLE_MSGSPEC": "1",
    "TRTLLM_TORCH_COMPILE_CONTEXT_ONLY": "1",
    "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "1.78",
}


def minimax_trt_batch_sizes(max_batch: int) -> list[int]:
    """Reproduce the launcher's CUDA-graph batch ladder for one concurrency."""
    if max_batch <= 20:
        return list(range(1, max_batch + 1))
    if max_batch == 25:
        return list(range(1, 16)) + [17, 19, 21, 23, 25]
    if max_batch == 30:
        return list(range(1, 13)) + [14, 16, 18, 20, 22, 24, 27, 30]
    if max_batch == 35:
        return list(range(1, 12)) + [14, 17, 20, 23, 26, 29, 32, 35]
    if max_batch == 40:
        return list(range(2, 42, 2))
    if max_batch == 45:
        return list(range(2, 20, 2)) + [21, 24, 27, 30, 33, 36, 39, 42, 45]
    raise SystemExit(f"no CUDA-graph batch ladder published for CONC={max_batch}")


def minimax_trtllm(arm: Arm) -> dict:
    b300 = arm.gpu_dir == "B300"
    if b300:
        host_cache = 309_237_645_312 if arm.tp == 8 else 388_554_555_392
    else:
        host_cache = 214_748_364_800 if arm.tp == 8 else 268_435_456_000

    cfg = OrderedDict(
        [
            ("tensor_parallel_size", arm.tp),
            ("max_seq_len", 1048576),
            ("max_num_tokens", 16384),
            (
                "torch_compile_config",
                OrderedDict(
                    [
                        ("enable_fullgraph", True),
                        ("enable_inductor", False),
                        ("enable_piecewise_cuda_graph", True),
                        ("capture_num_tokens", [1, 512, 1024, 2048]),
                        ("enable_userbuffers", True),
                        ("max_num_streams", 3),
                    ]
                ),
            ),
            (
                "moe_config",
                OrderedDict(
                    [("backend", "TRTLLM"), ("use_low_precision_moe_combine", True)]
                ),
            ),
            (
                "sparse_attention_config",
                OrderedDict(
                    [
                        ("algorithm", "minimax_m3"),
                        ("implementation", "msa"),
                        ("indexer_kv_dtype", "fp8"),
                        ("sparse_disable_index_value", True),
                        ("fuse_qkv_index_projection", True),
                    ]
                ),
            ),
            (
                "kv_cache_config",
                OrderedDict(
                    [
                        ("free_gpu_memory_fraction", 0.94),
                        ("enable_block_reuse", True),
                        ("block_reuse_policy", "per_conversation"),
                        ("tokens_per_block", 128),
                        ("use_kv_cache_manager_v2", True),
                        ("dtype", "fp8"),
                        ("event_buffer_max_size", 0),
                        ("host_cache_size", host_cache),
                    ]
                ),
            ),
            (
                "speculative_config",
                OrderedDict(
                    [
                        ("decoding_type", "Eagle3"),
                        ("max_draft_len", 3),
                        ("speculative_model", DRAFT_MINIMAX),
                    ]
                ),
            ),
            (
                "scheduler_config",
                OrderedDict([("capacity_scheduler_policy", "MAX_UTILIZATION")]),
            ),
            ("enable_chunked_prefill", True),
            ("enable_autotuner", True),
            ("trust_remote_code", True),
            ("reasoning_parser", "minimax_m3"),
            ("stream_interval", 20),
            ("print_iter_log", True),
            ("num_postprocess_workers", 8),
            ("enable_attention_dp", False),
        ]
    )

    def per_conc(conc: int) -> dict:
        return {
            "max_batch_size": conc,
            "cuda_graph_config": OrderedDict(
                [
                    ("enable_padding", True),
                    ("batch_sizes", minimax_trt_batch_sizes(conc)),
                ]
            ),
        }

    return {"env": dict(MINIMAX_TRT_ENV), "config": cfg, "per_conc": per_conc}


BACKENDS = {
    ("dsv4", "sglang"): dsv4_sglang,
    ("dsv4", "vllm"): dsv4_vllm,
    ("qwen3.5", "sglang"): qwen_sglang,
    ("kimik3", "vllm"): kimi_vllm,
    ("minimaxm3", "vllm"): minimax_vllm,
    ("minimaxm3", "trt"): minimax_trtllm,
}


# --------------------------------------------------------------------------
# Recipe assembly
# --------------------------------------------------------------------------


def header(arm: Arm, stem: str, conc_list: list[int]) -> list[str]:
    campaigns = [arm.campaign] + [
        campaign for campaign in arm.also_campaigns if campaign != arm.campaign
    ]
    lines = [
        "# InferenceX AgentX source:",
        f"#   {BLOB}/benchmarks/single_node/agentic/{arm.launcher}",
        f"#   {BLOB}/configs/nvidia-master.yaml  (job {', '.join(campaigns)})",
        "# See recipes/single-node/AGENTX.md",
        "#",
        "# Single-node AgentX has no published YAML: InferenceX runs it as a",
        "# master-config search space driven by the launcher above. This recipe is one",
        "# published serving configuration, taken from the InferenceX dashboard API,",
        "# with the launcher's engine configuration declared here and the client side",
        "# left on InferenceX agentic_srt.sh:",
        f"#   {AGENTIC_SRT}",
        "#",
        f"# Serving config: tp={arm.tp} ep={arm.ep} dp-attn={str(arm.dp_attn).lower()} "
        f"dcp={arm.dcp} kv-offloading={arm.offloading}"
        + (f"/{arm.offload_backend}" if arm.offload_backend else "")
        + f" spec-decoding={arm.spec}",
        f"# Published concurrencies: {', '.join(str(c) for c in conc_list)}",
        "# Every concurrency above is a published datapoint, and the launcher derives",
        "# engine settings from it, so each is a zip_override_conc variant instead of",
        "# being flattened away:",
        "#   srtctl apply -f <this file>:zip_override_conc[<index>]",
    ]
    if arm.offload_from_published:
        state = (
            f"{arm.offloading}/{arm.offload_backend}"
            if arm.offload_backend
            else "none"
        )
        lines += [
            "#",
            f"# kv-offloading={state} is the published runs' setting; the master-config",
            "# job above declares the opposite for this topology, having moved on since",
            "# these points were measured. The launcher branches on KV_OFFLOADING and",
            "# supports exactly one DRAM backend, so this side of the branch comes from",
            "# the sibling arm on the same launcher.",
        ]
    runs = []
    for point in arm.points:
        if point.conc in conc_list and point.run_url not in runs:
            runs.append(point.run_url)
    if runs:
        lines += ["#", "# Published runs:"] + [f"#   {run}" for run in runs]
    if arm.offloading == "dram":
        lines += [
            "#",
            f"# TOTAL_CPU_DRAM_GB={arm.total_dram_gb}, reproduced from configs/runners.yaml"
            f" ({arm.runner})",
            f"# and dram-utilization={arm.utilization} by the agentic_dram_offload_gb rule"
            " in",
            "# utils/matrix_logic/generate_sweep_configs.py. Re-derive it for a fleet with",
            "# different host DRAM.",
        ]
    if arm.framework_dir == "trtllm":
        lines += [
            "#",
            "# ENROOT_ALLOW_DEV is 'y' where the launcher exports 'yes': srtctl re-reads",
            "# resolved override variants with a YAML 1.1 loader, which turns bare yes",
            "# into a boolean and fails env validation. Enroot enables the setting on any",
            "# non-empty value, so the two are equivalent.",
        ]
    if (arm.prefix, arm.gpu_dir) in UNDATED_ALIAS_LAUNCHERS:
        lines += [
            "#",
            "# WEKA_LOADER_OVERRIDE names the dated 062126 corpus where the launcher",
            "# pins semianalysis_cc_traces_weka_with_subagents_256k. That alias is",
            "# frozen, not latest, and resolves to a different corpus per aiperf build",
            "# (061526 in InferenceX's pinned aiperf, 052726 in older checkouts), so it",
            "# cannot name a corpus reproducibly here. Expect the published numbers for",
            "# this arm to have come from whatever the alias resolved to when measured.",
        ]
    if arm.router:
        lines += [
            "#",
            f"# The launcher fronts the engine with {arm.router} for session-affine",
            "# routing across the DP ranks. srt-slurm has no stage for that router, so the",
            "# equivalent affinity is expressed through the Dynamo frontend below",
            "# (nginx session affinity + kv router mode).",
        ]
    return lines


def build_recipe(arm: Arm, stem: str, conc_list: list[int], per_conc: list[dict]) -> str:
    backend = BACKENDS[(arm.prefix, arm.framework)](arm)
    weka = WEKA_OVERRIDES.get(arm.prefix)

    config_key = f"{arm.framework_dir}_config"
    base: OrderedDict = OrderedDict()
    base["name"] = stem
    base["model"] = OrderedDict(
        [
            ("path", f"hf:{arm.model}"),
            ("container", arm.image),
            ("precision", arm.precision),
        ]
    )
    base["identity"] = OrderedDict(
        [
            ("model", OrderedDict([("repo", arm.model)])),
            ("container", OrderedDict([("image", arm.image)])),
        ]
    )
    base["dynamo"] = OrderedDict([("install", False)])
    base["slurm"] = OrderedDict([("time_limit", "08:00:00")])
    base["health_check"] = OrderedDict(
        [("max_attempts", 1440), ("interval_seconds", 10)]
    )
    base["resources"] = OrderedDict(
        [
            ("gpu_type", arm.gpu_dir.lower()),
            ("gpus_per_node", arm.gpus_per_node),
            ("agg_nodes", 1),
            ("agg_workers", 1),
            # DCP shards decode KV across the TP ranks, so it adds no GPUs.
            ("gpus_per_agg", arm.tp),
        ]
    )

    frontend: OrderedDict = OrderedDict([("type", "dynamo")])
    if arm.router:
        frontend["nginx_session_affinity"] = True
        frontend["nginx_session_affinity_header"] = "X-Dynamo-Session-ID"
        frontend["enable_multiple_frontends"] = True
        frontend["num_additional_frontends"] = 4
        frontend["args"] = OrderedDict(
            [
                ("router-mode", "kv"),
                ("router-session-affinity-ttl-secs", "3600"),
            ]
        )
    else:
        frontend["enable_multiple_frontends"] = False
    base["frontend"] = frontend

    base["infra"] = OrderedDict([("nats_max_payload_mb", 32)])

    aggregated_env = OrderedDict([("PYTHONUNBUFFERED", "1"), ("DYN_HEALTH_CHECK_ENABLED", "false")])
    for key in sorted(backend["env"]):
        aggregated_env[key] = backend["env"][key]

    backend_section = OrderedDict([("type", arm.framework_dir)])
    if "mooncake" in backend:
        # srt-slurm owns the Mooncake master lifecycle and injects the generated
        # MOONCAKE_CONFIG_PATH into the vLLM worker.
        backend_section["connector"] = None
        backend_section["mooncake_kv_store"] = OrderedDict(
            [("store_config", backend["mooncake"])]
        )
    backend_section["aggregated_environment"] = aggregated_env
    backend_section[config_key] = OrderedDict([("aggregated", backend["config"])])
    base["backend"] = backend_section

    benchmark_env = OrderedDict(
        [
            ("INFMAX_CONTAINER_WORKSPACE", "/infmax-workspace"),
            ("RESULT_DIR", "/logs/agentic"),
            ("PORT", "8000"),
            ("MODEL", arm.model),
            ("MODEL_PREFIX", arm.prefix),
            ("FRAMEWORK", arm.framework),
            ("PRECISION", arm.precision),
            ("DURATION", str(AGENTIC_DURATION_SECONDS)),
            ("WEKA_LOADER_OVERRIDE", weka),
            # Aggregated single-node: one worker owns prefill and decode, so the
            # result processor must use the single-node GPU accounting.
            ("IS_MULTINODE", "false"),
            ("TP", str(arm.tp)),
            ("EP", str(arm.ep)),
            ("DP_ATTENTION", "true" if arm.dp_attn else "false"),
            ("KV_OFFLOADING", arm.offloading),
            ("TOTAL_CPU_DRAM_GB", str(arm.total_dram_gb)),
            (
                "AIPERF_REQUIRED_SERVER_METRIC_PREFIX",
                {"sglang": "sglang:", "vllm": "vllm:", "trt": "tensorrt_llm:"}[arm.framework],
            ),
        ]
    )
    if arm.offload_backend:
        benchmark_env["KV_OFFLOAD_BACKEND"] = arm.offload_backend
    if arm.router:
        benchmark_env["AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"] = "true"

    base["benchmark"] = OrderedDict(
        [
            ("type", "custom"),
            ("command", AGENTIC_SRT_COMMAND),
            ("env", benchmark_env),
        ]
    )

    # zip_override_conc: one variant per published concurrency point.
    override: OrderedDict = OrderedDict()
    override["name"] = [f"{stem}-c{conc}" for conc in conc_list]
    keys = sorted({key for values in per_conc for key in values})
    if keys:
        aggregated: OrderedDict = OrderedDict()
        for key in keys:
            column = []
            for values, conc in zip(per_conc, conc_list):
                if key not in values:
                    raise SystemExit(
                        f"{stem}: {key} missing for CONC={conc}; a zipped key must "
                        "be present at every concurrency"
                    )
                column.append(values[key])
            aggregated[key] = column
        override["backend"] = OrderedDict(
            [(config_key, OrderedDict([("aggregated", aggregated)]))]
        )
    override["benchmark"] = OrderedDict(
        [
            (
                "env",
                OrderedDict(
                    [
                        ("CONC", [str(conc) for conc in conc_list]),
                        (
                            "RESULT_FILENAME",
                            [
                                f"{arm.prefix}_tp{arm.tp}_conc{conc}_{arm.offload_label}"
                                for conc in conc_list
                            ],
                        ),
                    ]
                ),
            )
        ]
    )

    document = OrderedDict([("base", base), ("zip_override_conc", override)])
    body = yaml.dump(
        json.loads(json.dumps(document)),
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    body = body.replace(
        f"command: {AGENTIC_SRT_COMMAND}",
        f"# {AGENTIC_SRT}\n    command: {AGENTIC_SRT_COMMAND}",
        1,
    )
    return "\n".join(header(arm, stem, conc_list)) + "\n\n" + body


def build_recipes(arm: Arm) -> dict[Path, str]:
    """Return the recipes for one arm.

    A few launchers switch engine regime inside an arm — Kimi-K3 drops drafting
    above concurrency 16, for instance — which changes which settings exist, not
    just their values. Those regimes cannot share one zip group, and the launcher
    keeps them disjoint in concurrency, so each becomes its own recipe.
    """
    backend = BACKENDS[(arm.prefix, arm.framework)](arm)
    groups: "OrderedDict[str, tuple[list[int], list[dict]]]" = OrderedDict()
    for conc in arm.conc_list:
        values = dict(backend["per_conc"](conc))
        label = values.pop("_label", "")
        concs, rows = groups.setdefault(label, ([], []))
        concs.append(conc)
        rows.append(values)

    recipes: dict[Path, str] = {}
    for label, (concs, rows) in groups.items():
        stem = f"{arm.stem}-{label}" if label and len(groups) > 1 else arm.stem
        path = arm.path.with_name(f"{stem}.yaml")
        recipes[path] = build_recipe(arm, stem, concs, rows)
    return recipes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inferencex",
        type=Path,
        default=DEFAULT_INFERENCEX,
        help="path to an InferenceX checkout (default: sibling of this repo)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the tree is out of date",
    )
    parser.add_argument(
        "--refresh-points",
        action="store_true",
        help="re-fetch the published points from the dashboard API",
    )
    args = parser.parse_args()

    if not (args.inferencex / "configs" / "nvidia-master.yaml").is_file():
        print(f"no InferenceX checkout at {args.inferencex}", file=sys.stderr)
        return 1

    published = load_published_points(args.refresh_points)
    arms = load_arms(args.inferencex)
    by_signature = index_by_signature(arms)

    recipes: dict[Path, str] = {}
    variants = 0
    unmatched: list[tuple] = []
    for signature, points in published.items():
        candidates = resolve_arm(signature, by_signature, arms)
        if not candidates:
            unmatched.append(signature)
            continue
        arm = candidates[0].as_published(points)
        arm.also_campaigns = [candidate.campaign for candidate in candidates]
        for path, text in build_recipes(arm).items():
            if path in recipes:
                raise SystemExit(f"two configurations resolved to {path}")
            recipes[path] = text
        variants += len(points)

    if unmatched:
        for signature in unmatched:
            print(
                "published point has no master-config arm: "
                + " ".join(str(field) for field in signature),
                file=sys.stderr,
            )
        return 1

    stale = [
        path
        for path in sorted(RECIPES_ROOT.glob("*/*/*/agentic/*.yaml"))
        if path not in recipes
    ]
    changed = list(stale)
    for path, text in recipes.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            changed.append(path)

    if args.check:
        if changed:
            for path in sorted(changed):
                print(f"out of date: {path.relative_to(REPO_ROOT)}")
            return 1
        print(f"up to date: {len(recipes)} single-node AgentX recipes")
        return 0

    for path in stale:
        path.unlink()
    for path, text in recipes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for directory in sorted(RECIPES_ROOT.glob("*/*/*/agentic")):
        if not any(directory.iterdir()):
            shutil.rmtree(directory)

    print(
        f"wrote {len(recipes)} single-node AgentX recipes "
        f"({variants} published points)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
