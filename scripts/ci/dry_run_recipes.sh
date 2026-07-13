#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${RECIPES_ROOT:?RECIPES_ROOT must point to the recipes directory}"
: "${SRTCTL_WORKDIR:?SRTCTL_WORKDIR must point to the srt-slurm checkout}"
: "${SRTSLURM_CONFIG:?SRTSLURM_CONFIG must point to the CI srtslurm.yaml path}"
: "${DRY_RUN_LOG_DIR:?DRY_RUN_LOG_DIR must point to a log output directory}"

SRTCTL_CMD="${SRTCTL_CMD:-uv run srtctl}"
SRTCTL_OUTPUT_DIR="${SRTCTL_OUTPUT_DIR:-${DRY_RUN_LOG_DIR}/../srtctl-output}"
SRTCTL_ROOT="${SRTCTL_ROOT:-${SRTCTL_WORKDIR}}"
DRY_RUN_CONFIG_DIR="${DRY_RUN_CONFIG_DIR:-${DRY_RUN_LOG_DIR}/configs}"
DRY_RUN_JOBS="${DRY_RUN_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 16)}"

if [[ ! -d "${RECIPES_ROOT}" ]]; then
  echo "Recipes directory not found: ${RECIPES_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SRTCTL_WORKDIR}" ]]; then
  echo "srt-slurm checkout not found: ${SRTCTL_WORKDIR}" >&2
  exit 1
fi

if ! [[ "${DRY_RUN_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DRY_RUN_JOBS must be a positive integer; got: ${DRY_RUN_JOBS}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${SRTSLURM_CONFIG}")" "${DRY_RUN_LOG_DIR}" "${SRTCTL_OUTPUT_DIR}" "${DRY_RUN_CONFIG_DIR}"

write_srtslurm_config() {
  local config_path="$1"
  local output_dir="$2"

  mkdir -p "$(dirname -- "${config_path}")" "${output_dir}"
  cat >"${config_path}" <<EOF
default_account: ci
default_partition: ci
default_time_limit: "00:30:00"
srtctl_root: "${SRTCTL_ROOT}"
output_dir: "${output_dir}"
EOF
}

write_srtslurm_config "${SRTSLURM_CONFIG}" "${SRTCTL_OUTPUT_DIR}"

read -r -a srtctl_cmd <<<"${SRTCTL_CMD}"
failures=()
recipes=()
running_jobs=0
declare -A pid_to_log=()
declare -A pid_to_recipe=()

mapfile -d '' -t recipes < <(find "${RECIPES_ROOT}" -type f -name '*.yaml' -print0 | sort -z)

finish_one_job() {
  local finished_pid status rel log

  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi

  running_jobs=$((running_jobs - 1))
  rel="${pid_to_recipe[${finished_pid}]}"
  log="${pid_to_log[${finished_pid}]}"
  unset "pid_to_recipe[${finished_pid}]" "pid_to_log[${finished_pid}]"

  if (( status != 0 )); then
    failures+=("${rel}")
    echo "failed: ${rel}"
    tail -n 120 "${log}"
  fi
}

for recipe in "${recipes[@]}"; do
  rel="${recipe#${RECIPES_ROOT}/}"
  safe_rel="${rel//\//__}"
  log="${DRY_RUN_LOG_DIR}/${safe_rel}.log"
  job_config="${DRY_RUN_CONFIG_DIR}/${safe_rel}.yaml"
  job_output_dir="${SRTCTL_OUTPUT_DIR}/${safe_rel}"

  echo "dry-run: ${rel}"
  write_srtslurm_config "${job_config}" "${job_output_dir}"
  (cd "${SRTCTL_WORKDIR}" && SRTSLURM_CONFIG="${job_config}" "${srtctl_cmd[@]}" dry-run -f "${recipe}") >"${log}" 2>&1 &
  pid=$!
  pid_to_recipe["${pid}"]="${rel}"
  pid_to_log["${pid}"]="${log}"
  running_jobs=$((running_jobs + 1))

  if (( running_jobs >= DRY_RUN_JOBS )); then
    finish_one_job
  fi
done

while (( running_jobs > 0 )); do
  finish_one_job
done

if (( ${#recipes[@]} == 0 )); then
  echo "No recipe YAML files found under ${RECIPES_ROOT}" >&2
  exit 1
fi

if (( ${#failures[@]} )); then
  printf '\nFailed recipe dry-runs:\n'
  printf ' - %s\n' "${failures[@]}"
  exit 1
fi

echo "Validated ${#recipes[@]} recipe YAML files with ${DRY_RUN_JOBS} parallel job(s)."
