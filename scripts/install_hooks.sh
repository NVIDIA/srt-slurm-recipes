#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

hook_source="../../.githooks/pre-commit"
hook_target=".git/hooks/pre-commit"

if [[ -e "$hook_target" && ! -L "$hook_target" ]]; then
  echo "Refusing to overwrite existing $hook_target." >&2
  echo "Move it aside or install .githooks/pre-commit manually." >&2
  exit 1
fi

mkdir -p .git/hooks
chmod +x .githooks/pre-commit
ln -sfn "$hook_source" "$hook_target"

echo "Installed $hook_target -> $hook_source"
