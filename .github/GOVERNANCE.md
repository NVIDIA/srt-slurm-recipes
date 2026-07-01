# Governance

`srt-slurm-recipes` is an NVIDIA open-source project. This document describes how
the project is governed and how decisions are made.

## Roles

- **Maintainers** — Have write access and are responsible for reviewing and
  merging changes, triaging issues, cutting releases, and upholding the
  [Code of Conduct](CODE_OF_CONDUCT.md). The current maintainers are listed in
  [MAINTAINERS.md](MAINTAINERS.md).
- **Contributors** — Anyone who proposes changes via pull requests or files
  issues. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## Scope

This repository is the source of truth for official `srt-slurm` sweep
configurations only. Changes to benchmark orchestration, CLI behavior, launch
scripts, or analysis code belong upstream in
[NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm).

## Decision Making

- Routine changes (new or updated recipes, documentation fixes) are approved by
  at least one maintainer through the normal pull request review process.
- Changes that affect repository structure, recipe conventions, or tooling are
  decided by maintainer consensus. If consensus cannot be reached, the lead
  maintainer makes the final decision.
- Maintainers are added or removed by consensus of the existing maintainers.

## Code of Conduct

All participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).
