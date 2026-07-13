# Contributing to srt-slurm Recipes

Thank you for contributing official sweep recipes for `srt-slurm`.

This repository stores configuration only. Changes to benchmark orchestration, CLI behavior, launch scripts, or analysis code should be made upstream in [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm).

## Recipe Guidelines

Place each recipe under the most specific leaf in the matrix:

```text
recipes/<single-node|multi-node>/<model>/<GPU>/<framework>/
```

When adding a recipe:

1. Use a descriptive `*.yaml` filename such as `isl_osl.yaml`, `isl_osl_performance-domain.yaml`, etc.
2. Use the model name only, without the model owner, for example `DeepSeek-R1`.
3. Keep model, GPU, framework, and node-count assumptions explicit in the config or adjacent notes.
4. Run `srtctl dry-run -f <recipe.yaml>` before opening a merge request.
5. Include enough tags in example commands or notes for downstream filtering, for example `official,<model>,<gpu>,<framework>`.
6. Prefer small, reviewable changes that add or update one sweep family at a time.

## Developer Certificate of Origin (DCO)

By contributing to this project, you agree to the Developer Certificate of Origin
(DCO) Version 1.1. This certifies that you have the right to submit your
contribution under the open source license used by the project.

The full DCO text is available at https://developercertificate.org/ and is
reproduced below:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## How to Sign Off

Add a `Signed-off-by` trailer to each commit message:

```bash
git commit -s -m "Your commit message"
```

This produces a commit message footer like:

```text
Signed-off-by: Jane Doe <jane.doe@example.com>
```

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](../LICENSE).

For new source files copied or adapted from upstream `srt-slurm`, preserve the
upstream SPDX license header where applicable:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Recipe YAML files should include license headers only if the surrounding file format and tooling support them cleanly.

## Merge Request Checklist

1. The recipe path matches the official matrix.
2. `srtctl dry-run -f <recipe.yaml>` succeeds, or the merge request explains why it could not be run. CI also dry-runs all recipe YAML files.
3. Any required cluster, partition, container, dataset, or model-access assumptions are documented.
4. Commits are signed off with `git commit -s`.
