## Description
<!-- Provide a standalone description of the changes in this PR. -->
<!-- Reference any issues closed by this PR with "closes #1234". -->

## Type of change
- [ ] New recipe
- [ ] Update to an existing recipe
- [ ] Tooling / scripts
- [ ] Documentation
- [ ] Other

## Checklist
- [ ] I have read the [Contributing Guidelines](CONTRIBUTING.md).
- [ ] The recipe is placed under the correct `recipes/<single-node|multi-node>/<model>/<GPU>/<framework>/` path.
- [ ] No internal-only paths or hostnames (e.g. cluster-specific filesystem paths or mounts) are included.
- [ ] `python scripts/update_recipe_table.py` has been run and the README table is up to date.
- [ ] The recipe was validated with `srtctl dry-run` locally or by CI.
