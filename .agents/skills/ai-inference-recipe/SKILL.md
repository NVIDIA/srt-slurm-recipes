---
name: ai-inference-recipe
description: Creates single-point srt-slurm recipes from NVIDIA AI Inference benchmark rows. Use when the user mentions ai-inference, NVIDIA inference performance pages, benchmark rows, or asks to create a non-sweep recipe from model/GPU/framework/sequence/concurrency details.
---

# AI Inference Recipe

Use this skill to turn a user description of an NVIDIA AI Inference benchmark row into a local single-point `srt-slurm` recipe.

In this repository, single-point means the generated recipe may keep a `sweep` block, but every sweep key has exactly one value so it expands to one run.

## Workflow

1. Extract the benchmark row fields from the user request: `model`, `hw`, `framework`, `sequence`, `precision`, `tp`, `ep`, `conc`, `spec_decoding`, and `date` when available.
2. Prefer the raw Markdown source unless the user provided a local copy:
   `https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference.md`
3. Run `scripts/map_inference_row_to_recipe.py` with enough `--where FIELD=VALUE` filters to select exactly one row.
4. Emit the recipe with `--emit-pruned` and `--single-point`.
5. Read the JSON report. If multiple rows matched, ask for the missing discriminator. If cautions remain, tell the user what was inferred or left unchanged.

## Command Template

```bash
python scripts/map_inference_row_to_recipe.py \
  --where model=DeepSeek-R1 \
  --where hw=B300 \
  --where sequence=1K/1K \
  --where framework=DYNAMO-TRT \
  --where conc=271 \
  --emit-pruned tmp/deepseek-r1-b300-dynamo-trt-1k1k-c271.yaml \
  --single-point \
  --json
```

Use `--sweep-value KEY=VALUE` for sweep keys that are not present in the NVIDIA row but are specified by the user, such as `pp`, `dp`, or framework-specific tuning keys.

## Rules

- Do not hand-map rows to recipes when the script can do it.
- Do not modify the source recipe when producing a one-off recipe; write a new path with `--emit-pruned`.
- Treat `ai-inference` and `ai-inference.md` as equivalent user references, but prefer `ai-inference.md` for row selection.
- If `--single-point` collapses an unspecified sweep key to the first value, surface that caution to the user.
