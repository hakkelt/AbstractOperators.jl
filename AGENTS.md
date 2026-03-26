# AGENTS.md

This repository uses layered guidance. Follow it in this order:

1. Read this file first.
2. Read any applicable files under `.github/instructions/` whose `applyTo` pattern matches the files you will edit.
3. Read the matching skill under `.github/skills/` when the task clearly matches a skill's scope.
4. Then inspect the target source files before editing.

## How to choose guidance

- Use `julia-operator-engineering.instructions.md` for changes under `src/**/*.jl`.
- Use `julia-performance.instructions.md` for code under `src/**/*.jl` and `benchmark/**/*.jl` when performance is relevant.
- Use `julia-testing-and-jet.instructions.md` for `test/**/*.jl` and docs-backed test guidance.
- Use `.github/skills/julia-long-test-workflow/SKILL.md` for long Julia test runs, filtered `TestItemRunner` work, JET triage, and AirspeedVelocity comparisons.
- Use `.github/skills/julia-gpu-implementation/SKILL.md` for GPU operator implementations, GPU extensions, GPU-specific tests, and GPU benchmark validation.

## Working rules

- Prefer the smallest skill and instruction set that fully covers the task.
- Do not ignore a matching instruction file because a skill also exists; use both when they apply.
- If multiple instruction files match, combine them rather than choosing only one.
- If a task touches both implementation and tests, read both the source and test instruction files before editing.
- Keep temporary artifacts under `.temp/`.
- When in doubt, inspect the relevant files before making changes.
