---
name: julia-long-test-workflow
description: 'Use for long-running Julia test suites, TestItemRunner filtering, JET triage, benchmark-driven refactoring, and AirspeedVelocity branch-vs-master comparisons in AbstractOperators.jl.'
argument-hint: 'Describe the operator, test group, or benchmark comparison you want to run'
user-invocable: true
---

# Julia Long Test Workflow

## When To Use

- Iterating on a failing Julia test suite that is too slow to rerun wholesale.
- Narrowing failures with TestItemRunner tags or filenames.
- Verifying JET coverage for public API.
- Refactoring performance-sensitive operator code and checking for regressions.
- Comparing the current branch against `master` using AirspeedVelocity.

## Workflow

1. Start from the smallest relevant test scope.
2. Prefer a persistent Julia REPL for repeated filtered `TestItemRunner.run_tests(...)` calls.
3. Fix real implementation bugs in source instead of weakening tests.
4. Capture all run logs under `.temp/`.
5. For performance-sensitive changes, benchmark before and after.
6. Run focused ASV filters first, then a single full ASV comparison for final validation.
7. Treat `speedup + uncertainty < 0.95` (master/dirty ratio) as a significant regression.
8. Prefer representative large inputs (for this repo use `32768` baseline elements where feasible) to reduce microbenchmark noise.
9. Use AirspeedVelocity with an explicit script path when comparing against revisions that do not yet contain the benchmark file.

## Common Commands

Filtered test run:

```julia
using TestItemRunner
TestItemRunner.run_tests(pwd(); filter = ti -> :MatrixOp in ti.tags)
```

AirSpeedVelocity comparison:

```sh
benchpkg \
  --path . \
  --rev master,dirty \
  --script benchmark/benchmarks.jl \
  --output-dir .temp/asv \
  --exeflags="--threads=4"
```

Render a comparison table:

```sh
benchpkgtable \
  --path . \
  --rev master,dirty \
  --input-dir .temp/asv \
  --ratio \
  --mode time,memory
```

## Done Criteria

- Targeted tests pass.
- JET coverage remains complete for public API touched.
- Benchmark deltas are measured and reported.
- Logs are saved under `.temp/`.

## Session Lessons (AbstractOperators GPU migration)

- Julia package extensions can only `import` the parent package, trigger package(s), and stdlib; if extension code needs a parent dependency API, expose it from the parent module first.
- For FFT plans, prefer `inv(plan)` (AbstractFFTs-generic) over backend-specific `FFTW.plan_inv(...)` to keep CUDA/AMDGPU compatibility.
- With JLArrays/GPUArrays, avoid `copyto!(gpu, cpu_view)` where source is a `SubArray`; materialize first (for example `src[1:n]`) or copy from a plain array.
- During large rebases, resolve conflicts by preserving known-good behavior first, then run focused `TestItemRunner` filters before full-suite reruns.
- Prefer `@testitem` over nested `@testset`; split oversized files into helper modules + multiple test items.
- For honest GPU coverage, keep JLArray checks separate from real device checks and add backend-specific tags (`:cuda`, `:amdgpu`) plus runtime skip guards.
- In `test/runtests.jl`, filter backend-tagged testitems when corresponding runtimes are unavailable, but keep per-test safety checks too.
- Add explicit tests for `domain_storage_type`/`codomain_storage_type` and that `op * x` allocates on the active backend.
