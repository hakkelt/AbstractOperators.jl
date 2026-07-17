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
8. Prefer representative large inputs for linear and nonlinear operators to reduce microbenchmark noise, but wrap only fast operators in calculus operators to measure the calculus overhead itself.
9. Use AirspeedVelocity with an explicit script path when comparing against revisions that do not yet contain the benchmark file.

## Common Commands

Main package coverage:

```sh
julia --project=test --code-coverage=user test/runtests.jl
```

Subpackage coverage (DSPOperators, FFTWOperators, NFFTOperators, WaveletOperators have **no** standalone `test/` directory):

> All subpackage code and their GPU extensions are exercised by the parent package's
> `test/` project. Run the same coverage command above; the `.cov` files under each
> subpackage's `src/` will be populated automatically.

Process coverage after a local run:

```sh
julia -e 'using Coverage; Coverage.LCOV.writefile("lcov.info", Coverage.process_folder())'
```

Filtered test run:

```julia
using TestItemRunner
TestItemRunner.run_tests(pwd(); filter = ti -> :MatrixOp in ti.tags) # example of filtering by tag
TestItemRunner.run_tests(pwd(); filter = ti -> ti.name == "DCT") # example of filtering by test name instead of tags
```

### Local benchmark comparison with AirspeedVelocity

AirspeedVelocity works well for local branch-vs-master comparisons and is the
recommended tool for interactive performance investigation:

```sh
mkdir -p .temp/asv
benchpkg \
  --path . \
  --rev master,dirty \
  --script benchmark/benchmarks.jl \
  --output-dir .temp/asv \
  --exeflags="--threads=4"
```

Filtered AirSpeedVelocity comparison for a single benchmark family:

```sh
mkdir -p .temp/asv
benchpkg \
  --path . \
  --rev master,dirty \
  --script benchmark/benchmarks.jl \
  --output-dir .temp/asv \
  --exeflags="--threads=4" \
  --add RecursiveArrayTools \
  --filter MIMOFilt
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

> **Note:** Use AirspeedVelocity with an explicit `--script` path when comparing
> against revisions that do not yet contain the benchmark file.

### CI benchmark comparison (GitHub Actions)

The GitHub Actions benchmark CI does **not** use the AirspeedVelocity action
because the root-level Julia workspace (`[workspace]` in `Project.toml`) causes
that action's revision-management to mis-resolve the monorepo subprojects.
Instead, two workflows implement a fork-safe two-stage approach:

- **`benchmark.yml`** – unprivileged `pull_request` job that checks out both
  the base and head revisions, runs `benchmark/compare.jl` against explicit
  worktree paths, and uploads `body.md`, `pr_number.txt`, and
  `julia_version.txt` as an artifact.
- **`post_benchmark_comment.yml`** – privileged `workflow_run` job that
  downloads the artifact and creates or updates the PR comment.

The comparison table mirrors AirspeedVelocity output with separate Time and
Memory sections, base/head columns, a ratio column, and emoji indicators:
- 🚀 significant speedup: `ratio − ratio_err > 1.2` (time) or `ratio < 0.5` (memory)
- 🐢 significant slowdown: `ratio + ratio_err < 0.8` (time) or `ratio > 1.5` (memory)

To run the comparison locally with the same script used by CI:

```sh
# Check out base separately, e.g. in a worktree:
git worktree add .temp/base master

julia --project=benchmark benchmark/compare.jl \
  --base-dir  .temp/base \
  --head-dir  . \
  --output-dir .temp/bench-compare \
  --pr        0 \
  --julia-version "$(julia -e 'print(VERSION)')"

cat .temp/bench-compare/body.md
```

## Done Criteria

- Targeted tests pass.
- JET coverage remains complete for public API touched.
- Benchmark deltas are measured and reported.
- Logs are saved under `.temp/`.
