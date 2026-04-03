---
name: julia-gpu-implementation
description: 'Use for GPU operator implementations, GPU extension fixes, backend-specific testitems, and GPU benchmark validation in AbstractOperators.jl.'
argument-hint: 'Describe the operator, GPU backend, or benchmark you want to implement or validate'
user-invocable: true
---

# Julia GPU Implementation

## When To Use

- Implementing or fixing GPU overrides under `ext/GpuExt/`.
- Adding or updating CUDA/AMDGPU testitems.
- Debugging backend-specific dispatch, storage traits, or array conversion issues.
- Extending benchmark coverage for GPU behavior.
- Checking whether a CPU operator should get a GPU path or stay CPU-only.

## Implementation Rules

- Julia package extensions can only `import` the parent package, trigger package(s), and stdlib; if extension code needs a parent dependency API, expose it from the parent module first.
- For FFT plans, prefer `inv(plan)` (AbstractFFTs-generic) over backend-specific `FFTW.plan_inv(...)` to keep CUDA/AMDGPU compatibility.
- With JLArrays/GPUArrays, avoid `copyto!(gpu, cpu_view)` where the source is a `SubArray`; materialize first, for example with `src[1:n]`, or copy from a plain array.
- Preserve backend storage semantics and trait dispatch when adding GPU methods.
- Keep CPU-only implementation details out of GPU overrides unless the backend truly supports them.
- For GPU `GetIndex` overrides, keep boolean-mask and integer-vector fancy indexing in CPU paths unless the backend support is verified.
- When overriding a threaded operator for GPU, delegate to the non-threaded variant; threading strategy is CPU-only.
- Prefer direct `CuArray(arr)` / `CUDA.zeros(...)` / `AMDGPU.ROCArray(arr)` / `AMDGPU.zeros(...)` calls over intermediate conversion variables.
- Benchmark setup code should normalize wrapped domain and codomain type traits to scalar element types before calling `randn` or `zeros`.

## Testing Rules

- For honest GPU coverage, keep JLArray checks separate from real device checks and add backend-specific tags such as `:cuda` and `:amdgpu` plus runtime skip guards.
- In `test/runtests.jl`, filter backend-tagged testitems when the runtime is unavailable, but keep per-test safety checks too.
- Add explicit tests for `domain_storage_type` and `codomain_storage_type`, and verify that `op * x` allocates on the active backend.
- When adding CUDA/AMDGPU companion tests, prefer direct backend array construction instead of temporary conversion variables.
- For GPU `GetIndex` tests, restrict indices to ranges, colons, and scalar integers; bool-mask and integer-vector `view` forms are not universally supported across GPU backends.
- Migrate GPU-backend storage-type assertions from central quality files into each operator's own CUDA/AMDGPU `@testitem` so they run with the functional tests.
- Use direct `import CUDA` / `import AMDGPU` plus `functional()` guards in testitems; avoid try/catch gating.

## Benchmarking Rules

- Benchmark scripts under `benchmark/` must prefer local workspace package paths over registry-installed copies, otherwise GPU fixes in sibling packages can be silently skipped.
- Use representative large inputs for GPU crossover studies and keep the measurement setup deterministic.
- Capture benchmark logs and generated reports under `.temp/`.

## Tooling Reminders

- Agent sub-tasks frequently generate `Eye(T, dims, array_type)` with three positional arguments instead of `Eye(T, dims; array_type=...)` with a keyword; verify this pattern.
- JET `@test_opt` catches runtime dispatch from `array_type::Type` when it is unparameterized; use `array_type::Type{<:AbstractArray}` and avoid kwarg-to-kwarg forwarding by routing through an internal helper.
- When fixing an "unexpected pass" Aqua error, remove the workaround and use `Aqua.test_all(pkg)` once the underlying issue is fixed.
