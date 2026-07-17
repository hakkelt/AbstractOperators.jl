---
description: "Use when editing Julia operator implementations in AbstractOperators.jl. Covers operator completeness, package boundaries, storage traits, and behavior-preserving refactors."
name: "Julia Operator Engineering"
applyTo: "src/**/*.jl"
---

# Julia Operator Engineering

- Respect package boundaries:
  - `src/linearoperators/` for concrete linear operators.
  - `src/nonlinearoperators/` for nonlinear operators.
  - `src/calculus/` for composition and operator calculus.
  - `src/batching/` for batch operators.
- For new or changed operators, keep implementation complete:
  - constructors,
  - forward `mul!`,
  - adjoint `mul!` where applicable,
  - size/domain/codomain/storage traits,
  - property traits such as linearity, diagonal structure, and rank-related predicates.
- `check` utility function must be called in all effective `mul!` paths to ensure consistent argument validation and error messages.
- Preserve `domain_array_type` and `codomain_array_type` semantics and dispatch compatibility.
- Constructors should expose an `array_type` keyword where storage backend selection is meaningful.
- `domain_array_type`/`codomain_array_type` must remain consistent with constructor-selected storage.
- When storage checks become stricter, fix operator traits and tests instead of relaxing `check`.
- Prefer behavior-preserving refactors: extract helpers, separate setup from kernels, reduce method size, but do not weaken checks.
- If modifying copy semantics, preserve the package convention that immutable/read-only arrays are shared while mutable working buffers are copied deliberately.
- Keep source formatted with Runic-compatible Julia style.
- GPU extensions live under `ext/GpuExt/` (triggered by `GPUArrays`). Override `mul!` there for any operator whose base implementation uses scalar indexing loops (`@nloops`, `@nref`, `@inbounds y[i] = b[j]`); replace with broadcast-over-view (`y .= view(b, idx...)`).
- When overriding a threaded operator (e.g. `Variation{..., true}`) for GPU, delegate to the non-threaded variant (`Variation{..., false}`) — the threading strategy is CPU-only.
