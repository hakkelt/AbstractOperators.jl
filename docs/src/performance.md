# Performance

For end users, the main performance recommendations are:

1. Prefer `mul!` over `*` to avoid allocations.
2. Preallocate with `allocate_in_domain(op)` / `allocate_in_codomain(op)`.
3. Use operator constructors with array inputs to preserve storage type (CPU vs GPU).

Developer-oriented performance internals (threading heuristics, storage traits, backend caveats) were moved to [Custom Operators](@ref) to keep this page concise.

## Fused element-wise operators

Inside a `Compose`, consecutive element-wise operators (diagonals, masks, broadcasts and their
adjoints, a scaling, a DFT's normalization) run as one loop, which reads the run's input once and
writes its output once instead of passing every intermediate result through memory. The result is
the same bit for bit as applying the operators one by one. To measure what this is worth, switch
it off at run time:

```julia
AbstractOperators.POINTWISE_FUSION[] = false   # every operator runs its own mul!
AbstractOperators.POINTWISE_FUSION[] = true    # the default
```

```@docs
AbstractOperators.POINTWISE_FUSION
```
