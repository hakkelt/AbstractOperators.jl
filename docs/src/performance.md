# Performance

AbstractOperators.jl is designed for high-performance linear algebra. This page documents the threading model, broadcast optimizations, and tips for maximizing throughput.

## Threading Model

Many operators support multi-threaded execution via [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)'s `@batch` macro. Threading is enabled **automatically** based on array size and the number of available threads.

### The `_should_thread` Function

The internal `_should_thread(x)` function decides whether threading should be enabled for an operator constructed from array `x`:

```julia
_should_thread(d) = length(d) > 2^16 && Threads.nthreads() > 1
```

This returns `true` only when:
1. The array has more than 65,536 elements
2. More than 1 Julia thread is available

For GPU arrays, `GpuExt` overrides this to always return `false`:
```julia
_should_thread(::AbstractGPUArray) = false
```

### Controlling Threading

Most operators accept a `threaded` keyword argument:

```julia
D = DiagOp(d; threaded=true)   # use default heuristic (recommended)
D = DiagOp(d; threaded=false)  # always single-threaded
```

For `Variation` and `BroadCast`, threading applies to the forward pass:

```julia
V = Variation((1000, 1000); threaded=true)   # @batch parallel gradient
B = BroadCast(op, (100, 100); threaded=true)  # parallel broadcasting
```

### Running with Multiple Threads

Start Julia with multiple threads:

```bash
julia --threads=auto --project=test test/runtests.jl
```

Or set the environment variable:

```bash
JULIA_NUM_THREADS=8 julia script.jl
```

## FastBroadcast Optimizations

`DiagOp` and related operators use [FastBroadcast.jl](https://github.com/YingboMa/FastBroadcast.jl)'s `@..` macro for efficient elementwise operations. The threading parameter controls whether `@..` uses multi-threaded SIMD:

```julia
@.. thread=true  y = d * x   # multi-threaded when B=True()
@.. thread=false y = d * x   # single-threaded
```

The `B` type parameter in `DiagOp{B, D, C, N, dS, cS, T}` encodes this choice as a compile-time constant (`FastBroadcast.True()` or `FastBroadcast.False()`).

## Smart Operator Copying

When using operators in multi-threaded or GPU contexts, use `copy_operator` instead of `deepcopy`:

```julia
op = DiagOp(rand(1000)) + DiagOp(rand(1000))  # Sum with internal buffers
op2 = copy_operator(op)  # efficient copy: shares immutable data, copies buffers only
```

### `storage_type` Conversion

`copy_operator` supports converting an operator to use a different storage type:

```julia
using CUDA

op_cpu = Compose(DiagOp(rand(1000)), FiniteDiff(CuArray{Float32}, (1000,)))
op_gpu = copy_operator(op_cpu; storage_type=CuArray)  # convert buffers to GPU
```

### `has_mutable_buffers` Trait

Check if an operator has mutable buffers:

```julia
using AbstractOperators: has_mutable_buffers

has_mutable_buffers(typeof(DiagOp(rand(5))))  # false — safe to share
has_mutable_buffers(typeof(D1 + D2))          # true — needs copy for parallel use
```

Operators with mutable buffers (`true`): `Compose`, `HCAT`, `VCAT`, `Sum`, `HadamardProd`, `OperatorBroadCast`.

## Memory Allocation Tips

1. **Prefer `mul!` over `*`**: The `*` operator allocates output; `mul!` is in-place.
2. **Pre-allocate output**: Use `allocate_in_codomain(op)` to get a properly-typed output buffer.
3. **Avoid nested composition when possible**: Deeply nested compositions allocate intermediate buffers at construction time. These buffers are reused across calls.

```julia
op = DiagOp(d1) * FiniteDiff((n,)) * DiagOp(d2)  # buffers allocated once
y = allocate_in_codomain(op)
x = allocate_in_domain(op)
mul!(y, op, x)  # no allocation
```

## Benchmarking

Run the built-in tests with `@time` or use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl):

```julia
using BenchmarkTools, AbstractOperators

D = DiagOp(rand(100_000))
x = rand(100_000); y = similar(x)

@benchmark mul!($y, $D, $x)
```
