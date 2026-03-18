# GPU Support

AbstractOperators.jl provides GPU compatibility through a lightweight extension model. All major operators work transparently with GPU arrays — no code changes are needed for most use cases.

## Using GPU Arrays

Simply pass a GPU array to the `mul!` function or the `*` operator:

```julia
using AbstractOperators, CUDA  # or any GPU backend

x_gpu = CuArray(randn(Float32, 10))
y_gpu = CuArray(zeros(Float32, 9))

F = FiniteDiff(Float32, (10,))  # CPU operator, GPU arrays as input
mul!(y_gpu, F, x_gpu)           # works transparently
```

For operators that carry internal arrays (like `DiagOp`), construct them from GPU arrays to get a GPU-typed operator:

```julia
d_gpu = CuArray(rand(Float32, 10))
D = DiagOp(d_gpu)                # GPU DiagOp — threading disabled automatically
y_gpu = similar(x_gpu)
mul!(y_gpu, D, x_gpu)
```

## Threading and GPU

CPU threading (via [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)'s `@batch`) is automatically disabled when GPU arrays are detected. This happens through the `_should_thread` mechanism:

- For operators with array fields (e.g., `DiagOp`), the constructor checks `_should_thread(d)` which returns `false` for `AbstractGPUArray`s.
- For GPU arrays passed to operators at construction time (e.g., `Variation(gpu_x)`), threading is disabled automatically.
- The `GpuExt` extension overrides `_should_thread(::AbstractGPUArray) = false`.

## Storage Type Tracking

Every operator tracks its storage type via `domain_storage_type` and `codomain_storage_type`. This enables correct intermediate buffer allocation in composed operators:

```julia
using AbstractOperators, CUDA

# Create GPU operators
D = DiagOp(CuArray(rand(Float32, 5)))
domain_storage_type(D)     # CuArray{Float32}
codomain_storage_type(D)   # CuArray{Float32}

# Composition allocates GPU buffers automatically
F = FiniteDiff(CuArray{Float32}, (10,))  # storage_type-aware FiniteDiff
domain_storage_type(F)     # CuArray{Float32}
```

For operators without an array at construction time, pass a GPU array to the array-based constructor:

```julia
F = FiniteDiff(CuArray(ones(Float32, 5, 4)), 1)  # deduces storage type from input
V = Variation(CuArray(ones(Float32, 4, 3)))        # threaded=false automatically
```

Or use the `storage_type` keyword argument (for operators that support it):

```julia
Z = Zeros(Float32, (4,), Float32, (3,); storage_type=CuArray)
```

## GpuExt Extension

GPU-specific overrides live in `ext/GpuExt/`. The extension is loaded automatically when `GPUArrays.jl` is loaded (directly or via any GPU backend like CUDA.jl).

The extension provides:
- `_should_thread(::AbstractGPUArray) = false` — disables CPU threading for GPU arrays
- `storage_type_display_string(::Type{<:AbstractGPUArray}) = "ᵍᵖᵘ"` — shows ᵍᵖᵘ superscript in operator display
- Variation adjoint GPU override — vectorized reshape-based stencil (no scalar indexing)
- BroadCast threading GPU override — falls back to non-threaded path

## Testing with JLArrays

The test suite includes GPU tests using [JLArrays.jl](https://github.com/JuliaGPU/JLArrays.jl), a simple CPU-backed array type that mimics GPU behavior (no scalar indexing, GPUArrays interface). GPU tests are tagged `:gpu`:

```bash
julia --project=test -e '
    using TestItemRunner
    @run_package_tests filter=ti -> :gpu in ti.tags
'
```

## Limitations

- **`Axt_mul_Bx`**: The forward `mul!` uses scalar GPU indexing (`y[1] = dot(...)`) — not GPU-compatible.
- **Boolean mask indexing**: `GetIndex` with a boolean CPU mask on a GPU array is backend-specific (e.g., works with CUDA.jl but not JLArrays). Use integer-vector indices instead.
- **DCT/IDCT** (FFTWOperators): Use CPU FFTW plans — not GPU-compatible. Use `DFT`/`RDFT` for GPU (via `AbstractFFTs` interface).
- **Batching operators**: `SimpleBatchOp` and `SpreadingBatchOp` are not yet tested with GPU arrays.

## NFFTOperators GPU Support (CUDA)

NFFTOperators.jl supports GPU via the `array_type` keyword argument. This requires loading CUDA.jl (or any package that loads GPUArrays + Adapt):

```julia
using NFFTOperators, CUDA

image_size = (128, 128)
trajectory = rand(Float32, 2, 128, 50) .- 0.5f0
dcf = rand(Float32, 128, 50)

# Create GPU NFFT operator — ksp_buffer and dcf are CuArrays
op_gpu = NFFTOp(image_size, trajectory, dcf; array_type=CuArray, threaded=false)

x_gpu = CUDA.randn(ComplexF32, image_size)
y_gpu = op_gpu * x_gpu         # forward: image → k-space (on GPU)
img_rec = op_gpu' * y_gpu      # adjoint: k-space → image (on GPU)
```

**Note:** GPU NFFT requires CUDA.jl because it relies on cuFFT plans. JLArrays (used in tests) does not support FFT plans and cannot be used with NFFTOperators GPU.

The trajectory `k` is kept on CPU (as required by NFFT.jl's GPU plan); only the computation buffers (`ksp_buffer`, `dcf`) and input/output arrays are on GPU.

## CpuOperatorWrapper

For operators that do not natively support GPU arrays (e.g., FFTWOperators DCT, custom CPU-only operators), use `CpuOperatorWrapper`:

```julia
using AbstractOperators, CUDA

# Any CPU operator
op = FiniteDiff(Float32, (64,))  # or any FFTWOperators, etc.

# Wrap it — preallocates CPU buffers for domain and codomain
wrapper = CpuOperatorWrapper(op)

x_gpu = CUDA.randn(Float32, 64)
y_gpu = similar(x_gpu, 63)

mul!(y_gpu, wrapper, x_gpu)   # GPU in → CPU compute → GPU out
mul!(x_gpu, wrapper', y_gpu)  # GPU in → CPU adjoint → GPU out
```

The wrapper preallocates CPU buffers (`dom_buf`, `cod_buf`) to avoid allocations during `mul!`. For parallel use, create independent copies per thread:

```julia
wrappers = [CpuOperatorWrapper(op) for _ in 1:Threads.nthreads()]
```
