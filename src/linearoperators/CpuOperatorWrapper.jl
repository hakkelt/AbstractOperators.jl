"""
    CpuOperatorWrapper{A, DomBuf, CodBuf}

A wrapper that makes any CPU operator usable with GPU input/output arrays.

When `mul!(y::AbstractGPUArray, wrapper, x::AbstractGPUArray)` is called
(after loading GPUArrays or CUDA):
1. Copies `x` (GPU) to the preallocated CPU input buffer
2. Executes the wrapped operator on CPU: `mul!(cpu_output, op, cpu_input)`
3. Copies the CPU output back to `y` (GPU)

This enables GPU-transparent use of operators that do not natively support GPU arrays
(e.g., operators using FFTW, custom CPU-only algorithms).

# Construction
    CpuOperatorWrapper(op::AbstractOperator)

Preallocates CPU buffers matching the domain and codomain of `op`.
The `op` must have CPU storage types (i.e., `Array`-backed).

# Notes
- Thread safety: each wrapper has its own buffers; use `copy_operator` for parallel use.
- GPU `mul!` methods are provided by the GpuExt extension (requires loading GPUArrays).

# Example (with CUDA)
```julia
using AbstractOperators, CUDA
op = FiniteDiff(Float32, (64,))  # CPU operator
wrapper = CpuOperatorWrapper(op)
x_gpu = CUDA.randn(Float32, 64)
y_gpu = similar(x_gpu, 63)
mul!(y_gpu, wrapper, x_gpu)  # GPU in/out, CPU compute
```
"""
struct CpuOperatorWrapper{
        A <: AbstractOperator,
        DomBuf <: AbstractArray,
        CodBuf <: AbstractArray,
    } <: AbstractOperator
    op::A
    dom_buf::DomBuf   # preallocated CPU domain buffer
    cod_buf::CodBuf   # preallocated CPU codomain buffer
end

"""
    CpuOperatorWrapper(op::AbstractOperator)

Create a GPU-transparent wrapper around the CPU operator `op`, preallocating
CPU buffers for domain and codomain arrays.
"""
CpuOperatorWrapper(op::AbstractOperator) =
    CpuOperatorWrapper(op, allocate_in_domain(op), allocate_in_codomain(op))

# Delegate is_linear to the wrapped operator
import OperatorCore: is_linear
is_linear(A::CpuOperatorWrapper) = is_linear(A.op)

# CPU mul! — routes through the internal buffers (used as fallback and for testing)
function mul!(y::AbstractArray, A::CpuOperatorWrapper, x::AbstractArray)
    check(y, A, x)
    copyto!(A.dom_buf, x)
    mul!(A.cod_buf, A.op, A.dom_buf)
    copyto!(y, A.cod_buf)
    return y
end

function mul!(
        y::AbstractArray,
        Ac::AdjointOperator{<:CpuOperatorWrapper},
        x::AbstractArray,
    )
    A = Ac.A
    check(y, Ac, x)
    copyto!(A.cod_buf, x)
    mul!(A.dom_buf, A.op', A.cod_buf)
    copyto!(y, A.dom_buf)
    return y
end

# Properties
Base.size(A::CpuOperatorWrapper) = size(A.op)
fun_name(A::CpuOperatorWrapper) = "CPU[$(fun_name(A.op))]"
domain_type(A::CpuOperatorWrapper) = domain_type(A.op)
codomain_type(A::CpuOperatorWrapper) = codomain_type(A.op)
