module GpuExt

using GPUArrays
using DSPOperators
import LinearAlgebra: mul!
import DSPOperators: Filt, MIMOFilt, _rfft, _irfft
import AbstractOperators: AdjointOperator, check

# Upload CPU array padded to `fftlen` → GPU. Uses full-array copyto! (not a view) to avoid scalar indexing.
function _upload_padded(ref::AbstractGPUArray{T}, b_cpu, fftlen::Int) where {T}
    b_pad_cpu = zeros(T, fftlen)
    b_pad_cpu[1:length(b_cpu)] .= b_cpu
    buf = similar(ref, fftlen)
    copyto!(buf, b_pad_cpu)   # full Vector → GPU (supported by JLArrays and CUDA)
    return buf
end

# Pad x (GPU) to length fftlen in a zero-initialized GPU buffer.
function _pad_gpu(x::AbstractGPUArray{T}, fftlen::Int) where {T}
    buf = similar(x, fftlen)
    fill!(buf, zero(T))
    # Use view on GPU side (GPU view ← GPU array is fine, no scalar indexing)
    copyto!(view(buf, 1:length(x)), x)
    return buf
end

# Copy first n elements of `result` to `y`. Works whether result is GPU or CPU (JLArrays fallback).
function _copy_head!(y::AbstractGPUArray, result, n::Int)
    # `result[1:n]` materializes to either a GPU array slice or a CPU Vector;
    # both are safe to copyto! into a GPU destination (avoids SubArray scalar indexing issue).
    copyto!(y, result[1:n])
end

# FFT-based FIR forward: y[1:n] = (b ★ x)[1:n]  (causal, truncated)
function _fir_fwd_gpu!(y::AbstractGPUArray{T}, b_cpu, x::AbstractGPUArray{T}) where {T <: Real}
    n = length(x)
    fftlen = nextpow(2, n + length(b_cpu) - 1)
    b_pad = _upload_padded(x, b_cpu, fftlen)
    x_pad = _pad_gpu(x, fftlen)
    Xf = _rfft(x_pad)
    Xf .*= _rfft(b_pad)
    _copy_head!(y, _irfft(Xf, fftlen), n)
    return y
end

# FFT-based FIR adjoint: y[1:n] = cross-corr(x, b)[1:n]  (adjoint of causal FIR)
function _fir_adj_gpu!(y::AbstractGPUArray{T}, b_cpu, x::AbstractGPUArray{T}) where {T <: Real}
    n = length(x)
    fftlen = nextpow(2, n + length(b_cpu) - 1)
    b_pad = _upload_padded(y, b_cpu, fftlen)
    x_pad = _pad_gpu(x, fftlen)
    Xf = _rfft(x_pad)
    Xf .*= conj.(_rfft(b_pad))
    _copy_head!(y, _irfft(Xf, fftlen), n)
    return y
end

# Accumulate: y += fir_fwd(b, x)[1:n]
function _add_fir_fwd_gpu!(y::AbstractGPUArray{T}, b_cpu, x::AbstractGPUArray{T}) where {T <: Real}
    n = length(x)
    fftlen = nextpow(2, n + length(b_cpu) - 1)
    b_pad = _upload_padded(x, b_cpu, fftlen)
    x_pad = _pad_gpu(x, fftlen)
    Xf = _rfft(x_pad)
    Xf .*= _rfft(b_pad)
    result = _irfft(Xf, fftlen)
    y .+= result[1:n]
    return y
end

# Accumulate adjoint: y += fir_adj(b, x)[1:n]
function _add_fir_adj_gpu!(y::AbstractGPUArray{T}, b_cpu, x::AbstractGPUArray{T}) where {T <: Real}
    n = length(x)
    fftlen = nextpow(2, n + length(b_cpu) - 1)
    b_pad = _upload_padded(y, b_cpu, fftlen)
    x_pad = _pad_gpu(x, fftlen)
    Xf = _rfft(x_pad)
    Xf .*= conj.(_rfft(b_pad))
    result = _irfft(Xf, fftlen)
    y .+= result[1:n]
    return y
end

# ─── Filt GPU mul! ────────────────────────────────────────────────────────────

function mul!(y::AbstractGPUArray{T}, A::Filt, b::AbstractGPUArray{T}) where {T}
    check(y, A, b)
    length(A.a) == 1 ||
        throw(ArgumentError("IIR Filt is not supported on GPU arrays. Use CpuOperatorWrapper to run on CPU."))
    for col in 1:size(b, 2)
        _fir_fwd_gpu!(view(y, :, col), A.b, view(b, :, col))
    end
    return y
end

function mul!(y::AbstractGPUArray{T}, L::AdjointOperator{<:Filt}, b::AbstractGPUArray{T}) where {T}
    check(y, L, b)
    A = L.A
    length(A.a) == 1 ||
        throw(ArgumentError("IIR Filt adjoint is not supported on GPU arrays. Use CpuOperatorWrapper."))
    for col in 1:size(b, 2)
        _fir_adj_gpu!(view(y, :, col), A.b, view(b, :, col))
    end
    return y
end

# ─── MIMOFilt GPU mul! ────────────────────────────────────────────────────────

function mul!(y::AbstractGPUArray{T}, L::MIMOFilt, b::AbstractGPUArray{T}) where {T}
    check(y, L, b)
    for a in L.A
        length(a) == 1 ||
            throw(ArgumentError("IIR MIMOFilt is not supported on GPU arrays. Use CpuOperatorWrapper."))
    end
    fill!(y, zero(T))
    cnt = 0
    for cy in 1:L.dim_out[2]
        for cx in 1:L.dim_in[2]
            cnt += 1
            _add_fir_fwd_gpu!(view(y, :, cy), L.B[cnt], view(b, :, cx))
        end
    end
    return y
end

function mul!(y::AbstractGPUArray{T}, M::AdjointOperator{<:MIMOFilt}, b::AbstractGPUArray{T}) where {T}
    check(y, M, b)
    L = M.A
    for a in L.A
        length(a) == 1 ||
            throw(ArgumentError("IIR MIMOFilt adjoint is not supported on GPU arrays. Use CpuOperatorWrapper."))
    end
    fill!(y, zero(T))
    cnt = 0
    for cy in 1:L.dim_out[2]
        for cx in 1:L.dim_in[2]
            cnt += 1
            _add_fir_adj_gpu!(view(y, :, cx), L.B[cnt], view(b, :, cy))
        end
    end
    return y
end

end # module GpuExt
