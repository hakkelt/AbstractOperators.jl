module GpuExt

using GPUArrays
using FFTWOperators
import FFTWOperators: DFT, Normalization, UNNORMALIZED, _dft_scaling, _plan_fft, _plan_bfft

# GPU constructor for DFT with real input — avoids FFTW-specific `flags`/`timelimit`/`num_threads` kwargs
# that are not accepted by GPU FFT backends (e.g. CUDA.CUFFT, AMDGPU.rocFFT).
function FFTWOperators.DFT(
        x::AbstractGPUArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = UNNORMALIZED,
        kwargs...,  # silently ignore FFTW-specific kwargs (flags, timelimit, num_threads)
    ) where {N, D <: Real}
    xc = similar(x, Complex{D})
    A = _plan_fft(xc, dims)
    At = _plan_bfft(xc, dims)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims_t = tuple(dims...)
    scaling = _dft_scaling(size(x), dims_t, normalization)
    return DFT{N, Complex{D}, D, dims_t, S, typeof(A), typeof(At), D}(
        size(x), A, At, normalization, scaling,
    )
end

# GPU constructor for DFT with complex input
function FFTWOperators.DFT(
        x::AbstractGPUArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = UNNORMALIZED,
        kwargs...,
    ) where {N, D <: Complex}
    A = _plan_fft(x, dims)
    At = _plan_bfft(x, dims)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims_t = tuple(dims...)
    scaling = _dft_scaling(size(x), dims_t, normalization)
    return DFT{N, D, D, dims_t, S, typeof(A), typeof(At), real(D)}(
        size(x), A, At, normalization, scaling,
    )
end

end # module GpuExt
