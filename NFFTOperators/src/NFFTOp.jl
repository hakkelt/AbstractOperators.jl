struct NFFTOp{
        T,
        D,
        P <: NFFT.AbstractNFFTPlan{T, D},
        K <: AbstractMatrix{Complex{T}},
        DC <: AbstractMatrix{T},
    } <: AbstractOperators.LinearOperator
    plan::P
    ksp_buffer::K
    dcf::DC
    threaded::Bool
end

"""
	NFFTOp(image_size::NTuple{D,Int}, trajectory::AbstractArray{T}, dcf::Union{Nothing,Symbol,AbstractArray}=nothing; threaded::Bool=true, kwargs...)

Create a non-uniform fast Fourier transform operator [1]. The operator is created with a given image
size, trajectory, and density compensation function (dcf). The dcf, when applied, corrects for the
non-uniform sample density of the trajectory in the *adjoint* direction (`op' * ksp`); the forward
direction (`op * image`) never uses it. The operator can be used to transform images to k-space and
back.

<em>To use the operator, the NFFT package must be explicitly imported!</em>

# Arguments
- `image_size::NTuple{D,Int}`: The size of the image to transform.
- `trajectory::AbstractArray{T}`: The trajectory of the samples in k-space. The first dimension
  of the trajectory must match the number of image dimensions. The trajectory must have at least
  two dimensions.
- `dcf::Union{Nothing,Symbol,AbstractArray}=nothing`: Controls density compensation:
  - `nothing` (the default): **no** density compensation is applied — the dcf is an array of ones,
    so `op'` is the *true* mathematical adjoint of `op`. This is what any algorithm that assumes
    `A'` is the adjoint (operator-norm estimation via power iteration, CG/CGNR, ...) requires.
  - `:auto`: estimate the dcf with the iterative sample density compensation method of Pipe &
    Menon [2] (`NFFTTools.sdc`), exactly as this constructor always did before this keyword
    existed. With `:auto`, `op'` is **not** the true adjoint of `op` — it is a density-compensated
    approximate inverse, useful for a quick direct (gridding) reconstruction but wrong as the
    adjoint fed to an algorithm that relies on the adjoint relationship.
  - An `AbstractArray`: used as given (its shape from the second dimension of `trajectory` on must
    match, and its element type must match `trajectory`'s). Same caveat as `:auto`: a non-trivial
    dcf makes `op'` a weighted approximate inverse, not the true adjoint.
- `threaded::Bool=true`: Whether to use threading when applying the operator. Defaults to `true`.
- `dcf_estimation_iterations::Int=20`: The number of iterations to use when estimating the dcf.
  Only used when `dcf = :auto`.
- `dcf_correction_function::Function=identity`: A correction function to apply to the estimated dcf.
  Defaults to the identity function. Only used when `dcf = :auto`.
- `kwargs...`: Additional keyword arguments to pass to the NFFTPlan constructor.

# References
1. Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation.
IEEE Transactions on Signal Processing, 51(2), 560-574.
2. Pipe, J. G., & Menon, P. (1999). Sampling density compensation in MRI: rationale and an iterative numerical solution.

# Examples
```jldoctest
julia> using NFFTOperators

julia> image_size = (128, 128);

julia> trajectory = rand(2, 128, 50) .- 0.5;

julia> dcf = rand(128, 50);

julia> op = NFFTOp(image_size, trajectory, dcf)
𝒩  ℂ^(128, 128) -> ℂ^(128, 50)

julia> image = rand(ComplexF64, image_size);

julia> ksp = op * image;

julia> image_reconstructed = op' * ksp;

```
"""
function NFFTOp(
        image_size::NTuple{D, Int},
        trajectory::AbstractArray{T},
        dcf::Union{Nothing, Symbol, AbstractArray} = nothing;
        threaded::Bool = true,
        array_type::Type = Array{T},
        dcf_estimation_iterations::Int = 20,
        dcf_correction_function::Function = identity,
        kwargs...,
    ) where {T, D}
    check_traj(trajectory, D)
    arr_wrapper = _array_wrapper_type(array_type)
    plan = _nfft_plan(arr_wrapper, trajectory, image_size, threaded; kwargs...)
    ksp_shape = size(trajectory)[2:end]
    ksp_buffer = _nfft_adapt(arr_wrapper, zeros(complex(T), ksp_shape...))
    dcf_cpu = _resolve_dcf(dcf, plan, trajectory, ksp_shape, T, D, dcf_estimation_iterations, dcf_correction_function)
    adapted_dcf = _nfft_adapt(arr_wrapper, collect(dcf_cpu))
    threaded_flag = threaded && arr_wrapper === Array
    return NFFTOp{T, D, typeof(plan), typeof(ksp_buffer), typeof(adapted_dcf)}(plan, ksp_buffer, adapted_dcf, threaded_flag)
end

"""
    _resolve_dcf(dcf, plan, trajectory, ksp_shape, T, D, dcf_estimation_iterations, dcf_correction_function)

Resolve the `dcf` keyword of [`NFFTOp`](@ref) into a concrete dcf array:
- `nothing` -> an array of ones (no density compensation, `op'` is the true adjoint).
- `:auto` -> estimate with `NFFTTools.sdc` (the pre-existing automatic behaviour).
- an `AbstractArray` -> used as given, after validating its shape/eltype against `trajectory`.
"""
function _resolve_dcf(::Nothing, plan, trajectory, ksp_shape, T, D, dcf_estimation_iterations, dcf_correction_function)
    return ones(T, ksp_shape...)
end
function _resolve_dcf(dcf::Symbol, plan, trajectory, ksp_shape, T, D, dcf_estimation_iterations, dcf_correction_function)
    dcf === :auto || throw(ArgumentError("dcf as a Symbol must be :auto, got :$dcf"))
    raw_dcf = NFFTTools.sdc(plan; iters = dcf_estimation_iterations)
    return dcf_correction_function(reshape(raw_dcf, ksp_shape))
end
function _resolve_dcf(dcf::AbstractArray, plan, trajectory, ksp_shape, T, D, dcf_estimation_iterations, dcf_correction_function)
    check_traj_and_dcf(trajectory, dcf, D)
    return dcf
end

# Default (CPU) implementations — overridden by NFFTOperatorsGPUArraysExt for GPU types
function _nfft_plan(::Type{Array}, trajectory, image_size, threaded; kwargs...)
    return create_plan(trajectory, image_size, threaded; kwargs...)
end
_nfft_adapt(::Type{Array}, arr::AbstractArray) = collect(arr)

function set_nfft_threading_expr(threading_state_expr, thread_count_expr, body_expr)
    return quote
        local prev_nfft_threading_state = NFFT._use_threads[]
        NFFT._use_threads[] = $threading_state_expr
        local res = $(set_thread_counts_expr(thread_count_expr, body_expr))
        NFFT._use_threads[] = prev_nfft_threading_state
        res
    end
end

macro enable_nfft_threading(expr)
    return set_nfft_threading_expr(true, nthreads(), expr)
end

macro disable_nfft_threading(expr)
    return set_nfft_threading_expr(false, 1, expr)
end

function mul!(ksp::AbstractArray, op::NFFTOp, img::AbstractArray)
    AbstractOperators.check(ksp, op, img)
    if op.threaded
        @enable_nfft_threading mul!(vec(ksp), op.plan, img)
    else
        @disable_nfft_threading mul!(vec(ksp), op.plan, img)
    end
    return ksp
end

function mul!(
        img::AbstractArray,
        adjop::AbstractOperators.AdjointOperator{<:NFFTOp},
        ksp::AbstractArray,
    )
    AbstractOperators.check(img, adjop, ksp)
    op = adjop.A
    if op.threaded
        @.. thread = true op.ksp_buffer = ksp * op.dcf
        @enable_nfft_threading mul!(img, op.plan', vec(op.ksp_buffer))
    else
        @.. op.ksp_buffer = ksp * op.dcf
        @disable_nfft_threading mul!(img, op.plan', vec(op.ksp_buffer))
    end
    return img
end

# Properties

size(L::NFFTOp) = size(L.ksp_buffer), NFFT.size_in(L.plan)
fun_name(::NFFTOp) = "𝒩"
domain_type(::NFFTOp{T}) where {T} = complex(T)
codomain_type(::NFFTOp{T}) where {T} = complex(T)
domain_array_type(op::NFFTOp) = typeof(op.plan.tmpVec)
codomain_array_type(op::NFFTOp{T, D, P, K}) where {T, D, P, K} = K

# Utility

function check_traj(traj, D)
    @assert size(traj, 1) == D "The first dimension of the trajectory must match the number of image dimensions"
    return @assert ndims(traj) > 1 "The trajectory must have at least two dimensions"
end

function check_traj_and_dcf(traj, dcf, D)
    check_traj(traj, D)
    @assert tuple(size(traj)[2:end]...) == size(dcf) "Shape of the trajectory from the second dimension must match the shape of the dcf array"
    return @assert eltype(traj) == eltype(dcf) "The element type of the trajectory must match the element type of the dcf array"
end

function create_plan(trajectory, image_size, threaded; kwargs...)
    traj = reshape(trajectory, size(trajectory, 1), :)
    return if threaded
        return @enable_nfft_threading NFFTPlan(traj, image_size; kwargs...)
    else
        return @disable_nfft_threading NFFTPlan(traj, image_size; kwargs...)
    end
end

# Helper to create matched forward/backward FFT plans so that JET can track
# that both plans have the same element type T and dimension D.
struct _MatchedFFTPlans{T, D}
    forward::FFTW.cFFTWPlan{Complex{T}, -1, true, D, UnitRange{Int64}}
    backward::FFTW.cFFTWPlan{Complex{T}, 1, true, D, UnitRange{Int64}}
end

function _make_matched_fft_plans(tmpVec::Array{Complex{T}, D}, dims_; kwargs...) where {T, D}
    FP = FFTW.plan_fft!(tmpVec, dims_; kwargs...)::FFTW.cFFTWPlan{Complex{T}, -1, true, D, UnitRange{Int64}}
    BP = FFTW.plan_bfft!(tmpVec, dims_; kwargs...)::FFTW.cFFTWPlan{Complex{T}, 1, true, D, UnitRange{Int64}}
    return _MatchedFFTPlans{T, D}(FP, BP)
end

function NFFTPlan(
        k::Matrix{T},
        N::NTuple{D, Int};
        dims::Union{Integer, UnitRange{Int64}} = 1:D,
        fftflags = nothing,
        kwargs...,
    ) where {T, D}
    NFFT.checkNodes(k)

    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)

    if length(NOut) > 1
        params.precompute = NFFT.LINEAR
    end

    tmpVec = Array{Complex{T}, D}(undef, Ñ)

    fftflags_ = (fftflags !== nothing) ? (flags = fftflags,) : NamedTuple()
    plans = _make_matched_fft_plans(tmpVec, dims_; num_threads = FFTW.get_num_threads(), fftflags_...)
    FP = plans.forward
    BP = plans.backward

    calcBlocks =
        (
        params.precompute == NFFT.LINEAR ||
            params.precompute == NFFT.TENSOR ||
            params.precompute == NFFT.POLYNOMIAL
    ) &&
        params.blocking &&
        length(dims_) == D

    blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor = NFFT.precomputeBlocks(
        k, Ñ, params, calcBlocks
    )

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(
        k, N[dims_], Ñ[dims_], params
    )

    U = params.storeDeconvolutionIdx ? N : ntuple(d -> 0, D)
    tmpVecHat = Array{Complex{T}, D}(undef, U)

    return NFFT.NFFTPlan(
        N,
        NOut,
        J,
        k,
        Ñ,
        dims_,
        params,
        FP,
        BP,
        tmpVec,
        tmpVecHat,
        deconvolveIdx,
        windowHatInvLUT,
        windowLinInterp,
        windowPolyInterp,
        blocks,
        nodesInBlocks,
        blockOffsets,
        idxInBlock,
        windowTensor,
        B,
    )
end
