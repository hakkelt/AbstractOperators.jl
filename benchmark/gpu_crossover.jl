using Pkg

Pkg.activate(temp=true; io = devnull)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.develop(path = joinpath(@__DIR__, "..", "DSPOperators"))
Pkg.develop(path = joinpath(@__DIR__, "..", "FFTWOperators"))
Pkg.add(PackageSpec(name = "RecursiveArrayTools"))
Pkg.instantiate()

using AbstractOperators
using BenchmarkTools
using DSPOperators
using FFTWOperators
using LinearAlgebra
using Random
using RecursiveArrayTools: ArrayPartition

# ── GPU backend detection ────────────────────────────────────────────────────
# Try each backend in preference order; stop at the first functional one.
# `Base.require` loads the package without a top-level `using` statement so we
# can stay in a conditional context.
const _GPU = let
    found = nothing
    for name in ("CUDA", "AMDGPU", "Metal", "oneAPI")
        Pkg.add(PackageSpec(name = name))
        #try
            m = Base.require(Main, Symbol(name))
            if @invokelatest m.functional()
                found = m
                break
            end
        #catch
        #end
    end
    found
end

if _GPU === nothing
    @warn "No functional GPU backend found (tried CUDA, AMDGPU, Metal, oneAPI)"
    exit(0)
end

# Identify which backend we have (for synchronization dispatch).
const _GPU_SYM = let
    for (sym, name) in [(:cuda, "CUDA"), (:amdgpu, "AMDGPU"), (:metal, "Metal"), (:oneapi, "oneAPI")]
        try
            m = Base.require(Main, Symbol(name))
            m === _GPU && (@eval(using $(Symbol(name))); break)
        catch
        end
    end
    if isdefined(Main, :CUDA) && Main.CUDA === _GPU
        :cuda
    elseif isdefined(Main, :AMDGPU) && Main.AMDGPU === _GPU
        :amdgpu
    elseif isdefined(Main, :Metal) && Main.Metal === _GPU
        :metal
    else
        :oneapi
    end
end

@info "GPU benchmark using backend: $_GPU_SYM"

# ── Backend-agnostic helpers used by builder functions ───────────────────────
# Transfer a CPU array to the elected GPU backend.
function _gpu_to(x::AbstractArray)
    if _GPU_SYM == :cuda
        return _GPU.cu(x)
    elseif _GPU_SYM == :amdgpu
        return _GPU.roc(x)
    elseif _GPU_SYM == :metal
        return _GPU.mtl(x)
    else  # oneapi
        return _GPU.oneArray(x)
    end
end

function _gpu_synchronize()
    hasproperty(_GPU, :synchronize) && _GPU.synchronize()
    return nothing
end

# ── Timing utilities ─────────────────────────────────────────────────────────
function _measure!(y, op, x, is_gpu::Bool; samples::Int)
    mul!(y, op, x)
    is_gpu && _gpu_synchronize()
    return @belapsed begin
        mul!($y, $op, $x)
        $is_gpu && $_gpu_synchronize()
    end samples = samples evals = 1
end

function _threaded_time(threaded_builder, n; samples)
    threaded_builder === nothing && return missing
    op, x, y = threaded_builder(n)
    return _measure!(y, op, x, false; samples)
end

function _gpu_time(gpu_builder, n; samples)
    op, x, y = gpu_builder(n)
    return _measure!(y, op, x, true; samples)
end

function _shape2(n)
    m = max(8, round(Int, sqrt(n)))
    return (m, max(8, cld(n, m)))
end

function _broad_shape(n)
    base = max(128, n)
    return (base, 8)
end

function _diag_cpu(n; threaded)
    d = randn(Float32, n)
    op = DiagOp(d; threaded)
    x = randn(Float32, n)
    y = zeros(Float32, n)
    return op, x, y
end

function _diag_gpu(n)
    d = _GPU.randn(Float32, n)
    op = DiagOp(d; threaded = false)
    x = _GPU.randn(Float32, n)
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

function _finitediff_cpu(n; threaded)
    op = FiniteDiff(Float32, (n,), 1)
    x = randn(Float32, n)
    y = zeros(Float32, n - 1)
    return op, x, y
end

function _finitediff_gpu(n)
    x = _GPU.randn(Float32, n)
    op = FiniteDiff(x, 1)
    y = _GPU.zeros(Float32, n - 1)
    return op, x, y
end

function _getindex_cpu(n; threaded)
    dims = _shape2(n)
    x = randn(Float32, dims...)
    idx = (2:(dims[1] - 1), 2:(dims[2] - 1))
    op = GetIndex(Float32, dims, idx)
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _getindex_gpu(n)
    dims = _shape2(n)
    x = _GPU.randn(Float32, dims...)
    idx = (2:(dims[1] - 1), 2:(dims[2] - 1))
    op = GetIndex(x, idx)
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

# GetIndex: bool-mask variant
function _getindex_boolmask_cpu(n; threaded)
    x = randn(Float32, n)
    mask = [i % 3 == 0 for i in 1:n]
    op = GetIndex(Float32, (n,), mask)
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _getindex_boolmask_gpu(n)
    x = _GPU.randn(Float32, n)
    mask = [i % 3 == 0 for i in 1:n]
    op = GetIndex(x, mask)
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

# GetIndex: integer-vector variant (select every 2nd element → output ≈ n/2)
function _getindex_intvec_cpu(n; threaded)
    x = randn(Float32, n)
    idx = collect(1:2:n)
    op = GetIndex(Float32, (n,), idx)
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _getindex_intvec_gpu(n)
    x = _GPU.randn(Float32, n)
    idx = collect(1:2:n)
    op = GetIndex(x, idx)
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

# Eye
function _eye_cpu(n; threaded)
    x = randn(Float32, n)
    op = Eye(Float32, (n,))
    y = zeros(Float32, n)
    return op, x, y
end

function _eye_gpu(n)
    x = _GPU.randn(Float32, n)
    op = Eye(Float32, (n,); array_type = typeof(x))
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# MatrixOp (dense matrix-vector product)
function _matrixop_cpu(n; threaded)
    A = randn(Float32, n, n)
    op = MatrixOp(Float32, (n,), A)
    x = randn(Float32, n)
    y = zeros(Float32, n)
    return op, x, y
end

function _matrixop_gpu(n)
    A = _GPU.randn(Float32, n, n)
    op = MatrixOp(A)
    x = _GPU.randn(Float32, n)
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# Scale (scales an Eye operator by 2)
function _scale_cpu(n; threaded)
    A = Eye(Float32, (n,))
    op = Scale(Float32(2.0), A; threaded)
    x = randn(Float32, n)
    y = zeros(Float32, n)
    return op, x, y
end

function _scale_gpu(n)
    x = _GPU.randn(Float32, n)
    A = Eye(Float32, (n,); array_type = typeof(x))
    op = Scale(Float32(2.0), A; threaded = false)
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# HadamardProd  (Sin .* Cos)
function _hadamardprod_cpu(n; threaded)
    A = Sin(Float32, (n,))
    B = Cos(Float32, (n,))
    op = HadamardProd(A, B)
    x = randn(Float32, n)
    y = zeros(Float32, n)
    return op, x, y
end

function _hadamardprod_gpu(n)
    x = _GPU.randn(Float32, n)
    A = Sin(Float32, (n,); array_type = typeof(x))
    B = Cos(Float32, (n,); array_type = typeof(x))
    op = HadamardProd(A, B)
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# Nonlinear: Sin
function _sin_cpu(n; threaded)
    x = randn(Float32, n)
    op = Sin(Float32, (n,))
    y = zeros(Float32, n)
    return op, x, y
end

function _sin_gpu(n)
    x = _GPU.randn(Float32, n)
    op = Sin(Float32, (n,); array_type = typeof(x))
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# Nonlinear: Tanh
function _tanh_cpu(n; threaded)
    x = randn(Float32, n)
    op = Tanh(Float32, (n,))
    y = zeros(Float32, n)
    return op, x, y
end

function _tanh_gpu(n)
    x = _GPU.randn(Float32, n)
    op = Tanh(Float32, (n,); array_type = typeof(x))
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# Nonlinear: Exp
function _exp_cpu(n; threaded)
    x = randn(Float32, n)
    op = Exp(Float32, (n,))
    y = zeros(Float32, n)
    return op, x, y
end

function _exp_gpu(n)
    x = _GPU.randn(Float32, n)
    op = Exp(Float32, (n,); array_type = typeof(x))
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

# DCAT
function _dcat_cpu(n; threaded)
    x1 = randn(Float32, n)
    x2 = randn(Float32, n)
    x = ArrayPartition(x1, x2)
    op = DCAT(Eye(Float32, (n,)), DiagOp(randn(Float32, n); threaded = false))
    y = ArrayPartition(zeros(Float32, n), zeros(Float32, n))
    return op, x, y
end

function _dcat_gpu(n)
    x1 = _GPU.randn(Float32, n)
    x2 = _GPU.randn(Float32, n)
    x = ArrayPartition(x1, x2)
    op = DCAT(
        Eye(Float32, (n,); array_type = typeof(x1)), DiagOp(_GPU.randn(Float32, n); threaded = false)
    )
    y = ArrayPartition(_GPU.zeros(Float32, n), _GPU.zeros(Float32, n))
    return op, x, y
end

# AffineAdd
function _affineadd_cpu(n; threaded)
    x = randn(Float32, n)
    b = randn(Float32, n)
    op = AffineAdd(DiagOp(randn(Float32, n); threaded = false), b)
    y = zeros(Float32, n)
    return op, x, y
end

function _affineadd_gpu(n)
    x = _GPU.randn(Float32, n)
    b = _GPU.randn(Float32, n)
    op = AffineAdd(DiagOp(_GPU.randn(Float32, n); threaded = false), b)
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

function _zeropad_cpu(n; threaded)
    dims = _shape2(n)
    x = randn(Float32, dims...)
    op = ZeroPad(Float32, dims, (0, 8))
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _zeropad_gpu(n)
    dims = _shape2(n)
    x = _GPU.randn(Float32, dims...)
    op = ZeroPad(x, (0, 8))
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _variation_cpu(n; threaded)
    dims = _shape2(n)
    x = randn(Float32, dims...)
    op = Variation(Float32, dims; threaded)
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _variation_gpu(n)
    dims = _shape2(n)
    x = _GPU.randn(Float32, dims...)
    op = Variation(x; threaded = false)
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _variation_adj_cpu(n; threaded)
    dims = _shape2(n)
    op = adjoint(Variation(Float32, dims; threaded))
    b = zeros(Float32, prod(dims), length(dims))
    y = zeros(Float32, dims...)
    return op, b, y
end

function _variation_adj_gpu(n)
    dims = _shape2(n)
    x = _GPU.randn(Float32, dims...)
    op = adjoint(Variation(x; threaded = false))
    b = _GPU.zeros(Float32, prod(dims), length(dims))
    y = _GPU.zeros(Float32, dims...)
    return op, b, y
end

function _compose_cpu(n; threaded)
    d = randn(Float32, n - 1)
    fd = FiniteDiff(Float32, (n,), 1)
    op = Compose(DiagOp(d; threaded = false), fd)
    x = randn(Float32, n)
    y = zeros(Float32, n - 1)
    return op, x, y
end

function _compose_gpu(n)
    d = _gpu_to(randn(Float32, n - 1))
    x = _GPU.randn(Float32, n)
    op = Compose(DiagOp(d; threaded = false), FiniteDiff(x, 1))
    y = _GPU.zeros(Float32, n - 1)
    return op, x, y
end

function _sum_cpu(n; threaded)
    d = randn(Float32, n)
    op = Sum(Eye(Float32, (n,)), DiagOp(d; threaded = false))
    x = randn(Float32, n)
    y = zeros(Float32, n)
    return op, x, y
end

function _sum_gpu(n)
    x = _GPU.randn(Float32, n)
    d = _gpu_to(randn(Float32, n))
    op = Sum(Eye(Float32, (n,); array_type = typeof(x)), DiagOp(d; threaded = false))
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

function _broadcast_cpu(n; threaded)
    dims = _broad_shape(n)
    x = randn(Float32, dims[1])
    op = BroadCast(Eye(Float32, (dims[1],)), dims; threaded)
    y = zeros(Float32, dims...)
    return op, x, y
end

function _broadcast_gpu(n)
    dims = _broad_shape(n)
    x = _GPU.randn(Float32, dims[1])
    op = BroadCast(Eye(x), dims; threaded = false)
    y = _GPU.zeros(Float32, dims...)
    return op, x, y
end

function _hcat_cpu(n; threaded)
    x1 = randn(Float32, n)
    x2 = randn(Float32, n)
    x = ArrayPartition(x1, x2)
    op = HCAT(Eye(Float32, (n,)), DiagOp(randn(Float32, n); threaded = false))
    y = zeros(Float32, n)
    return op, x, y
end

function _hcat_gpu(n)
    x1 = _GPU.randn(Float32, n)
    x2 = _GPU.randn(Float32, n)
    x = ArrayPartition(x1, x2)
    op = HCAT(
        Eye(Float32, (n,); array_type = typeof(x1)), DiagOp(_GPU.randn(Float32, n); threaded = false)
    )
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

function _vcat_cpu(n; threaded)
    op = VCAT(Eye(Float32, (n,)), DiagOp(randn(Float32, n); threaded = false))
    x = randn(Float32, n)
    y = ArrayPartition(zeros(Float32, n), zeros(Float32, n))
    return op, x, y
end

function _vcat_gpu(n)
    x = _GPU.randn(Float32, n)
    op = VCAT(
        Eye(Float32, (n,); array_type = typeof(x)), DiagOp(_GPU.randn(Float32, n); threaded = false)
    )
    y = ArrayPartition(_GPU.zeros(Float32, n), _GPU.zeros(Float32, n))
    return op, x, y
end

function _filt_cpu(n; threaded)
    taps = randn(Float32, 7)
    op = Filt(Float32, (n,), taps)
    x = randn(Float32, n)
    y = zeros(Float32, n)
    return op, x, y
end

function _filt_gpu(n)
    x = _GPU.randn(Float32, n)
    taps = randn(Float32, 7)
    op = Filt(x, taps)
    y = _GPU.zeros(Float32, n)
    return op, x, y
end

function _mimofilt_cpu(n; threaded)
    len = max(64, n)
    taps = [randn(Float32, 5), randn(Float32, 3), randn(Float32, 4), randn(Float32, 5)]
    coeffs = [Float32[1] for _ in taps]
    op = MIMOFilt(Float32, (len, 2), taps, coeffs)
    x = randn(Float32, len, 2)
    y = zeros(Float32, len, 2)
    return op, x, y
end

function _mimofilt_gpu(n)
    len = max(64, n)
    taps = [randn(Float32, 5), randn(Float32, 3), randn(Float32, 4), randn(Float32, 5)]
    coeffs = [Float32[1] for _ in taps]
    x = _GPU.randn(Float32, len, 2)
    op = MIMOFilt(x, taps, coeffs)
    y = _GPU.zeros(Float32, len, 2)
    return op, x, y
end

function _conv_cpu(n; threaded)
    h = randn(Float32, 17)
    op = Conv(Float32, (n,), h)
    x = randn(Float32, n)
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _conv_gpu(n)
    h = _gpu_to(randn(Float32, 17))
    op = Conv(Float32, (n,), h)
    x = _GPU.randn(Float32, n)
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _xcorr_cpu(n; threaded)
    h = randn(Float32, 17)
    op = Xcorr(Float32, (n,), h)
    x = randn(Float32, n)
    y = zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _xcorr_gpu(n)
    h = _gpu_to(randn(Float32, 17))
    op = Xcorr(Float32, (n,), h)
    x = _GPU.randn(Float32, n)
    y = _GPU.zeros(Float32, size(op, 1)...)
    return op, x, y
end

function _dft_cpu(n; threaded)
    dims = _shape2(n)
    x = randn(ComplexF32, dims...)
    op = DFT(x)
    y = zeros(ComplexF32, dims...)
    return op, x, y
end

function _dft_gpu(n)
    dims = _shape2(n)
    x = _GPU.randn(ComplexF32, dims...)
    op = DFT(x)
    y = _GPU.zeros(ComplexF32, dims...)
    return op, x, y
end

const CONFIGS = [
    (
        name = "DiagOp",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _diag_cpu(n; threaded = false),
        cpu_threaded = Threads.nthreads() > 1 ? (n -> _diag_cpu(n; threaded = true)) : nothing,
        gpu = _diag_gpu,
    ),
    (
        name = "FiniteDiff",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _finitediff_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _finitediff_gpu,
    ),
    (
        name = "GetIndex",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _getindex_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _getindex_gpu,
    ),
    (
        name = "GetIndex_boolmask",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _getindex_boolmask_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _getindex_boolmask_gpu,
    ),
    (
        name = "GetIndex_intvec",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _getindex_intvec_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _getindex_intvec_gpu,
    ),
    (
        name = "Eye",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _eye_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _eye_gpu,
    ),
    (
        name = "MatrixOp",
        sizes = [2^k for k in 6:12],
        cpu_single = n -> _matrixop_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _matrixop_gpu,
    ),
    (
        name = "Scale",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _scale_cpu(n; threaded = false),
        cpu_threaded = Threads.nthreads() > 1 ? (n -> _scale_cpu(n; threaded = true)) : nothing,
        gpu = _scale_gpu,
    ),
    (
        name = "HadamardProd",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _hadamardprod_cpu(n; threaded = false),
        cpu_threaded = Threads.nthreads() > 1 ? (n -> _hadamardprod_cpu(n; threaded = true)) : nothing,
        gpu = _hadamardprod_gpu,
    ),
    (
        name = "DCAT",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _dcat_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _dcat_gpu,
    ),
    (
        name = "AffineAdd",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _affineadd_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _affineadd_gpu,
    ),
    (
        name = "Sin",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _sin_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _sin_gpu,
    ),
    (
        name = "Tanh",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _tanh_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _tanh_gpu,
    ),
    (
        name = "Exp",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _exp_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _exp_gpu,
    ),
    (
        name = "ZeroPad",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _zeropad_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _zeropad_gpu,
    ),
    (
        name = "Variation",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _variation_cpu(n; threaded = false),
        cpu_threaded = Threads.nthreads() > 1 ? (n -> _variation_cpu(n; threaded = true)) : nothing,
        gpu = _variation_gpu,
    ),
    (
        name = "Variation_adj",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _variation_adj_cpu(n; threaded = false),
        cpu_threaded = Threads.nthreads() > 1 ? (n -> _variation_adj_cpu(n; threaded = true)) : nothing,
        gpu = _variation_adj_gpu,
    ),
    (
        name = "Compose",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _compose_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _compose_gpu,
    ),
    (
        name = "Sum",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _sum_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _sum_gpu,
    ),
    (
        name = "BroadCast",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _broadcast_cpu(n; threaded = false),
        cpu_threaded = Threads.nthreads() > 1 ? (n -> _broadcast_cpu(n; threaded = true)) : nothing,
        gpu = _broadcast_gpu,
    ),
    (
        name = "HCAT",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _hcat_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _hcat_gpu,
    ),
    (
        name = "VCAT",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _vcat_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _vcat_gpu,
    ),
    (
        name = "Filt",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _filt_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _filt_gpu,
    ),
    (
        name = "MIMOFilt",
        sizes = [2^k for k in 8:16],
        cpu_single = n -> _mimofilt_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _mimofilt_gpu,
    ),
    (
        name = "Conv",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _conv_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _conv_gpu,
    ),
    (
        name = "Xcorr",
        sizes = [2^k for k in 10:20],
        cpu_single = n -> _xcorr_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _xcorr_gpu,
    ),
    (
        name = "DFT",
        sizes = [2^k for k in 10:18],
        cpu_single = n -> _dft_cpu(n; threaded = false),
        cpu_threaded = nothing,
        gpu = _dft_gpu,
    ),
]

function main()
    Random.seed!(1234)
    _GPU.functional() || error("GPU backend ($_GPU_SYM) is not functional")

    selected_names = split(get(ENV, "ABSTRACTOPERATORS_GPU_BENCH_OPERATORS", ""), ',')
    selected_names = filter!(!isempty, selected_names)
    configs = isempty(selected_names) ? CONFIGS : filter(c -> c.name in selected_names, CONFIGS)
    samples = parse(Int, get(ENV, "ABSTRACTOPERATORS_GPU_BENCH_SAMPLES", "20"))
    size_override = split(get(ENV, "ABSTRACTOPERATORS_GPU_BENCH_SIZES", ""), ',')
    size_override =
    if isempty(size_override) || (length(size_override) == 1 && isempty(size_override[1]))
        nothing
    else
        parse.(Int, size_override)
    end

    out_dir = joinpath(@__DIR__, "..", ".temp", "gpu-crossover")
    mkpath(out_dir)
    out_file = joinpath(out_dir, "results.tsv")
    md_file = joinpath(out_dir, "report.md")

    # Collect per-operator summary for the markdown report
    summary_rows = NamedTuple[]

    open(out_file, "w") do io
        println(
            io,
            "operator\tsize\tcpu_single_s\tcpu_threaded_s\tgpu_s\tbest_cpu_s\tgpu_faster\tnote",
        )
        for config in configs
            crossover = nothing
            last_gpu = nothing          # for "GPU faster at max size?" check
            last_best_cpu = nothing
            sizes = size_override === nothing ? config.sizes : size_override
            println("=== ", config.name, " ===")
            for n in sizes
                try
                    op_cpu, x_cpu, y_cpu = config.cpu_single(n)
                    cpu_single = _measure!(y_cpu, op_cpu, x_cpu, false; samples)
                    cpu_threaded = _threaded_time(config.cpu_threaded, n; samples)
                    gpu = _gpu_time(config.gpu, n; samples)
                    best_cpu = cpu_threaded === missing ? cpu_single : min(cpu_single, cpu_threaded)
                    gpu_faster = gpu < best_cpu
                    if crossover === nothing && gpu_faster
                        crossover = n
                    end
                    last_gpu = gpu
                    last_best_cpu = best_cpu
                    println(
                        io,
                        join(
                            (
                                config.name,
                                n,
                                cpu_single,
                                cpu_threaded,
                                gpu,
                                best_cpu,
                                gpu_faster,
                                "",
                            ),
                            '\t',
                        ),
                    )
                    println(
                        config.name,
                        " n=",
                        n,
                        " cpu_single=",
                        cpu_single,
                        " cpu_threaded=",
                        cpu_threaded,
                        " gpu=",
                        gpu,
                    )
                catch err
                    println(
                        io,
                        join(
                            (
                                config.name,
                                n,
                                missing,
                                missing,
                                missing,
                                missing,
                                missing,
                                sprint(showerror, err),
                            ),
                            '\t',
                        ),
                    )
                    println(config.name, " n=", n, " ERROR: ", sprint(showerror, err))
                end
            end
            range_str = "n=$(first(sizes))..$(last(sizes))"
            if crossover === nothing
                gpu_at_max = if (last_gpu !== nothing && last_best_cpu !== nothing)
                    (last_gpu < last_best_cpu)
                else
                    missing
                end
                msg =
                    "no crossover detected in range [$range_str]" * (
                    if gpu_at_max === true
                        " (GPU faster at max)"
                    elseif gpu_at_max === false
                        " (CPU still faster at max)"
                    else
                        ""
                    end
                )
                println(config.name, " → ", msg)
            else
                println(config.name, " → crossover at n=", crossover)
            end
            push!(
                summary_rows,
                (
                    operator = config.name,
                    crossover = crossover,
                    range = range_str,
                    gpu_at_max = if (last_gpu !== nothing && last_best_cpu !== nothing)
                        (last_gpu < last_best_cpu)
                    else
                        missing
                    end,
                ),
            )
        end
    end

    # Write markdown report
    open(md_file, "w") do io
        println(io, "# GPU Crossover Benchmark Report\n")
        println(
            io,
            "Each row shows the smallest problem size at which the GPU becomes faster than the best CPU time.",
        )
        println(io, "\"—\" means no crossover was found in the tested range.\n")
        println(io, "| Operator | Crossover size | Tested range | GPU faster at max size |")
        println(io, "|----------|---------------|-------------|----------------------|")
        for row in summary_rows
            xover = row.crossover === nothing ? "—" : string(row.crossover)
            at_max = row.gpu_at_max === missing ? "n/a" : (row.gpu_at_max ? "yes" : "no")
            println(io, "| $(row.operator) | $xover | $(row.range) | $at_max |")
        end
    end

    println("\nWrote results to ", out_file)
    return println("Wrote markdown report to ", md_file)
end

main()
