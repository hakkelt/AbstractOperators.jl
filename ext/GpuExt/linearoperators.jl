# GetIndex: vectorized GPU mul! to avoid scalar indexing
function mul!(y::AbstractGPUArray, L::GetIndex{<:NTuple{K, Any}}, b::AbstractGPUArray) where {K}
    check(y, L, b)
    y .= view(b, L.idx...)
    return y
end

function mul!(y::AbstractGPUArray, Lc::AdjointOperator{<:GetIndex{<:NTuple{K, Any}}}, b::AbstractGPUArray) where {K}
    check(y, Lc, b)
    fill!(y, zero(eltype(y)))
    view(y, Lc.A.idx...) .= b
    return y
end

# ZeroPad: vectorized GPU mul! to avoid scalar indexing in @generated loop
function mul!(y::AbstractGPUArray, L::ZeroPad{N}, b::AbstractGPUArray) where {N}
    check(y, L, b)
    fill!(y, zero(eltype(y)))
    dst = view(y, ntuple(i -> 1:L.dim_in[i], N)...)
    copyto!(dst, b)
    return y
end

function mul!(y::AbstractGPUArray, Lc::AdjointOperator{<:ZeroPad{N}}, b::AbstractGPUArray) where {N}
    check(y, Lc, b)
    src = view(b, ntuple(i -> 1:Lc.A.dim_in[i], N)...)
    copyto!(y, src)
    return y
end

# Variation: forward (threaded) — delegate to non-threaded for GPU since @batch is CPU-only
function mul!(y::AbstractGPUArray, A::Variation{T, N, true, S}, b::AbstractGPUArray) where {T, N, S}
    mul!(y, Variation{T, N, false, S}(A.dim_in), b)
end

# Helper: GPU-compatible Variation adjoint.
function _variation_adjoint_gpu!(y, A, b)
    check(y, A, b)
    Av = A.A
    dim_in = Av.dim_in
    T = domain_type(Av)
    N = length(dim_in)
    fill!(y, zero(T))
    n = prod(dim_in)
    stride = 1
    for d in 1:N
        nd = dim_in[d]
        rest = n ÷ (stride * nd)
        b_col = reshape(view(b, :, d), stride, nd, rest)
        y_3d = reshape(y, stride, nd, rest)

        y_3d[:, 1, :] .+= -(b_col[:, 1, :] .+ b_col[:, 2, :])

        if nd >= 3
            y_3d[:, 2, :] .+= b_col[:, 2, :] .+ b_col[:, 1, :] .- b_col[:, 3, :]
        elseif nd == 2
            y_3d[:, 2, :] .+= b_col[:, 2, :]
        end

        for k in 3:(nd - 1)
            y_3d[:, k, :] .+= b_col[:, k, :] .- b_col[:, k + 1, :]
        end

        if nd >= 3
            y_3d[:, nd, :] .+= b_col[:, nd, :]
        end

        stride *= nd
    end
    return y
end

function mul!(y::AbstractGPUArray, A::AdjointOperator{<:Variation{T, N, false}}, b::AbstractGPUArray) where {T, N}
    _variation_adjoint_gpu!(y, A, b)
end

function mul!(y::AbstractGPUArray, A::AdjointOperator{<:Variation{T, N, true}}, b::AbstractGPUArray) where {T, N}
    _variation_adjoint_gpu!(y, A, b)
end

