# Variation: KernelAbstractions @kernel implementations.

@kernel function _var_fwd_dim1_kernel!(y_col, @Const(b_flat), stride)
    cnt = @index(Global, Linear)
    n = length(b_flat)
    i1 = (cnt - 1) % stride + 1
    if i1 == 1
        next = cnt + 1 <= n ? cnt + 1 : cnt
        y_col[cnt] = b_flat[next] - b_flat[cnt]
    else
        y_col[cnt] = b_flat[cnt] - b_flat[cnt - 1]
    end
end

@kernel function _var_fwd_dimd_kernel!(y_col, @Const(b_flat), stride, nd)
    cnt = @index(Global, Linear)
    k = (cnt - 1) ÷ stride
    pos = k % nd
    if pos == 0
        y_col[cnt] = b_flat[cnt + stride] - b_flat[cnt]
    else
        y_col[cnt] = b_flat[cnt] - b_flat[cnt - stride]
    end
end

@kernel function _var_adj_kernel!(y_flat, @Const(b), @Const(dim_in_ntuple))
    cnt = @index(Global, Linear)
    dim_in = dim_in_ntuple
    N = length(dim_in)
    T = eltype(y_flat)
    acc = zero(T)
    stride = 1
    for d in 1:N
        nd = dim_in[d]
        i_d = ((cnt - 1) ÷ stride) % nd + 1
        acc += if i_d == 1
            -(b[cnt, d] + b[cnt + stride, d])
        elseif i_d == nd
            b[cnt, d]
        elseif i_d == 2
            b[cnt, d] + b[cnt - stride, d] - b[cnt + stride, d]
        else
            b[cnt, d] - b[cnt + stride, d]
        end
        stride *= nd
    end
    y_flat[cnt] = acc
end

function mul!(y::AbstractGPUArray, A::Variation{T, N, false}, b::AbstractGPUArray) where {T, N}
    check(y, A, b)
    backend = KernelAbstractions.get_backend(b)
    n = length(b)
    b_flat = reshape(b, n)

    stride1 = size(b, 1)
    ker1 = _var_fwd_dim1_kernel!(backend, 256)
    ker1(view(y, :, 1), b_flat, stride1; ndrange = n)

    stride = stride1
    for d in 2:N
        nd = size(b, d)
        kerd = _var_fwd_dimd_kernel!(backend, 256)
        kerd(view(y, :, d), b_flat, stride, nd; ndrange = n)
        stride *= nd
    end
    applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)
    return y
end

function mul!(y::AbstractGPUArray, A::Variation{T, N, true, S}, b::AbstractGPUArray) where {T, N, S}
    return mul!(y, Variation{T, N, false, S}(A.dim_in), b)
end

function _var_adj_gpu!(y::AbstractGPUArray, dim_in, b::AbstractGPUArray)
    backend = KernelAbstractions.get_backend(y)
    n = length(y)
    ker = _var_adj_kernel!(backend, 256)
    ker(reshape(y, n), b, dim_in; ndrange = n)
    applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)
    return y
end

function mul!(
        y::AbstractGPUArray, A::AdjointOperator{<:Variation{T, N, false}}, b::AbstractGPUArray
    ) where {T, N}
    check(y, A, b)
    return _var_adj_gpu!(y, A.A.dim_in, b)
end

function mul!(
        y::AbstractGPUArray, A::AdjointOperator{<:Variation{T, N, true}}, b::AbstractGPUArray
    ) where {T, N}
    check(y, A, b)
    return _var_adj_gpu!(y, A.A.dim_in, b)
end
