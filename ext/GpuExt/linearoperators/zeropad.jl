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
