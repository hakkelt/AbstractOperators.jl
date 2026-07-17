export SoftMax

"""
	SoftMax([domain_type=Float64::Type,] dim_in::Tuple)

Creates the softmax non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{e^{\\mathbf{x} }}{ \\sum e^{\\mathbf{x} } }
```

"""
struct SoftMax{T, N, B <: AbstractArray{T, N}} <: NonLinearOperator
    dim::NTuple{N, Int}
    buf::B
end

function SoftMax(x::AbstractArray{T, N}; array_type::Type = _array_wrapper(x)) where {T, N}
    S = _normalize_array_type(array_type, T)
    buf = similar(S, size(x))
    return SoftMax{T, N, typeof(buf)}(size(x), buf)
end

function SoftMax(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}; array_type::Type = Array{T}
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    buf = similar(S, DomainDim)
    fill!(buf, zero(T))
    return SoftMax{T, N, typeof(buf)}(DomainDim, buf)
end

function SoftMax(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return SoftMax(Float64, DomainDim; array_type)
end

function mul!(y::AbstractArray, L::SoftMax, x::AbstractArray)
    check(y, L, x)
    y .= exp.(x .- maximum(x))
    return y ./= sum(y)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftMax}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    L.A.buf .= exp.(L.x .- maximum(L.x))
    L.A.buf ./= sum(L.A.buf)
    d = dot(L.A.buf, b)
    y .= L.A.buf .* (b .- d)
    return y
end

fun_name(L::SoftMax) = "σ"

size(L::SoftMax) = (L.dim, L.dim)

domain_type(::SoftMax{T}) where {T} = T
codomain_type(::SoftMax{T}) where {T} = T
domain_array_type(::SoftMax{T, N, B}) where {T, N, B} = _array_wrapper_type(B){T}
codomain_array_type(::SoftMax{T, N, B}) where {T, N, B} = _array_wrapper_type(B){T}
is_thread_safe(::SoftMax) = false
