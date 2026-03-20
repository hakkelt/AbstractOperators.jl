export Pow

"""
	Pow([domain_type=Float64::Type,] dim_in::Tuple)

Elementwise power `p` non-linear operator with input dimensions `dim_in`.

"""
struct Pow{T, N, I <: Real, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
    p::I
end

function Pow(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}, p::I; array_type::Type = Array{T}
    ) where {T, N, I <: Real}
    S = _normalize_array_type(array_type, T)
    return Pow{T, N, I, S}(DomainDim, p)
end

Pow(DomainDim::NTuple{N, Int}, p::I; array_type::Type = Array{Float64}) where {N, I <: Real} =
    Pow(Float64, DomainDim, p; array_type)

function Pow(x::AbstractArray{T}, p::I; array_type::Type = _array_wrapper(x)) where {T, I <: Real}
    S = _normalize_array_type(array_type, T)
    return Pow{T, ndims(x), I, S}(size(x), p)
end

function mul!(y::AbstractArray, L::Pow, x::AbstractArray)
    check(y, L, x)
    return y .= x .^ L.p
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Pow}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(L.A.p .* (L.x) .^ (L.A.p - 1)) .* b
end

fun_name(L::Pow) = "『"

size(L::Pow) = (L.dim, L.dim)

domain_type(::Pow{T, N}) where {T, N} = T
codomain_type(::Pow{T, N}) where {T, N} = T
domain_storage_type(::Pow{T, N, I, S}) where {T, N, I, S} = S
codomain_storage_type(::Pow{T, N, I, S}) where {T, N, I, S} = S
is_thread_safe(::Pow) = true
