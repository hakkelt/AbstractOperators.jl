export Exp

"""
	Exp([domain_type=Float64::Type,] dim_in::Tuple)

Creates the exponential non-linear operator with input dimensions `dim_in`:
```math
e^{ \\mathbf{x} }.
```

"""
struct Exp{T, N, S <: AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Exp(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}; array_type::Type = Array{T}
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Exp{T, N, S}(DomainDim)
end

function Exp(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return Exp(Float64, DomainDim; array_type)
end
Exp(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}) = Exp(Float64, DomainDim; array_type)

function Exp(x::AbstractArray{T}; array_type::Type = _array_wrapper(x)) where {T}
    S = _normalize_array_type(array_type, T)
    return Exp{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::Exp, x::AbstractArray)
    check(y, L, x)
    return y .= exp.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Exp}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(exp.(L.x)) .* b
end

fun_name(L::Exp) = "e"

size(L::Exp) = (L.dim, L.dim)

domain_type(::Exp{T, N}) where {T, N} = T
codomain_type(::Exp{T, N}) where {T, N} = T
domain_array_type(::Exp{T, N, S}) where {T, N, S} = S
codomain_array_type(::Exp{T, N, S}) where {T, N, S} = S
is_thread_safe(::Exp) = true
