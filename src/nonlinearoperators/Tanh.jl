export Tanh

"""
	Tanh([domain_type=Float64::Type,] dim_in::Tuple)

Creates an hyperbolic tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{tanh} ( \\mathbf{x} ).
```

"""
struct Tanh{T, N, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Tanh(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}; array_type::Type = Array{T}
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Tanh{T, N, S}(DomainDim)
end

Tanh(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N} =
    Tanh(Float64, DomainDim; array_type)
Tanh(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}) =
    Tanh(Float64, DomainDim; array_type)

function Tanh(x::AbstractArray{T}; array_type::Type = _array_wrapper(x)) where {T}
    S = _normalize_array_type(array_type, T)
    return Tanh{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::Tanh, x::AbstractArray)
    check(y, L, x)
    return y .= tanh.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Tanh}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(sech.(L.x) .^ 2) .* b
end

fun_name(L::Tanh) = "tanh"

size(L::Tanh) = (L.dim, L.dim)

domain_type(::Tanh{T, N}) where {T, N} = T
codomain_type(::Tanh{T, N}) where {T, N} = T
domain_storage_type(::Tanh{T, N, S}) where {T, N, S} = S
codomain_storage_type(::Tanh{T, N, S}) where {T, N, S} = S
is_thread_safe(::Tanh) = true
