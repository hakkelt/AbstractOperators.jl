export Atan

"""
	Atan([domain_type=Float64::Type,] dim_in::Tuple)

Creates an inverse tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{atan} ( \\mathbf{x} ).
```

"""
struct Atan{T, N, S <: AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Atan(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}; array_type::Type = Array{T}
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Atan{T, N, S}(DomainDim)
end

function Atan(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return Atan(Float64, DomainDim; array_type)
end
Atan(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}) = Atan(Float64, DomainDim; array_type)

function Atan(x::AbstractArray{T}; array_type::Type = _array_wrapper(x)) where {T}
    S = _normalize_array_type(array_type, T)
    return Atan{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::Atan, x::AbstractArray)
    check(y, L, x)
    return y .= atan.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Atan}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(1.0 ./ (1.0 .+ L.x .^ 2)) .* b
end

fun_name(L::Atan) = "atan"

size(L::Atan) = (L.dim, L.dim)

domain_type(::Atan{T, N}) where {T, N} = T
codomain_type(::Atan{T, N}) where {T, N} = T
domain_storage_type(::Atan{T, N, S}) where {T, N, S} = S
codomain_storage_type(::Atan{T, N, S}) where {T, N, S} = S
is_thread_safe(::Atan) = true
