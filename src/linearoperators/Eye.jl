export Eye

abstract type AbstractEye{T, N, S <: AbstractArray} <: LinearOperator end

"""
	Eye([domain_type=Float64::Type,] dim_in::Tuple)
	Eye([domain_type=Float64::Type,] dims...)

Create the identity operator.

```jldoctest
julia> op = Eye(Float64,(4,))
I  ℝ^4 -> ℝ^4

julia> op = Eye(2,3,4)
I  ℝ^(2, 3, 4) -> ℝ^(2, 3, 4)

julia> op*ones(2,3,4) == ones(2,3,4)
true
	
```
"""
struct Eye{T, N, S <: AbstractArray{T}} <: AbstractEye{T, N, S}
    dim::NTuple{N, Int}
end

# Constructors
@inline _make_eye(::Type{T}, dims::NTuple{N, Int}, ::Type{S}) where {T, N, S <: AbstractArray} =
    Eye{T, N, _normalize_array_type(S, T)}(dims)

function Eye(
        domain_type::Type{T}, domainDim::NTuple{N, <:Integer}; array_type::Type{<:AbstractArray} = Array{T}
    ) where {N, T}
    return _make_eye(domain_type, map(Int, domainDim), array_type)
end

Eye(t::Type{T}, dims::Vararg{Integer}; array_type::Type{<:AbstractArray} = Array{T}) where {T} =
    _make_eye(t, map(Int, dims), array_type)
Eye(dims::NTuple{N, Integer}; array_type::Type{<:AbstractArray} = Array{Float64}) where {N} =
    _make_eye(Float64, map(Int, dims), array_type)
Eye(dims::Vararg{Integer}; array_type::Type{<:AbstractArray} = Array{Float64}) =
    _make_eye(Float64, map(Int, dims), array_type)
Eye(x::A) where {A <: AbstractArray} = _make_eye(eltype(x), size(x), _array_wrapper(x){eltype(x)})

# Mappings

function mul!(y::AbstractArray, L::AbstractEye, b::AbstractArray)
    check(y, L, b)
    y .= b
    return y
end

# Properties
diag(::AbstractEye) = 1.0
diag_AcA(::AbstractEye) = 1.0
diag_AAc(::AbstractEye) = 1.0

domain_type(::AbstractEye{T, N}) where {T, N} = T
codomain_type(::AbstractEye{T, N}) where {T, N} = T
domain_storage_type(::AbstractEye{T, N, S}) where {T, N, S} = S
codomain_storage_type(::AbstractEye{T, N, S}) where {T, N, S} = S
is_thread_safe(::Eye) = true

size(L::AbstractEye) = (L.dim, L.dim)

fun_name(::AbstractEye) = "I"

is_eye(::AbstractEye) = true
is_diagonal(::AbstractEye) = true
is_orthogonal(::AbstractEye) = true
is_invertible(::AbstractEye) = true
is_full_row_rank(::AbstractEye) = true
is_full_column_rank(::AbstractEye) = true
is_symmetric(::AbstractEye) = true
is_positive_definite(::AbstractEye) = true
is_positive_semidefinite(::AbstractEye) = true

has_optimized_normalop(::AbstractEye) = true
get_normal_op(L::AbstractEye) = L

has_fast_opnorm(::AbstractEye) = true
LinearAlgebra.opnorm(L::AbstractEye) = one(real(domain_type(L)))
AdjointOperator(L::AbstractEye) = L
