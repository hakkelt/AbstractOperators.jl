export Eye

abstract type AbstractEye{T,N,S<:AbstractArray} <: LinearOperator end

"""
	Eye([domainType=Float64::Type,] dim_in::Tuple)
	Eye([domainType=Float64::Type,] dims...)

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
struct Eye{T,N,S<:AbstractArray{T}} <: AbstractEye{T,N,S}
	dim::NTuple{N,Integer}
end

# Constructors
###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
function Eye(
	domainType::Type{T}, domainDim::NTuple{N,Int}, storageType::Type{S}=Array{T}
) where {N,T,S<:AbstractArray{T}}
	return Eye{domainType,N,storageType}(domainDim)
end
###

Eye(t::Type, dims::Vararg{Integer}) = Eye(t, dims)
Eye(dims::NTuple{N,Integer}) where {N} = Eye(Float64, dims)
Eye(dims::Vararg{Integer}) = Eye(Float64, dims)
Eye(x::A) where {A<:AbstractArray} = Eye(eltype(x), size(x), Array{eltype(x)})

# Mappings

mul!(y::AbstractArray{T,N}, ::AbstractEye{T,N}, b::AbstractArray{T,N}) where {T,N} = y .= b
function mul!(
	y::AbstractArray{T,N}, ::AdjointOperator{E}, b::AbstractArray{T,N}
) where {T,N,E<:AbstractEye{T,N}}
	return y .= b
end

# Properties
diag(::AbstractEye) = 1.0
diag_AcA(::AbstractEye) = 1.0
diag_AAc(::AbstractEye) = 1.0

domainType(::AbstractEye{T,N}) where {T,N} = T
codomainType(::AbstractEye{T,N}) where {T,N} = T
domain_storage_type(::AbstractEye{T,N,S}) where {T,N,S} = S
codomain_storage_type(::AbstractEye{T,N,S}) where {T,N,S} = S
is_thread_safe(::Eye) = true

size(L::AbstractEye) = (L.dim, L.dim)

fun_name(::AbstractEye) = "I"

is_eye(::AbstractEye) = true
is_diagonal(::AbstractEye) = true
is_AcA_diagonal(::AbstractEye) = true
is_AAc_diagonal(::AbstractEye) = true
is_orthogonal(::AbstractEye) = true
is_invertible(::AbstractEye) = true
is_full_row_rank(::AbstractEye) = true
is_full_column_rank(::AbstractEye) = true

has_optimized_normalop(::AbstractEye) = true
get_normal_op(L::AbstractEye) = L

LinearAlgebra.opnorm(L::AbstractEye) = one(real(domainType(L)))
