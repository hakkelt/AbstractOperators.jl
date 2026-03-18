export Exp

"""
	Exp([domain_type=Float64::Type,] dim_in::Tuple)

Creates the exponential non-linear operator with input dimensions `dim_in`:
```math
e^{ \\mathbf{x} }.
```

"""
struct Exp{T, N, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Exp(domain_type::Type{T}, DomainDim::NTuple{N, Int}) where {T, N}
    return Exp{T, N, Array{T}}(DomainDim)
end

Exp(DomainDim::NTuple{N, Int}) where {N} = Exp{Float64, N, Array{Float64}}(DomainDim)
Exp(DomainDim::Vararg{Int}) = Exp{Float64, length(DomainDim), Array{Float64}}(DomainDim)

function Exp(x::AbstractArray{T}) where {T}
    S = _array_wrapper(x){T}
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
domain_storage_type(::Exp{T, N, S}) where {T, N, S} = S
codomain_storage_type(::Exp{T, N, S}) where {T, N, S} = S
is_thread_safe(::Exp) = true
