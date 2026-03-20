export Sech

"""
	Sech([domain_type=Float64::Type,] dim_in::Tuple)

Creates an hyperbolic secant non-linear operator with input dimensions `dim_in`:
```math
\\text{sech} ( \\mathbf{x} ).
```

"""
struct Sech{T, N, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Sech(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}; array_type::Type = Array{T}
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Sech{T, N, S}(DomainDim)
end

Sech(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N} =
    Sech(Float64, DomainDim; array_type)
Sech(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}) =
    Sech(Float64, DomainDim; array_type)

function Sech(x::AbstractArray{T}; array_type::Type = _array_wrapper(x)) where {T}
    S = _normalize_array_type(array_type, T)
    return Sech{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::Sech, x::AbstractArray)
    check(y, L, x)
    return y .= sech.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sech}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= -conj.(tanh.(L.x) .* sech.(L.x)) .* b
end

fun_name(L::Sech) = "sech"

size(L::Sech) = (L.dim, L.dim)

domain_type(::Sech{T, N}) where {T, N} = T
codomain_type(::Sech{T, N}) where {T, N} = T
domain_storage_type(::Sech{T, N, S}) where {T, N, S} = S
codomain_storage_type(::Sech{T, N, S}) where {T, N, S} = S
is_thread_safe(::Sech) = true
