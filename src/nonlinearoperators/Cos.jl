export Cos

"""
	Cos([domain_type=Float64::Type,] dim_in::Tuple)

Creates a cosine non-linear operator with input dimensions `dim_in`:
```math
\\cos (\\mathbf{x} ).
```

"""
struct Cos{T, N, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Cos(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}; array_type::Type = Array{T}
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Cos{T, N, S}(DomainDim)
end

function Cos(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return Cos(Float64, DomainDim; array_type)
end
Cos(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}) = Cos(Float64, DomainDim; array_type)

function Cos(x::AbstractArray{T}; array_type::Type = _array_wrapper(x)) where {T}
    S = _normalize_array_type(array_type, T)
    return Cos{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::Cos, x::AbstractArray)
    check(y, L, x)
    return y .= cos.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Cos}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= -conj.(sin.(L.x)) .* b
end

fun_name(L::Cos) = "cos"

size(L::Cos) = (L.dim, L.dim)

domain_type(::Cos{T, N}) where {T, N} = T
codomain_type(::Cos{T, N}) where {T, N} = T
domain_storage_type(::Cos{T, N, S}) where {T, N, S} = S
codomain_storage_type(::Cos{T, N, S}) where {T, N, S} = S
is_thread_safe(::Cos) = true
