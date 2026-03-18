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

function Cos(domain_type::Type{T}, DomainDim::NTuple{N, Int}) where {T, N}
    return Cos{T, N, Array{T}}(DomainDim)
end

Cos(DomainDim::NTuple{N, Int}) where {N} = Cos{Float64, N, Array{Float64}}(DomainDim)
Cos(DomainDim::Vararg{Int}) = Cos{Float64, length(DomainDim), Array{Float64}}(DomainDim)

function Cos(x::AbstractArray{T}) where {T}
    S = _array_wrapper(x){T}
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
