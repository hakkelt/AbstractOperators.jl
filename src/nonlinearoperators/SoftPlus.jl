export SoftPlus

"""
	SoftPlus([domain_type=Float64::Type,] dim_in::Tuple)

Creates the softplus non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\log (1 + e^{x} )
```

"""
struct SoftPlus{T, N, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function SoftPlus(domain_type::Type{T}, DomainDim::NTuple{N, Int}) where {T, N}
    return SoftPlus{T, N, Array{T}}(DomainDim)
end

SoftPlus(DomainDim::NTuple{N, Int}) where {N} = SoftPlus{Float64, N, Array{Float64}}(DomainDim)

function SoftPlus(x::AbstractArray{T}) where {T}
    S = _array_wrapper(x){T}
    return SoftPlus{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::SoftPlus, x::AbstractArray)
    check(y, L, x)
    return y .= log.(1 .+ exp.(x))
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftPlus}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= 1 ./ (1 .+ exp.(-L.x)) .* b
end

fun_name(L::SoftPlus) = "σ"

size(L::SoftPlus) = (L.dim, L.dim)

domain_type(::SoftPlus{T, N}) where {T, N} = T
codomain_type(::SoftPlus{T, N}) where {T, N} = T
domain_storage_type(::SoftPlus{T, N, S}) where {T, N, S} = S
codomain_storage_type(::SoftPlus{T, N, S}) where {T, N, S} = S
is_thread_safe(::SoftPlus) = true
