export Sigmoid

"""
	Sigmoid([domain_type=Float64::Type,] dim_in::Tuple, γ = 1.)

Creates the sigmoid non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{1}{1+e^{-\\gamma \\mathbf{x} } }
```

"""
struct Sigmoid{T, N, G <: Real, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
    gamma::G
end

function Sigmoid(domain_type::Type{T}, DomainDim::NTuple{N, Int}, gamma::G = 1.0) where {T, N, G <: Real}
    return Sigmoid{T, N, G, Array{T}}(DomainDim, gamma)
end

function Sigmoid(DomainDim::NTuple{N, Int}, gamma::G = 1.0) where {N, G}
    return Sigmoid{Float64, N, G, Array{Float64}}(DomainDim, gamma)
end

function Sigmoid(x::AbstractArray{T}; gamma::G = 1.0) where {T, G <: Real}
    S = _array_wrapper(x){T}
    return Sigmoid{T, ndims(x), G, S}(size(x), gamma)
end

function mul!(y::AbstractArray, L::Sigmoid, x::AbstractArray)
    check(y, L, x)
    return y .= (1 .+ exp.(-L.gamma .* x)) .^ (-1)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sigmoid}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    y .= exp.(-L.A.gamma .* L.x)
    y ./= (1 .+ y) .^ 2
    y .= conj.(L.A.gamma .* y)
    return y .*= b
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domain_type(::Sigmoid{T, N, D}) where {T, N, D} = T
codomain_type(::Sigmoid{T, N, D}) where {T, N, D} = T
domain_storage_type(::Sigmoid{T, N, D, S}) where {T, N, D, S} = S
codomain_storage_type(::Sigmoid{T, N, D, S}) where {T, N, D, S} = S
is_thread_safe(::Sigmoid) = true
