export Sin

"""
	Sin([domain_type=Float64::Type,] dim_in::Tuple)

Creates a sinusoid non-linear operator with input dimensions `dim_in`:
```math
\\sin( \\mathbf{x} ).
```

"""
struct Sin{T, N, S<:AbstractArray{T}} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Sin(domain_type::Type{T}, DomainDim::NTuple{N, Int}) where {T, N}
    return Sin{T, N, Array{T}}(DomainDim)
end

Sin(DomainDim::NTuple{N, Int}) where {N} = Sin{Float64, N, Array{Float64}}(DomainDim)
Sin(DomainDim::Vararg{Int}) = Sin{Float64, length(DomainDim), Array{Float64}}(DomainDim)

function Sin(x::AbstractArray{T}) where {T}
    S = _array_wrapper(x){T}
    return Sin{T, ndims(x), S}(size(x))
end

function mul!(y::AbstractArray, L::Sin, x::AbstractArray)
    check(y, L, x)
    return y .= sin.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sin}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(cos.(L.x)) .* b
end

fun_name(L::Sin) = "sin"

size(L::Sin) = (L.dim, L.dim)

domain_type(::Sin{T, N}) where {T, N} = T
codomain_type(::Sin{T, N}) where {T, N} = T
domain_storage_type(::Sin{T, N, S}) where {T, N, S} = S
codomain_storage_type(::Sin{T, N, S}) where {T, N, S} = S
is_thread_safe(::Sin) = true
