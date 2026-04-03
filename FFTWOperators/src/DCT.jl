export DCT, IDCT

abstract type CosineTransform{N, C, T1, T2} <: LinearOperator end

"""
	DCT([domain_type=Float64::Type,] dim_in::Tuple)
	DCT(dim_in...)
	DCT(x::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Inverse Discrete Cosine Transform of `x`.

```jldoctest
julia> using FFTWOperators

julia> DCT(Complex{Float64},(10,10))
ℱc  ℂ^(10, 10) -> ℂ^(10, 10)

julia> DCT(10,10)
ℱc  ℝ^(10, 10) -> ℝ^(10, 10)

julia> A = DCT(ones(3))
ℱc  ℝ^3 -> ℝ^3

julia> A*ones(3)
3-element Vector{Float64}:
 1.7320508075688772
 0.0
 0.0
	
```
"""
struct DCT{N, C, T1 <: AbstractFFTs.Plan, T2 <: AbstractFFTs.Plan, B} <: CosineTransform{N, C, T1, T2}
    dim_in::NTuple{N, Int}
    A::T1
    At::T2
    buf::B
end

"""
	IDCT([domain_type=Float64::Type,] dim_in::Tuple)
	IDCT(dim_in...)
	IDCT(x::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional inverse Discrete Cosine Transform of `x`.

```jldoctest
julia> using FFTWOperators

julia> IDCT(Complex{Float64},(10,10))
ℱc⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> IDCT(10,10)
ℱc⁻¹  ℝ^(10, 10) -> ℝ^(10, 10)

julia> A = IDCT(ones(3))
ℱc⁻¹  ℝ^3 -> ℝ^3

julia> A*[1.;0.;0.]
3-element Vector{Float64}:
 0.5773502691896258
 0.5773502691896258
 0.5773502691896258

```
"""
struct IDCT{N, C, T1 <: AbstractFFTs.Plan, T2 <: AbstractFFTs.Plan, B} <: CosineTransform{N, C, T1, T2}
    dim_in::NTuple{N, Int}
    A::T1
    At::T2
    buf::B
end

# Constructors
#standard constructor
DCT(T::Type, dim_in::NTuple{N, Int}) where {N} = DCT(zeros(T, dim_in))

DCT(dim_in::NTuple{N, Int}) where {N} = DCT(zeros(dim_in))
DCT(dim_in::Vararg{Int64}) = DCT(dim_in)
DCT(T::Type, dim_in::Vararg{Int64}) = DCT(T, dim_in)

function DCT(x::AbstractArray{C, N}) where {N, C}
    A, At = plan_dct(x), plan_idct(x)
    buf = similar(x)
    return DCT{N, C, typeof(A), typeof(At), typeof(buf)}(size(x), A, At, buf)
end

#standard constructor
IDCT(T::Type, dim_in::NTuple{N, Int}) where {N} = IDCT(zeros(T, dim_in))

IDCT(dim_in::NTuple{N, Int}) where {N} = IDCT(zeros(dim_in))
IDCT(dim_in::Vararg{Int64}) = IDCT(dim_in)
IDCT(T::Type, dim_in::Vararg{Int64}) = IDCT(T, dim_in)

function IDCT(x::AbstractArray{C, N}) where {N, C}
    A, At = plan_idct(x), plan_dct(x)
    buf = similar(x)
    return IDCT{N, C, typeof(A), typeof(At), typeof(buf)}(size(x), A, At, buf)
end

# Mappings

function mul!(y::AbstractArray, A::DCT, b::AbstractArray)
    check(y, A, b)
    mul!(y, A.A, b)  # DCT plan (REDFT10): non-destructive to input
    return y
end

function mul!(y::AbstractArray, A::AdjointOperator{<:DCT}, b::AbstractArray)
    check(y, A, b)
    # IDCT plan (REDFT01) modifies its input in-place; use scratch buffer
    copyto!(A.A.buf, b)
    mul!(y, A.A.At, A.A.buf)
    return y
end

function mul!(y::AbstractArray, A::IDCT, b::AbstractArray)
    check(y, A, b)
    # IDCT plan (REDFT01) modifies its input in-place; use scratch buffer
    copyto!(A.buf, b)
    mul!(y, A.A, A.buf)
    return y
end

function mul!(y::AbstractArray, A::AdjointOperator{<:IDCT}, b::AbstractArray)
    check(y, A, b)
    mul!(y, A.A.At, b)  # DCT plan (REDFT10): non-destructive to input
    return y
end

# Properties

size(L::CosineTransform) = (L.dim_in, L.dim_in)

fun_name(A::DCT) = "ℱc"
fun_name(A::IDCT) = "ℱc⁻¹"

domain_type(::CosineTransform{N, C}) where {N, C} = C
codomain_type(::CosineTransform{N, C}) where {N, C} = C
domain_storage_type(::DCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
domain_storage_type(::IDCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
codomain_storage_type(::DCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
codomain_storage_type(::IDCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
is_thread_safe(::CosineTransform) = false

is_AcA_diagonal(L::CosineTransform) = true
is_AAc_diagonal(L::CosineTransform) = true
is_orthogonal(L::CosineTransform) = true
is_invertible(L::CosineTransform) = true
is_full_row_rank(L::CosineTransform) = true
is_full_column_rank(L::CosineTransform) = true

diag_AcA(L::CosineTransform) = 1.0
diag_AAc(L::CosineTransform) = 1.0

has_optimized_normalop(L::CosineTransform) = true
get_normal_op(L::CosineTransform) = Eye(allocate_in_domain(L))

has_fast_opnorm(::CosineTransform) = true
LinearAlgebra.opnorm(L::CosineTransform) = one(real(domain_type(L)))
