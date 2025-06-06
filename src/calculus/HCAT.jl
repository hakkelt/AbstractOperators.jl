export HCAT

"""
	HCAT(A::AbstractOperator...)

Shorthand constructors:

	[A1 A2 ...]
	hcat(A...)

Horizontally concatenate `AbstractOperator`s. Notice that all the operators must share the same codomain dimensions and type, e.g. `size(A1,1) == size(A2,1)` and `codomainType(A1) == codomainType(A2)`.

```jldoctest
julia> HCAT(DFT(10),DCT(Complex{Float64},20)[1:10])
[ℱ,↓*ℱc]  ℝ^10  ℂ^20 -> ℂ^10

julia> H = [Eye(10) DiagOp(2*ones(10))]
[I,╲]  ℝ^10  ℝ^10 -> ℝ^10

julia> hcat(H,DCT(10))
HCAT  ℝ^10  ℝ^10  ℝ^10 -> ℝ^10

julia> # To evaluate `HCAT` operators multiply them with a `Tuple` of `AbstractArray` of the correct dimensions and type.

julia> using RecursiveArrayTools

julia> H*ArrayPartition(ones(10),ones(10))
10-element Vector{Float64}:
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
	
```
"""
struct HCAT{
	N, # number of AbstractOperator
	L<:NTuple{N,AbstractOperator},
	P<:Tuple,
	C<:AbstractArray,
} <: AbstractOperator
	A::L     # tuple of AbstractOperators
	idxs::P  # indices
	# H = HCAT(Eye(n),HCAT(Eye(n),Eye(n))) has H.idxs = (1,2,3)
	# `AbstractOperators` are flatten
	# H = HCAT(Eye(n),Compose(MatrixOp(randn(n,n)),HCAT(Eye(n),Eye(n))))
	# has H.idxs = (1,(2,3))
	# `AbstractOperators` are stack
	buf::C   # buffer memory
	function HCAT(A::L, idxs::P, buf::C) where {N,L<:NTuple{N,AbstractOperator},P<:Tuple,C}
		if any([size(A[1], 1) != size(a, 1) for a in A])
			throw(DimensionMismatch("operators must have the same codomain dimension!"))
		end
		if any([codomainType(A[1]) != codomainType(a) for a in A])
			throw(error("operators must all share the same codomainType!"))
		end
		return new{N,L,P,C}(A, idxs, buf)
	end
end

function HCAT(A::Vararg{AbstractOperator})
	if any((<:).(typeof.(A), HCAT)) #there are HCATs in A
		AA = ()
		for a in A
			if typeof(a) <: HCAT # flatten
				AA = (AA..., a.A...)
			else                 # stack
				AA = (AA..., a)
			end
		end
		# use buffer from HCAT in A
		buf = A[findfirst((<:).(typeof.(A), HCAT))].buf
	else
		AA = A
		# generate buffer
		buf = allocate_in_codomain(AA[1])
	end

	return HCAT(AA, buf)
end

function HCAT(AA::NTuple{N,AbstractOperator}, buf::C) where {N,C}
	if N == 1
		return AA[1]
	else
		# build H.idxs
		K = 0
		idxs = []
		for i in eachindex(ndoms.(AA, 2))
			if ndoms(AA[i], 2) == 1 # flatten operator
				K += 1
				push!(idxs, K)
			else                   # stacked operator
				idxs = push!(idxs, (collect((K + 1):(K + ndoms(AA[i], 2)))...,))
				K += ndoms(AA[i], 2)
			end
		end
		return HCAT(AA, (idxs...,), buf)
	end
end

HCAT(A::AbstractOperator) = A

# Mappings
function mul!(y::C, H::HCAT{N,L,P,C}, b::DD) where {N,L,P,C,DD<:ArrayPartition}
	return mul!(y, H, b.x)
end

@generated function mul!(y::C, H::HCAT{N,L,P,C}, b::DD) where {N,L,P,C,DD<:Tuple}
	ex = :()

	if fieldtype(P, 1) <: Int
		# flatten operator
		# build mul!(y, H.A[1], b[H.idxs[1]])
		bb = :(b[H.idxs[1]])
	else
		# stacked operator
		# build mul!(y, H.A[1],ArrayPartition( b[H.idxs[1][1]], b[H.idxs[1][2]] ...  ))
		bb = [:(b[H.idxs[1][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, 1)))]
		bb = :(ArrayPartition($(bb...)))
	end
	ex = :($ex; mul!(y, H.A[1], $bb)) # write on y

	for i in 2:N
		if fieldtype(P, i) <: Int
			# flatten operator
			# build mul!(H.buf, H.A[i], b[H.idxs[i]])
			bb = :(b[H.idxs[$i]])
		else
			# stacked operator
			# build mul!(H.buf, H.A[i],( b[H.idxs[i][1]], b[H.idxs[i][2]] ...  ))
			bb = [:(b[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))]
			bb = :(ArrayPartition($(bb...)))
		end
		ex = :($ex; mul!(H.buf, H.A[$i], $bb)) # write on H.buf
		# sum H.buf with y
		ex = :($ex; y .+= H.buf)
	end
	ex = :($ex; return y)
	return ex
end

function mul!(
	y::DD, A::AdjointOperator{HCAT{N,L,P,C}}, b::C
) where {N,L,P,C,DD<:ArrayPartition}
	mul!(y.x, A, b)
	return y
end

@generated function mul!(
	y::DD, A::AdjointOperator{HCAT{N,L,P,C}}, b::C
) where {N,L,P,C,DD<:Tuple}
	ex = :(H = A.A)
	for i in 1:N
		if fieldtype(P, i) <: Int
			# flatten operator
			# build mul!(y[H.idxs[i]], H.A[i]', b)
			yy = :(y[H.idxs[$i]])
		else
			# stacked operator
			# build mul!(ArrayPartition( y[.xH.idxs[i][1]], y[H.idxs[i][2]] ...  ), H.A[i]', b)
			yy = [:(y[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))]
			yy = :(ArrayPartition($(yy...)))
		end
		ex = :($ex; mul!($yy, H.A[$i]', b))
	end
	ex = :($ex; return y)
	return ex
end

## same as mul! but skips `Zeros`
@generated function mul_skipZeros!(
	y::C, H::HCAT{N,L,P,C}, b::DD
) where {N,L,P,C,DD<:ArrayPartition}
	ex = :()

	if fieldtype(P, 1) <: Int
		# flatten operator
		# build mul!(y, H.A[1], b.x[H.idxs[1]])
		bb = :(b.x[H.idxs[1]])
	else
		# stacked operator
		# build mul!(y, H.A[1],ArrayPartition( b.x[H.idxs[1][1]], b.x[H.idxs[1][2]] ...  ))
		bb = [:(b.x[H.idxs[1][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, 1)))]
		bb = :(ArrayPartition($(bb...)))
	end
	ex = :($ex; mul!(y, H.A[1], $bb)) # write on y

	for i in 2:N
		if !(fieldtype(L, i) <: Zeros)
			if fieldtype(P, i) <: Int
				# flatten operator
				# build mul!(H.buf, H.A[i], b.x[H.idxs[i]])
				bb = :(b.x[H.idxs[$i]])
			else
				# stacked operator
				# build mul!(H.buf, H.A[i],( b.x[H.idxs[i][1]], b.x[H.idxs[i][2]] ...  ))
				bb = [
					:(b.x[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))
				]
				bb = :(ArrayPartition($(bb...)))
			end
			ex = :($ex; mul!(H.buf, H.A[$i], $bb)) # write on H.buf
			# sum H.buf with y
			ex = :($ex; y .+= H.buf)
		end
	end
	ex = :($ex; return y)
	return ex
end

@generated function mul_skipZeros!(
	y::DD, A::AdjointOperator{HCAT{N,L,P,C}}, b::C
) where {N,L,P,C,DD<:ArrayPartition}
	ex = :(H = A.A)
	for i in 1:N
		if !(fieldtype(L, i) <: Zeros)
			if fieldtype(P, i) <: Int
				# flatten operator
				# build mul!(y.x[H.idxs[i]], H.A[i]', b)
				yy = :(y.x[H.idxs[$i]])
			else
				# stacked operator
				# build mul!(ArrayPartition( y[.xH.idxs[i][1]], y.x[H.idxs[i][2]] ...  ), H.A[i]', b)
				yy = [
					:(y.x[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))
				]
				yy = :(ArrayPartition($(yy...)))
			end
			ex = :($ex; mul!($yy, H.A[$i]', b))
		end
	end
	ex = :($ex; return y)
	return ex
end

# Properties

function size(H::HCAT)
	size_in = []
	for s in size.(H.A, 2)
		eltype(s) <: Int ? push!(size_in, s) : push!(size_in, s...)
	end
	p = vcat([[idx...] for idx in H.idxs]...)
	invpermute!(size_in, p)

	return size(H.A[1], 1), (size_in...,)
end

function fun_name(L::HCAT)
	if length(L.A) == 2
		if L.idxs[1] == 2 || L.idxs[2] == 1
			return "[" * fun_name(L.A[2]) * "," * fun_name(L.A[1]) * "]"
		else
			return "[" * fun_name(L.A[1]) * "," * fun_name(L.A[2]) * "]"
		end
	else
		return "HCAT"
	end
end

function domainType(H::HCAT)
	domain = vcat([typeof(d) <: Tuple ? [d...] : d for d in domainType.(H.A)]...)
	p = vcat([[idx...] for idx in H.idxs]...)
	invpermute!(domain, p)
	return (domain...,)
end
codomainType(L::HCAT) = codomainType.(Ref(L.A[1]))
is_thread_safe(::HCAT) = false

is_linear(L::HCAT) = all(is_linear.(L.A))
is_AAc_diagonal(L::HCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::HCAT) = any(is_full_row_rank.(L.A))

is_sliced(L::HCAT) = any(is_sliced.(L.A))
function get_slicing_expr(L::HCAT)
	exprs = ()
	for i in eachindex(L.A)
		expr = get_slicing_expr(L[i])
		if expr isa Tuple && all(e -> e isa Tuple, expr)
			exprs = (exprs..., expr...)
		else
			exprs = (exprs..., expr)
		end
	end
	if length(exprs) == 1
		return exprs[1]
	else
		return exprs
	end
end
get_slicing_mask(L::HCAT) = get_slicing_mask.(L[i] for i in eachindex(L.A))
remove_slicing(L::HCAT) = HCAT(remove_slicing.(Tuple(A for A in L.A)), L.idxs, L.buf)

diag_AAc(L::HCAT) = (+).(diag_AAc.(L[i] for i in eachindex(L.A))...)

# utils
function permute(H::HCAT, p::AbstractVector{Int})
	unfolded = vcat([[idx...] for idx in H.idxs]...)
	invpermute!(unfolded, p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxs)
		new_part = (
			new_part..., z == 1 ? unfolded[cnt + 1] : (unfolded[(cnt + 1):(z + cnt)]...,)
		)
		cnt += z
	end

	return HCAT(H.A, new_part, H.buf)
end

remove_displacement(H::HCAT) = HCAT(remove_displacement.(H.A), H.idxs, H.buf)
