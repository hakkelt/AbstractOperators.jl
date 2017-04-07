immutable Sum <: LinearOperator
	A::Vector{LinearOperator}
	mid::AbstractArray
	function Sum(A, mid)
		if any(size.(A[2:end]) .!= size(A[1]))
			throw(DimensionMismatch("cannot sum operator of different sizes"))
		end
		if size(mid) != size(A[1], 2)
			throw(error("buffer must be of the correct size"))
		end
		new(A, mid)
	end
end

# import Base: +, -
#
# size(S::Sum) = size(S.A[1])
# sign(S::Sum) = true
# -{D1,D2}(A::Sum{D1,D2}) = Sum{D1,D2}((-).(A.A), A.mid)
#
# +{D1,D2}(A::LinearOperator{D1,D2}, B::LinearOperator{D1,D2}) = Sum{D1,D2}([A, B],Array{D2}(size(A,2)))
# -{D1,D2}(A::LinearOperator{D1,D2}, B::LinearOperator{D1,D2}) = Sum{D1,D2}([A,(-B)],Array{D2}(size(A,2)))
#
# +{D1,D2}(A::LinearOperator{D1,D2}, B::Sum{D1,D2})     = Sum{D1,D2}([A,B.A...],B.mid)
# -{D1,D2}(A::LinearOperator{D1,D2}, B::Sum{D1,D2})     =
# Sum{D1,D2}([A,(-B.A)...],B.mid)
#
# +{D1,D2}(A::Sum{D1,D2}, B::LinearOperator{D1,D2} ) = B+A
# -{D1,D2}(A::Sum{D1,D2}, B::LinearOperator{D1,D2} ) = Sum{D1,D2}([A.A...,(-B)],A.mid)

fun_name(S::Sum) = "Sum of linear operators"

# transpose{D1,D2}(S::Sum{D1,D2}) = Sum{D2,D1}((S.A.')[:],Array{D1}(size(S,1)))

function A_mul_B!(y::AbstractArray,S::Sum,b::AbstractArray)
	A_mul_B!(y,S.A[1],b)
	for i = 2:length(S.A)
		A_mul_B!(S.mid,S.A[i],b)
		y .= (+).(y,S.mid)
	end
end

################################################################################
# FROM HERE ON IT IS USERS' SYNTAX
################################################################################

+(A::Affine,b::Float64) = b == 0.0 ? A : error("cannot sum $(typeof(A)) with $(typeof(b))")
-(A::Affine,b::Float64) = b == 0.0 ? A : error("cannot sum $(typeof(A)) with $(typeof(b))")

-(A::Affine) = isnull(A.b) ? Affine(A.x,-A.A,-A.At,A.b) : Affine(A.x,-A.A,-A.At,Nullable(-get(A.b)))
-(x::Variable) = -(eye(x))

+(A::Affine,b::AbstractArray)  =
(isnull(A.b) ? Affine(A.x,A.A,A.At,Nullable(b)) : Affine(A.x,A.A,A.At,Nullable(get(A.b)+b)))
+(b::AbstractArray,A::Affine)  = A+b
-(A::Affine,b::AbstractArray)  = A+(-b)
-(b::AbstractArray, A::Affine) =  (-A)+b

+(x::Variable,b::AbstractArray) = eye(x)+b
-(x::Variable,b::AbstractArray) = eye(x)-b
+(b::AbstractArray,x::Variable) = b+eye(x)
-(b::AbstractArray,x::Variable) = b+(-eye(x))

+(x::Variable,A::Affine) = eye(x)+A
-(x::Variable,A::Affine) = eye(x)-A
+(A::Affine,x::Variable) = A+eye(x)
-(A::Affine,x::Variable) = A-eye(x)

+(x::Variable,y::Variable) = eye(x)+eye(y)
-(x::Variable,y::Variable) = eye(x)-eye(y)

function +(A::Affine, B::Affine)
	if variable(A) == variable(B)
		if size(operator(A)) == size(operator(B))
			S = operator(A)+operator(B)
			x = A.x
		else
			dimension_error(operator(A),operator(B))
		end
	else
		if size(operator(A),2) == size(operator(B),2)
			#HCAT
			S,x = unsigned_sum(variable(A),operator(A),variable(B),operator(B),true)
		else
			dimension_error(operator(A),operator(B))
		end
	end
	b = sum_b(A, B, true)
	return Affine(x, S, S',b)
end

function -(A::Affine, B::Affine)
	if variable(A) == variable(B)
		if size(operator(A)) == size(operator(B))
			S = operator(A)-operator(B)
			x = A.x
		else
			dimension_error(operator(A),operator(B))
		end
	else
		if size(operator(A),2) == size(operator(B),2)
			#HCAT
			S,x = unsigned_sum(variable(A),operator(A),variable(B),operator(B),false)
		else
			dimension_error(operator(A),operator(B))
		end
	end
	b = sum_b(A, B, false)
	return Affine(x, S, S',b)
end

function sum_b(A::Affine,B::Affine, sign::Bool)
	if isnull(A.b) == true && isnull(B.b) == true
		b = Nullable{Vector{AbstractArray}}()
	elseif isnull(A.b) == false && isnull(B.b) == true
		b = Nullable(get(A.b))
	elseif isnull(A.b) == true && isnull(B.b) == false
		sign ? b = Nullable(get(B.b)) : b = Nullable(-get(B.b))
	elseif isnull(A.b) == false && isnull(B.b) == false
		sign ? b = Nullable(get(A.b)+get(B.b)) : b = Nullable(get(A.b)-get(B.b))
	end
	return b
end

function unsigned_sum{D1,D2,D3}(xa::Vector{AbstractVariable}, A::LinearOperator{D1,D3},
				xb::Vector{AbstractVariable}, B::LinearOperator{D2,D3}, sign::Bool)
	sign ? (hcat(A,B), [xa[1],xb[1]]) : (hcat(A,-B), [xa[1],xb[1]])

end

function unsigned_sum{
		      T1<:AbstractVariable,
		      T2<:AbstractVariable}(xa::Vector{T1}, A::HCAT,
			    		  xb::Vector{T2}, B::LinearOperator, sign::Bool)
	H = copy(A.A)
	x = copy(xa)

	if any(x .== xb[1])
		idx = find(x .== xb[1])[1]
		sign ? H[idx] = H[idx] + B : H[idx] =  H[idx] - B
	else
		push!(x,xb[1])
		sign ? push!(H,B) : push!(H,-B)
	end
	return HCAT(H,A.mid), x
end

unsigned_sum(xa::Vector{AbstractVariable}, A::LinearOperator,
		    xb::Vector{AbstractVariable}, B::HCAT, sign::Bool) = unsigned_sum(xb,B,xa,A,sign)

function unsigned_sum{
		      T1<:AbstractVariable,
		      T2<:AbstractVariable}(xa::Vector{T1}, A::HCAT,
			    		  xb::Vector{T2}, B::HCAT, sign::Bool)

	H, x = unsigned_sum(xa,A,[xb[1]],B.A[1],sign)
	for i = 2:length(xb)
		H, x = unsigned_sum(x,H,[xb[i]],B.A[i],sign)
	end
	return H, x
end

dimension_error(A::LinearOperator,B::LinearOperator) =
throw(DimensionMismatch("cannot sum operator of size $(size(A)) with operator of size$(size(B))"))

dimension_error() =
throw(DimensionMismatch("cannot sum operator of different sizes"))