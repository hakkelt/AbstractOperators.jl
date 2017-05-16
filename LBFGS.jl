export LBFGS, update!

immutable LBFGS{M, T<:RealOrComplex, A<:AbstractArray{T}, N} <:LinearOperator
	domainType::Type
	dim::NTuple{N,Int}
	currmem::Array{Int64,1}
	curridx::Array{Int64,1}
	s::A
	y::A
	s_m::NTuple{M,A}
	y_m::NTuple{M,A}
	ys_m::Array{T,1}
	alphas::Array{T,1}
	H::Array{T,1}
end
size(A::LBFGS) = (A.dim,A.dim)

function LBFGS{T<:RealOrComplex, A<:AbstractArray{T}}(x::A, mem::Int)

	s_m = ([similar(x) for i = 1:mem]...)
	y_m = ([similar(x) for i = 1:mem]...)

	s = similar(x)
	y = similar(x)

	ys_m = zeros(Float64,mem)
	alphas = zeros(Float64,mem)

	return LBFGS{mem,T,A,ndims(x)}(eltype(x), size(x), [0], [0], s, y, s_m, y_m, ys_m, alphas, [1.])
end

#create an array of LBFGS Op. which all share some stuff
function LBFGS{N}(x::NTuple{N,Any},mem::Int64)
	return LBFGS.(x,mem)
	
end

function update!{M, 
		 T<:RealOrComplex, 
		 A<:AbstractArray{T},N}(
			L::LBFGS{M,T,A,N}, 
			x::A, 
			x_prev::A, 
			gradx::A, 
			gradx_prev::A)

	L.s .= (-).(x, x_prev)
	L.y .= (-).(gradx, gradx_prev)
	ys = real(vecdot(L.s,L.y))

	if ys > 0
		L.curridx[1] += 1
		if L.curridx[1] > M L.curridx[1] = 1 end
		L.currmem[1] += 1
		if L.currmem[1] > M L.currmem[1] = M end

		L.s_m[L.curridx[1]] .=  L.s
		L.y_m[L.curridx[1]] .=  L.y
		L.ys_m[L.curridx[1]] = ys
		L.H[1] = ys/real(vecdot(L.y,L.y))
	end
	return L
end

function update!{M,N2,A <: NTuple{N2,Any}}(L::NTuple{N2,LBFGS{M}}, 
					 x::A, 
					 x_prev::A, 
					 gradx::A, 
					 gradx_prev::A)

	ys = 0.
	for i in eachindex(L)
		L[i].s .= (-).(x[i], x_prev[i])
		L[i].y .= (-).(gradx[i], gradx_prev[i])
		ys += real(vecdot(L[i].s, L[i].y))
	end

	if ys > 0
		L[1].curridx[1] += 1
		if L[1].curridx[1] > M L[1].curridx[1] = 1 end
		L[1].currmem[1] += 1
		if L[1].currmem[1] > M L[1].currmem[1] = M end

		L[1].H[1] = 0.
		for i in eachindex(L)
			copy!(L[i].s_m[L[1].curridx[1]], L[i].s)
			copy!(L[i].y_m[L[1].curridx[1]], L[i].y)
			L[1].H[1] += real(vecdot(L[i].y,L[i].y))
		end
		L[1].ys_m[L[1].curridx[1]] = ys
		L[1].H[1] = (L[1].H[1]/ys)^(-1)
	end
	return L
end

function A_mul_B!{M, T<:RealOrComplex, A<:AbstractArray{T},N}(d::A, L::LBFGS{M,T,A,N}, gradx::A)
	d .= (-).(gradx)
	idx = L.curridx[1]
	for i=1:L.currmem[1]
		L.alphas[idx] = real(vecdot(L.s_m[idx], d))/L.ys_m[idx]
		d .= (-).(d, (*).(L.alphas[idx], L.y_m[idx]))
		idx -= 1
		if idx == 0 idx = M end
	end
	d .= (*).(L.H[1], d)
	for i=1:L.currmem[1]
		idx += 1
		if idx > M idx = 1 end
		beta = real(vecdot(L.y_m[idx], d))/L.ys_m[idx]
		d .= (+).(d, (*).((L.alphas[idx]-beta), L.s_m[idx]))
	end
end

function A_mul_B!{M,N2,A<:NTuple{N2,Any}}(d::A, L::NTuple{N2,LBFGS{M}}, gradx::A)
	for ii in 1:N2
		d[ii] .= (-).(gradx[ii])
	end
	idx = L[1].curridx[1]
	for i=1:L[1].currmem[1]
		
		L[1].alphas[idx] = 0.
		for ii in eachindex(L)
			L[1].alphas[idx] += real(vecdot(L[ii].s_m[idx], d[ii]))
		end
		L[1].alphas[idx] /= L[1].ys_m[idx]
		
		for ii in eachindex(L)
			d[ii] .= (-).(d[ii], (*).(L[1].alphas[idx], L[ii].y_m[idx]))
		end
		
		idx -= 1
		if idx == 0 idx = M end
	end
	for ii in 1:N2
		d[ii] .= (*).(L[1].H[1], d[ii])
	end
	for i=1:L[1].currmem[1]
		idx += 1
		if idx > M idx = 1 end
		
		beta = 0.
		for ii in eachindex(L)
			beta += real(vecdot(L[ii].y_m[idx], d[ii]))
		end
		beta /= L[1].ys_m[idx]

		for ii in eachindex(L)
			d[ii] .= (+).(d[ii], (*).((L[1].alphas[idx]-beta), L[ii].s_m[idx]))
		end
	end
end

function reset(A::LBFGS)
	A.currmem[1] = 0
	A.curridx[1] = 0
end

function reset(A::Array{LBFGS,1})
	A[1].currmem[1] = 0
	A[1].curridx[1] = 0
end

fun_name(A::LBFGS)  = "LBFGS Operator"
