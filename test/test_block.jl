@printf("\nTesting BlockArrays\n")

using AbstractOperators:BlockArrays

T = Float64
x = [2.0; 3.0]
xb = ([2.0; 3.0], Complex{T}[4.0; 5.0; 6.0], [1.0 2.0 3.0; 4.0 5.0 6.0])

x2 = randn(2)
x2b = (randn(2), randn(3)+im.*randn(3), randn(2,3))

@test blocksize(x) == (2,)
@test blocksize(xb) == ((2,),(3,),(2,3))

@test blockeltype(x) == T
@test blockeltype(xb) == (T, Complex{T},T)

@test blocklength(x) == 2
@test blocklength(xb) == 2+3+6

@test blockvecnorm(x) == vecnorm(x)
@test blockvecnorm(xb) == sqrt(vecnorm(xb[1])^2+vecnorm(xb[2])^2+vecnorm(xb[3])^2)

@test blockmaxabs(x) == 3.0
@test blockmaxabs(xb) == 6.0

@test typeof(blocksimilar(x)) == typeof(x)
@test typeof(blocksimilar(xb)) == typeof(xb)

@test blockcopy(x) == x
@test blockcopy(xb) == xb

y = blocksimilar(x)
yb = blocksimilar(xb)
blockcopy!(y,x)
blockcopy!(yb,xb)

@test y == x
@test yb == xb

y = blocksimilar(x)
yb = blocksimilar(xb)
blockset!(y,x)
blockset!(yb,xb)

@test y == x
@test yb == xb

z = blockvecdot(x,x2)
zb = blockvecdot(xb,x2b)
@test z == vecdot(x,x2)
@test zb == vecdot(xb[1],x2b[1])+vecdot(xb[2],x2b[2])+vecdot(xb[3],x2b[3])

y = blockzeros(x)
yb = blockzeros(xb)
@test y == zeros(2)
@test yb == (zeros(2),zeros(3)+im*zeros(3),zeros(2,3))

y = blockzeros(blocksize(x))
yb = blockzeros(blocksize(xb))
@test y == zeros(2)
@test yb == (zeros(2),zeros(3)+im*zeros(3),zeros(2,3))

y = blockzeros(blockeltype(x),blocksize(x))
yb = blockzeros(blockeltype(xb),blocksize(xb))
@test y == zeros(2)
@test yb == (zeros(2),zeros(3)+im*zeros(3),zeros(2,3))

blockaxpy!(y,x,2,x2)
blockaxpy!(yb,xb,2,x2b)

@test y == x .+ 2.*x2
@test yb == (xb[1] .+ 2.*x2b[1], xb[2] .+ 2.*x2b[2], xb[3] .+ 2.*x2b[3])

x = rand(Float64, 5)
y = (x, x)
@test blockiszero(x) == blockiszero(y) == false

x = rand(Complex{Float64}, 5)
y = (x, x)
@test blockiszero(x) == blockiszero(y) == false

x = zeros(Float64, 5)
y = (x, x)
@test blockiszero(x) == blockiszero(y) == true

x = zeros(Complex{Float64}, 5)
y = (x, x)
@test blockiszero(x) == blockiszero(y) == true
