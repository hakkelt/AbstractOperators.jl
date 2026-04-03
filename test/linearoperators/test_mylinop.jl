@testitem "MyLinOp basic" tags = [:linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing MyLinOp --- ")

    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    op_array_type = MyLinOp(
        Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x); array_type = Array{ComplexF32, 2}
    )
    @test domain_storage_type(op_array_type) == Array{Float64}
    @test codomain_storage_type(op_array_type) == Array{Float64}
    x1 = randn(m)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = A * x1
    @test y1 ≈ y2

    # size & types
    @test size(op) == ((n,), (m,))
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
end

@testitem "MyLinOp mul and scaling" tags = [:linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    x1 = randn(m)
    z = randn(n)
    out_adj = similar(x1)
    mul!(out_adj, op', z)
    @test out_adj ≈ A' * z
    X = randn(m, 3)
    Y = zeros(n, 3)
    for j in 1:3
        mul!(view(Y, :, j), op, view(X, :, j))
    end
    @test Y ≈ A * X
    Sop = Scale(3.0, op)
    @test Sop * x1 ≈ 3.0 * (op * x1)
    @test_throws ErrorException Scale(1 + 2im, op)
end

@testitem "MyLinOp constructors and errors" tags = [:linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    x1 = randn(m)
    @test_throws DimensionMismatch op * randn(m + 1)
    io = IOBuffer()
    show(io, op)
    s = String(take!(io))
    @test occursin("A", s)
    op2 = MyLinOp(Float64, (m,), Float64, (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    @test size(op2) == ((n,), (m,))
    @test op2 * x1 ≈ A * x1
end

@testitem "MyLinOp (JLArray)" tags = [:linearoperator, :MyLinOp, :gpu, :jlarray] setup = [TestUtils, GpuTestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 4
    A = jl(randn(n, m))
    AT = Base.typename(typeof(jl(randn(1)))).wrapper
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x); array_type = AT)
    x = jl(randn(m))
    y = jl(randn(n))
    test_op(op, x, y, false)
    @test domain_storage_type(op) <: AT
end

@testitem "MyLinOp (CUDA)" tags = [:gpu, :cuda, :linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)
        n, m = 5, 4
        A = CuArray(randn(n, m))
        AT = Base.typename(typeof(CUDA.zeros(Float64, 1))).wrapper
        op = MyLinOp(
            Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x); array_type = AT
        )
        x = CuArray(randn(m))
        y = CuArray(randn(n))
        test_op(op, x, y, false)
        @test domain_storage_type(op) <: AT
    end
end

@testitem "MyLinOp (AMDGPU)" tags = [:gpu, :amdgpu, :linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)
        n, m = 5, 4
        A = AMDGPU.ROCArray(randn(n, m))
        AT = Base.typename(typeof(AMDGPU.zeros(Float64, 1))).wrapper
        op = MyLinOp(
            Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x); array_type = AT
        )
        x = AMDGPU.ROCArray(randn(m))
        y = AMDGPU.ROCArray(randn(n))
        test_op(op, x, y, false)
        @test domain_storage_type(op) <: AT
    end
end
