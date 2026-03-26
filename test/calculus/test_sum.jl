@testitem "Sum: basic mul" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Sum --- ")

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    A3 = randn(m, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opS = Sum(opA1, opA2, opA3)
    x1 = randn(n)
    y1 = test_op(opS, x1, randn(m), verb)
    @test norm(y1 - (A1 * x1 + A2 * x1 + A3 * x1)) <= 1.0e-12

    @test_throws Exception Sum(opA1, MatrixOp(randn(m, m)))
    @test is_full_row_rank(opS) == true
    @test is_full_column_rank(opS) == false
end

@testitem "Sum: displacement" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    A3 = randn(m, n)
    d1 = randn(m)
    d2 = pi
    d3 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opA3 = AffineAdd(MatrixOp(A3), d3)
    opS = Sum(opA1, opA2, opA3)
    x1 = randn(n)
    @test norm(opS * x1 - (A1 * x1 + A2 * x1 + A3 * x1 + d1 .+ d2 + d3)) <= 1.0e-12
    @test norm(displacement(opS) - (d1 .+ d2 + d3)) <= 1.0e-12
    @test norm(remove_displacement(opS) * x1 - (A1 * x1 + A2 * x1 + A3 * x1)) <= 1.0e-12
end

@testitem "Sum: properties" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 5, 7
    Aeq = MatrixOp(randn(m, n))
    Beq = MatrixOp(randn(m, n))
    S1 = Sum(Aeq, Beq)
    S2 = Sum(Aeq, Beq)
    S3 = Sum(Beq, Aeq)
    @test S1 == S2 && S1 != S3
    @test Sum(Aeq) === Aeq

    z = Zeros(Float64, (n,), Float64, (m,))
    @test Sum(Aeq, z) === Aeq

    d = randn(10)
    op = Sum(Scale(-3.1, Eye(10)), DiagOp(d))
    @test is_diagonal(op) == true
    @test norm(diag(op) - (d .- 3.1)) < 1.0e-12

    @test domain_storage_type(S1) !== nothing
    @test codomain_storage_type(S1) !== nothing
    @test is_thread_safe(S1) == false
end

@testitem "Sum: nonlinear" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m = 5
    x = randn(m)
    r = randn(m)
    A = randn(m, m)
    opB = Sigmoid(Float64, (m,), 2)
    op = Sum(MatrixOp(A), opB)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(A * x + opB * x - y) < 1.0e-8
end

@testitem "Sum (GPU)" tags = [:gpu, :calculus, :Sum] setup=[TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    n = 5
    opS = Sum(DiagOp(jl(ones(n))), DiagOp(jl(2 * ones(n))))
    test_op(opS, jl(randn(n)), jl(randn(n)), false)

    m, n2 = 5, 7
    A1 = jl(randn(m, n2))
    A2 = jl(randn(m, n2))
    opS2 = Sum(MatrixOp(A1), MatrixOp(A2))
    test_op(opS2, jl(randn(n2)), jl(randn(m)), false)
end

@testitem "Sum (CUDA)" tags = [:gpu, :cuda, :calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)

        n = 5
        opS = Sum(DiagOp(CuArray(ones(n))), DiagOp(CuArray(2 * ones(n))))
        test_op(opS, CuArray(randn(n)), CuArray(randn(n)), false)

        m, n2 = 5, 7
        A1 = CuArray(randn(m, n2))
        A2 = CuArray(randn(m, n2))
        opS2 = Sum(MatrixOp(A1), MatrixOp(A2))
        test_op(opS2, CuArray(randn(n2)), CuArray(randn(m)), false)
    end
end

@testitem "Sum (AMDGPU)" tags = [:gpu, :amdgpu, :calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)

        n = 5
        opS = Sum(DiagOp(AMDGPU.ROCArray(ones(n))), DiagOp(AMDGPU.ROCArray(2 * ones(n))))
        test_op(opS, AMDGPU.ROCArray(randn(n)), AMDGPU.ROCArray(randn(n)), false)

        m, n2 = 5, 7
        A1 = AMDGPU.ROCArray(randn(m, n2))
        A2 = AMDGPU.ROCArray(randn(m, n2))
        opS2 = Sum(MatrixOp(A1), MatrixOp(A2))
        test_op(opS2, AMDGPU.ROCArray(randn(n2)), AMDGPU.ROCArray(randn(m)), false)
    end
end
