@testitem "VCAT: basic mul" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing VCAT --- ")

    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opV = VCAT(opA1, opA2)
    x1 = randn(n)
    y1 = test_op(opV, x1, ArrayPartition(randn(m1), randn(m2)), verb)
    @test norm(y1 - ArrayPartition(A1 * x1, A2 * x1)) .<= 1.0e-12

    m1, m2, m3, n = 4, 7, 3, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    A3 = randn(m3, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opV = VCAT(opA1, opA2, opA3)
    x1 = randn(n)
    y1 = test_op(opV, x1, ArrayPartition(randn(m1), randn(m2), randn(m3)), verb)
    @test norm(y1 - ArrayPartition(A1 * x1, A2 * x1, A3 * x1)) .<= 1.0e-12

    # VCAT of VCAT (flattening)
    opVV = VCAT(opV, opA3)
    y1 = test_op(opVV, x1, ArrayPartition(randn(m1), randn(m2), randn(m3), randn(m3)), verb)
    @test norm(y1 .- ArrayPartition(A1 * x1, A2 * x1, A3 * x1, A3 * x1)) <= 1.0e-12
end

@testitem "VCAT: properties" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, m3, n = 4, 7, 3, 5
    op = VCAT(MatrixOp(randn(m1, n)), MatrixOp(randn(m2, n)), MatrixOp(randn(m3, n)))
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == false
    @test is_full_column_rank(op) == true

    d = randn(5) .+ im .* randn(5)
    op2 = VCAT(DiagOp(d), Eye(ComplexF64, 5))
    @test is_AcA_diagonal(op2) == true
    @test diag_AcA(op2) == d .* conj(d) .+ 1

    m1, m2, n = 4, 7, 5
    A1 = MatrixOp(randn(m1, n))
    A2 = MatrixOp(randn(m2, n))
    op3 = VCAT(A1, A2)
    @test domain_storage_type(op3) !== nothing
    @test codomain_storage_type(op3) !== nothing
    @test is_thread_safe(op3) == false
end

@testitem "VCAT: displacement" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    d1 = randn(m1)
    d2 = randn(m2)
    opV = VCAT(AffineAdd(MatrixOp(A1), d1), AffineAdd(MatrixOp(A2), d2))
    x1 = randn(n)
    @test norm(opV * x1 - ArrayPartition(A1 * x1 + d1, A2 * x1 + d2)) <= 1.0e-12
    @test norm(remove_displacement(opV) * x1 - ArrayPartition(A1 * x1, A2 * x1)) <= 1.0e-12

    A1b = MatrixOp(randn(m1, n))
    A2b = MatrixOp(randn(m2, n))
    op = VCAT(A1b, A2b)
    @test remove_displacement(op) == op
    opd = VCAT(AffineAdd(A1b, d1), AffineAdd(A2b, d2))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed
end

@testitem "VCAT: nonlinear operators" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 4, 3
    x = randn(m)
    r = ArrayPartition(randn(n), randn(m))
    A = randn(n, m)
    B = Sigmoid(Float64, (m,), 2)
    op = VCAT(MatrixOp(A), B)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(ArrayPartition(A * x, B * x) - y) < 1.0e-8
end

@testitem "VCAT: slicing utilities" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    g1 = GetIndex(Float64, (n,), (1:5,))
    g2 = GetIndex(Float64, (n,), (6:10,))
    Vs = VCAT(g1, g2)
    @test is_sliced(Vs) == true
    exprs = AbstractOperators.get_slicing_expr(Vs)
    @test exprs[1] == (1:5,) && exprs[2] == (6:10,)
    @test !is_sliced(AbstractOperators.remove_slicing(Vs))

    # fun_name and equality
    A1 = Eye(3)
    A2 = Eye(3)
    A3 = Eye(3)
    V2 = VCAT(A1, A2)
    V3 = VCAT(A1, A2, A3)
    name2 = AbstractOperators.fun_name(V2)
    @test occursin("[", name2) || occursin("]", name2)
    @test AbstractOperators.fun_name(V3) == "VCAT"
    # Use different operators for equality/inequality test
    Aeq1 = MatrixOp(randn(3, 4))
    Aeq2 = MatrixOp(randn(5, 4))
    @test VCAT(Aeq1, Aeq2) == VCAT(Aeq1, Aeq2)
    @test VCAT(Aeq1, Aeq2) != VCAT(Aeq2, Aeq1)
end

@testitem "VCAT (GPU)" tags = [:gpu, :calculus, :VCAT] setup = [TestUtils, GpuTestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    n = 4
    opV = VCAT(DiagOp(jl(ones(n))), DiagOp(jl(2 * ones(n))))
    test_op(opV, jl(randn(n)), ArrayPartition(jl(randn(n)), jl(randn(n))), false)

    m1, m2, n = 4, 7, 5
    A1 = jl(randn(m1, n))
    A2 = jl(randn(m2, n))
    opV2 = VCAT(MatrixOp(A1), MatrixOp(A2))
    test_op(opV2, jl(randn(n)), ArrayPartition(jl(randn(m1)), jl(randn(m2))), false)
end

@testitem "VCAT (CUDA)" tags = [:gpu, :cuda, :calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)

        n = 4
        opV = VCAT(DiagOp(CuArray(ones(n))), DiagOp(CuArray(2 * ones(n))))
        test_op(opV, CuArray(randn(n)), ArrayPartition(CuArray(randn(n)), CuArray(randn(n))), false)

        m1, m2, n = 4, 7, 5
        A1 = CuArray(randn(m1, n))
        A2 = CuArray(randn(m2, n))
        opV2 = VCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(opV2, CuArray(randn(n)), ArrayPartition(CuArray(randn(m1)), CuArray(randn(m2))), false)
    end
end

@testitem "VCAT (AMDGPU)" tags = [:gpu, :amdgpu, :calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)

        n = 4
        opV = VCAT(DiagOp(AMDGPU.ROCArray(ones(n))), DiagOp(AMDGPU.ROCArray(2 * ones(n))))
        test_op(
            opV,
            AMDGPU.ROCArray(randn(n)),
            ArrayPartition(AMDGPU.ROCArray(randn(n)), AMDGPU.ROCArray(randn(n))),
            false,
        )

        m1, m2, n = 4, 7, 5
        A1 = AMDGPU.ROCArray(randn(m1, n))
        A2 = AMDGPU.ROCArray(randn(m2, n))
        opV2 = VCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(
            opV2,
            AMDGPU.ROCArray(randn(n)),
            ArrayPartition(AMDGPU.ROCArray(randn(m1)), AMDGPU.ROCArray(randn(m2))),
            false,
        )
    end
end

@testitem "VCAT: copy_operator" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(4)

    n, m1, m2 = 8, 5, 6
    opV = VCAT(MatrixOp(randn(m1, n)), MatrixOp(randn(m2, n)))
    opV2 = copy_operator(opV)
    @test opV2 isa VCAT
    x = randn(n)
    y1 = opV * x
    y2 = opV2 * x
    @test collect(y1) ≈ collect(y2)
    # Verify independence: forward into opV2 alone
    x2 = randn(n)
    @test collect(opV2 * x2) ≈ collect(opV * x2)
end
