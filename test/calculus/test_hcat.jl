@testitem "HCAT: basic mul" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing HCAT --- ")

    m, n1, n2 = 4, 7, 5
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opH = HCAT(opA1, opA2)
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = test_op(opH, ArrayPartition(x1, x2), randn(m), verb)
    y2 = A1 * x1 + A2 * x2
    @test norm(y1 - y2) <= 1.0e-12

    # permutation
    p = [2; 1]
    opHp = opH[p]
    y1 = test_op(opHp, ArrayPartition(x2, x1), randn(m), verb)
    @test norm(y1 - y2) <= 1.0e-12

    m, n1, n2, n3 = 4, 7, 5, 6
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    A3 = randn(m, n3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opH = HCAT(opA1, opA2, opA3)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(opH, ArrayPartition(x1, x2, x3), randn(m), verb)
    @test norm(y1 - (A1 * x1 + A2 * x2 + A3 * x3)) <= 1.0e-12

    # HCAT of HCAT (flattening)
    opHH = HCAT(opH, opA2, opA3)
    y1 = test_op(opHH, ArrayPartition(x1, x2, x3, x2, x3), randn(m), verb)
    @test norm(y1 - (A1 * x1 + A2 * x2 + A3 * x3 + A2 * x2 + A3 * x3)) <= 1.0e-12
end

@testitem "HCAT: properties" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    m, n1, n2, n3 = 4, 7, 5, 6
    opA1 = MatrixOp(randn(m, n1))
    opA2 = MatrixOp(randn(m, n2))
    opA3 = MatrixOp(randn(m, n3))
    op = HCAT(opA1, opA2, opA3)
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false

    d1 = randn(n1) .+ im .* randn(n1)
    d2 = randn(n1) .+ im .* randn(n1)
    op2 = HCAT(DiagOp(d1), DiagOp(d2))
    @test is_AAc_diagonal(op2) == true
    @test diag_AAc(op2) == d1 .* conj(d1) .+ d2 .* conj(d2)
    y1 = randn(n1) .+ im .* randn(n1)
    @test norm(op2 * (op2' * y1) .- diag_AAc(op2) .* y1) < 1.0e-12

    # storage type and thread safety
    A1 = MatrixOp(randn(m, n1))
    A2 = MatrixOp(randn(m, n2))
    op3 = HCAT(A1, A2)
    @test domain_storage_type(op3) !== nothing
    @test codomain_storage_type(op3) !== nothing
    @test is_thread_safe(op3) == false
end

@testitem "HCAT: displacement" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n1, n2 = 4, 7, 5
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    d1 = randn(m)
    d2 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opH = HCAT(opA1, opA2)
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = opH * ArrayPartition(x1, x2)
    @test norm(y1 - (A1 * x1 + d1 + A2 * x2 + d2)) <= 1.0e-12
    y2 = remove_displacement(opH) * ArrayPartition(x1, x2)
    @test norm(y2 - (A1 * x1 + A2 * x2)) <= 1.0e-12

    # remove_displacement idempotence
    A1b = MatrixOp(randn(m, n1))
    A2b = MatrixOp(randn(m, n2))
    op = HCAT(A1b, A2b)
    @test remove_displacement(op) == op
    opd = HCAT(AffineAdd(A1b, d1), AffineAdd(A2b, d2))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed
end

@testitem "HCAT: slicing and permute utilities" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 8
    op1 = GetIndex(Float64, (n,), (1:4,))
    op2 = GetIndex(Float64, (n,), (5:8,))
    Hs = HCAT(op1, op2)
    @test is_sliced(Hs) == true
    exprs = AbstractOperators.get_slicing_expr(Hs)
    @test length(exprs) == 2
    @test exprs[1] == (1:4,) && exprs[2] == (5:8,)
    masks = AbstractOperators.get_slicing_mask(Hs)
    @test length(masks) == 2
    @test sum(masks[1]) == 4 && sum(masks[2]) == 4
    @test !is_sliced(AbstractOperators.remove_slicing(Hs))

    d1, d2 = randn(5), randn(5)
    D1 = DiagOp(d1) * GetIndex((10,), 1:5)
    D2 = DiagOp(d2) * GetIndex((10,), 6:10)
    Hs_comp = HCAT(D1, D2)
    @test is_sliced(Hs_comp)
    @test AbstractOperators.get_slicing_expr(Hs_comp) == ((1:5,), (6:10,))
    @test !is_sliced(AbstractOperators.remove_slicing(Hs_comp))

    m, n1, n2 = 4, 3, 2
    Aeq = MatrixOp(randn(m, n1))
    Beq = MatrixOp(randn(m, n2))
    H1a = HCAT(Aeq, Beq)
    p2 = collect(Iterators.reverse(1:ndoms(H1a, 2)))
    Hp = AbstractOperators.permute(H1a, p2)
    @test typeof(Hp) <: HCAT
    xA = randn(size(Aeq, 2))
    xB = randn(size(Beq, 2))
    y_orig = H1a * ArrayPartition(xA, xB)
    xin = p2 == [2, 1] ? ArrayPartition(xB, xA) : ArrayPartition(xA, xB)
    @test y_orig ≈ Hp * xin
end

@testitem "HCAT: nonlinear operators" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 4, 3
    x = ArrayPartition(randn(n), randn(m))
    r = randn(m)
    A = randn(m, n)
    B = Sigmoid(Float64, (m,), 2)
    op = HCAT(MatrixOp(A), B)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(A * x.x[1] + B * x.x[2] - y) < 1.0e-8

    n, m = 5, 3
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m)
    A_sin = Sin(Float64, (m,))
    M = randn(m, n)
    op2 = HCAT(A_sin, MatrixOp(M))
    y2, grad2 = test_NLop(op2, x, r, verb)
    @test norm(A_sin * x.x[1] + M * x.x[2] - y2) < 1.0e-8
end

@testitem "HCAT constructor errors" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    A1 = MatrixOp(randn(4, 3))
    A2 = MatrixOp(randn(5, 2))
    @test_throws DimensionMismatch HCAT(A1, A2)

    A1 = MatrixOp(randn(4, 3))
    A2 = MatrixOp(randn(ComplexF64, 4, 2))
    @test_throws Exception HCAT(A1, A2)
end

@testitem "HCAT flattening" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n1, n2, n3 = 4, 3, 2, 5
    A1 = MatrixOp(randn(m, n1))
    A2 = MatrixOp(randn(m, n2))
    A3 = MatrixOp(randn(m, n3))
    H1 = HCAT(A1, A2)
    H2 = HCAT(H1, A3)

    x1, x2, x3 = randn(n1), randn(n2), randn(n3)
    y = H2 * ArrayPartition(x1, x2, x3)
    y_expected = A1 * x1 + A2 * x2 + A3 * x3
    @test norm(y - y_expected) < 1.0e-12

    y_test = randn(m)
    x_adj = H2' * y_test
    @test length(x_adj.x) == 3

    H3 = HCAT(A1, A2)
    H4 = HCAT(A2, A3)
    H5 = HCAT(H3, H4)
    x_full = ArrayPartition(x1, x2, x2, x3)
    y2 = H5 * x_full
    y2_expected = A1 * x1 + A2 * x2 + A2 * x2 + A3 * x3
    @test norm(y2 - y2_expected) < 1.0e-12
end

@testitem "HCAT single operator" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    A = MatrixOp(randn(4, 3))
    H_single = HCAT(A)
    @test H_single === A
end

@testitem "HCAT (GPU)" tags = [:gpu, :calculus, :HCAT] setup=[TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    n = 4
    opH = HCAT(DiagOp(jl(ones(n))), DiagOp(jl(2 * ones(n))))
    test_op(opH, ArrayPartition(jl(randn(n)), jl(randn(n))), jl(randn(n)), false)

    m, n1, n2 = 4, 7, 5
    A1 = jl(randn(m, n1))
    A2 = jl(randn(m, n2))
    opH2 = HCAT(MatrixOp(A1), MatrixOp(A2))
    test_op(opH2, ArrayPartition(jl(randn(n1)), jl(randn(n2))), jl(randn(m)), false)
end

@testitem "HCAT (CUDA)" tags = [:gpu, :cuda, :calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)

        n = 4
        opH = HCAT(DiagOp(CuArray(ones(n))), DiagOp(CuArray(2 * ones(n))))
        test_op(opH, ArrayPartition(CuArray(randn(n)), CuArray(randn(n))), CuArray(randn(n)), false)

        m, n1, n2 = 4, 7, 5
        A1 = CuArray(randn(m, n1))
        A2 = CuArray(randn(m, n2))
        opH2 = HCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(opH2, ArrayPartition(CuArray(randn(n1)), CuArray(randn(n2))), CuArray(randn(m)), false)
    end
end

@testitem "HCAT (AMDGPU)" tags = [:gpu, :amdgpu, :calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)

        n = 4
        opH = HCAT(DiagOp(AMDGPU.ROCArray(ones(n))), DiagOp(AMDGPU.ROCArray(2 * ones(n))))
        test_op(
            opH,
            ArrayPartition(AMDGPU.ROCArray(randn(n)), AMDGPU.ROCArray(randn(n))),
            AMDGPU.ROCArray(randn(n)),
            false,
        )

        m, n1, n2 = 4, 7, 5
        A1 = AMDGPU.ROCArray(randn(m, n1))
        A2 = AMDGPU.ROCArray(randn(m, n2))
        opH2 = HCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(
            opH2,
            ArrayPartition(AMDGPU.ROCArray(randn(n1)), AMDGPU.ROCArray(randn(n2))),
            AMDGPU.ROCArray(randn(m)),
            false,
        )
    end
end
