@testitem "DCAT: basic mul" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing DCAT --- ")

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    A3 = randn(m3, n3)
    opD = DCAT(MatrixOp(A1), MatrixOp(A2), MatrixOp(A3))
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(
        opD, ArrayPartition(x1, x2, x3), ArrayPartition(randn(m1), randn(m2), randn(m3)), verb
    )
    @test norm(y1 .- ArrayPartition(A1 * x1, A2 * x2, A3 * x3)) .<= 1.0e-12

    n1, n2 = 4, 7
    opEye = Eye(ArrayPartition(randn(n1), randn(n2)))
    @test is_eye(opEye) == true && is_orthogonal(opEye) == true
end

@testitem "DCAT: properties" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    opD = DCAT(MatrixOp(randn(m1, n1)), MatrixOp(randn(m2, n2)), MatrixOp(randn(m3, n3)))
    @test is_linear(opD) == true
    @test is_null(opD) == false
    @test is_eye(opD) == false
    @test is_diagonal(opD) == false
    @test is_AcA_diagonal(opD) == false
    @test is_full_row_rank(opD) == false
    @test is_full_column_rank(opD) == false

    m1, n1, m2, n2 = 4, 7, 5, 2
    A1 = MatrixOp(randn(m1, n1))
    A2 = MatrixOp(randn(m2, n2))
    op = DCAT(A1, A2)
    @test domain_storage_type(op) !== nothing
    @test codomain_storage_type(op) !== nothing
    @test remove_displacement(op) == op
    opd = DCAT(AffineAdd(A1, randn(m1)), AffineAdd(A2, randn(m2)))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed
end

@testitem "DCAT: displacement" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    A3 = randn(m3, n3)
    d1 = randn(m1)
    d2 = randn(m2)
    d3 = randn(m3)
    opD = DCAT(
        AffineAdd(MatrixOp(A1), d1), AffineAdd(MatrixOp(A2), d2), AffineAdd(MatrixOp(A3), d3)
    )
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = opD * ArrayPartition(x1, x2, x3)
    @test norm(y1 .- ArrayPartition(A1 * x1 + d1, A2 * x2 + d2, A3 * x3 + d3)) .<= 1.0e-12
    @test norm(displacement(opD) .- ArrayPartition(d1, d2, d3)) .<= 1.0e-12
    y_rd = remove_displacement(opD) * ArrayPartition(x1, x2, x3)
    @test norm(y_rd .- ArrayPartition(A1 * x1, A2 * x2, A3 * x3)) .<= 1.0e-12
end

@testitem "DCAT: nonlinear" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 4, 3
    x = ArrayPartition(randn(n), randn(m))
    r = ArrayPartition(randn(n), randn(m))
    A = randn(n, n)
    B = Sigmoid(Float64, (m,), 2)
    op = DCAT(MatrixOp(A), B)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(ArrayPartition(A * x.x[1], B * x.x[2]) - y) < 1.0e-8
end

@testitem "DCAT (GPU)" tags = [:gpu, :calculus, :DCAT] setup = [TestUtils, GpuTestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    n1, n2 = 3, 4
    opD = DCAT(DiagOp(jl(ones(n1))), DiagOp(jl(2 * ones(n2))))
    test_op(
        opD,
        ArrayPartition(jl(randn(n1)), jl(randn(n2))),
        ArrayPartition(jl(randn(n1)), jl(randn(n2))),
        false,
    )

    m1, n1, m2, n2 = 4, 7, 5, 2
    A1 = jl(randn(m1, n1))
    A2 = jl(randn(m2, n2))
    opD2 = DCAT(MatrixOp(A1), MatrixOp(A2))
    test_op(
        opD2,
        ArrayPartition(jl(randn(n1)), jl(randn(n2))),
        ArrayPartition(jl(randn(m1)), jl(randn(m2))),
        false,
    )
end

@testitem "DCAT (CUDA)" tags = [:gpu, :cuda, :calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)

        n1, n2 = 3, 4
        opD = DCAT(DiagOp(CuArray(ones(n1))), DiagOp(CuArray(2 * ones(n2))))
        test_op(
            opD,
            ArrayPartition(CuArray(randn(n1)), CuArray(randn(n2))),
            ArrayPartition(CuArray(randn(n1)), CuArray(randn(n2))),
            false,
        )

        m1, n1, m2, n2 = 4, 7, 5, 2
        A1 = CuArray(randn(m1, n1))
        A2 = CuArray(randn(m2, n2))
        opD2 = DCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(
            opD2,
            ArrayPartition(CuArray(randn(n1)), CuArray(randn(n2))),
            ArrayPartition(CuArray(randn(m1)), CuArray(randn(m2))),
            false,
        )
    end
end

@testitem "DCAT (AMDGPU)" tags = [:gpu, :amdgpu, :calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)

        n1, n2 = 3, 4
        opD = DCAT(DiagOp(AMDGPU.ROCArray(ones(n1))), DiagOp(AMDGPU.ROCArray(2 * ones(n2))))
        test_op(
            opD,
            ArrayPartition(AMDGPU.ROCArray(randn(n1)), AMDGPU.ROCArray(randn(n2))),
            ArrayPartition(AMDGPU.ROCArray(randn(n1)), AMDGPU.ROCArray(randn(n2))),
            false,
        )

        m1, n1, m2, n2 = 4, 7, 5, 2
        A1 = AMDGPU.ROCArray(randn(m1, n1))
        A2 = AMDGPU.ROCArray(randn(m2, n2))
        opD2 = DCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(
            opD2,
            ArrayPartition(AMDGPU.ROCArray(randn(n1)), AMDGPU.ROCArray(randn(n2))),
            ArrayPartition(AMDGPU.ROCArray(randn(m1)), AMDGPU.ROCArray(randn(m2))),
            false,
        )
    end
end
