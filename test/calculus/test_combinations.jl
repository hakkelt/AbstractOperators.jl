@testitem "Combinations: HCAT and Compose" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(42)
    verb && println(" --- Testing Combinations: HCAT and Compose --- ")

    m1, m2, m3, m4 = 4, 7, 3, 2
    A1 = randn(m3, m1)
    A2 = randn(m3, m2)
    A3 = randn(m4, m3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opH = HCAT(opA1, opA2)
    opC = Compose(opA3, opH)
    x1, x2 = randn(m1), randn(m2)
    y1 = test_op(opC, ArrayPartition(x1, x2), randn(m4), verb)

    y2 = A3 * (A1 * x1 + A2 * x2)

    @test norm(y1 - y2) < 1.0e-9

    opCp = AbstractOperators.permute(opC, [2, 1])
    y1 = test_op(opCp, ArrayPartition(x2, x1), randn(m4), verb)
    @test norm(y1 - y2) < 1.0e-9

    m5 = 10
    A4 = randn(m4, m5)
    x3 = randn(m5)
    opHC = HCAT(opC, MatrixOp(A4))
    x = ArrayPartition(x1, x2, x3)
    y1 = test_op(opHC, x, randn(m4), verb)
    @test norm(y1 - (y2 + A4 * x3)) < 1.0e-9

    p = randperm(ndoms(opHC, 2))
    opHP = AbstractOperators.permute(opHC, p)
    xp = ArrayPartition(x.x[p]...)
    y1 = test_op(opHP, xp, randn(m4), verb)

    pp = randperm(ndoms(opHC, 2))
    opHPP = AbstractOperators.permute(opHC, pp)
    xpp = ArrayPartition(x.x[pp]...)
    y1 = test_op(opHPP, xpp, randn(m4), verb)
end

@testitem "Combinations: VCAT and HCAT mixtures" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(43)
    verb && println(" --- Testing Combinations: VCAT/HCAT --- ")

    # VCAT of HCATs
    m1, m2, n1 = 4, 7, 3
    A1 = randn(n1, m1)
    A2 = randn(n1, m2)
    opH1 = HCAT(MatrixOp(A1), MatrixOp(A2))
    m1, m2, n2 = 4, 7, 5
    A3 = randn(n2, m1)
    A4 = randn(n2, m2)
    opH2 = HCAT(MatrixOp(A3), MatrixOp(A4))
    opV = VCAT(opH1, opH2)
    x1, x2 = randn(m1), randn(m2)
    y1 = test_op(opV, ArrayPartition(x1, x2), ArrayPartition(randn(n1), randn(n2)), verb)
    y2 = ArrayPartition(A1 * x1 + A2 * x2, A3 * x1 + A4 * x2)
    @test norm(y1 - y2) <= 1.0e-12

    # VCAT of HCATs with complex
    m1, m2, n1 = 4, 7, 5
    A1c = randn(n1, m1) + im * randn(n1, m1)
    d1 = rand(ComplexF64, n1)
    opH1c = HCAT(MatrixOp(A1c), DiagOp(Float64, (n1,), d1))
    m1, m2, n2 = 4, 7, 5
    A3c = randn(n2, m1) + im * randn(n2, m1)
    d2 = rand(ComplexF64, n2)
    opH2c = HCAT(MatrixOp(A3c), DiagOp(Float64, (n2,), d2))
    opVc = VCAT(opH1c, opH2c)
    x1c = randn(m1) + im * randn(m1)
    x2c = randn(n2)
    y1c = test_op(
        opVc,
        ArrayPartition(x1c, x2c),
        ArrayPartition(randn(n1) + im * randn(n1), randn(n2) + im * randn(n2)),
        verb,
    )
    y2c = ArrayPartition(A1c * x1c + x2c .* d1, A3c * x1c + x2c .* d2)
    @test norm(y1c - y2c) <= 1.0e-12

    # HCAT of VCATs
    n1, n2, m1, m2 = 3, 5, 4, 7
    A = randn(m1, n1)
    B = randn(m1, n2)
    C = randn(m2, n1)
    D = randn(m2, n2)
    opV2 = HCAT(VCAT(MatrixOp(A), MatrixOp(C)), VCAT(MatrixOp(B), MatrixOp(D)))
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = test_op(opV2, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A * x1 + B * x2, C * x1 + D * x2)
    @test norm(y1 - y2) <= 1.0e-12
end

@testitem "Combinations: Sum structures" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(44)
    verb && println(" --- Testing Combinations: Sum --- ")

    # Sum of HCATs
    m, n1, n2, n3 = 4, 7, 5, 3
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    A3 = randn(m, n3)
    B1 = randn(m, n1)
    B2 = randn(m, n2)
    B3 = randn(m, n3)
    opHA = HCAT(MatrixOp(A1), MatrixOp(A2), MatrixOp(A3))
    opHB = HCAT(MatrixOp(B1), MatrixOp(B2), MatrixOp(B3))
    opS = Sum(opHA, opHB)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(opS, ArrayPartition(x1, x2, x3), randn(m), verb)
    y2 = A1 * x1 + B1 * x1 + A2 * x2 + B2 * x2 + A3 * x3 + B3 * x3
    @test norm(y1 - y2) <= 1.0e-12

    p = [3; 2; 1]
    opSp = AbstractOperators.permute(opS, p)
    y1 = test_op(opSp, ArrayPartition(((x1, x2, x3)[p])...), randn(m), verb)

    # Sum of VCATs
    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    B1 = randn(m1, n)
    B2 = randn(m2, n)
    C1 = randn(m1, n)
    C2 = randn(m2, n)
    opVA = VCAT(MatrixOp(A1), MatrixOp(A2))
    opVB = VCAT(MatrixOp(B1), MatrixOp(B2))
    opVC = VCAT(MatrixOp(C1), MatrixOp(C2))
    opS = Sum(opVA, opVB, opVC)
    x = randn(n)
    y1 = test_op(opS, x, ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A1 * x + B1 * x + C1 * x, A2 * x + B2 * x + C2 * x)
    @test norm(y1 - y2) .<= 1.0e-12
end

@testitem "Combinations: Scale structures" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(45)
    verb && println(" --- Testing Combinations: Scale --- ")

    # Scale of DCAT
    m1, n1 = 4, 7
    m2, n2 = 3, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    opD = DCAT(MatrixOp(A1), MatrixOp(A2))
    coeff = randn()
    opS = Scale(coeff, opD)
    x1 = randn(n1)
    x2 = randn(n2)
    y = test_op(opS, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    z = ArrayPartition(coeff * A1 * x1, coeff * A2 * x2)
    @test norm(y - z) <= 1.0e-12

    # Scale of VCAT
    m1, m2, n = 4, 3, 7
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opV = VCAT(MatrixOp(A1), MatrixOp(A2))
    coeff = randn()
    opS = Scale(coeff, opV)
    x = randn(n)
    y = test_op(opS, x, ArrayPartition(randn(m1), randn(m2)), verb)
    z = ArrayPartition(coeff * A1 * x, coeff * A2 * x)
    @test norm(y - z) <= 1.0e-12

    # Scale of HCAT
    m, n1, n2 = 4, 3, 7
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    opH = HCAT(MatrixOp(A1), MatrixOp(A2))
    coeff = randn()
    opS = Scale(coeff, opH)
    x1 = randn(n1)
    x2 = randn(n2)
    y = test_op(opS, ArrayPartition(x1, x2), randn(m), verb)
    z = coeff * (A1 * x1 + A2 * x2)
    @test norm(y - z) <= 1.0e-12

    # DCAT of HCATs
    m1, m2, n1, n2 = 2, 3, 4, 5
    A1 = randn(m1, n1)
    A2 = randn(m1, n2)
    B1 = randn(m2, n1)
    B2 = randn(m2, n2)
    B3 = randn(m2, n2)
    opH1 = HCAT(MatrixOp(A1), MatrixOp(A2))
    opH2 = HCAT(MatrixOp(B1), MatrixOp(B2), MatrixOp(B3))
    op = DCAT(MatrixOp(A1), opH2)
    x = ArrayPartition(randn.(size(op, 2))...)
    y0 = ArrayPartition(randn.(size(op, 1))...)
    y = test_op(op, x, y0, verb)
    op2 = DCAT(opH1, opH2)
    x = ArrayPartition(randn.(size(op2, 2))...)
    y0 = ArrayPartition(randn.(size(op2, 1))...)
    y = test_op(op2, x, y0, verb)
    p = randperm(ndoms(op2, 2))
    y2 = op2[p] * ArrayPartition(x.x[p]...)
    @test norm(y - y2) <= 1.0e-8

    # Scale of Sum and Compose
    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    opSum = Sum(MatrixOp(A1), MatrixOp(A2))
    coeff = pi
    opSS = Scale(coeff, opSum)
    x1 = randn(n)
    y1 = test_op(opSS, x1, randn(m), verb)
    y2 = coeff * (A1 * x1 + A2 * x1)
    @test norm(y1 - y2) <= 1.0e-12

    m1, m2, m3 = 4, 7, 3
    Ac1 = randn(m2, m1)
    Ac2 = randn(m3, m2)
    opC = Compose(MatrixOp(Ac2), MatrixOp(Ac1))
    opSC = Scale(coeff, opC)
    x = randn(m1)
    y1 = test_op(opSC, x, randn(m3), verb)
    y2 = coeff * (Ac2 * Ac1 * x)
    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "Combinations: Nonlinear" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(46)
    verb && println(" --- Testing Combinations: Nonlinear --- ")

    # Nonlinear HCAT of VCAT
    n, m1, m2, m3 = 4, 3, 2, 7
    x1 = randn(m1)
    x2 = randn(m2)
    x3 = randn(m3)
    x = ArrayPartition(x1, x2, x3)
    r = ArrayPartition(randn(n), randn(m1))
    A1 = randn(n, m1)
    A2 = randn(n, m2)
    A3 = randn(n, m3)
    B1 = Sigmoid(Float64, (m1,), 2)
    B2 = randn(m1, m2)
    B3 = randn(m1, m3)
    op1 = VCAT(MatrixOp(A1), B1)
    op2 = VCAT(MatrixOp(A2), MatrixOp(B2))
    op3 = VCAT(MatrixOp(A3), MatrixOp(B3))
    op = HCAT(op1, op2, op3)
    y, grad = test_NLop(op, x, r, verb)
    Y = ArrayPartition(A1 * x1 + A2 * x2 + A3 * x3, B1 * x1 + B2 * x2 + B3 * x3)
    @test norm(Y - y) < 1.0e-8

    # Nonlinear VCAT of HCAT
    m1, m2, m3, n1, n2 = 3, 4, 5, 6, 7
    x1 = randn(m1)
    x2 = randn(n1)
    x3 = randn(m3)
    x = ArrayPartition(x1, x2, x3)
    r = ArrayPartition(randn(n1), randn(n2))
    A1 = randn(n1, m1)
    B1 = Sigmoid(Float64, (n1,), 2)
    C1 = randn(n1, m3)
    A2 = randn(n2, m1)
    B2 = randn(n2, n1)
    C2 = randn(n2, m3)
    op = VCAT(HCAT(MatrixOp(A1), B1, MatrixOp(C1)), HCAT(MatrixOp(A2), MatrixOp(B2), MatrixOp(C2)))
    y, grad = test_NLop(op, x, r, verb)
    Y = ArrayPartition(A1 * x1 + B1 * x2 + C1 * x3, A2 * x1 + B2 * x2 + C2 * x3)
    @test norm(Y - y) < 1.0e-8

    # Nonlinear AffineAdd and Compose
    n = 10
    d1 = randn(n)
    d2 = randn(n)
    T = Compose(AffineAdd(Sin(n), d2), AffineAdd(Eye(n), d1))
    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (sin.(x + d1) + d2)) < 1.0e-8

    d3 = pi
    T2 = Compose(
        AffineAdd(Sin(n), d3), Compose(AffineAdd(Exp(n), d2, false), AffineAdd(Eye(n), d1))
    )
    r = randn(n)
    x = randn(size(T2, 2))
    y, grad = test_NLop(T2, x, r, verb)
    @test norm(y - (sin.(exp.(x + d1) - d2) .+ d3)) < 1.0e-8
end

@testitem "Combinations (GPU)" tags = [:gpu, :calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    # Compose of HCAT with GPU matrices
    m1, m2, m3, m4 = 4, 7, 3, 2
    A1 = jl(randn(m3, m1))
    A2 = jl(randn(m3, m2))
    A3 = jl(randn(m4, m3))
    opH = HCAT(MatrixOp(A1), MatrixOp(A2))
    opC = Compose(MatrixOp(A3), opH)
    x1, x2 = jl(randn(m1)), jl(randn(m2))
    test_op(opC, ArrayPartition(x1, x2), jl(randn(m4)), false)
end

@testitem "Combinations (CUDA)" tags = [:gpu, :cuda, :calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)

        m1, m2, m3, m4 = 4, 7, 3, 2
        A1 = CuArray(randn(m3, m1))
        A2 = CuArray(randn(m3, m2))
        A3 = CuArray(randn(m4, m3))
        opH = HCAT(MatrixOp(A1), MatrixOp(A2))
        opC = Compose(MatrixOp(A3), opH)
        x1, x2 = CuArray(randn(m1)), CuArray(randn(m2))
        test_op(opC, ArrayPartition(x1, x2), CuArray(randn(m4)), false)
    end
end

@testitem "Combinations (AMDGPU)" tags = [:gpu, :amdgpu, :calculus, :Combinations] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)

        m1, m2, m3, m4 = 4, 7, 3, 2
        A1 = AMDGPU.ROCArray(randn(m3, m1))
        A2 = AMDGPU.ROCArray(randn(m3, m2))
        A3 = AMDGPU.ROCArray(randn(m4, m3))
        opH = HCAT(MatrixOp(A1), MatrixOp(A2))
        opC = Compose(MatrixOp(A3), opH)
        x1, x2 = AMDGPU.ROCArray(randn(m1)), AMDGPU.ROCArray(randn(m2))
        test_op(opC, ArrayPartition(x1, x2), AMDGPU.ROCArray(randn(m4)), false)
    end
end
