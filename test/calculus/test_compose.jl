@testitem "Compose: basic mul" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Compose --- ")

    m1, m2 = 4, 7
    A1 = randn(m2, m1)
    opA1 = MatrixOp(A1); opF = FiniteDiff((m2,))
    opC = Compose(opF, opA1)
    x = randn(m1)
    y1 = test_op(opC, x, randn(m2 - 1), verb)
    @test y1 == diff(A1 * x)

    m1, m2, m3 = 4, 7, 3
    A1 = randn(m2, m1); A2 = randn(m3, m2 - 1)
    opA1 = MatrixOp(A1); opA2 = MatrixOp(A2)
    opC1 = Compose(opA2, Compose(FiniteDiff((m2,)), opA1))
    opC2 = Compose(Compose(opA2, FiniteDiff((m2,))), opA1)
    x = randn(m1)
    y1 = test_op(opC1, x, randn(m3), verb)
    y2 = test_op(opC2, x, randn(m3), verb)
    @test all(norm.(y1 .- A2 * diff(A1 * x)) .<= 1.0e-12)
    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    @test typeof(opA1 * Eye(m1)) == typeof(opA1)
    @test typeof(Eye(m2) * opA1) == typeof(opA1)
    @test typeof(Eye(m2) * Eye(m2)) == typeof(Eye(m2))
    @test_throws Exception Compose(MatrixOp(randn(5, 4)), MatrixOp(randn(3, 2)))
end

@testitem "Compose: properties" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    m1, m2, m3 = 4, 7, 3
    A1 = randn(m2, m1); A2 = randn(m3, m2 - 1)
    opA1 = MatrixOp(A1); opA2 = MatrixOp(A2); opF = FiniteDiff((m2,))
    opC = Compose(opF, opA1)
    opC1 = Compose(opA2, Compose(opF, opA1))
    @test is_sliced(opC) == false
    @test is_linear(opC1) == true
    @test is_null(opC1) == false
    @test is_eye(opC1) == false
    @test is_diagonal(opC1) == false
    @test is_orthogonal(opC1) == false
    @test is_invertible(opC1) == false

    d = randn(5)
    opC2 = DiagOp(d) * GetIndex((10,), 1:5)
    @test is_sliced(opC2) == true
    @test is_diagonal(opC2) == true
    @test diag(opC2) == d

    Z = Zeros(Float64, (m2,), Float64, (m2 - 1,))
    ZC = Compose(opA2, Z)
    @test is_null(ZC)
end

@testitem "Compose: displacement" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, m3, m4, m5 = 4, 7, 3, 2, 11
    A1 = randn(m2, m1); A2 = randn(m3, m2); A3 = randn(m4, m3); A4 = randn(m5, m4)
    d1 = randn(m2); d2 = pi; d3 = 0.0; d4 = randn(m5)
    opA1 = AffineAdd(MatrixOp(A1), d1); opA2 = AffineAdd(MatrixOp(A2), d2)
    opA3 = MatrixOp(A3); opA4 = AffineAdd(MatrixOp(A4), d4, false)
    opC = Compose(Compose(Compose(opA4, opA3), opA2), opA1)
    x = randn(m1)
    @test norm(opC * x - (A4 * (A3 * (A2 * (A1 * x + d1) .+ d2) .+ d3) - d4)) < 1.0e-9
    @test norm(displacement(opC) - (A4 * (A3 * (A2 * d1 .+ d2) .+ d3) - d4)) < 1.0e-9
    @test norm(remove_displacement(opC) * x - (A4 * (A3 * (A2 * (A1 * x))))) < 1.0e-9
end

@testitem "Compose: nonlinear" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    l, n, m = 5, 4, 3
    x = randn(m); r = randn(l)
    A = randn(l, n); C = randn(n, m)
    opA = MatrixOp(A); opB = Sigmoid(Float64, (n,), 2); opC = MatrixOp(C)
    op = Compose(opA, Compose(opB, opC))
    y, grad = test_NLop(op, x, r, verb)
    @test norm(A * (opB * (opC * x)) - y) < 1.0e-8
end

@testitem "Compose internal buffer mismatch" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = MatrixOp(randn(3, 3)); B = MatrixOp(randn(3, 3))
    @test_throws DimensionMismatch AbstractOperators.Compose((B, A), ())
end

@testitem "Adjacent adjoint optimized normal (GetIndex*GetIndex')" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex((5,), 2:4)
    L = AbstractOperators.Compose((G, G'), (randn(size(G, 1)),))
    x = randn(5)
    @test L * x == G' * (G * x)
    @test is_diagonal(L)
end

@testitem "Combine branch producing nested Compose is inlined" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    M = MatrixOp(randn(3, 3)); S = Scale(2.0, Eye(3))
    L = AbstractOperators.Compose((S, M), (zeros(3),))
    @test L * randn(3) isa AbstractVector
    x = randn(3); @test L * x ≈ 2.0 * (M * x)
    @test L isa AbstractOperator
end

@testitem "remove_slicing first op GetIndex" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex((5,), 2:4); A2 = FiniteDiff((3,)); A3 = MatrixOp(randn(2, 2))
    L = Compose(A3, Compose(A2, G)); L2 = AbstractOperators.remove_slicing(L)
    @test !is_sliced(L2)
    @test size(L2, 2) == (3,)
    v = randn(3); xfull = zeros(5); xfull[2:4] .= v
    @test L * xfull ≈ L2 * v
end

@testitem "remove_slicing first op Scale(GetIndex)" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex((5,), 2:4); S = 3.0 * G; A2 = FiniteDiff((3,))
    L = Compose(A2, S); L2 = AbstractOperators.remove_slicing(L)
    @test !is_sliced(L2)
    @test size(L2, 2) == (3,)
    v = randn(3); xfull = zeros(5); xfull[2:4] .= v
    @test L * xfull ≈ L2 * v
    A3 = MatrixOp(randn(2, 2)); L3 = Compose(A3, L); L4 = AbstractOperators.remove_slicing(L3)
    @test !is_sliced(L4)
    @test size(L4, 2) == (3,)
    @test L3 * xfull ≈ L4 * v
end

@testitem "remove_slicing error path first not sliced" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    op1 = MyLinOp(Float64, (3,), Float64, (3,), (y, x) -> (y .= x), (y, x) -> (y .= x))
    op2 = MatrixOp(randn(2, 3))
    L = AbstractOperators.Compose((op1, op2), (zeros(3),))
    @test_throws ArgumentError AbstractOperators.remove_slicing(L)
end

@testitem "diag_AAc on Compose ok and error" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    d = randn(5); sel = 1:3
    L_ok = Compose(DiagOp(d[sel]), GetIndex((length(d),), sel))
    @test is_AAc_diagonal(L_ok)
    @test AbstractOperators.diag_AAc(L_ok) == diag_AAc(DiagOp(d[sel]))
    L_bad = Compose(MatrixOp(randn(3, 3)), MatrixOp(randn(3, 3)))
    @test !is_AAc_diagonal(L_bad)
    @test_throws ErrorException AbstractOperators.diag_AAc(L_bad)
end

@testitem "get_normal_op on Compose fallback path" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    L = Compose(FiniteDiff((3,)), MatrixOp(randn(3, 3)))
    N = AbstractOperators.get_normal_op(L); x = randn(3)
    @test N isa AbstractOperator
    @test N * x ≈ L' * (L * x)
end

@testitem "Scale(coeff, L::Compose) specialized paths" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    Llin = FiniteDiff((3,)) * MatrixOp(randn(3, 3)) * FiniteDiff((4,))
    Slin = Scale(1.7, Llin); x = randn(4)
    @test Slin isa Compose
    @test Slin * x ≈ 1.7 * (Llin * x)
    Lnl = Compose(FiniteDiff((3,)), Sigmoid(Float64, (3,), 2))
    Snl = Scale(2.0, Lnl); x2 = randn(3)
    @test Snl isa Scale
    @test Snl * x2 ≈ 2.0 * (Lnl * x2)
end

@testitem "get_normal_op(Compose) else branch" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    F1 = FiniteDiff((5,)); F2 = FiniteDiff((6,)); L = Compose(F1, F2)
    N = AbstractOperators.get_normal_op(L); x = randn(6)
    @test N isa AbstractOperator
    @test N * x ≈ L' * (L * x)
end

@testitem "Buffer reuse in 4-operator chain" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    F1 = FiniteDiff((10,)); F2 = MatrixOp(rand(10, 9)); F3 = FiniteDiff((10,)); F4 = MatrixOp(rand(10, 10))
    L = F1 * F2 * F3 * F4
    @test L isa Compose
    @test length(L * randn(10)) == 9
    @test length(L' * randn(9)) == 10
    @test L.buf[1] === L.buf[3]
end

@testitem "DEBUG_COMPOSE logging branches" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    old_debug = AbstractOperators.DEBUG_COMPOSE[]
    try
        AbstractOperators.DEBUG_COMPOSE[] = true
        original_stdout = stdout
        (read_pipe, write_pipe) = redirect_stdout()
        G = GetIndex((5,), 2:4)
        _ = AbstractOperators.Compose((G, G'), (randn(3),))
        D1 = DiagOp(randn(3)); D2 = DiagOp(randn(3))
        _ = AbstractOperators.Compose((D1, D2), (randn(3),))
        redirect_stdout(original_stdout)
        close(write_pipe)
        log_str = read(read_pipe, String)
        @test occursin("Replacing", log_str) || occursin("Combining", log_str)
    finally
        AbstractOperators.DEBUG_COMPOSE[] = old_debug
    end
end

@testitem "Triple combination path" tags = [:calculus, :Compose] setup=[TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    struct TripleCombTestOp <: LinearOperator end
    LinearAlgebra.size(::TripleCombTestOp) = ((5,), (5,))
    AbstractOperators.domain_type(::TripleCombTestOp) = Float64
    AbstractOperators.codomain_type(::TripleCombTestOp) = Float64
    AbstractOperators.fun_name(::TripleCombTestOp) = "TCT"
    AbstractOperators.can_be_combined(::TripleCombTestOp, ::TripleCombTestOp, ::TripleCombTestOp) = true
    AbstractOperators.combine(::TripleCombTestOp, ::TripleCombTestOp, ::TripleCombTestOp) = DiagOp(3 .* ones(5))
    AbstractOperators.can_be_combined(::FiniteDiff, ::DiagOp, ::TripleCombTestOp) = true
    AbstractOperators.combine(L1::FiniteDiff, L2::DiagOp, ::TripleCombTestOp) = L1 * L2
    AbstractOperators.can_be_combined(::TripleCombTestOp, ::DiagOp, ::FiniteDiff) = true
    AbstractOperators.combine(::TripleCombTestOp, L1::DiagOp, L2::FiniteDiff) = L1 * L2

    T1, T2, T3 = TripleCombTestOp(), TripleCombTestOp(), TripleCombTestOp()
    D1, D2 = DiagOp(randn(5)), DiagOp(randn(4))
    F1, F2 = FiniteDiff((5,)), FiniteDiff((6,))
    old_debug = AbstractOperators.DEBUG_COMPOSE[]
    try
        AbstractOperators.DEBUG_COMPOSE[] = true
        original_stdout = stdout
        (read_pipe, write_pipe) = redirect_stdout()

        L = T1 * T2 * T3
        @test L isa DiagOp
        @test diag(L) == 3 .* ones(5)
        L = F1 * T1 * T2 * T3; x = randn(5); @test L isa Compose; @test L * x ≈ F1 * (3 .* x)
        L = T1 * T2 * T3 * F2; x = randn(6); @test L isa Compose; @test L * x ≈ 3 .* (F2 * x)
        L = F1 * D1 * T1; x = randn(5); @test L isa Compose; @test length(L.A) == 2; @test L * x ≈ F1 * (D1 * x)
        L = F1 * D1 * T1 * F2; x = randn(6); @test L isa Compose; @test length(L.A) == 3; @test L * x ≈ F1 * (D1 * (F2 * x))
        L = D2 * F1 * D1 * T1; x = randn(5); @test L isa Compose; @test length(L.A) == 3; @test L * x ≈ D2 * (F1 * (D1 * x))
        L = D2 * F1 * D1 * T1 * F2; x = randn(6); @test L isa Compose; @test length(L.A) == 4; @test L * x ≈ D2 * (F1 * (D1 * (F2 * x)))
        L = F1 * D1 * T1 * D1; x = randn(5); @test L isa Compose; @test length(L.A) == 2; @test L * x ≈ F1 * ((diag(D1) .^ 2) .* x)
        L = D1 * T1 * D1 * F2; x = randn(6); @test L isa Compose; @test length(L.A) == 2; @test L * x ≈ ((diag(D1) .^ 2) .* (F2 * x))

        redirect_stdout(original_stdout)
        close(write_pipe)
        log_str = read(read_pipe, String)
        @test occursin("Replacing", log_str) || occursin("Combining", log_str)
    finally
        AbstractOperators.DEBUG_COMPOSE[] = old_debug
    end
end

@testitem "Compose (GPU)" tags = [:gpu, :calculus, :Compose] setup=[TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    # Both operators need matching GPU storage types for Compose
    m1, m2 = 4, 7
    A1 = jl(randn(m2, m1))
    opC = Compose(FiniteDiff(jl(zeros(Float64, m2))), MatrixOp(A1))
    test_op(opC, jl(randn(m1)), jl(randn(m2 - 1)), false)

    # Chain of DiagOps
    n = 5
    opC2 = DiagOp(jl(randn(n))) * DiagOp(jl(randn(n)))
    test_op(opC2, jl(randn(n)), jl(randn(n)), false)
end
