@testitem "AffineAdd" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing AffineAdd --- ")

    n, m = 5, 6
    A = randn(n, m)
    opA = MatrixOp(A)
    d = randn(n)
    T = AffineAdd(opA, d)

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 + d)) < 1.0e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1.0e-9
    @test displacement(T) == d
    @test norm(remove_displacement(T) * x1 - A * x1) < 1.0e-9

    # with sign
    T = AffineAdd(opA, d, false)
    @test sign(T) == -1

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 - d)) < 1.0e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1.0e-9
    @test displacement(T) == -d
    @test norm(remove_displacement(T) * x1 - A * x1) < 1.0e-9

    # with scalar
    T = AffineAdd(opA, pi)
    @test sign(T) == 1

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 .+ pi)) < 1.0e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1.0e-9
    @test displacement(T) .- pi < 1.0e-9
    @test norm(remove_displacement(T) * x1 - A * x1) < 1.0e-9

    @test_throws DimensionMismatch AffineAdd(MatrixOp(randn(2, 5)), randn(5))
    opD = DiagOp(Float64, (4,), randn(ComplexF64, 4))
    @test_throws ErrorException AffineAdd(opD, randn(4))
    AffineAdd(opD, pi)
    @test_throws ErrorException AffineAdd(Eye(4), im * pi)

    # with scalar and vector
    d = randn(n)
    T = AffineAdd(AffineAdd(opA, pi), d, false)

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 .+ pi .- d)) < 1.0e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1.0e-9
    @test norm(displacement(T) .- (pi .- d)) < 1.0e-9

    T2 = remove_displacement(T)
    @test norm(T2 * x1 - (A * x1)) < 1.0e-9

    # permute AddAffine
    n, m = 5, 6
    A = randn(n, m)
    d = randn(n)
    opH = HCAT(Eye(n), MatrixOp(A))
    x = ArrayPartition(randn(n), randn(m))
    opHT = AffineAdd(opH, d)

    @test norm(opHT * x - (x.x[1] + A * x.x[2] .+ d)) < 1.0e-12
    p = [2; 1]
    @test norm(
        AbstractOperators.permute(opHT, p) * ArrayPartition(x.x[p]...) -
            (x.x[1] + A * x.x[2] .+ d),
    ) < 1.0e-12

    n, m = 5, 6
    A = randn(n, m)
    opA = MatrixOp(A)
    d = randn(n)
    Tplus = AffineAdd(opA, d)
    Tminus = AffineAdd(opA, d, false)
    Tscalar = AffineAdd(opA, pi)
    io = IOBuffer(); show(io, Tplus); splus = String(take!(io)); @test occursin("+d", splus)
    io = IOBuffer(); show(io, Tminus); sminus = String(take!(io)); @test occursin("-d", sminus)
    io = IOBuffer(); show(io, Tscalar); sscal = String(take!(io)); @test occursin("+d", sscal) || occursin("-d", sscal)

    # Scaling of AffineAdd (array displacement)
    α = 2.0
    Tscaled = Scale(α, Tplus)
    x = randn(m)
    @test Tscaled * x ≈ α * (A * x + d)
    # scaling of negative sign variant
    Tscaled_neg = Scale(α, Tminus)
    @test Tscaled_neg * x ≈ α * (A * x - d)

    # Diagonal passthrough and diag/diag_AcA/diag_AAc
    dvec = randn(n)
    D = DiagOp(dvec)
    Tdiag = AffineAdd(D, zeros(n))
    @test is_diagonal(Tdiag) == true
    @test diag(Tdiag) == diag(D)

    # is_null / is_eye cases
    Zop = Zeros(Float64, (n,), Float64, (n,))
    Tzero = AffineAdd(Zop, zeros(n))
    @test is_null(Tzero) == true
    Eop = Eye(n)
    Teye = AffineAdd(Eop, zeros(n))
    @test is_eye(Teye) == true

    # remove_displacement idempotence (already covered for simple, add explicit check)
    rd1 = remove_displacement(Tplus)
    rd2 = remove_displacement(rd1)
    @test rd1 * x == rd2 * x

    # AffineAdd with NonLinearOperator
    n = 10
    d = randn(n)
    T = AffineAdd(Exp(n), d, false)

    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (exp.(x) - d)) < 1.0e-8

end

@testitem "AffineAdd equality operator" tags = [:calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 6
    A = MatrixOp(randn(n, m)); d1 = randn(n); d2 = randn(n)
    @test AffineAdd(A, d1) == AffineAdd(A, d1)
    @test !(AffineAdd(A, d1) == AffineAdd(A, d2))
end

@testitem "AffineAdd property delegations (invertible, rank, diagonal)" tags = [:calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 5
    E = Eye(n); TE = AffineAdd(E, randn(n)); @test is_invertible(TE) == is_invertible(E)
    D = DiagOp(randn(n)); TD = AffineAdd(D, randn(n))
    @test is_AcA_diagonal(TD) == is_AcA_diagonal(D)
    @test is_AAc_diagonal(TD) == is_AAc_diagonal(D)
    m = 6
    A = MatrixOp(randn(n, m)); TA = AffineAdd(A, randn(n))
    @test is_full_row_rank(TA) == is_full_row_rank(A)
    @test is_full_column_rank(TA) == is_full_column_rank(A)
    D2 = DiagOp(randn(n)); TD2 = AffineAdd(D2, zeros(n))
    @test diag_AcA(TD2) == diag_AcA(D2)
    @test diag_AAc(TD2) == diag_AAc(D2)
end

@testitem "AffineAdd slicing property delegation" tags = [:calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex(Float64, (10,), 2:5); TG = AffineAdd(G, randn(4))
    @test is_sliced(TG) == is_sliced(G)
    @test AbstractOperators.get_slicing_expr(TG) == AbstractOperators.get_slicing_expr(G)
    @test AbstractOperators.get_slicing_mask(TG) == AbstractOperators.get_slicing_mask(G)
    @test AbstractOperators.remove_slicing(TG) isa Eye
end

@testitem "AffineAdd normal operator" tags = [:calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 6
    G = GetIndex(Float64, (n, m), (1:3, :)); d = randn(3, m); TG = AffineAdd(G, d)
    @test AbstractOperators.has_optimized_normalop(TG) == AbstractOperators.has_optimized_normalop(G)
    N = AbstractOperators.get_normal_op(TG); x = randn(n, m)
    @test N * x ≈ TG.A' * (TG.A * x + TG.d)
end

@testitem "AffineAdd is_thread_safe delegation" tags = [:calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test is_thread_safe(AffineAdd(DiagOp(randn(5)), randn(5))) == true
    C = Compose(FiniteDiff((6,)), DiagOp(randn(6)))
    @test is_thread_safe(AffineAdd(C, randn(5))) == false
end

@testitem "AffineAdd sign function" tags = [:calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 6
    opA = MatrixOp(randn(n, m)); d = randn(n)
    @test sign(AffineAdd(opA, d, true)) == 1
    @test sign(AffineAdd(opA, d, false)) == -1
end

@testitem "AffineAdd (GPU)" tags = [:gpu, :calculus, :AffineAdd] setup=[TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)

    n, m = 5, 6
    A = randn(n, m)
    d = randn(n)
    # AffineAdd is affine (not linear), so we test mul! directly rather than test_op
    T = AffineAdd(MatrixOp(jl(A)), jl(d))
    x1 = jl(randn(m))
    y1 = T * x1
    y1_buf = similar(y1)
    mul!(y1_buf, T, x1)
    @test collect(y1) ≈ collect(y1_buf)

    # adjoint (adjoint of AffineAdd is adjoint of linear part)
    r = jl(randn(n))
    r_adj = T' * r
    r_adj2 = similar(r_adj)
    mul!(r_adj2, T', r)
    @test collect(r_adj) ≈ collect(r_adj2)
end
