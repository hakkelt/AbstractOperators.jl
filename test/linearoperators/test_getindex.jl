@testitem "GetIndex" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing GetIndex --- ")

    function test_getindex_mul(conv, verb)
        n, k = 5, 3
        op = GetIndex(conv(zeros(Float64, n)), (1:k,))
        test_op(op, conv(randn(n)), conv(randn(k)), verb)

        n, m = 5, 4
        k = 3
        op = GetIndex(conv(zeros(Float64, n, m)), (1:k, :))
        test_op(op, conv(randn(n, m)), conv(randn(k, m)), verb)
    end

    test_getindex_mul(identity, verb)

    n, m = 5, 4
    k = 3
    op = GetIndex(Float64, (n,), (1:k,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(k), verb)
    op_array_type = GetIndex(Float64, (n,), (1:k,); array_type = Array{ComplexF32, 2})
    @test domain_storage_type(op_array_type) == Array{Float64}
    @test codomain_storage_type(op_array_type) == Array{Float64}

    @test all(norm.(y1 .- x1[1:k]) .<= 1.0e-12)

    n, m = 5, 4
    k = 3
    op = GetIndex(Float64, (n, m), (1:k, :))
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(k, m), verb)

    @test all(norm.(y1 .- x1[1:k, :]) .<= 1.0e-12)

    n, m = 5, 4
    op = GetIndex(Float64, (n, m), (:, 2))
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n), verb)

    @test all(norm.(y1 .- x1[:, 2]) .<= 1.0e-12)

    n, m, l = 5, 4, 3
    op = GetIndex(Float64, (n, m, l), (1:3, 2, :))
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, randn(3, 3), verb)

    @test all(norm.(y1 .- x1[1:3, 2, :]) .<= 1.0e-12)

    # other constructors
    GetIndex((n, m), (1:k, :))
    GetIndex(x1, (1:k, :, :))

    # SubArray constructor path
    xv = randn(5, 5)
    sv = @view xv[1:3, :]
    g_sub = GetIndex(sv, 1:2)
    @test size(g_sub, 1) == (2,)

    @test_throws BoundsError GetIndex(Float64, (n, m), (1:k, :, :))
    op = GetIndex(Float64, (n, m), (1:n, 1:m))
    @test typeof(op) <: Eye

    op = GetIndex(Float64, (n,), (1:k,))

    ##properties
    @test is_sliced(op) == true
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false

    @test diag_AAc(op) == 1

    # Boolean mask partial selection
    mask = falses(n, m)
    mask[1:2, 2] .= true
    op_mask = GetIndex(Float64, (n, m), mask)
    xmask = randn(n, m)
    ymask = op_mask * xmask
    @test length(ymask) == sum(mask)
    @test all(ymask .== xmask[mask])

    # Boolean mask selecting all => should behave like Eye reshaped
    fullmask = trues(n, m)
    op_full = GetIndex(Float64, (n, m), fullmask)
    xfull = randn(n, m)
    yfull = op_full * xfull
    @test prod(size(xfull)) == length(yfull)
    # Applying adjoint should scatter back
    back = zeros(size(xfull))
    mul!(back, op_full', yfull)
    @test back ≈ xfull

    # Vector of CartesianIndex selection
    cart = collect(CartesianIndices((n, m))[1:4])
    op_cart = GetIndex(Float64, (n, m), cart)
    xcart = randn(n, m)
    ycart = op_cart * xcart
    @test ycart == xcart[cart]

    # Normal operator A'A (should be diagonal identity restricted)
    A = GetIndex(Float64, (n, m), (1:3, :))
    xA = randn(n, m)
    yA = A * xA
    normal = AbstractOperators.get_normal_op(A)
    # normal maps full shape -> full shape
    tmp = similar(xA)
    mul!(tmp, normal, xA)
    # For GetIndex selecting subset rows, A'A should be a projector with ones on selected entries
    proj = zeros(size(xA))
    proj[1:3, :] .= xA[1:3, :]
    @test tmp == proj

    # normal of adjoint is Eye on domain
    normal_adj = AbstractOperators.get_normal_op(A')
    @test typeof(normal_adj) <: Eye

    # Slicing helpers
    @test AbstractOperators.get_slicing_expr(A) == (1:3, :)
    mask_expr = AbstractOperators.get_slicing_mask(op_mask)
    @test sum(mask_expr) == sum(mask)

    # remove_slicing returns Eye of original domain
    base_eye = AbstractOperators.remove_slicing(A)
    @test typeof(base_eye) <: Eye
    @test size(base_eye) == (size(A, 1), size(A, 1))

    # opnorm vs estimate_opnorm
    @test AbstractOperators.has_fast_opnorm(A) == true
    @test opnorm(A) == estimate_opnorm(A)

    # show output should contain arrow-like symbol for GetIndex
    io = IOBuffer(); show(io, A); strA = String(take!(io)); @test occursin("↓", strA)

    # Dimension mismatch errors in mul!
    bad_y = zeros(size(A, 1)..., 2)  # deliberately wrong extra dim
    @test_throws DimensionMismatch mul!(bad_y, A, xA)

end

@testitem "GetIndex (JLArray)" tags = [:linearoperator, :GetIndex, :gpu, :jlarray] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m, k = 5, 4, 3
    op = GetIndex(jl(zeros(Float64, n)), (1:k,))
    test_op(op, jl(randn(n)), jl(randn(k)), false)
    op2 = GetIndex(jl(zeros(Float64, n, m)), (1:k, :))
    test_op(op2, jl(randn(n, m)), jl(randn(k, m)), false)
end

@testitem "GetIndex scalar Int specialized path" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 6, 7
    op = GetIndex(Float64, (n, m), (2, 3))
    x = randn(n, m); y = op * x
    @test size(op) == ((1,), (n, m))
    @test length(y) == 1
    @test y[1] == x[2, 3]
end

@testitem "GetIndex boolean mask inside tuple branch" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m, l = 5, 4, 3
    mask_first = falses(n); mask_first[2:4] .= true
    op = GetIndex(Float64, (n, m, l), (mask_first, :, 2))
    x = randn(n, m, l); y = op * x
    @test length(y) == sum(mask_first) * m
    @test y == x[mask_first, :, 2]
end

@testitem "GetIndex AbstractArray of Int indices (multi-dim)" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    arr_idx = reshape([1, 2, 3, 4], 2, 2)
    op = GetIndex(Float64, (10,), (arr_idx,))
    x = randn(10); y = op * x
    @test size(y) == (2, 2)
    @test y == x[arr_idx]
end

@testitem "GetIndex unsupported index type error path" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test_throws ArgumentError GetIndex(Float64, (5, 4), (1:2, "bad"))
end

@testitem "NormalGetIndex non-tuple idx conversion" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 8
    N1 = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (n,), 3:3)
    @test size(N1) == ((n,), (n,))
    @test AbstractOperators.domain_type(N1) == Float64
    @test AbstractOperators.domain_storage_type(N1) == Array{Float64}
    expected = zeros(n); expected[3] = 1
    @test AbstractOperators.diag(N1) == expected
end

@testitem "NormalGetIndex vector idx conversion" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    idx_vec = [2, 4, 5]
    N2 = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (9,), idx_vec)
    d = AbstractOperators.diag(N2)
    @test d[idx_vec] == ones(length(idx_vec))
    @test sum(d) == length(idx_vec)
end

@testitem "GetIndex get_idx accessor" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test AbstractOperators.get_idx(GetIndex(Float64, (7,), (2:5,))) == (2:5,)
end

@testitem "GetIndex array-first overloads and BitArray mask" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    x2 = randn(4, 3); op_eye_x = GetIndex(x2, (:, :))
    @test op_eye_x isa Eye
    @test op_eye_x * x2 == x2
    xv = randn(6); op_eye_vec = GetIndex(xv, collect(1:length(xv)))
    @test op_eye_vec isa Eye
    @test op_eye_vec * xv == xv
    n, m = 3, 5
    bmask = trues(n, m); op_bmask = GetIndex(Float64, (n, m), bmask)
    if op_bmask isa GetIndex
        @test AbstractOperators.get_slicing_mask(op_bmask) === bmask
    else
        @test size(op_bmask) == ((n * m,), (n, m))
    end
end

@testitem "get_dim_out missing indices error" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test_throws ErrorException AbstractOperators.get_dim_out((2, 3))
end

@testitem "AdjointOperator(NormalGetIndex) identity" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    N = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (5,), 2:3)
    @test AbstractOperators.AdjointOperator(N) === N
    @test AbstractOperators.codomain_storage_type(N) == Array{Float64}
end

@testitem "GetIndex vector of CartesianIndex via tuple" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 6, 5
    cart = collect(CartesianIndices((n, m))[1:6])
    op = GetIndex(Float64, (n, m), (cart,))
    x = randn(n, m); y = op * x
    @test size(y) == (length(cart),)
    @test y == x[cart]
end
