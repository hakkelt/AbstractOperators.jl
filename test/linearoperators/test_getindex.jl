@testitem "GetIndex: basic mul" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing GetIndex --- ")

    n, k = 5, 3
    test_op(GetIndex(zeros(Float64, n), (1:k,)), randn(n), randn(k), verb)
    n, m = 5, 4
    k = 3
    test_op(GetIndex(zeros(Float64, n, m), (1:k, :)), randn(n, m), randn(k, m), verb)

    op = GetIndex(Float64, (n,), (1:k,); array_type = Array{ComplexF32, 2})
    @test domain_storage_type(op) == Array{Float64}
    @test codomain_storage_type(op) == Array{Float64}

    op2d = GetIndex(Float64, (n, m), (:, 2))
    x1 = randn(n, m)
    y1 = test_op(op2d, x1, randn(n), verb)
    @test all(norm.(y1 .- x1[:, 2]) .<= 1.0e-12)

    @test_throws BoundsError GetIndex(Float64, (n, m), (1:k, :, :))
    @test typeof(GetIndex(Float64, (n, m), (1:n, 1:m))) <: Eye
end

@testitem "GetIndex: properties" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m, k = 5, 4, 3
    op = GetIndex(Float64, (n,), (1:k,))
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
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == estimate_opnorm(op)
end

@testitem "GetIndex: boolean mask and CartesianIndex" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    mask = falses(n, m)
    mask[1:2, 2] .= true
    op_mask = GetIndex(Float64, (n, m), mask)
    xmask = randn(n, m)
    ymask = op_mask * xmask
    @test length(ymask) == sum(mask) && all(ymask .== xmask[mask])

    fullmask = trues(n, m)
    op_full = GetIndex(Float64, (n, m), fullmask)
    xfull = randn(n, m)
    yfull = op_full * xfull
    back = zeros(size(xfull))
    mul!(back, op_full', yfull)
    @test back ≈ xfull

    mask_vec = [isodd(i) for i in 1:n]
    op_mask_vec = GetIndex(Float64, (n,), mask_vec)
    xmask_vec = randn(n)
    @test op_mask_vec * xmask_vec == xmask_vec[mask_vec]

    op_mask_vec_tuple = GetIndex(Float64, (n,), (mask_vec,))
    @test op_mask_vec_tuple * xmask_vec == xmask_vec[mask_vec]

    cart = collect(CartesianIndices((n, m))[1:4])
    op_cart = GetIndex(Float64, (n, m), cart)
    xcart = randn(n, m)
    @test op_cart * xcart == xcart[cart]
end

@testitem "GetIndex: normal op and slicing helpers" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    A = GetIndex(Float64, (n, m), (1:3, :))
    xA = randn(n, m)
    normal = AbstractOperators.get_normal_op(A)
    tmp = similar(xA)
    mul!(tmp, normal, xA)
    proj = zeros(size(xA))
    proj[1:3, :] .= xA[1:3, :]
    @test tmp == proj
    @test typeof(AbstractOperators.get_normal_op(A')) <: Eye

    @test AbstractOperators.get_slicing_expr(A) == (1:3, :)
    mask = falses(n, m)
    mask[1:2, 2] .= true
    op_mask = GetIndex(Float64, (n, m), mask)
    @test sum(AbstractOperators.get_slicing_mask(op_mask)) == sum(mask)

    base_eye = AbstractOperators.remove_slicing(A)
    @test typeof(base_eye) <: Eye
    @test size(base_eye) == (size(A, 1), size(A, 1))

    io = IOBuffer()
    show(io, A)
    @test occursin("↓", String(take!(io)))
end

@testitem "GetIndex (JLArray)" tags = [:linearoperator, :GetIndex, :gpu, :jlarray] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m, k = 5, 4, 3
    gpu_wrapper = Base.typename(typeof(jl(randn(1)))).wrapper
    # 1-D range
    test_op(GetIndex(jl(zeros(Float64, n)), (1:k,)), jl(randn(n)), jl(randn(k)), false)
    # 2-D range + colon
    test_op(GetIndex(jl(zeros(Float64, n, m)), (1:k, :)), jl(randn(n, m)), jl(randn(k, m)), false)
    # colon + scalar int  (column selection)
    test_op(GetIndex(jl(zeros(Float64, n, m)), (:, 2)), jl(randn(n, m)), jl(randn(n)), false)
    # range + range  (submatrix)
    test_op(
        GetIndex(jl(zeros(Float64, n, m)), (2:k, 2:m)),
        jl(randn(n, m)),
        jl(randn(k - 1, m - 1)),
        false,
    )
    # range + scalar int  (rows of a specific column)
    test_op(GetIndex(jl(zeros(Float64, n, m)), (1:k, 2)), jl(randn(n, m)), jl(randn(k)), false)

    idx_vec = collect(1:2:n)
    op_vec = GetIndex(jl(zeros(Float64, n)), idx_vec)
    @test Base.typename(typeof(op_vec.idx[1])).wrapper == gpu_wrapper
    @test eltype(op_vec.idx[1]) <: Integer
    test_op(op_vec, jl(randn(n)), jl(randn(length(idx_vec))), false)

    mask = falses(n)
    mask[2:2:n] .= true
    op_mask = GetIndex(jl(zeros(Float64, n)), mask)
    @test Base.typename(typeof(op_mask.idx[1])).wrapper == gpu_wrapper
    @test eltype(op_mask.idx[1]) <: Integer
    @test length(op_mask.idx[1]) == sum(mask)
    test_op(op_mask, jl(randn(n)), jl(randn(sum(mask))), false)
end

@testitem "GetIndex (CUDA)" tags = [:gpu, :cuda, :linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(0)
        n, m, k = 5, 4, 3
        # 1-D range
        test_op(GetIndex(CUDA.zeros(Float64, n), (1:k,)), CuArray(randn(n)), CuArray(randn(k)), false)
        # 2-D range + colon
        test_op(
            GetIndex(CUDA.zeros(Float64, n, m), (1:k, :)),
            CuArray(randn(n, m)),
            CuArray(randn(k, m)),
            false,
        )
        # colon + scalar int  (column selection)
        test_op(
            GetIndex(CUDA.zeros(Float64, n, m), (:, 2)), CuArray(randn(n, m)), CuArray(randn(n)), false
        )
        # range + range  (submatrix)
        test_op(
            GetIndex(CUDA.zeros(Float64, n, m), (2:k, 2:m)),
            CuArray(randn(n, m)),
            CuArray(randn(k - 1, m - 1)),
            false,
        )
        # range + scalar int  (rows of a specific column)
        test_op(
            GetIndex(CUDA.zeros(Float64, n, m), (1:k, 2)),
            CuArray(randn(n, m)),
            CuArray(randn(k)),
            false,
        )
        # vector of indices
        idx_vec = collect(1:2:n)
        op_vec = GetIndex(CUDA.zeros(Float64, n), idx_vec)
        @test op_vec.idx[1] isa CUDA.CuArray{Int, 1}
        test_op(op_vec, CuArray(randn(n)), CuArray(randn(length(idx_vec))), false)
        # vector of indices with array_type keyword
        op_vec_kw = GetIndex(Float64, (n,), idx_vec; array_type = CUDA.CuArray{Float64, 1})
        @test op_vec_kw.idx[1] isa CUDA.CuArray{Int, 1}
        # boolean mask
        mask = falses(n)
        mask[2:2:n] .= true
        op_mask = GetIndex(CUDA.zeros(Float64, n), mask)
        @test op_mask.idx[1] isa CUDA.CuArray{Int, 1}
        @test length(op_mask.idx[1]) == sum(mask)
        test_op(op_mask, CuArray(randn(n)), CuArray(randn(sum(mask))), false)
        # boolean mask with array_type keyword
        op_mask_kw = GetIndex(Float64, (n,), mask; array_type = CUDA.CuArray{Float64, 1})
        @test op_mask_kw.idx[1] isa CUDA.CuArray{Int, 1}
    end
end

@testitem "GetIndex (AMDGPU)" tags = [:gpu, :amdgpu, :linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)
        n, m, k = 5, 4, 3
        # 1-D range
        test_op(
            GetIndex(AMDGPU.zeros(Float64, n), (1:k,)),
            AMDGPU.ROCArray(randn(n)),
            AMDGPU.ROCArray(randn(k)),
            false,
        )
        # 2-D range + colon
        test_op(
            GetIndex(AMDGPU.zeros(Float64, n, m), (1:k, :)),
            AMDGPU.ROCArray(randn(n, m)),
            AMDGPU.ROCArray(randn(k, m)),
            false,
        )
        # colon + scalar int  (column selection)
        test_op(
            GetIndex(AMDGPU.zeros(Float64, n, m), (:, 2)),
            AMDGPU.ROCArray(randn(n, m)),
            AMDGPU.ROCArray(randn(n)),
            false,
        )
        # range + range  (submatrix)
        test_op(
            GetIndex(AMDGPU.zeros(Float64, n, m), (2:k, 2:m)),
            AMDGPU.ROCArray(randn(n, m)),
            AMDGPU.ROCArray(randn(k - 1, m - 1)),
            false,
        )
        # range + scalar int  (rows of a specific column)
        test_op(
            GetIndex(AMDGPU.zeros(Float64, n, m), (1:k, 2)),
            AMDGPU.ROCArray(randn(n, m)),
            AMDGPU.ROCArray(randn(k)),
            false,
        )
        # vector of indices
        idx_vec = collect(1:2:n)
        op_vec = GetIndex(AMDGPU.zeros(Float64, n), idx_vec)
        @test op_vec.idx[1] isa AMDGPU.ROCArray{Int, 1}
        test_op(op_vec, AMDGPU.ROCArray(randn(n)), AMDGPU.ROCArray(randn(length(idx_vec))), false)
        # vector of indices with array_type keyword
        op_vec_kw = GetIndex(Float64, (n,), idx_vec; array_type = AMDGPU.ROCArray{Float64, 1})
        @test op_vec_kw.idx[1] isa AMDGPU.ROCArray{Int, 1}
        # boolean mask
        mask = falses(n)
        mask[2:2:n] .= true
        op_mask = GetIndex(AMDGPU.zeros(Float64, n), mask)
        @test op_mask.idx[1] isa AMDGPU.ROCArray{Int, 1}
        @test length(op_mask.idx[1]) == sum(mask)
        test_op(op_mask, AMDGPU.ROCArray(randn(n)), AMDGPU.ROCArray(randn(sum(mask))), false)
        # boolean mask with array_type keyword
        op_mask_kw = GetIndex(Float64, (n,), mask; array_type = AMDGPU.ROCArray{Float64, 1})
        @test op_mask_kw.idx[1] isa AMDGPU.ROCArray{Int, 1}
    end
end

@testitem "GetIndex scalar Int specialized path" tags = [:linearoperator, :GetIndex] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 6, 7
    op = GetIndex(Float64, (n, m), (2, 3))
    x = randn(n, m)
    y = op * x
    @test size(op) == ((1,), (n, m))
    @test length(y) == 1
    @test y[1] == x[2, 3]
end

@testitem "GetIndex boolean mask inside tuple branch" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m, l = 5, 4, 3
    mask_first = falses(n)
    mask_first[2:4] .= true
    op = GetIndex(Float64, (n, m, l), (mask_first, :, 2))
    x = randn(n, m, l)
    y = op * x
    @test length(y) == sum(mask_first) * m
    @test y == x[mask_first, :, 2]
end

@testitem "GetIndex AbstractArray of Int indices (multi-dim)" tags = [:linearoperator, :GetIndex] setup=[TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    arr_idx = reshape([1, 2, 3, 4], 2, 2)
    op = GetIndex(Float64, (10,), (arr_idx,))
    x = randn(10)
    y = op * x
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
    expected = zeros(n)
    expected[3] = 1
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
    x2 = randn(4, 3)
    op_eye_x = GetIndex(x2, (:, :))
    @test op_eye_x isa Eye
    @test op_eye_x * x2 == x2
    xv = randn(6)
    op_eye_vec = GetIndex(xv, collect(1:length(xv)))
    @test op_eye_vec isa Eye
    @test op_eye_vec * xv == xv
    n, m = 3, 5
    bmask = trues(n, m)
    op_bmask = GetIndex(Float64, (n, m), bmask)
    if op_bmask isa GetIndex
        @test AbstractOperators.get_slicing_mask(op_bmask) === bmask
    else
        @test size(op_bmask) == ((n * m,), (n, m))
    end

    bool_vec = [i % 2 == 0 for i in 1:length(xv)]
    op_bool_vec = GetIndex(xv, bool_vec)
    @test op_bool_vec * xv == xv[bool_vec]
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
    x = randn(n, m)
    y = op * x
    @test size(y) == (length(cart),)
    @test y == x[cart]
end
