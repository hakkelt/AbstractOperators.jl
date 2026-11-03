# Standalone: julia --project=test test/jet/test_threading_api.jl
#
# JET coverage for the threading API introduced alongside `threading_policy.jl`. Kept in
# all three modes (@test_opt, @test_call, and the package-level check in test_package.jl)
# because the policy functions take type arguments, which is exactly where unparameterized
# `::Type` keywords would silently introduce runtime dispatch.
@testitem "@test_opt threading policy" tags = [:jet, :threading] begin
    using JET, AbstractOperators
    const AO = AbstractOperators
    n = 8
    x = randn(n)

    # `default_threaded` / `_default_threaded` route the storage type through typed
    # positional parameters rather than a `storage_type::Type` keyword. If that ever
    # regresses to an unparameterized keyword, this is what catches it.
    @test_opt target_modules = (AO,) AO.default_threaded(
        typeof(FiniteDiff((n,))), Float64, (n,), Array{Float64}
    )
    @test_opt target_modules = (AO,) AO.threading_threshold(typeof(FiniteDiff((n,))))
    @test_opt target_modules = (AO,) AO._total_elements((4, 5))

    # Constructors with the `threaded` keyword.
    @test_opt target_modules = (AO,) FiniteDiff(Float64, (n,); threaded = true)
    @test_opt target_modules = (AO,) FiniteDiff(Float64, (n,); threaded = false)
    @test_opt target_modules = (AO,) Sin(Float64, (n,); threaded = true)
    @test_opt target_modules = (AO,) Pow(Float64, (n,), 2; threaded = true)
    @test_opt target_modules = (AO,) Sigmoid(Float64, (n,), 1.0; threaded = true)

    # Traits.
    @test_opt target_modules = (AO,) is_threaded(FiniteDiff((n,)))
    @test_opt target_modules = (AO,) supports_threading(FiniteDiff((n,)))
    @test_opt target_modules = (AO,) is_threaded(Compose(Eye(n), DiagOp(randn(n))))
end

@testitem "@test_call threading policy" tags = [:jet, :threading] begin
    using JET, AbstractOperators, LinearAlgebra
    const AO = AbstractOperators
    n = 1 << 13
    x = randn(n)
    y = zeros(n - 1)

    @test_call target_modules = (AO,) FiniteDiff(Float64, (n,); threaded = true)
    @test_call target_modules = (AO,) is_threaded(FiniteDiff((n,)))
    @test_call target_modules = (AO,) adapt_operator(FiniteDiff((n,)); threaded = false)
    @test_call target_modules = (AO,) copy_operator(FiniteDiff((n,)); threaded = true)

    # Both threaded and serial `mul!` paths, forward and adjoint.
    for threaded in (false, true)
        op = FiniteDiff(Float64, (n,); threaded)
        @test_call target_modules = (AO,) mul!(y, op, x)
        @test_call target_modules = (AO,) mul!(zeros(n), op', randn(n - 1))
    end

    xs = randn(n)
    for threaded in (false, true)
        s = Sin(Float64, (n,); threaded)
        @test_call target_modules = (AO,) mul!(similar(xs), s, xs)
    end
end

@testitem "JET: threading traits derived from `_children`" tags = [:jet, :threading] begin
    using JET, AbstractOperators
    const AO = AbstractOperators
    n = 8
    M = MatrixOp(randn(n, n))
    F = FiniteDiff((n,))
    # Every forwarder below relies on the default `is_threaded`/`supports_threading`, which
    # combine `_self_threaded` with the traits of `_children`. Each must still infer.
    ops = (
        F',
        AffineAdd(M, randn(n)),
        Compose(DiagOp(randn(n - 1)), F),
        Sum(M, Eye(n)),
        Reshape(M, 2, 4),
        Scale(2.0, Eye(n)),
        VCAT(M, Eye(n)),
        HCAT(M, Eye(n)),
        DCAT(M, Eye(n)),
        BroadCast(MatrixOp(randn(4, n)), (4, 3)),
        HadamardProd(Sin(Float64, (n,)), Cos(Float64, (n,))),
        Ax_mul_Bx(MatrixOp(randn(2, 2), 2), MatrixOp(randn(2, 2), 2)),
        jacobian(Compose(Sin(Float64, (n,)), M), randn(n)),
        BatchOp(M, 3, (:b, :_)),
    )
    for op in ops
        @test_opt target_modules = (AO,) is_threaded(op)
        @test_opt target_modules = (AO,) supports_threading(op)
        @test_call target_modules = (AO,) is_threaded(op)
        @test_call target_modules = (AO,) supports_threading(op)
    end
    # The SpreadingBatchOp structs store `operators::Array{opT}` without a dimension. The
    # threaded variants are threaded by their own loop, so `is_threaded` never reaches the
    # field; the serial one asks its operators and dispatches once at runtime on it.
    # `@test_call` still holds there, and `mul!` is unaffected (it only reads elements,
    # whose type is known).
    serial = BatchOp([MatrixOp(randn(n, n)) for _ in 1:3], (3,), (:s, :_))
    @test serial isa AO.SpreadingBatchOpSingleThreaded
    @test_call target_modules = (AO,) is_threaded(serial)
    @test_opt target_modules = (AO,) supports_threading(serial)
    if Threads.nthreads() > 1
        big = AO.MIN_BATCH_WORK_FOR_PARALLEL
        lbfgs() = [LBFGS(zeros(big), 1) for _ in 1:Threads.nthreads()]
        copying = BatchOp(
            lbfgs(), (Threads.nthreads(),), (:s, :_);
            threading_strategy = AO.ThreadingStrategy.COPYING,
        )
        locking = BatchOp(
            lbfgs(), (Threads.nthreads(),), (:s, :_);
            threading_strategy = AO.ThreadingStrategy.LOCKING,
        )
        @test copying isa AO.SpreadingBatchOpCopying
        @test locking isa AO.SpreadingBatchOpLocking
        for sp in (copying, locking)
            @test_opt target_modules = (AO,) is_threaded(sp)
            @test_opt target_modules = (AO,) supports_threading(sp)
            @test_call target_modules = (AO,) AO._children(sp)
        end
    end
end
