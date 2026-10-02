@testitem "Transform Combinations" tags = [:fftw, :CombinationRules] begin
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: can_be_combined, combine

    n = 8  # Power of 2 for DCT

    # Test DCT combinations
    dct_op = DCT(n)
    idct_op = IDCT(n)

    @test can_be_combined(dct_op, idct_op)
    @test can_be_combined(idct_op, dct_op)

    combined_dct = combine(dct_op, idct_op)
    @test combined_dct isa Eye

    # Test DFT combinations
    dft_op = DFT(ComplexF64, n)
    idft_op = IDFT(n)

    @test can_be_combined(dft_op, idft_op)
    @test can_be_combined(idft_op, dft_op)

    combined_dft = combine(dft_op, idft_op)
    @test combined_dft isa Eye
end

@testitem "A SignAlternation outside the batch dimensions folds into the batch" tags = [
    :fftw, :CombinationRules, :batching,
] begin
    using LinearAlgebra, Random
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: can_be_combined, _slice_operator

    Random.seed!(13)
    B = BatchOp(DiagOp(randn(ComplexF64, 4, 6); threaded = false), (5,); threaded = false)
    x = randn(ComplexF64, 4, 6, 5)
    y = randn(ComplexF64, 4, 6, 5)

    # `dirs` avoids dimension 3, so the operator is the same sign pattern on every slice.
    S = SignAlternation(ComplexF64, (4, 6, 5), (1, 2); threaded = false)
    @test _slice_operator(S, (false, false, true)) isa SignAlternation
    @test can_be_combined(S, B)
    @test can_be_combined(B, S)
    @test S * B isa AbstractOperators.SimpleBatchOp
    @test B * S isa AbstractOperators.SimpleBatchOp
    @test (S * B) * x ≈ S * (B * x)
    @test (S * B)' * y ≈ B' * (S' * y)
    @test (B * S) * x ≈ B * (S * x)
    @test (B * S)' * y ≈ S' * (B' * y)

    # `dirs` reaching the batch dimension makes the sign depend on which slice it is, so the
    # operator is not one per-slice factor and must stay outside.
    Sbad = SignAlternation(ComplexF64, (4, 6, 5), (1, 3); threaded = false)
    @test _slice_operator(Sbad, (false, false, true)) === nothing
    @test !can_be_combined(Sbad, B)
    @test Sbad * B isa Compose
    @test (Sbad * B) * x ≈ Sbad * (B * x)

    # A shift is separable on the same grounds, and by the same test on its `dirs`.
    @test _slice_operator(FFTShift(ComplexF64, (4, 6, 5), (1, 2)), (false, false, true)) isa FFTShift
    @test _slice_operator(IFFTShift(ComplexF64, (4, 6, 5), (1, 2)), (false, false, true)) isa IFFTShift
    @test _slice_operator(FFTShift(ComplexF64, (4, 6, 5), (1, 3)), (false, false, true)) === nothing

    # It also folds through a batch whose slices differ from one another.
    SP = BatchOp(
        [DiagOp(randn(ComplexF64, 4, 6); threaded = false) for _ in 1:5], (:_, :_, :s);
        threaded = false,
    )
    @test can_be_combined(S, SP)
    @test S * SP isa AbstractOperators.SpreadingBatchOp
    @test (S * SP) * x == S * (SP * x)
    @test (SP * S) * x == SP * (S * x)
    @test (S * SP)' * y == SP' * (S' * y)
end

@testitem "A transform chain over a batched weighting loses a factor" tags = [
    :fftw, :CombinationRules, :batching,
] begin
    using LinearAlgebra, Random
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: opnorm_bound, powerit

    # The shape a multi-channel transform chain takes when the channel weights are batched over a
    # further dimension: weight, alternate signs, transform. The sign alternation runs over the
    # transformed dimensions only, so it belongs inside the batch and the chain is one factor
    # shorter than it was written.
    Random.seed!(21)
    nx, ny, nc, nb = 16, 16, 4, 3
    weights = randn(ComplexF32, nx, ny, nc)
    W = DiagOp(weights; threaded = false) *
        BroadCast(Eye(zeros(ComplexF32, nx, ny)), (nx, ny, nc); threaded = false)
    B = BatchOp(W, (nb,); threaded = false)
    F = DFT(ComplexF32, (nx, ny, nc, nb), (1, 2))
    S = SignAlternation(ComplexF32, (nx, ny, nc, nb), (1, 2); threaded = false)

    chain = F * S * B
    @test chain isa Compose
    @test length(chain.A) == 2
    @test !any(op -> op isa SignAlternation, chain.A)

    x = randn(ComplexF32, nx, ny, nb)
    k = randn(ComplexF32, nx, ny, nc, nb)
    @test chain * x == F * (S * (B * x))
    @test chain' * k == B' * (S' * (F' * k))

    # Folding a factor away must not cost the closed-form norm bound its certificate.
    @test opnorm_bound(chain) >= powerit(chain; maxit = 500, rel_margin = 1.0e-12)
end
