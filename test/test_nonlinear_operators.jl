@testitem "NonlinearOp: Sigmoid" tags = [:nonlinearoperator, :Sigmoid] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 4
    x = randn(n)
    r = randn(n)
    op = Sigmoid(Float64, (n,), 2)
    y, grad = test_NLop(op, x, r, verb)

    n, m, l = 4, 5, 6
    x = randn(n, m)
    r = randn(n, m)
    op = Sigmoid((n, m), 2)
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: SoftMax" tags = [:nonlinearoperator, :SoftMax] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = SoftMax(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = SoftMax(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: SoftPlus" tags = [:nonlinearoperator, :SoftPlus] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = SoftPlus(Float64, (n,))

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = SoftPlus(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Exp" tags = [:nonlinearoperator, :Exp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = Exp(n, m, l)
    op = Exp(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Sin" tags = [:nonlinearoperator, :Sin] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = Sin(n, m, l)
    op = Sin(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Cos" tags = [:nonlinearoperator, :Cos] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m, l = 4, 5, 6
    x = randn(n, m, l)
    r = randn(n, m, l)
    op = Cos(n, m, l)
    op = Cos(Float64, (n, m, l))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Atan" tags = [:nonlinearoperator, :Atan] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Atan(n)
    op = Atan(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Tanh" tags = [:nonlinearoperator, :Tanh] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Tanh(n)
    op = Tanh(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Sech" tags = [:nonlinearoperator, :Sech] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Sech(n)
    op = Sech(Float64, (n,))
    y, grad = test_NLop(op, x, r, verb)
end

@testitem "NonlinearOp: Pow" tags = [:nonlinearoperator, :Pow] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    x = randn(n)
    r = randn(n)
    op = Pow(Float64, (n,), 2)
    y, grad = test_NLop(op, x, r, verb)

    x = abs.(randn(n))
    r = abs.(randn(n))
    op = Pow(Float64, (n,), 0.5)
    y, grad = test_NLop(op, x, r, verb)
end

# ─── GPU test items for nonlinear operators ───────────────────────────────────

@testitem "NonlinearOp: Sigmoid (GPU)" tags = [:gpu, :nonlinearoperator, :Sigmoid] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 4
    x = jl(randn(n))
    # Construct from GPU array to get GPU storage type
    op = Sigmoid(x; gamma = 2.0)
    test_NLop_gpu(op, x, jl(randn(n)), false)
end

@testitem "NonlinearOp: SoftMax (GPU)" tags = [:gpu, :nonlinearoperator, :SoftMax] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 10
    x = jl(randn(n))
    op = SoftMax(x)  # construct from GPU array so buffer is GPU-typed
    test_NLop_gpu(op, x, jl(randn(n)), false)
end

@testitem "NonlinearOp: SoftPlus (GPU)" tags = [:gpu, :nonlinearoperator, :SoftPlus] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 10
    x = jl(randn(n))
    op = SoftPlus(x)
    test_NLop_gpu(op, x, jl(randn(n)), false)
end

@testitem "NonlinearOp: Exp (GPU)" tags = [:gpu, :nonlinearoperator, :Exp] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n, m = 4, 5
    x = jl(randn(n, m))
    op = Exp(x)
    test_NLop_gpu(op, x, jl(randn(n, m)), false)
end

@testitem "NonlinearOp: Sin (GPU)" tags = [:gpu, :nonlinearoperator, :Sin] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n, m = 4, 5
    x = jl(randn(n, m))
    op = Sin(x)
    test_NLop_gpu(op, x, jl(randn(n, m)), false)
end

@testitem "NonlinearOp: Cos (GPU)" tags = [:gpu, :nonlinearoperator, :Cos] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n, m = 4, 5
    x = jl(randn(n, m))
    op = Cos(x)
    test_NLop_gpu(op, x, jl(randn(n, m)), false)
end

@testitem "NonlinearOp: Atan (GPU)" tags = [:gpu, :nonlinearoperator, :Atan] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 10
    x = jl(randn(n))
    op = Atan(x)
    test_NLop_gpu(op, x, jl(randn(n)), false)
end

@testitem "NonlinearOp: Tanh (GPU)" tags = [:gpu, :nonlinearoperator, :Tanh] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 10
    x = jl(randn(n))
    op = Tanh(x)
    test_NLop_gpu(op, x, jl(randn(n)), false)
end

@testitem "NonlinearOp: Sech (GPU)" tags = [:gpu, :nonlinearoperator, :Sech] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 10
    x = jl(randn(n))
    op = Sech(x)
    test_NLop_gpu(op, x, jl(randn(n)), false)
end

@testitem "NonlinearOp: Pow (GPU)" tags = [:gpu, :nonlinearoperator, :Pow] setup = [TestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(0)
    n = 10
    x = jl(randn(n))
    op = Pow(x, 2)
    test_NLop_gpu(op, x, jl(randn(n)), false)

    x2 = jl(abs.(randn(n)))
    op2 = Pow(x2, 0.5)
    test_NLop_gpu(op2, x2, jl(abs.(randn(n))), false)
end
