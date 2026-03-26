@testitem "CpuOperatorWrapper: CPU mul! forward and adjoint" tags = [
    :linearoperator, :CpuOperatorWrapper,
] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(42)

    n = 8
    cpu_op = FiniteDiff(Float64, (n,), 1)           # n-1 output
    wrapper = CpuOperatorWrapper(cpu_op)

    x = randn(n)
    y = zeros(n - 1)
    mul!(y, wrapper, x)
    y_ref = zeros(n - 1)
    mul!(y_ref, cpu_op, x)
    @test y ≈ y_ref

    b = randn(n - 1)
    z = zeros(n)
    mul!(z, wrapper', b)
    z_ref = zeros(n)
    mul!(z_ref, cpu_op', b)
    @test z ≈ z_ref

    @test size(wrapper) == size(cpu_op)
    @test domain_type(wrapper) == domain_type(cpu_op)
    @test codomain_type(wrapper) == codomain_type(cpu_op)
    @test is_linear(wrapper) == is_linear(cpu_op)

    io = IOBuffer()
    show(io, wrapper)
    @test occursin("CPU[", String(take!(io)))
end

@testitem "CpuOperatorWrapper: CPU mul! via test_op" tags = [:linearoperator, :CpuOperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    Random.seed!(42)

    n = 6
    op = MatrixOp(randn(n, n))
    wrapper = CpuOperatorWrapper(op)

    test_op(wrapper, randn(n), randn(n), verb)
end

@testitem "CpuOperatorWrapper (CUDA)" tags = [:gpu, :cuda, :linearoperator, :CpuOperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(42)

        n = 32
        op = FiniteDiff(Float32, (n,), 1)
        wrapper = CpuOperatorWrapper(op)
        x = CUDA.randn(Float32, n)
        y = CUDA.zeros(Float32, n - 1)
        mul!(y, wrapper, x)
        @test y isa CUDA.CuArray
        @test collect(y) ≈ op * collect(x)

        r = CUDA.randn(Float32, n - 1)
        z = CUDA.zeros(Float32, n)
        mul!(z, wrapper', r)
        @test z isa CUDA.CuArray
        ref = zeros(Float32, n)
        mul!(ref, op', collect(r))
        @test collect(z) ≈ ref
    end
end

@testitem "CpuOperatorWrapper (AMDGPU)" tags = [:gpu, :amdgpu, :linearoperator, :CpuOperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(42)

        n = 32
        op = FiniteDiff(Float32, (n,), 1)
        wrapper = CpuOperatorWrapper(op)
        x = AMDGPU.ROCArray(randn(Float32, n))
        y = AMDGPU.zeros(Float32, n - 1)
        mul!(y, wrapper, x)
        @test y isa AMDGPU.ROCArray
        @test collect(y) ≈ op * collect(x)

        r = AMDGPU.ROCArray(randn(Float32, n - 1))
        z = AMDGPU.zeros(Float32, n)
        mul!(z, wrapper', r)
        @test z isa AMDGPU.ROCArray
        ref = zeros(Float32, n)
        mul!(ref, op', collect(r))
        @test collect(z) ≈ ref
    end
end
