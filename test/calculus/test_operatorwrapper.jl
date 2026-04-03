@testitem "OperatorWrapper: CPU mul! forward and adjoint" tags = [
    :calculus, :OperatorWrapper,
] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(42)

    n = 8
    cpu_op = FiniteDiff(Float64, (n,), 1)           # n-1 output
    wrapper = OperatorWrapper(cpu_op)

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
    @test domain_storage_type(wrapper) == Array{Float64, 1}
    @test codomain_storage_type(wrapper) == Array{Float64, 1}

    io = IOBuffer()
    show(io, wrapper)
    @test occursin("CPU[", String(take!(io)))
end

@testitem "OperatorWrapper: properties forwarded" tags = [:calculus, :OperatorWrapper] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(42)

    n = 6
    op = MatrixOp(randn(n, n))
    wrapper = OperatorWrapper(op)
    @test is_full_row_rank(wrapper) == is_full_row_rank(op)
    @test is_full_column_rank(wrapper) == is_full_column_rank(op)
    @test is_thread_safe(wrapper) == false  # always false regardless of inner op
end

@testitem "OperatorWrapper: CPU mul! via test_op" tags = [:calculus, :OperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    Random.seed!(42)

    n = 6
    op = MatrixOp(randn(n, n))
    wrapper = OperatorWrapper(op)

    test_op(wrapper, randn(n), randn(n), verb)
end

@testitem "OperatorWrapper (JLArray)" tags = [:gpu, :jlarray, :calculus, :OperatorWrapper] setup = [TestUtils, GpuTestUtils] begin
    using Random, AbstractOperators, JLArrays
    Random.seed!(42)
    n = 32
    op = FiniteDiff(Float32, (n,), 1)
    wrapper = OperatorWrapper(op; array_type = JLArray{Float32})
    @test domain_storage_type(wrapper) == JLArray{Float32, 1}
    @test codomain_storage_type(wrapper) == JLArray{Float32, 1}

    x = jl(randn(Float32, n))
    y = jl(zeros(Float32, n - 1))
    mul!(y, wrapper, x)
    @test y isa JLArray
    @test Array(y) ≈ op * Array(x)

    r = jl(randn(Float32, n - 1))
    z = jl(zeros(Float32, n))
    mul!(z, wrapper', r)
    @test z isa JLArray
    ref = zeros(Float32, n)
    mul!(ref, op', Array(r))
    @test Array(z) ≈ ref
end

@testitem "OperatorWrapper (CUDA)" tags = [:gpu, :cuda, :calculus, :OperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    using CUDA
    if CUDA.functional()
        Random.seed!(42)

        n = 32
        op = FiniteDiff(Float32, (n,), 1)
        wrapper = OperatorWrapper(op; array_type = CuArray{Float32})
        @test domain_storage_type(wrapper) == CuArray{Float32, 1}
        @test codomain_storage_type(wrapper) == CuArray{Float32, 1}

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

@testitem "OperatorWrapper (AMDGPU)" tags = [:gpu, :amdgpu, :calculus, :OperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(42)

        n = 32
        op = FiniteDiff(Float32, (n,), 1)
        wrapper = OperatorWrapper(op; array_type = AMDGPU.ROCArray{Float32})
        @test domain_storage_type(wrapper) == AMDGPU.ROCArray{Float32, 1}

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
