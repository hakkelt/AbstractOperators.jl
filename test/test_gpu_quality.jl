@testitem "GpuExt Quality" tags = [:quality, :gpu] begin
    using AbstractOperators, JLArrays, LinearAlgebra

    # Loading JLArrays triggers GpuExt (GPUArrays is transitive dep)
    # Verify GpuExt loaded by checking GPU-specific dispatch
    d = jl(randn(4))
    op = DiagOp(d)
    x = jl(randn(4))
    y = jl(zeros(4))
    mul!(y, op, x)
    @test collect(y) ≈ collect(d) .* collect(x)

    # Verify guards work: GPU output + CPU input → error
    op_gi = GetIndex(Float64, (4,), (2:4,))
    x_cpu = randn(4)
    y_gpu = jl(zeros(3))
    @test_throws ArgumentError mul!(y_gpu, op_gi, x_cpu)

    # CPU output + GPU input → error
    x_gpu = jl(randn(4))
    y_cpu = zeros(3)
    @test_throws ArgumentError mul!(y_cpu, op_gi, x_gpu)

    # Storage type and allocation checks for GPU operators
    @test domain_storage_type(op) <: JLArrays.JLArray
    @test codomain_storage_type(op) <: JLArrays.JLArray
    y_alloc = op * x
    @test y_alloc isa JLArrays.JLArray
end

@testitem "GpuExt JET" tags = [:jet, :gpu] begin
    using AbstractOperators, JLArrays, LinearAlgebra, JET

    # Verify key GPU-dispatched functions are type-stable (no dynamic dispatch)
    n = 8
    d = jl(randn(n))
    x = jl(randn(n))
    y = jl(zeros(n))

    # DiagOp GPU mul!  — target_modules restricts JET to AbstractOperators code only,
    # avoiding false positives from KernelAbstractions (transitive dep of JLArrays).
    op_diag = DiagOp(d)
    @test_call target_modules=(AbstractOperators,) mul!(y, op_diag, x)

    # GetIndex GPU mul!
    op_gi = GetIndex(d, (2:4,))
    y3 = jl(zeros(3))
    @test_call target_modules=(AbstractOperators,) mul!(y3, op_gi, x)
    x3 = jl(randn(3))
    @test_call target_modules=(AbstractOperators,) mul!(y, op_gi', x3)

    # ZeroPad GPU mul!  — zp is a flat NTuple{N,Int} giving per-dimension padding
    op_zp = ZeroPad(d, (2,))
    y10 = jl(zeros(n + 2))
    @test_call target_modules=(AbstractOperators,) mul!(y10, op_zp, x)
    @test_call target_modules=(AbstractOperators,) mul!(x, op_zp', y10)

    # _should_thread dispatches to false for GPU arrays
    @test AbstractOperators._should_thread(typeof(d)) == false
    @test AbstractOperators._should_thread(Array{Float64}) == true
end

@testitem "Gpu storage type (CUDA)" tags = [:quality, :gpu, :cuda] setup = [TestUtils] begin
    using AbstractOperators
    cuda = try
        @eval import CUDA
        @eval CUDA
    catch
        nothing
    end
    has_cuda = !(cuda === nothing) && try
        cuda.functional()
    catch
        false
    end
    if has_cuda
        conv = cuda.cu
        x = conv(randn(8))
        op = DiagOp(x)
        @test domain_storage_type(op) <: typeof(x).name.wrapper
        @test codomain_storage_type(op) <: typeof(x).name.wrapper
        y = op * x
        @test y isa typeof(x)
    else
        @test_skip "CUDA not functional"
    end
end

@testitem "Gpu storage type (AMDGPU)" tags = [:quality, :gpu, :amdgpu] setup = [TestUtils] begin
    using AbstractOperators
    amdgpu = try
        @eval import AMDGPU
        @eval AMDGPU
    catch
        nothing
    end
    has_amdgpu = !(amdgpu === nothing) && try
        amdgpu.functional()
    catch
        false
    end
    if has_amdgpu
        conv = amdgpu.ROCArray
        x = conv(randn(8))
        op = DiagOp(x)
        @test domain_storage_type(op) <: typeof(x).name.wrapper
        @test codomain_storage_type(op) <: typeof(x).name.wrapper
        y = op * x
        @test y isa typeof(x)
    else
        @test_skip "AMDGPU not functional"
    end
end
