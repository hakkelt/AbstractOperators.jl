@testitem "Conv" tags = [:dsp, :Conv] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n, m = 5, 6
    h = randn(m)
    op = Conv(Float64, (n,), h)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n + m - 1), verb)
    y2 = conv(x1, h)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    z1 = op' * y1
    z2 = xcorr(y1, h)[size(op.h, 1)[1]:(end - length(op.h) + 1)]
    @test all(norm.(z1 .- z2) .<= 1.0e-12)

    # other constructors
    op = Conv(x1, h)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true
end

@testitem "Filt: IIR and FIR mappings" tags = [:dsp, :Filt] setup = [TestUtils, GpuTestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    n, m = 15, 2
    b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
    op = Filt(Float64, (n,), b, a)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = filt(b, a, x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    h = randn(10)
    op = Filt(Float64, (n, m), h)
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n, m), verb)
    y2 = filt(h, [1.0], x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    x_gpu = jl(randn(n, m))
    gpu_op = Filt(x_gpu, h)
    y_gpu = gpu_op * x_gpu
    @test y_gpu isa typeof(x_gpu)
    @test collect(y_gpu) ≈ filt(h, [1.0], collect(x_gpu))
end

@testitem "Filt: constructors and properties" tags = [:dsp, :Filt] setup = [TestUtils] begin
    using DSPOperators
    using AbstractOperators:
        is_linear,
        is_null,
        is_eye,
        is_diagonal,
        is_AcA_diagonal,
        is_AAc_diagonal,
        is_orthogonal,
        is_invertible,
        is_full_row_rank,
        is_full_column_rank

    n, m = 15, 2
    b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
    h = randn(10)
    x1 = randn(n, m)
    op = Filt(Float64, (n, m), h)

    # other constructors
    Filt(n, b, a)
    Filt((n, m), b, a)
    Filt(n, h)
    Filt((n,), h)
    Filt(x1, b, a)
    Filt(x1, b)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true
end

@testitem "MIMOFilt: 2-input 1-output mapping" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    m, n = 10, 2
    b = [[1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 0.0; 1.0; 0.0; 0.0]]
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 1), verb)
    y2 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2])

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "MIMOFilt: 3-input 2-output mapping" tags = [:dsp, :MIMOFilt] setup = [TestUtils, GpuTestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    m, n = 10, 3  #time samples, number of inputs
    b = [
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
    ]
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0], [3.0], [4.0], [5.0], [6.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 2), verb)
    col1 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2]) + filt(b[3], a[3], x1[:, 3])
    col2 = filt(b[4], a[4], x1[:, 1]) + filt(b[5], a[5], x1[:, 2]) + filt(b[6], a[6], x1[:, 3])
    y2 = [col1 col2]

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "MIMOFilt: FIR mapping and GPU allocation" tags = [:dsp, :MIMOFilt, :gpu] setup = [
    TestUtils,
    GpuTestUtils,
] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 2), verb)
    col1 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2]) + filt(b[3], a[3], x1[:, 3])
    col2 = filt(b[4], a[4], x1[:, 1]) + filt(b[5], a[5], x1[:, 2]) + filt(b[6], a[6], x1[:, 3])
    y2 = [col1 col2]

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    x_gpu = jl(randn(m, n))
    gpu_op = MIMOFilt(x_gpu, b, a)
    y_gpu = gpu_op * x_gpu
    @test y_gpu isa typeof(similar(x_gpu, Float64, m, 2))
    x_gpu_cpu = collect(x_gpu)
    expected_gpu = hcat(
        filt(b[1], a[1], x_gpu_cpu[:, 1]) +
            filt(b[2], a[2], x_gpu_cpu[:, 2]) +
            filt(b[3], a[3], x_gpu_cpu[:, 3]),
        filt(b[4], a[4], x_gpu_cpu[:, 1]) +
            filt(b[5], a[5], x_gpu_cpu[:, 2]) +
            filt(b[6], a[6], x_gpu_cpu[:, 3]),
    )
    @test collect(y_gpu) ≈ expected_gpu
end

@testitem "MIMOFilt: constructors and errors" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    x1 = randn(m, n)

    ## other constructors
    MIMOFilt((10, 3), b, a)
    MIMOFilt((10, 3), b)
    MIMOFilt(x1, b, a)
    MIMOFilt(x1, b)

    #errors
    @test_throws ErrorException MIMOFilt(Float64, (10, 3, 2), b, a)
    a2 = [[1.0f0], [1.0f0], [1.0f0], [1.0f0], [1.0f0], [1.0f0]]
    b2 = convert.(Array{Float32, 1}, b)
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b2, a2)
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b, a[1:(end - 1)])
    push!(a2, [1.0f0])
    push!(b2, randn(Float32, 10))
    @test_throws ErrorException MIMOFilt(Float32, (m, n), b2, a2)
    a[1][1] = 0.0
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b, a)
end

@testitem "MIMOFilt: properties" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators
    using AbstractOperators:
        is_linear,
        is_null,
        is_eye,
        is_diagonal,
        is_AcA_diagonal,
        is_AAc_diagonal,
        is_orthogonal,
        is_invertible,
        is_full_row_rank,
        is_full_column_rank

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    ##properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true
end

@testitem "Conv (CUDA)" tags = [:dsp, :gpu, :cuda, :Conv] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using CUDA
    if CUDA.functional()
        Random.seed!(0)
        n, m = 20, 6
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = CuArray(h_cpu)
        op = Conv(Float64, (n,), h)
        x = CuArray(x_cpu)
        y = CUDA.zeros(Float64, n + m - 1)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        op_cpu = Conv(Float64, (n,), h_cpu)
        y_cpu = zeros(n + m - 1)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        b_cpu = randn(n + m - 1)
        b = CuArray(b_cpu)
        z = CUDA.zeros(Float64, n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Conv (AMDGPU)" tags = [:dsp, :gpu, :amdgpu, :Conv] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)
        n, m = 20, 6
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = AMDGPU.ROCArray(h_cpu)
        op = Conv(Float64, (n,), h)
        x = AMDGPU.ROCArray(x_cpu)
        y = AMDGPU.zeros(n + m - 1)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        op_cpu = Conv(Float64, (n,), h_cpu)
        y_cpu = zeros(n + m - 1)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        b_cpu = randn(n + m - 1)
        b = AMDGPU.ROCArray(b_cpu)
        z = AMDGPU.zeros(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Xcorr (CUDA)" tags = [:dsp, :gpu, :cuda, :Xcorr] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using CUDA
    if CUDA.functional()
        Random.seed!(0)
        n, m = 15, 5
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = CuArray(h_cpu)
        outlen = 2 * max(n, m) - 1
        op = Xcorr(Float64, (n,), h)
        x = CuArray(x_cpu)
        y = CUDA.zeros(Float64, outlen)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        op_cpu = Xcorr(Float64, (n,), h_cpu)
        y_cpu = zeros(outlen)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        b_cpu = randn(outlen)
        b = CuArray(b_cpu)
        z = CUDA.zeros(Float64, n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Xcorr (AMDGPU)" tags = [:dsp, :gpu, :amdgpu, :Xcorr] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(0)
        n, m = 15, 5
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = AMDGPU.ROCArray(h_cpu)
        outlen = 2 * max(n, m) - 1
        op = Xcorr(Float64, (n,), h)
        x = AMDGPU.ROCArray(x_cpu)
        y = AMDGPU.zeros(outlen)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        op_cpu = Xcorr(Float64, (n,), h_cpu)
        y_cpu = zeros(outlen)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        b_cpu = randn(outlen)
        b = AMDGPU.ROCArray(b_cpu)
        z = AMDGPU.zeros(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Filt (CUDA, FIR)" tags = [:dsp, :gpu, :cuda, :Filt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using CUDA
    if CUDA.functional()
        Random.seed!(42)
        n = 20
        b = randn(5)
        x_cpu = randn(n)
        op_cpu = Filt(Float64, (n,), b)
        y_cpu = zeros(n)
        mul!(y_cpu, op_cpu, x_cpu)
        x = CuArray(x_cpu)
        op = Filt(x, b)
        y = CUDA.zeros(Float64, n)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        r_cpu = randn(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = CuArray(r_cpu)
        z = CUDA.zeros(Float64, n)
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Filt (AMDGPU, FIR)" tags = [:dsp, :gpu, :amdgpu, :Filt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(42)
        n = 20
        b = randn(5)
        x_cpu = randn(n)
        op_cpu = Filt(Float64, (n,), b)
        y_cpu = zeros(n)
        mul!(y_cpu, op_cpu, x_cpu)
        x = AMDGPU.ROCArray(x_cpu)
        op = Filt(x, b)
        y = AMDGPU.zeros(n)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        r_cpu = randn(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = AMDGPU.ROCArray(r_cpu)
        z = AMDGPU.zeros(n)
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "MIMOFilt (CUDA, FIR)" tags = [:dsp, :gpu, :cuda, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using CUDA
    if CUDA.functional()
        Random.seed!(7)
        m, n = 10, 3
        b = [randn(5), randn(3), randn(4), randn(5), randn(3), randn(4)]
        a = [[1.0] for _ in b]
        op_cpu = MIMOFilt(Float64, (m, n), b, a)
        x_cpu = randn(m, n)
        y_cpu = zeros(m, 2)
        mul!(y_cpu, op_cpu, x_cpu)
        x = CuArray(x_cpu)
        op = MIMOFilt(x, b, a)
        y = CUDA.zeros(Float64, m, 2)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        r_cpu = randn(m, 2)
        z_cpu = zeros(m, n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = CuArray(r_cpu)
        z = CUDA.zeros(Float64, m, n)
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "MIMOFilt (AMDGPU, FIR)" tags = [:dsp, :gpu, :amdgpu, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AMDGPU
    if AMDGPU.functional()
        Random.seed!(7)
        m, n = 10, 3
        b = [randn(5), randn(3), randn(4), randn(5), randn(3), randn(4)]
        a = [[1.0] for _ in b]
        op_cpu = MIMOFilt(Float64, (m, n), b, a)
        x_cpu = randn(m, n)
        y_cpu = zeros(m, 2)
        mul!(y_cpu, op_cpu, x_cpu)
        x = AMDGPU.ROCArray(x_cpu)
        op = MIMOFilt(x, b, a)
        y = AMDGPU.zeros(m, 2)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        r_cpu = randn(m, 2)
        z_cpu = zeros(m, n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = AMDGPU.ROCArray(r_cpu)
        z = AMDGPU.zeros(m, n)
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Conv (JLArray)" tags = [:dsp, :gpu, :jlarray, :Conv] setup = [TestUtils, GpuTestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random, JLArrays
    Random.seed!(0)
    n, m = 20, 6
    h_cpu = randn(m)
    x_cpu = randn(n)
    h = jl(h_cpu)
    op = Conv(Float64, (n,), h)
    x = jl(x_cpu)
    y = jl(zeros(Float64, n + m - 1))
    mul!(y, op, x)
    y_alloc = op * x
    @test y_alloc isa typeof(y)
    op_cpu = Conv(Float64, (n,), h_cpu)
    y_cpu = zeros(n + m - 1)
    mul!(y_cpu, op_cpu, x_cpu)
    @test collect(y) ≈ y_cpu atol = 1.0e-10
    @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
    b_cpu = randn(n + m - 1)
    b = jl(b_cpu)
    z = jl(zeros(Float64, n))
    z_cpu = zeros(n)
    mul!(z_cpu, op_cpu', b_cpu)
    mul!(z, op', b)
    @test collect(z) ≈ z_cpu atol = 1.0e-10
end

@testitem "Xcorr (JLArray)" tags = [:dsp, :gpu, :jlarray, :Xcorr] setup = [TestUtils, GpuTestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random, JLArrays
    Random.seed!(0)
    n, m = 15, 5
    h_cpu = randn(m)
    x_cpu = randn(n)
    h = jl(h_cpu)
    outlen = 2 * max(n, m) - 1
    op = Xcorr(Float64, (n,), h)
    x = jl(x_cpu)
    y = jl(zeros(Float64, outlen))
    mul!(y, op, x)
    y_alloc = op * x
    @test y_alloc isa typeof(y)
    op_cpu = Xcorr(Float64, (n,), h_cpu)
    y_cpu = zeros(outlen)
    mul!(y_cpu, op_cpu, x_cpu)
    @test collect(y) ≈ y_cpu atol = 1.0e-10
    @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
    b_cpu = randn(outlen)
    b = jl(b_cpu)
    z = jl(zeros(Float64, n))
    z_cpu = zeros(n)
    mul!(z_cpu, op_cpu', b_cpu)
    mul!(z, op', b)
    @test collect(z) ≈ z_cpu atol = 1.0e-10
end

@testitem "Filt (JLArray, FIR)" tags = [:dsp, :gpu, :jlarray, :Filt] setup = [TestUtils, GpuTestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random, JLArrays
    Random.seed!(42)
    n = 20
    b = randn(5)
    x_cpu = randn(n)
    op_cpu = Filt(Float64, (n,), b)
    y_cpu = zeros(n)
    mul!(y_cpu, op_cpu, x_cpu)
    x = jl(x_cpu)
    op = Filt(x, b)
    y = jl(zeros(Float64, n))
    mul!(y, op, x)
    y_alloc = op * x
    @test y_alloc isa typeof(y)
    @test collect(y) ≈ y_cpu atol = 1.0e-10
    @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
    r_cpu = randn(n)
    z_cpu = zeros(n)
    mul!(z_cpu, op_cpu', r_cpu)
    r = jl(r_cpu)
    z = jl(zeros(Float64, n))
    mul!(z, op', r)
    @test collect(z) ≈ z_cpu atol = 1.0e-10
end

@testitem "MIMOFilt (JLArray, FIR)" tags = [:dsp, :gpu, :jlarray, :MIMOFilt] setup = [TestUtils, GpuTestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random, JLArrays
    Random.seed!(7)
    m, n = 10, 3
    b = [randn(5), randn(3), randn(4), randn(5), randn(3), randn(4)]
    a = [[1.0] for _ in b]
    op_cpu = MIMOFilt(Float64, (m, n), b, a)
    x_cpu = randn(m, n)
    y_cpu = zeros(m, 2)
    mul!(y_cpu, op_cpu, x_cpu)
    x = jl(x_cpu)
    op = MIMOFilt(x, b, a)
    y = jl(zeros(Float64, m, 2))
    mul!(y, op, x)
    y_alloc = op * x
    @test y_alloc isa typeof(y)
    @test collect(y) ≈ y_cpu atol = 1.0e-10
    @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
    r_cpu = randn(m, 2)
    z_cpu = zeros(m, n)
    mul!(z_cpu, op_cpu', r_cpu)
    r = jl(r_cpu)
    z = jl(zeros(Float64, m, n))
    mul!(z, op', r)
    @test collect(z) ≈ z_cpu atol = 1.0e-10
end
