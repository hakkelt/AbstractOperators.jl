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

@testitem "Filt" tags = [:dsp, :Filt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

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

@testitem "MIMOFilt" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    m, n = 10, 2
    b = [[1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 0.0; 1.0; 0.0; 0.0]]
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 1), verb)
    y2 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2])

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

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

@testitem "Conv (CUDA)" tags = [:dsp, :gpu, :cuda, :Conv] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_cuda
        @test_skip "CUDA not functional"
    else
        Random.seed!(0)
        conv = cuda.cu
        n, m = 20, 6
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = conv(h_cpu)
        op = Conv(Float64, (n,), h)
        x = conv(x_cpu)
        y = conv(zeros(n + m - 1))
        mul!(y, op, x)
        op_cpu = Conv(Float64, (n,), h_cpu)
        y_cpu = zeros(n + m - 1)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu  atol=1e-10
        b_cpu = randn(n + m - 1)
        b = conv(b_cpu)
        z = conv(zeros(n))
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "Conv (AMDGPU)" tags = [:dsp, :gpu, :amdgpu, :Conv] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_amdgpu
        @test_skip "AMDGPU not functional"
    else
        Random.seed!(0)
        conv = amdgpu.ROCArray
        n, m = 20, 6
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = conv(h_cpu)
        op = Conv(Float64, (n,), h)
        x = conv(x_cpu)
        y = conv(zeros(n + m - 1))
        mul!(y, op, x)
        op_cpu = Conv(Float64, (n,), h_cpu)
        y_cpu = zeros(n + m - 1)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu  atol=1e-10
        b_cpu = randn(n + m - 1)
        b = conv(b_cpu)
        z = conv(zeros(n))
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "Xcorr (CUDA)" tags = [:dsp, :gpu, :cuda, :Xcorr] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_cuda
        @test_skip "CUDA not functional"
    else
        Random.seed!(0)
        conv = cuda.cu
        n, m = 15, 5
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = conv(h_cpu)
        outlen = 2 * max(n, m) - 1
        op = Xcorr(Float64, (n,), h)
        x = conv(x_cpu)
        y = conv(zeros(outlen))
        mul!(y, op, x)
        op_cpu = Xcorr(Float64, (n,), h_cpu)
        y_cpu = zeros(outlen)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu  atol=1e-10
        b_cpu = randn(outlen)
        b = conv(b_cpu)
        z = conv(zeros(n))
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "Xcorr (AMDGPU)" tags = [:dsp, :gpu, :amdgpu, :Xcorr] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_amdgpu
        @test_skip "AMDGPU not functional"
    else
        Random.seed!(0)
        conv = amdgpu.ROCArray
        n, m = 15, 5
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = conv(h_cpu)
        outlen = 2 * max(n, m) - 1
        op = Xcorr(Float64, (n,), h)
        x = conv(x_cpu)
        y = conv(zeros(outlen))
        mul!(y, op, x)
        op_cpu = Xcorr(Float64, (n,), h_cpu)
        y_cpu = zeros(outlen)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu  atol=1e-10
        b_cpu = randn(outlen)
        b = conv(b_cpu)
        z = conv(zeros(n))
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "Filt (CUDA, FIR)" tags = [:dsp, :gpu, :cuda, :Filt] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_cuda
        @test_skip "CUDA not functional"
    else
        Random.seed!(42)
        conv = cuda.cu
        n = 20
        b = randn(5)
        x_cpu = randn(n)
        op_cpu = Filt(Float64, (n,), b)
        y_cpu = zeros(n)
        mul!(y_cpu, op_cpu, x_cpu)
        x = conv(x_cpu)
        y = conv(zeros(n))
        op = Filt(Float64, (n,), b)
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10
        r_cpu = randn(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = conv(r_cpu)
        z = conv(zeros(n))
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "Filt (AMDGPU, FIR)" tags = [:dsp, :gpu, :amdgpu, :Filt] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_amdgpu
        @test_skip "AMDGPU not functional"
    else
        Random.seed!(42)
        conv = amdgpu.ROCArray
        n = 20
        b = randn(5)
        x_cpu = randn(n)
        op_cpu = Filt(Float64, (n,), b)
        y_cpu = zeros(n)
        mul!(y_cpu, op_cpu, x_cpu)
        x = conv(x_cpu)
        y = conv(zeros(n))
        op = Filt(Float64, (n,), b)
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10
        r_cpu = randn(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = conv(r_cpu)
        z = conv(zeros(n))
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "MIMOFilt (CUDA, FIR)" tags = [:dsp, :gpu, :cuda, :MIMOFilt] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_cuda
        @test_skip "CUDA not functional"
    else
        Random.seed!(7)
        conv = cuda.cu
        m, n = 10, 3
        b = [randn(5), randn(3), randn(4), randn(5), randn(3), randn(4)]
        a = [[1.0] for _ in b]
        op_cpu = MIMOFilt(Float64, (m, n), b, a)
        x_cpu = randn(m, n)
        y_cpu = zeros(m, 2)
        mul!(y_cpu, op_cpu, x_cpu)
        x = conv(x_cpu)
        y = conv(zeros(m, 2))
        op = MIMOFilt(Float64, (m, n), b, a)
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10
        r_cpu = randn(m, 2)
        z_cpu = zeros(m, n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = conv(r_cpu)
        z = conv(zeros(m, n))
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "MIMOFilt (AMDGPU, FIR)" tags = [:dsp, :gpu, :amdgpu, :MIMOFilt] setup=[TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
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
    if !has_amdgpu
        @test_skip "AMDGPU not functional"
    else
        Random.seed!(7)
        conv = amdgpu.ROCArray
        m, n = 10, 3
        b = [randn(5), randn(3), randn(4), randn(5), randn(3), randn(4)]
        a = [[1.0] for _ in b]
        op_cpu = MIMOFilt(Float64, (m, n), b, a)
        x_cpu = randn(m, n)
        y_cpu = zeros(m, 2)
        mul!(y_cpu, op_cpu, x_cpu)
        x = conv(x_cpu)
        y = conv(zeros(m, 2))
        op = MIMOFilt(Float64, (m, n), b, a)
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10
        r_cpu = randn(m, 2)
        z_cpu = zeros(m, n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = conv(r_cpu)
        z = conv(zeros(m, n))
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end
