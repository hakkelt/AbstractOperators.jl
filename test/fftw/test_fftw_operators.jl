@testitem "DCT" tags = [:fftw, :DCT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 4
    op = DCT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = dct(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    # other constructors
    op = DCT((n,))
    op = DCT(n, n)
    op = DCT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == true
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    m = 10
    op = DCT(n, m)
    x1 = randn(n, m)

    @test norm(op' * (op * x1) - x1) <= 1.0e-12
    @test diag_AAc(op) == 1.0
    @test diag_AcA(op) == 1.0
end

@testitem "IDCT" tags = [:fftw, :IDCT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 4
    op = IDCT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = idct(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    # other constructors
    op = IDCT((n,))
    op = IDCT(n, n)
    op = IDCT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == true
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    m = 10
    op = IDCT(n, m)
    x1 = randn(n, m)

    @test norm(op' * (op * x1) - x1) <= 1.0e-12
    @test diag_AAc(op) == 1.0
    @test diag_AcA(op) == 1.0
end

@testitem "DFT" tags = [:fftw, :DFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n, m = 4, 7

    op = DFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n,))
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n,))
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (n,), 1)
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n,), 1)
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (n, m))
    x1 = randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n, m))
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (m, n), 1)
    x1 = randn(m, n)
    y1 = test_op(op, x1, fft(randn(m, n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n, m), 2)
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    # other constructors
    op = DFT((n,))
    op = DFT(n, n)
    op = DFT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    op = DFT(n, m)
    x1 = randn(n, m)
    y1 = op * x1
    @test norm(op' * (op * x1) - diag_AcA(op) * x1) <= 1.0e-12
    @test norm(op * (op' * y1) - diag_AAc(op) * y1) <= 1.0e-12
end

@testitem "IDFT" tags = [:fftw, :IDFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n, m = 5, 6

    @test_throws AssertionError IDFT(Float64, (n,))

    op = IDFT(Complex{Float64}, (n,))
    x1 = randn(ComplexF64, n)
    @test op * x1 ≈ ifft(x1)

    @test_throws AssertionError IDFT(Float64, (n,), 1)

    op = IDFT(Complex{Float64}, (n,), 1)
    x1 = randn(ComplexF64, n)
    @test op * x1 ≈ ifft(x1, 1)

    @test_throws AssertionError IDFT(Float64, (n, m))

    op = IDFT(Complex{Float64}, (n, m))
    x1 = randn(ComplexF64, n, m)
    @test op * x1 ≈ ifft(x1)

    @test_throws AssertionError IDFT(Float64, (m, n), 1)

    op = IDFT(Complex{Float64}, (n, m), 2)
    x1 = randn(ComplexF64, n, m)
    @test op * x1 ≈ ifft(x1, 2)

    n, m, l = 4, 19, 5
    op = IDFT(Complex{Float64}, (n, m, l), 2)
    x1 = fft(randn(n, m, l), 2)
    @test op * x1 ≈ ifft(x1, 2)

    n, m, l = 4, 18, 5
    op = IDFT(Complex{Float64}, (n, m, l), (1, 2))
    x1 = fft(randn(n, m, l), (1, 2))
    @test op * x1 ≈ ifft(x1, (1, 2))

    op = IDFT(Complex{Float64}, (n, m, l), (3, 2))
    x1 = fft(randn(n, m, l), (3, 2))
    @test op * x1 ≈ ifft(x1, (3, 2))

    # other constructors
    op = IDFT((n,))
    op = IDFT(n, n)
    op = IDFT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    n, m = 4, 10
    op = IDFT(n, m)
    x1 = randn(ComplexF64, n, m)
    y1 = op * x1
    @test norm(op' * (op * x1) - diag_AcA(op) * x1) <= 1.0e-12
    @test norm(op * (op' * y1) - diag_AAc(op) * y1) <= 1.0e-12
end

@testitem "RDFT" tags = [:fftw, :RDFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 4
    op = RDFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, rfft(x1), verb)
    y2 = rfft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 4, 8, 5
    op = RDFT(Float64, (n, m, l), 2)
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, rfft(x1, 2), verb)
    y2 = rfft(x1, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    # other constructors
    op = RDFT((n,))
    op = RDFT(n, n)

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
    @test is_full_column_rank(op) == false
end

@testitem "IRDFT" tags = [:fftw, :IRDFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 12
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1,), n)
    x1 = rfft(randn(n))
    y1 = test_op(op, x1, irfft(randn(div(n, 2) + 1), n), verb)
    y2 = irfft(x1, n)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n = 11
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1,), n)
    x1 = rfft(randn(n))
    y1 = test_op(op, x1, irfft(randn(div(n, 2) + 1), n), verb)
    y2 = irfft(x1, n)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 4, 19, 5
    op = IRDFT(Complex{Float64}, (n, div(m, 2) + 1, l), m, 2)
    x1 = rfft(randn(n, m, l), 2)
    y1 = test_op(op, x1, irfft(x1, m, 2), verb)
    y2 = irfft(x1, m, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 4, 18, 5
    op = IRDFT(Complex{Float64}, (n, div(m, 2) + 1, l), m, 2)
    x1 = rfft(randn(n, m, l), 2)
    y1 = test_op(op, x1, irfft(x1, m, 2), verb)
    y2 = irfft(x1, m, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 5, 18, 5
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1, m, l), n, 1)
    x1 = rfft(randn(n, m, l), 1)
    y1 = test_op(op, x1, irfft(x1, n, 1), verb)
    y2 = irfft(x1, n, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    ## other constructors
    op = IRDFT((10,), 19)

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
    @test is_full_column_rank(op) == false
end

@testitem "DCT (GPU via CpuOperatorWrapper)" tags = [:fftw, :gpu, :DCT] setup=[TestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators, JLArrays, AbstractOperators

    n = 4
    cpu_op = DCT(Float64, (n,))
    gpu_op = CpuOperatorWrapper(cpu_op)

    x_gpu = jl(randn(n))
    y_gpu = jl(zeros(n))
    mul!(y_gpu, gpu_op, x_gpu)
    @test collect(y_gpu) ≈ dct(collect(x_gpu))

    b_gpu = jl(randn(n))
    z_gpu = jl(zeros(n))
    mul!(z_gpu, gpu_op', b_gpu)
    @test collect(z_gpu) ≈ idct(collect(b_gpu))
end

@testitem "DFT (JLArray)" tags = [:fftw, :gpu, :DFT] setup=[TestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators
    Random.seed!(1)

    for (backend_name, conv) in filter(b -> b[1] == "JLArray", GPU_BACKENDS)
        # Complex input
        n = 8
        x_cpu = randn(ComplexF64, n)
        op_cpu = DFT(x_cpu)
        y_cpu = zeros(ComplexF64, n)
        mul!(y_cpu, op_cpu, x_cpu)

        x = conv(x_cpu)
        op = DFT(x)
        y = conv(zeros(ComplexF64, n))
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10

        # Adjoint
        b_cpu = randn(ComplexF64, n)
        z_cpu = zeros(ComplexF64, n)
        mul!(z_cpu, op_cpu', b_cpu)
        b = conv(b_cpu)
        z = conv(zeros(ComplexF64, n))
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10

        # Real input (codomain is complex)
        xr_cpu = randn(Float64, n)
        opr_cpu = DFT(xr_cpu)
        yr_cpu = zeros(ComplexF64, n)
        mul!(yr_cpu, opr_cpu, xr_cpu)

        xr = conv(xr_cpu)
        opr = DFT(xr)
        yr = similar(xr, ComplexF64)
        mul!(yr, opr, xr)
        @test collect(yr) ≈ yr_cpu  atol=1e-10
    end
end

@testitem "RDFT (JLArray)" tags = [:fftw, :gpu, :RDFT] setup=[TestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators
    Random.seed!(2)

    for (backend_name, conv) in filter(b -> b[1] == "JLArray", GPU_BACKENDS)
        n = 8
        x_cpu = randn(Float64, n)
        op_cpu = RDFT(x_cpu)
        y_cpu = zeros(ComplexF64, n÷2+1)
        mul!(y_cpu, op_cpu, x_cpu)

        x = conv(x_cpu)
        op = RDFT(x)
        y = conv(zeros(ComplexF64, n÷2+1))
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10

        # Adjoint
        b_cpu = randn(ComplexF64, n÷2+1)
        z_cpu = zeros(Float64, n)
        mul!(z_cpu, op_cpu', b_cpu)
        b = conv(b_cpu)
        z = conv(zeros(Float64, n))
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "IRDFT (JLArray)" tags = [:fftw, :gpu, :IRDFT] setup=[TestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators
    Random.seed!(3)

    for (backend_name, conv) in filter(b -> b[1] == "JLArray", GPU_BACKENDS)
        n = 8
        x_cpu = randn(ComplexF64, n÷2+1)
        op_cpu = IRDFT(x_cpu, n)
        y_cpu = zeros(Float64, n)
        mul!(y_cpu, op_cpu, x_cpu)

        x = conv(x_cpu)
        op = IRDFT(x, n)
        y = conv(zeros(Float64, n))
        mul!(y, op, x)
        @test collect(y) ≈ y_cpu  atol=1e-10

        # Adjoint
        b_cpu = randn(Float64, n)
        z_cpu = zeros(ComplexF64, n÷2+1)
        mul!(z_cpu, op_cpu', b_cpu)
        b = conv(b_cpu)
        z = conv(zeros(ComplexF64, n÷2+1))
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu  atol=1e-10
    end
end

@testitem "DFT/RDFT/IRDFT (CUDA)" tags = [:fftw, :gpu, :cuda] setup=[TestUtils] begin
    using FFTWOperators, Random
    b = backend_by_tag(:cuda)
    if b === nothing
        @test_skip "CUDA not functional"
    else
        _, _, conv = b
    Random.seed!(11)

    n = 8
    x = conv(randn(ComplexF64, n))
    op = DFT(x)
    y = conv(zeros(ComplexF64, n))
    mul!(y, op, x)
    @test y isa typeof(x)

    xr = conv(randn(Float64, n))
    rop = RDFT(xr)
    ry = conv(zeros(ComplexF64, n ÷ 2 + 1))
    mul!(ry, rop, xr)
    @test ry isa typeof(ry)

    ir = IRDFT(ry, n)
    out = conv(zeros(Float64, n))
    mul!(out, ir, ry)
    @test out isa typeof(xr)
    end
end

@testitem "DFT/RDFT/IRDFT (AMDGPU)" tags = [:fftw, :gpu, :amdgpu] setup=[TestUtils] begin
    using FFTWOperators, Random
    b = backend_by_tag(:amdgpu)
    if b === nothing
        @test_skip "AMDGPU not functional"
    else
        _, _, conv = b
    Random.seed!(12)

    n = 8
    x = conv(randn(ComplexF64, n))
    op = DFT(x)
    y = conv(zeros(ComplexF64, n))
    mul!(y, op, x)
    @test y isa typeof(x)

    xr = conv(randn(Float64, n))
    rop = RDFT(xr)
    ry = conv(zeros(ComplexF64, n ÷ 2 + 1))
    mul!(ry, rop, xr)
    @test ry isa typeof(ry)

    ir = IRDFT(ry, n)
    out = conv(zeros(Float64, n))
    mul!(out, ir, ry)
    @test out isa typeof(xr)
    end
end
