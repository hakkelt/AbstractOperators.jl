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

@testitem "DCT (GPU via OperatorWrapper)" tags = [:fftw, :gpu, :jlarray, :DCT] setup = [TestUtils, GpuTestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators, JLArrays, AbstractOperators

    n = 4
    cpu_op = DCT(Float64, (n,))
    gpu_op = OperatorWrapper(cpu_op; array_type = JLArray{Float64})

    x_gpu = jl(randn(n))
    y_gpu = jl(zeros(n))
    mul!(y_gpu, gpu_op, x_gpu)
    @test collect(y_gpu) ≈ dct(collect(x_gpu))

    b_gpu = jl(randn(n))
    z_gpu = jl(zeros(n))
    mul!(z_gpu, gpu_op', b_gpu)
    @test collect(z_gpu) ≈ idct(collect(b_gpu))
end

@testitem "DCT/IDCT (CUDA via AcceleratedDCTs)" tags = [:fftw, :gpu, :cuda, :DCT, :IDCT] setup = [TestUtils] begin
    using AbstractOperators, FFTW, FFTWOperators, Random
    using AcceleratedDCTs, CUDA
    if CUDA.functional()
        Random.seed!(0)
        dims = (8, 10)
        x_cpu = randn(Float32, dims...)
        x_gpu = CuArray(x_cpu)

        dct_op = DCT(x_gpu)
        y_gpu = similar(x_gpu)
        mul!(y_gpu, dct_op, x_gpu)
        @test y_gpu isa typeof(x_gpu)
        @test collect(y_gpu) ≈ dct(x_cpu) rtol = 1.0e-4 atol = 1.0e-4

        idct_op = IDCT(x_gpu)
        x_rec_gpu = similar(x_gpu)
        mul!(x_rec_gpu, idct_op, y_gpu)
        @test collect(x_rec_gpu) ≈ x_cpu rtol = 1.0e-4 atol = 1.0e-4

        x_adj_gpu = similar(x_gpu)
        mul!(x_adj_gpu, dct_op', y_gpu)
        @test collect(x_adj_gpu) ≈ x_cpu rtol = 1.0e-4 atol = 1.0e-4
    end
end

@testitem "DCT/IDCT (AMDGPU via AcceleratedDCTs)" tags = [:fftw, :gpu, :amdgpu, :DCT, :IDCT] setup = [TestUtils] begin
    using AbstractOperators, FFTW, FFTWOperators, Random
    using AMDGPU
    using AcceleratedDCTs

    if AMDGPU.functional()
        Random.seed!(0)
        dims = (8, 10)
        x_cpu = randn(Float32, dims...)
        x_gpu = AMDGPU.ROCArray(x_cpu)

        dct_op = DCT(x_gpu)
        y_gpu = similar(x_gpu)
        mul!(y_gpu, dct_op, x_gpu)
        @test collect(y_gpu) ≈ dct(x_cpu) rtol = 1.0e-4 atol = 1.0e-4

        idct_op = IDCT(x_gpu)
        x_rec_gpu = similar(x_gpu)
        mul!(x_rec_gpu, idct_op, y_gpu)
        @test collect(x_rec_gpu) ≈ x_cpu rtol = 1.0e-4 atol = 1.0e-4
    end
end

@testitem "DFT/RDFT/IRDFT (JLArray)" tags = [:fftw, :gpu, :jlarray, :DFT, :RDFT, :IRDFT] setup = [TestUtils, GpuTestUtils] begin
    using FFTW, LinearAlgebra, Random, FFTWOperators, JLArrays, AbstractOperators
    Random.seed!(0)

    n, m = 8, 6

    # DFT with real input — exercises the GPU DFT constructor branch (D <: Real)
    x_cpu = randn(Float64, n, m)
    op = DFT(jl(x_cpu))
    @test op isa DFT
    x_gpu = jl(x_cpu)
    y_gpu = similar(x_gpu, Complex{Float64})
    mul!(y_gpu, op, x_gpu)
    @test Array(y_gpu) ≈ fft(x_cpu)
    r_gpu = similar(x_gpu, Float64)  # adjoint of real-input DFT outputs real
    mul!(r_gpu, op', y_gpu)
    @test Array(r_gpu) ≈ real.(bfft(Array(y_gpu)))

    # DFT with complex input — exercises the GPU DFT constructor branch (D <: Complex)
    xc_cpu = randn(ComplexF64, n, m)
    opc = DFT(jl(xc_cpu))
    @test opc isa DFT
    xc_gpu = jl(xc_cpu)
    yc_gpu = similar(xc_gpu)
    mul!(yc_gpu, opc, xc_gpu)
    @test Array(yc_gpu) ≈ fft(xc_cpu)

    # RDFT with real input
    opr = RDFT(jl(x_cpu))
    yr_gpu = similar(x_gpu, Complex{Float64}, n ÷ 2 + 1, m)
    mul!(yr_gpu, opr, x_gpu)
    @test Array(yr_gpu) ≈ rfft(x_cpu, 1)

    # IRDFT
    opir = IRDFT(jl(yr_gpu), n)
    out_gpu = similar(x_gpu)
    mul!(out_gpu, opir, yr_gpu)
    @test Array(out_gpu) ≈ x_cpu  rtol = 1e-12
end

@testitem "DFT/RDFT/IRDFT (CUDA)" tags = [:fftw, :gpu, :cuda] setup = [TestUtils] begin
    using FFTWOperators, LinearAlgebra, Random
    using CUDA
    if CUDA.functional()
        conv = CUDA.cu
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

@testitem "DFT/RDFT/IRDFT (AMDGPU)" tags = [:fftw, :gpu, :amdgpu] setup = [TestUtils] begin
    using FFTWOperators, LinearAlgebra, Random
    using AMDGPU
    if AMDGPU.functional()
        conv = AMDGPU.ROCArray
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
