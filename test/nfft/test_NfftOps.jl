@testmodule NFFTTestHelper begin
    using Test
    using AbstractOperators
    using LinearAlgebra, Random, NFFT, NFFTOperators

    function test_nufft_op(op, plan, image, dcf)
        ksp1 = similar(image, ComplexF64, size(op, 1))
        mul!(vec(ksp1), plan, image)
        ksp2 = similar(ksp1)
        mul!(ksp2, op, image)
        @test ksp2 == ksp1

        image2 = similar(image)
        if dcf === nothing
            mul!(image2, plan', vec(ksp2 .* op.dcf))
            @test norm(image2 .- image) / norm(image) < 0.5
        else
            image1 = similar(image)
            mul!(image1, plan', vec(ksp1 .*= dcf))
            mul!(image2, op', ksp2)
            @test image2 ≈ image1
        end

        normal_op = AbstractOperators.get_normal_op(op)
        image3 = similar(image)
        mul!(image3, normal_op, image)
        @test image3 ≈ image2
    end

    function test_2d_nufft(threaded)
        trajectory = rand(2, 128, 50) .- 0.5
        dcf = rand(128, 50)
        image_size = (128, 128)
        image = rand(ComplexF64, image_size)
        plan = plan_nfft(reshape(trajectory, 2, :), image_size)
        op = NFFTOp(image_size, trajectory, dcf; threaded)
        test_nufft_op(op, plan, image, dcf)
    end

    function test_3d_nufft(threaded)
        trajectory = rand(3, 128, 50) .- 0.5
        dcf = rand(128, 50)
        image_size = (64, 64, 64)
        image = rand(ComplexF64, image_size)
        plan = plan_nfft(reshape(trajectory, 3, :), image_size)
        op = NFFTOp(image_size, trajectory, dcf; threaded)
        test_nufft_op(op, plan, image, dcf)
    end

    function test_realistic_2d_nufft(threaded)
        trajectory = Array{Float64}(undef, 2, 256, 201)
        ϕstep = 2π / 201
        for i in 1:201
            ϕ = i * ϕstep
            trajectory[1, :, i] = cos(ϕ) .* ((-64:0.5:63.5) ./ 128)
            trajectory[2, :, i] = sin(ϕ) .* ((-64:0.5:63.5) ./ 128)
        end
        image_size = (128, 128)
        image = zeros(ComplexF64, image_size)
        for idx in CartesianIndices(image)
            d = norm([idx[1] - 64, idx[2] - 64])
            if d < 15
                image[idx] = 1.0
            end
        end
        plan = plan_nfft(reshape(trajectory, 2, :), image_size)
        op = NFFTOp(image_size, trajectory; threaded)
        test_nufft_op(op, plan, image, nothing)
    end

    function test_nfft_normal_op(threaded)
        trajectory = rand(2, 128, 50) .- 0.5
        dcf = rand(128, 50)
        image_size = (128, 128)
        image = rand(ComplexF64, image_size)
        op = NFFTOp(image_size, trajectory, dcf; threaded)
        normal_op = AbstractOperators.get_normal_op(op)

        image_out1 = similar(image)
        mul!(image_out1, normal_op, image)
        ksp = similar(image, ComplexF64, size(op, 1))
        mul!(ksp, op, image)
        image_out2 = similar(image)
        mul!(image_out2, op', ksp)
        @test image_out1 ≈ image_out2
    end
end

@testitem "NFFTOp 2D" tags = [:nfft, :NFFTOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_2d_nufft(false)
    NFFTTestHelper.test_2d_nufft(true)
end

@testitem "NFFTOp realistic 2D" tags = [:nfft, :NFFTOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_realistic_2d_nufft(false)
    NFFTTestHelper.test_realistic_2d_nufft(true)
end

@testitem "NFFTOp 3D" tags = [:nfft, :NFFTOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_3d_nufft(false)
    NFFTTestHelper.test_3d_nufft(true)
end

@testitem "NfftNormalOp" tags = [:nfft, :NfftNormalOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_nfft_normal_op(false)
    NFFTTestHelper.test_nfft_normal_op(true)
end

@testitem "NfftNormalOp (GPU)" tags = [:gpu, :nfft, :NfftNormalOp] setup = [TestUtils, GpuEnvSetup] begin
    using AbstractOperators, NFFTOperators, NFFT, GPUEnv, LinearAlgebra, Random
    using AbstractOperators: get_normal_op

    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(0)
        traj = Float32.(rand(2, 64, 20) .- 0.5)
        dcf = rand(Float32, 64, 20)
        x = randn(ComplexF32, 32, 32)
        x_gpu = to_gpu(backend, x)

        host = get_normal_op(NFFTOp((32, 32), traj, dcf)) * x
        op = NFFTOp((32, 32), traj, dcf; array_type = typeof(to_gpu(backend, dcf)))
        N = get_normal_op(op)
        y = N * x_gpu
        @test y isa typeof(x_gpu)
        @test collect(y) ≈ host rtol = 1.0e-4
        @test collect(y) ≈ collect(op' * (op * x_gpu)) rtol = 1.0e-4
    end
end

@testitem "NFFTOp (GPU)" tags = [:gpu, :nfft, :NFFTOp] setup = [TestUtils, GpuEnvSetup] begin
    using AbstractOperators, NFFTOperators, NFFT, GPUEnv, LinearAlgebra, Random

    # The device plan builds its interpolation matrix itself; it must be the one NFFT.jl's own
    # GPU plan computes on the host, and the operator must match the host operator.
    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true), D in (2, 3)
        Random.seed!(0)
        n = D == 2 ? 32 : 12
        traj = Float32.(rand(D, 64, 20) .- 0.5)
        x = randn(ComplexF32, ntuple(_ -> n, D))
        y = randn(ComplexF32, 64, 20)
        A = typeof(to_gpu(backend, y))
        op = NFFTOp(ntuple(_ -> n, D), traj; array_type = A)
        host = NFFTOp(ntuple(_ -> n, D), traj)
        @test collect(op * to_gpu(backend, x)) ≈ host * x rtol = 1.0e-4
        @test collect(op' * to_gpu(backend, y)) ≈ host' * y rtol = 1.0e-4

        reference = NFFT.plan_nfft(Base.typename(A).wrapper, reshape(traj, D, :), ntuple(_ -> n, D))
        @test collect(op.plan.B.rowVal) == collect(reference.B.rowVal)
        @test collect(op.plan.B.nzVal) ≈ collect(reference.B.nzVal) rtol = 1.0e-5
    end
end
