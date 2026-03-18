@testmodule TestUtils begin
using Test
using AbstractOperators
using RecursiveArrayTools
using RecursiveArrayTools: ArrayPartition
using LinearAlgebra
using Random
using JLArrays

export verb, test_op, test_NLop, gradient_fd, ArrayPartition, norm, dot, diag, opnorm
export jl, to_cpu, GPU_CONV_FUNCTIONS, test_NLop_gpu
export ALL_BACKENDS, GPU_BACKENDS, GPU_BACKENDS_TAGGED, HAS_CUDA, HAS_AMDGPU, true_gpu_backends, get_backend

const verb = false

to_cpu(x::AbstractArray) = collect(x)
to_cpu(x::RecursiveArrayTools.ArrayPartition) = RecursiveArrayTools.ArrayPartition(collect.(x.x)...)

const GPU_CONV_FUNCTIONS = [("CPU", identity), ("GPU (JLArray)", jl)]
const ALL_BACKENDS = Tuple{String, Function}[("CPU", identity), ("JLArray", jl)]

const HAS_CUDA = Ref(false)
const HAS_AMDGPU = Ref(false)

# GPU_BACKENDS: (name, conversion_fn) for existing tests.
const GPU_BACKENDS = Tuple{String, Function}[("JLArray", jl)]
# GPU_BACKENDS_TAGGED: (name, tag, conversion_fn) for backend-aware test dispatch.
const GPU_BACKENDS_TAGGED = Tuple{String, Symbol, Function}[("JLArray", :jlarray, jl)]

function __init__()
    try
        cuda = Base.require(Main, :CUDA)
        if cuda.functional()
            HAS_CUDA[] = true
            push!(GPU_BACKENDS, ("CuArray", cuda.cu))
            push!(GPU_BACKENDS_TAGGED, ("CuArray", :cuda, cuda.cu))
            push!(ALL_BACKENDS, ("CuArray", cuda.cu))
        end
    catch
    end
    try
        amdgpu = Base.require(Main, :AMDGPU)
        if amdgpu.functional()
            HAS_AMDGPU[] = true
            push!(GPU_BACKENDS, ("ROCArray", amdgpu.ROCArray))
            push!(GPU_BACKENDS_TAGGED, ("ROCArray", :amdgpu, amdgpu.ROCArray))
            push!(ALL_BACKENDS, ("ROCArray", amdgpu.ROCArray))
        end
    catch
    end
end

true_gpu_backends() = filter(b -> b[2] in (:cuda, :amdgpu), GPU_BACKENDS_TAGGED)
function get_backend(tag::Symbol)
    for (name, backend_tag, conv) in GPU_BACKENDS_TAGGED
        if backend_tag == tag
            return (name, backend_tag, conv)
        end
    end
    return nothing
end

function test_NLop_gpu(A::AbstractOperator, x, y, verb::Bool = false)
    verb && (println(), println(A))

    Ax = A * x
    Ax2 = similar(Ax)
    mul!(Ax2, A, x)
    @test norm(to_cpu(Ax) .- to_cpu(Ax2)) <= 1.0e-8

    @test_throws ErrorException A'

    J = Jacobian(A, x)
    grad = J' * y
    mul!(Ax2, A, x)  # redo forward
    grad2 = similar(grad)
    mul!(grad2, J', y)
    @test norm(to_cpu(grad) .- to_cpu(grad2)) < 1.0e-8

    return Ax, grad
end

########### Test for LinearOperators
function test_op(A::AbstractOperator, x, y, verb::Bool = false)
    verb && (println(); show(A); println())

    Ax = A * x
    Ax2 = similar(Ax)
    verb && println("forward preallocated")
    mul!(Ax2, A, x) #verify in-place linear operator works
    verb && @time mul!(Ax2, A, x)

    @test norm(to_cpu(Ax) .- to_cpu(Ax2)) <= 1.0e-8

    Acy = A' * y
    Acy2 = similar(Acy)
    verb && println("adjoint preallocated")
    At = AdjointOperator(A)
    mul!(Acy2, At, y) #verify in-place linear operator works
    verb && @time mul!(Acy2, At, y)

    @test norm(to_cpu(Acy) .- to_cpu(Acy2)) <= 1.0e-8

    s1 = real(dot(to_cpu(Ax2), to_cpu(y)))
    s2 = real(dot(to_cpu(x), to_cpu(Acy2)))
    @test abs(s1 - s2) < 1.0e-8

    return Ax
end

########### Test for NonLinearOperators
function test_NLop(A::AbstractOperator, x, y, verb::Bool = false)
    verb && (println(), println(A))

    Ax = A * x
    Ax2 = similar(Ax)
    verb && println("forward preallocated")
    mul!(Ax2, A, x) #verify in-place linear operator works
    verb && @time mul!(Ax2, A, x)

    @test_throws ErrorException A'

    @test norm(Ax .- Ax2) <= 1.0e-8

    J = Jacobian(A, x)
    verb && println(J)

    grad = J' * y
    mul!(Ax2, A, x) #redo forward
    verb && println("adjoint jacobian mul! preallocated")
    grad2 = similar(grad)
    mul!(grad2, J', y) #verify in-place linear operator works
    verb && mul!(Ax2, A, x) #redo forward
    verb && @time mul!(grad2, J', y)

    @test norm(grad .- grad2) < 1.0e-8

    if eltype(grad) <: Real  # currently finite difference gradient not working with complex variables
        grad3 = gradient_fd(A, Ax, x, y) #calculate gradient using finite differences

        @test norm(grad .- grad3) < 5.0e-4
    end

    return Ax, grad
end

############# Finite Diff for Jacobian tests

function gradient_fd(
        op::A, y0::AbstractArray, x0::AbstractArray, r::AbstractArray
    ) where {A <: AbstractOperator}
    y = copy(y0)
    J = zeros(prod(size(op, 1)), prod(size(op, 2)))
    h = sqrt(eps())
    for i in axes(J, 2)
        x = copy(x0)
        x[i] = x[i] + h
        mul!(y, op, x)
        J[:, i] .= vec((y .- y0) ./ h)
    end
    return reshape(J' * vec(r), size(op, 2))
end

function gradient_fd(
        op::A, y0::AbstractArray, x0::ArrayPartition, r::AbstractArray
    ) where {A <: AbstractOperator}
    N = length(x0.x)
    y = copy(y0)
    grad = zero(x0)
    J = [zeros(prod(size(op, 1)), prod(sz2)) for sz2 in size(op, 2)]

    h = sqrt(eps())
    for ii in eachindex(J)
        for i in 1:size(J[ii], 2)
            x = deepcopy(x0)
            x.x[ii][i] = x.x[ii][i] + h
            mul!(y, op, x)
            J[ii][:, i] .= vec((y .- y0) ./ h)
        end
        grad.x[ii] .= reshape(J[ii]' * vec(r), size(op, 2)[ii])
    end
    return grad
end

function gradient_fd(
        op::A, y0::ArrayPartition, x0::AbstractArray, r::ArrayPartition
    ) where {A <: AbstractOperator}
    N = length(y0.x)
    grad = zero(x0)
    y = zero(y0)
    J = [zeros(prod(sz1), prod(size(op, 2))) for sz1 in size(op, 1)]

    h = sqrt(eps())
    for i in eachindex(x0)
        x = deepcopy(x0)
        x[i] = x[i] + h
        mul!(y, op, x)
        for ii in eachindex(J)
            J[ii][:, i] .= vec((y.x[ii] .- y0.x[ii]) ./ h)
        end
    end
    for ii in eachindex(J)
        grad .+= reshape(J[ii]' * r.x[ii], size(op, 2))
    end
    return grad
end

function gradient_fd(
        op::A, y0::ArrayPartition, x0::ArrayPartition, r::ArrayPartition
    ) where {A <: AbstractOperator}
    grad = zero(x0)
    y = zero(y0)
    M = length(x0.x)
    N = length(y0.x)
    J = [zeros(prod(size(op, 1)[i]), prod(size(op, 2)[ii])) for ii in 1:M, i in 1:N]

    h = sqrt(eps())
    for i in 1:M
        for iii in eachindex(x0.x[i])
            x = deepcopy(x0)
            x.x[i][iii] = x.x[i][iii] + h
            mul!(y, op, x)

            for ii in 1:N
                J[i, ii][:, iii] .= vec((y.x[ii] .- y0.x[ii]) ./ h)
            end
        end
    end

    for ii in 1:N, i in 1:M
        grad.x[i] .+= reshape(J[i, ii]' * r.x[ii], size(op, 2)[i])
    end
    return grad
end

end  # @testmodule TestUtils
