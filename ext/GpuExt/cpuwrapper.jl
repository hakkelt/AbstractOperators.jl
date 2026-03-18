# GPU mul! overrides for CpuOperatorWrapper

# Forward: GPU → CPU → op → CPU → GPU
function mul!(y::AbstractGPUArray, A::CpuOperatorWrapper, x::AbstractGPUArray)
    check(y, A, x)
    copyto!(A.dom_buf, x)
    mul!(A.cod_buf, A.op, A.dom_buf)
    copyto!(y, A.cod_buf)
    return y
end

# Adjoint: GPU → CPU → op' → CPU → GPU
function mul!(
        y::AbstractGPUArray,
        Ac::AdjointOperator{<:CpuOperatorWrapper},
        x::AbstractGPUArray,
    )
    A = Ac.A
    check(y, Ac, x)
    copyto!(A.cod_buf, x)
    mul!(A.dom_buf, A.op', A.cod_buf)
    copyto!(y, A.dom_buf)
    return y
end
