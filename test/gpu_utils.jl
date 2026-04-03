@testmodule GpuTestUtils begin
    # JLArrays provides jl() to wrap arrays in a simulated GPU array (JLArray).
    # Loading JLArrays triggers the GPUArrays + KernelAbstractions extensions.
    using JLArrays
    using GPUArrays: AbstractGPUArray

    export jl, AbstractGPUArray
end
