module GpuExt

using GPUArrays
using AbstractOperators
import LinearAlgebra: mul!
import AbstractOperators:
    _should_thread, storage_type_display_string, check,
    AdjointOperator, Variation, NoOperatorBroadCast,
    domain_type, allocate_in_domain, allocate_in_codomain,
    GetIndex, ZeroPad, CpuOperatorWrapper
using RecursiveArrayTools: ArrayPartition

include("properties.jl")
include("linearoperators.jl")
include("cpuwrapper.jl")
include("guards.jl")

end # module GpuExt
