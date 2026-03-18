using TestItemRunner

try
    import CUDA
catch
end
try
    import AMDGPU
catch
end

const HAS_CUDA = (@isdefined CUDA) && CUDA.functional()
const HAS_AMDGPU = (@isdefined AMDGPU) && AMDGPU.functional()
const VERB = get(ENV, "ABSTRACTOPERATORS_TEST_VERBOSE", "false") == "true"
const FILTER = ti -> begin
    run_item = (!(:cuda in ti.tags) || HAS_CUDA) &&
               (!(:amdgpu in ti.tags) || HAS_AMDGPU)
    if VERB && run_item
        println("Running @testitem: ", ti.name)
    end
    run_item
end

@run_package_tests filter = FILTER
