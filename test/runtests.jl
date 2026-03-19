using TestItemRunner

if Base.find_package("CUDA") !== nothing
    @eval import CUDA
    const HAS_CUDA = CUDA.functional()
else
    const HAS_CUDA = false
end

if Base.find_package("AMDGPU") !== nothing
    @eval import AMDGPU
    const HAS_AMDGPU = AMDGPU.functional()
else
    const HAS_AMDGPU = false
end
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
