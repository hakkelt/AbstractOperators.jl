using TestItemRunner, CUDA, AMDGPU

const VERB = get(ENV, "ABSTRACTOPERATORS_TEST_VERBOSE", "false") == "true"
const FILTER = if length(ARGS) > 0
    @assert length(ARGS) == 1
    parts = split(ARGS[1], ",")
    tags = map(p -> Symbol(p[2:end]), filter(x -> startswith(x, ":"), parts))
    names = filter(x -> !startswith(x, ":"), parts)
    ti -> begin
        run_item = any(t -> t in ti.tags, tags) || any(n -> n == ti.name, names)
        if VERB && run_item
            println("Running @testitem: ", ti.name)
        end
        run_item
    end
else
    ti -> begin
        run_item = (!(:cuda in ti.tags) || CUDA.functional()) &&
                (!(:amdgpu in ti.tags) || AMDGPU.functional())
        if VERB && run_item
            println("Running @testitem: ", ti.name)
        end
        run_item
    end
end

@run_package_tests filter = FILTER
