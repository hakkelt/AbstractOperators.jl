using TestItemRunner

function _backend_functional(mod::Symbol)
    try
        m = Base.require(Main, mod)
        return getproperty(m, :functional)()
    catch
        return false
    end
end

const HAS_CUDA = _backend_functional(:CUDA)
const HAS_AMDGPU = _backend_functional(:AMDGPU)
const FILTER = ti ->
    (!(:cuda in ti.tags) || HAS_CUDA) &&
    (!(:amdgpu in ti.tags) || HAS_AMDGPU)

@run_package_tests filter = FILTER
