@testitem "Aqua" tags = [:quality, :wavelet] begin
    using Aqua, WaveletOperators
    Aqua.test_all(WaveletOperators; ambiguities = false, stale_deps = false, persistent_tasks = false)
end
