@testitem "Aqua" tags = [:quality, :dsp] begin
    using DSPOperators, Aqua
    Aqua.test_all(DSPOperators; ambiguities = false, stale_deps = false, persistent_tasks = false)
end
