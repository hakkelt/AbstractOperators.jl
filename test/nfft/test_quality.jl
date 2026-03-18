@testitem "Aqua" tags = [:quality, :nfft] begin
    using Aqua, NFFTOperators
    Aqua.test_all(NFFTOperators; ambiguities=false, stale_deps=false, persistent_tasks=false)
end
