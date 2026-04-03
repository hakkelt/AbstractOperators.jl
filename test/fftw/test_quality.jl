@testitem "Aqua" tags = [:quality, :fftw] begin
    using Aqua, AbstractOperators, FFTW, FFTWOperators
    Aqua.test_all(
        FFTWOperators;
        piracies = false, ambiguities = false, stale_deps = false, persistent_tasks = false
    )
    Aqua.test_piracies(FFTWOperators; treat_as_own = [FFTW, AbstractOperators])
end
