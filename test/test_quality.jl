@testitem "Documentation" tags = [:quality] begin
    using Documenter, AbstractOperators
    # Eagerly load weak-dep extensions so they're precompiled before the doctest
    # sandbox runs `using AbstractOperators, LinearMaps` (avoids stray precompile output).
    using LinearMaps
    DocMeta.setdocmeta!(
        AbstractOperators,
        :DocTestSetup,
        :(using AbstractOperators);
        recursive = true,
    )
    doctest(AbstractOperators; fix = false)
end

@testitem "Aqua" tags = [:quality] begin
    using Aqua, AbstractOperators

    Aqua.test_all(AbstractOperators; ambiguities=false, stale_deps=false, persistent_tasks=false)
end
