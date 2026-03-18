module DSPOperators

using AbstractOperators, FFTW
import AbstractFFTs
import LinearAlgebra: mul!
import Base: size, ndims

import AbstractOperators:
    domain_type,
    codomain_type,
    check,
    fun_name,
    get_normal_op,
    allocate_in_domain,
    allocate_in_codomain,
    domain_storage_type,
    codomain_storage_type,
    is_full_column_rank,
    is_full_row_rank,
    is_thread_safe

# Capture AbstractFFTs generic functions for use in package extensions.
# Extensions cannot directly `import AbstractFFTs` (it's a dep of this package, not the extension),
# but they can access these via `import DSPOperators: _rfft, _irfft`.
const _rfft = rfft
const _irfft = irfft

include("Conv.jl")
include("Filt.jl")
include("MIMOFilt.jl")
include("Xcorr.jl")

end # module DSPOperators
