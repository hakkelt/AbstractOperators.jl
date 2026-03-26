# Compile-time ndoms from operator type, for use in @generated constructors.
# Specializations for HCAT/VCAT/DCAT are added in their respective files.
_ndoms_from_type(::Type{<:AbstractOperator}, dim::Int) = 1

const thread_count_functions = Ref{Vector{Pair{Function, Function}}}(
    Pair{Function, Function}[
        BLAS.get_num_threads => BLAS.set_num_threads,
    ]
)

function set_thread_counts_expr(thread_count_expr, body_expr)
    return quote
        local prev_thread_counts = [pair.first() for pair in AbstractOperators.thread_count_functions[]]
        for pair in AbstractOperators.thread_count_functions[]
            pair.second($thread_count_expr)
        end
        local res
        try
            if $thread_count_expr == 1
                res = disable_polyester_threads() do
                    $(esc(body_expr))
                end
            else
                # Full threading enabled
                res = $(esc(body_expr))
            end
        finally
            # Restore previous thread counts
            for (i, pair) in enumerate(AbstractOperators.thread_count_functions[])
                pair.second(prev_thread_counts[i])
            end
        end
        res
    end
end

macro enable_full_threading(expr)
    return set_thread_counts_expr(nthreads(), expr)
end

macro restrict_threading(expr)
    return set_thread_counts_expr(1, expr)
end

_is_storage_compatible(a, ::Type{<:Array}) = a isa AbstractArray
_is_storage_compatible(a, ::Type{T}) where {T} = a isa T

@generated function _is_storage_compatible(a::ArrayPartition, ::Type{T}) where {T <: ArrayPartition}
    Tp = T.parameters[2]
    if !(Tp isa DataType && Tp <: Tuple)
        return :(a isa T)
    end
    types = Tp.parameters
    checks = [:(a.x[$i] isa $(types[i])) for i in 1:length(types)]
    cond = isempty(checks) ? :(true) : reduce((x, y) -> :($x && $y), checks)
    return :(length(a.x) == $(length(types)) && $cond)
end

_check_domain_storage(::ArrayPartition, op) = nothing
function _check_domain_storage(domain_array, op)
    return if !_is_storage_compatible(domain_array, domain_storage_type(op))
        throw(ArgumentError("Input must be an AbstractArray"))
    end
end

_check_codomain_storage(::ArrayPartition, op) = nothing
function _check_codomain_storage(codomain_array, op)
    return if !_is_storage_compatible(codomain_array, codomain_storage_type(op))
        throw(ArgumentError("Output must be an AbstractArray"))
    end
end

function check(codomain_array, op, domain_array)
    _check_domain_storage(domain_array, op)
    _check_codomain_storage(codomain_array, op)
    if (ndoms(op, 2) > 1) != (domain_array isa ArrayPartition)
        throw(ArgumentError("Input must be an ArrayPartition if and only if operator has multiple input domains"))
    end
    if domain_array isa ArrayPartition
        dtype = eltype.(domain_array.x)
    else
        dtype = eltype(domain_array)
    end
    # Use isequal instead of != to avoid Union{Missing,Bool} from tuple comparisons
    if !isequal(dtype, domain_type(op))
        throw(
            ArgumentError(
                "Input type $(dtype) does not match operator input type $(domain_type(op))",
            ),
        )
    end
    dim_in = domain_array isa ArrayPartition ? size.(domain_array.x) : size(domain_array)
    if !isequal(dim_in, size(op, 2))
        throw(
            DimensionMismatch(
                "Input size $(dim_in) does not match operator input size $(size(op, 2))",
            ),
        )
    end
    if (ndoms(op, 1) > 1) != (codomain_array isa ArrayPartition)
        throw(
            ArgumentError(
                "Output must be an ArrayPartition if and only if operator has multiple output domains",
            ),
        )
    end
    if codomain_array isa ArrayPartition
        dtype = eltype.(codomain_array.x)
    else
        dtype = eltype(codomain_array)
    end
    if !isequal(dtype, codomain_type(op))
        throw(
            ArgumentError(
                "Output type $(dtype) does not match operator output type $(codomain_type(op))",
            ),
        )
    end
    dim_out = codomain_array isa ArrayPartition ? size.(codomain_array.x) : size(codomain_array)
    return if !isequal(dim_out, size(op, 1))
        throw(
            DimensionMismatch(
                "Output size $(dim_out) does not match operator output size $(size(op, 1))",
            ),
        )
    end
end
