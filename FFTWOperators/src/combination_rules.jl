# DCT and IDCT
can_be_combined(T1::IDCT, T2::DCT) = true
can_be_combined(T1::DCT, T2::IDCT) = true
can_be_combined(T1::DCT, T2::AdjointOperator{<:DCT}) = true
can_be_combined(T1::IDCT, T2::AdjointOperator{<:IDCT}) = true
can_be_combined(T1::AdjointOperator{<:DCT}, T2::IDCT) = true
can_be_combined(T1::AdjointOperator{<:IDCT}, T2::DCT) = true
can_be_combined(T1::AdjointOperator{<:DCT}, T2::AdjointOperator{<:IDCT}) = true
can_be_combined(T1::AdjointOperator{<:IDCT}, T2::AdjointOperator{<:DCT}) = true
combine(::CosineTransform, T2::CosineTransform) = Eye(allocate_in_domain(T2))
function combine(::CosineTransform, T2::AdjointOperator{<:CosineTransform})
    return Eye(allocate_in_domain(T2))
end
function combine(::AdjointOperator{<:CosineTransform}, T2::CosineTransform)
    return Eye(allocate_in_domain(T2))
end
function combine(
        ::AdjointOperator{<:CosineTransform}, T2::AdjointOperator{<:CosineTransform}
    )
    return Eye(allocate_in_domain(T2))
end

# DFT
can_be_combined(T1::DFT{N, C, D, Dir}, T2::AdjointOperator{<:DFT{N, C, D, Dir}}) where {N, C, D, Dir} = true
can_be_combined(T1::AdjointOperator{<:DFT{N, C, D, Dir}}, T2::DFT{N, C, D, Dir}) where {N, C, D, Dir} = true
function combine(T1::DFT{N, C, D, Dir}, T2::AdjointOperator{<:DFT}) where {N, C, D, Dir}
    scaling = _dft_scaling(T1.dim_in, Dir, FORWARD)
    scaling /= _dft_scaling(T1.dim_in, Dir, T1.normalization)
    scaling /= _dft_scaling(T1.dim_in, Dir, T2.A.normalization)
    return scaling * Eye(domain_type(T2), T1.dim_in; array_type = domain_array_type(T2))
end
function combine(T1::AdjointOperator{<:DFT}, T2::DFT{N, C, D, Dir}) where {N, C, D, Dir}
    scaling = _dft_scaling(T2.dim_in, Dir, FORWARD)
    scaling /= _dft_scaling(T2.dim_in, Dir, T1.A.normalization)
    scaling /= _dft_scaling(T2.dim_in, Dir, T2.normalization)
    return scaling * Eye(domain_type(T2), T2.dim_in; array_type = domain_array_type(T2))
end

# FFTShift/IFFTShift with DFT: the shift becomes a phase ramp on the other side of the DFT
# (`SignAlternation` when every shifted length is even, a `DiagOp` otherwise; see Shift.jl).
# `F ∘ S`: a shift of the DFT's input.
function _can_absorb_input_shift(F, S)
    return _ramp_absorbs(F, _shift_dirs(S), :domain)
end
function _absorb_input_shift(F, S)
    dirs = _shift_dirs(S)
    if _all_even(size(F, 2), dirs)
        return SignAlternation(codomain_type(F), size(F, 1), dirs; array_type = codomain_array_type(F)) * F
    end
    return _phase_ramp_op(allocate_in_codomain(F), dirs, n -> _shift_amount(S, n), _dft_sign(F)) * F
end
# `S ∘ F`: a shift of the DFT's output.
function _can_absorb_output_shift(S, F)
    dirs = _shift_dirs(S)
    return all(in(_dft_dirs(F)), dirs) && (_all_even(size(F, 1), dirs) || domain_type(F) <: Complex)
end
function _absorb_output_shift(S, F)
    dirs = _shift_dirs(S)
    if _all_even(size(F, 1), dirs)
        return F * SignAlternation(domain_type(F), size(F, 2), dirs; array_type = domain_array_type(F))
    end
    return F * _phase_ramp_op(allocate_in_domain(F), dirs, n -> _shift_amount(S, n), -_dft_sign(F))
end
# Every shifted length even and every shifted dimension transformed: the rules with a
# `SignAlternation` between the shift and the DFT below need both.
function _even_shift_next_to_dft(F, S)
    dirs = _shift_dirs(S)
    return all(in(_dft_dirs(F)), dirs) && _all_even(size(F, 1), dirs)
end

for TF in (DFT, AdjointOperator{<:DFT}), TS in (ShiftOp, AdjointOperator{<:ShiftOp})
    @eval begin
        can_be_combined(T1::$TF, T2::$TS) = _can_absorb_input_shift(T1, T2)
        can_be_combined(T1::$TS, T2::$TF) = _can_absorb_output_shift(T1, T2)
        combine(T1::$TF, T2::$TS) = _absorb_input_shift(T1, T2)
        combine(T1::$TS, T2::$TF) = _absorb_output_shift(T1, T2)
    end
end

# FFTShift/IFFTShift with DFT and SignAlternation
function can_be_combined(T1::ShiftOp, ::SignAlternation, T3::DFT)
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::DFT, ::SignAlternation, T3::ShiftOp)
    return _even_shift_next_to_dft(T1, T3)
end
function can_be_combined(T1::AdjointOperator{<:ShiftOp}, ::SignAlternation, T3::DFT)
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::DFT, ::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return _even_shift_next_to_dft(T1, T3)
end
function can_be_combined(T1::ShiftOp, ::SignAlternation, T3::AdjointOperator{<:DFT})
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::AdjointOperator{<:DFT}, ::SignAlternation, T3::ShiftOp)
    return _even_shift_next_to_dft(T1, T3)
end
function can_be_combined(T1::AdjointOperator{<:ShiftOp}, ::SignAlternation, T3::AdjointOperator{<:DFT})
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::AdjointOperator{<:DFT}, ::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return _even_shift_next_to_dft(T1, T3)
end
function combine(T1::ShiftOp, T2::SignAlternation, T3::DFT)
    return T2 * combine(T1, T3)
end
function combine(T1::DFT, T2::SignAlternation, T3::ShiftOp)
    return combine(T1, T3) * T2
end
function combine(T1::AdjointOperator{<:ShiftOp}, T2::SignAlternation, T3::DFT)
    return T2 * combine(T1, T3)
end
function combine(T1::DFT, T2::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return combine(T1, T3) * T2
end
function combine(T1::ShiftOp, T2::SignAlternation, T3::AdjointOperator{<:DFT})
    return T2 * combine(T1, T3)
end
function combine(T1::AdjointOperator{<:DFT}, T2::SignAlternation, T3::ShiftOp)
    return combine(T1, T3) * T2
end
function combine(T1::AdjointOperator{<:ShiftOp}, T2::SignAlternation, T3::AdjointOperator{<:DFT})
    return T2 * combine(T1, T3)
end
function combine(T1::AdjointOperator{<:DFT}, T2::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return combine(T1, T3) * T2
end

# FFTShift/IFFTShift with FFTShift/IFFTShift
have_shifted_dims_even_length(T::ShiftOp) = all(iseven, size(T, 2)[collect(T.dirs)])
are_shifted_dims_disjoint(T1::ShiftOp, T2::ShiftOp) =
    all(d -> !(d in T1.dirs && d in T2.dirs), 1:ndims(T1, 2))
does_fully_cover(T1::ShiftOp, T2::ShiftOp) = all(d -> (d in T1.dirs && d in T2.dirs), T1.dirs) # T1 fully covers T2
function can_be_combined(T1::FFTShift, T2::FFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || are_shifted_dims_disjoint(T1, T2)
end
function can_be_combined(T1::IFFTShift, T2::IFFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || are_shifted_dims_disjoint(T1, T2)
end
function can_be_combined(T1::FFTShift, T2::IFFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || does_fully_cover(T1, T2) || does_fully_cover(T2, T1)
end
function can_be_combined(T1::IFFTShift, T2::FFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || does_fully_cover(T1, T2) || does_fully_cover(T2, T1)
end
function combine(T1::FFTShift, T2::FFTShift)
    new_dirs = Tuple(d for d in 1:ndims(T1, 2) if (d in T1.dirs) || (d in T2.dirs))
    return FFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
end
function combine(T1::IFFTShift, T2::IFFTShift)
    new_dirs = Tuple(d for d in 1:ndims(T1, 2) if (d in T1.dirs) || (d in T2.dirs))
    return IFFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
end
function combine(T1::FFTShift, T2::IFFTShift)
    if does_fully_cover(T1, T2)
        new_dirs = Tuple(d for d in T1.dirs if !(d in T2.dirs))
        if isempty(new_dirs)
            return Eye(domain_type(T1), size(T1, 2); array_type = domain_array_type(T1))
        else
            return FFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
        end
    else
        new_dirs = Tuple(d for d in T2.dirs if !(d in T1.dirs))
        return IFFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
    end
end
function combine(T1::IFFTShift, T2::FFTShift)
    if does_fully_cover(T1, T2)
        new_dirs = Tuple(d for d in T1.dirs if !(d in T2.dirs))
        if isempty(new_dirs)
            return Eye(domain_type(T1), size(T1, 2); array_type = domain_array_type(T1))
        else
            return IFFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
        end
    else
        new_dirs = Tuple(d for d in T2.dirs if !(d in T1.dirs))
        return FFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
    end
end

# SignAlternation with SignAlternation
can_be_combined(T1::SignAlternation, T2::SignAlternation) = true
function combine(T1::SignAlternation, T2::SignAlternation)
    new_dirs = Tuple(d for d in 1:ndims(T1, 1) if (d in T1.dirs) != (d in T2.dirs))
    if isempty(new_dirs)
        return Eye(domain_type(T1), size(T1, 2); array_type = domain_array_type(T1))
    else
        return SignAlternation(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
    end
end

# SignAlternation with DiagOp
can_be_combined(::SignAlternation, T2::DiagOp) = diag(T2) isa AbstractArray
can_be_combined(T1::DiagOp, ::SignAlternation) = diag(T1) isa AbstractArray
function combine(T1::SignAlternation, T2::DiagOp)
    return DiagOp(domain_type(T2), size(T2, 2), T1 * diag(T2))
end
function combine(T1::DiagOp, T2::SignAlternation)
    return DiagOp(domain_type(T1), size(T1, 2), T2 * diag(T1))
end
