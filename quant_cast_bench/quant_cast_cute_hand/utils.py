"""Shared helpers for handwritten CuTe DSL quantization kernels."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

from quant_cast_bench.quant_cast_cute.recipes import (
    _cvt_rn_satfinite_e2m1x2_f32_x4,
    _nvfp4_scale_e4m3,
    _philox_4x32,
)
from quant_cast_bench.quant_cast_cute_hand.blockscale_tma_plan import ScaleAlgo


COMPILE_CACHE: dict = {}


def _compiled(key, jit_fn, *cute_args):
    fn = COMPILE_CACHE.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, *cute_args)
        COMPILE_CACHE[key] = fn
    return fn


def _ceil_div(
    num: int | cutlass.Int32 | cutlass.Int64,
    den: int | cutlass.Int32 | cutlass.Int64 | cutlass.Constexpr,
) -> int | cutlass.Int32 | cutlass.Int64:
    return (num + den - 1) // den


def _validate_nvfp4_swizzle_inputs(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    kernel_name: str,
    k_multiple: int,
) -> tuple[int, int]:
    assert input.dim() == 2, f"{kernel_name} requires a 2-D input"
    assert input.is_contiguous(), f"{kernel_name} requires contiguous input"
    assert input.dtype == torch.bfloat16, f"{kernel_name} is bf16-only"
    assert outer_scale.device == input.device, (
        "input and outer scale must be on the same device"
    )
    assert outer_scale.dtype == torch.float32 and outer_scale.numel() == 1, (
        "outer scale must be a float32 scalar"
    )
    M, K = input.shape
    assert M > 0 and K > 0, f"{kernel_name} requires non-empty dimensions"
    assert K % k_multiple == 0, (
        f"{kernel_name} requires K % {k_multiple} == 0"
    )
    return M, K


def _allocate_nvfp4_swizzle_outputs(
    input: torch.Tensor,
    M: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    nrb, ncb = _ceil_div(M, 128), _ceil_div(K, 64)
    output = torch.empty(M, K // 2, dtype=torch.uint8, device=input.device)
    scale = torch.empty(
        nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device
    )
    return output, scale, nrb, ncb


@dsl_user_op
def view_as(
    x: cutlass.Numeric,
    dtype: type[cutlass.Numeric],
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """Bitcast one scalar to another scalar of equal width."""
    assert type(x).width == dtype.width
    # bitcast wants a signed IR type even for unsigned CUTLASS types.
    dst_type = (
        T.i(dtype.width)
        if ir.IntegerType.isinstance(dtype.mlir_type)
        else dtype.mlir_type
    )
    return dtype(
        arith.bitcast(dst_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def unpack(
    x: cutlass.Numeric,
    dtype: type[cutlass.Numeric],
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[cutlass.Numeric, ...]:
    """Unpack an integer carrier into a tuple of scalar values."""
    x = cute.typing.as_numeric(x)
    carrier_dtype = type(x)
    assert ir.IntegerType.isinstance(carrier_dtype.mlir_type)
    assert carrier_dtype.width % dtype.width == 0
    num_lanes = carrier_dtype.width // dtype.width
    # Integer vector lanes avoid a compiler crash with vector<N x FP8> (NVIDIA/cutlass#3342).
    lanes = llvm.bitcast(
        T.vector(num_lanes, T.i(dtype.width)),
        x.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return tuple(
        view_as(
            cute.typing.as_numeric(
                vector.extract(
                    lanes,
                    dynamic_position=[],
                    static_position=[i],
                    loc=loc,
                    ip=ip,
                )
            ),
            dtype,
            loc=loc,
            ip=ip,
        )
        for i in range(num_lanes)
    )


@dsl_user_op
def pack(
    *values: cutlass.Numeric,
    carrier: type[cutlass.Numeric] | None = None,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """Pack same-typed scalar values into an integer carrier."""
    assert len(values) > 0
    lane_dtype = type(values[0])
    assert all(type(value) is lane_dtype for value in values)
    lane_type = T.i(lane_dtype.width)
    lanes = vector.from_elements(
        T.vector(len(values), lane_type),
        tuple(
            arith.bitcast(
                lane_type,
                value.ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
            for value in values
        ),
        loc=loc,
        ip=ip,
    )
    packed_width = len(values) * lane_dtype.width
    packed = llvm.bitcast(T.i(packed_width), lanes, loc=loc, ip=ip)
    if carrier is None:
        return cute.typing.as_numeric(packed)
    assert ir.IntegerType.isinstance(carrier.mlir_type)
    assert carrier.width == packed_width
    return carrier(packed)


@dsl_user_op
def _cvt_f32_to_ue8m0(
    x: cutlass.Float32,
    *,
    rounding_mode: nvvm.FPRoundingMode,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float8E8M0FNU:
    """Convert x to a single E8M0 value without saturation (high input 0.0)."""
    packed = nvvm.cvt_packfloat_f32(
        cutlass.Float32(0.0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(x).ir_value(loc=loc, ip=ip),
        cutlass.Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        rnd=rounding_mode,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc,
        ip=ip,
    )
    return unpack(
        cutlass.Int32(packed),
        cutlass.Float8E8M0FNU,
        loc=loc,
        ip=ip,
    )[0]


@dsl_user_op
def _cvt_ue8m0_to_f32(
    x: cutlass.Float8E8M0FNU,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    """Convert one E8M0 value to FP32 through the supported BF16 path."""
    x_e8m0x2 = pack(x, cutlass.Float8E8M0FNU(0), loc=loc, ip=ip)
    x_u32 = llvm.zext(
        T.i32(), x_e8m0x2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    bf16x2_bits = nvvm.cvt_packfloat(
        x_u32,
        cutlass.Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        nvvm.CVTPackFloatKind.BF16x2,
        rnd=nvvm.FPRoundingMode.RN,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc,
        ip=ip,
    )
    low_bf16 = unpack(
        cutlass.Int32(bf16x2_bits),
        cutlass.BFloat16,
        loc=loc,
        ip=ip,
    )[0]
    return low_bf16.to(cutlass.Float32)


@cute.jit
def _reciprocal_scale(
    scale_e8m0: cutlass.Float8E8M0FNU,
) -> cutlass.Float32:
    scale_biased = view_as(scale_e8m0, cutlass.Uint8)
    reciprocal_biased = cutlass.Uint8(254) - scale_biased
    return _cvt_ue8m0_to_f32(
        view_as(reciprocal_biased, cutlass.Float8E8M0FNU)
    )


@cute.jit
def _e8m0(amax: cutlass.Float32) -> tuple[cutlass.Float32, cutlass.Uint8]:
    """Return the reciprocal FP32 scale and RCEIL E8M0 scale byte for an amax."""
    descale = amax * cutlass.Float32(1.0 / 448.0)
    scale_e8m0 = _cvt_f32_to_ue8m0(
        descale, rounding_mode=nvvm.FPRoundingMode.RP
    )
    rcp = _reciprocal_scale(scale_e8m0)
    return rcp, view_as(scale_e8m0, cutlass.Uint8)


@cute.jit
def _e8m0_with_max_pos(
    amax: cutlass.Float32,
    max_pos: cutlass.Constexpr,
) -> tuple[cutlass.Float32, cutlass.Uint8]:
    """Return an RCEIL E8M0 scale for a caller-specified element-format maximum."""
    descale = amax * cutlass.Float32(1.0 / max_pos)
    scale_e8m0 = _cvt_f32_to_ue8m0(
        descale, rounding_mode=nvvm.FPRoundingMode.RP
    )
    rcp = _reciprocal_scale(scale_e8m0)
    return rcp, view_as(scale_e8m0, cutlass.Uint8)


@dsl_user_op
def _cvt_rs_satfinite_e4m3x4_f32(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    rbits: cutlass.Uint32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Uint32:
    """Stochastically round four FP32 values to four packed E4M3 bytes."""
    # PTX places its first source in the high byte. Reverse the sources so the little-endian byte
    # view is [e4m3(v0), e4m3(v1), e4m3(v2), e4m3(v3)].
    args = [
        cutlass.Float32(v3).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v2).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
        cutlass.Uint32(rbits).ir_value(loc=loc, ip=ip),
    ]
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            "cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;",
            "=r,f,f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _e8m0_scale_store_as_uint(
    mScaleLogical: cute.Tensor,
    rScale: cute.Tensor,
    row: cutlass.Int32,
    col: cutlass.Int32,
    count: cutlass.Constexpr,
) -> None:
    """Store adjacent E8M0 bytes with one naturally sized integer write.

    Args:
        mScaleLogical: Global scale tensor with the logical-to-blocked layout.
        rScale: Register tensor holding the E8M0 byte values.
        row: Logical destination row.
        col: First logical destination column, aligned to ``count``.
        count: Compile-time store width in bytes; must be 1, 2, or 4.

    The explicit integer view preserves packed stores when dynamic outer extents
    prevent ``cute.copy`` from proving alignment and vectorizing the write.
    """
    flat = mScaleLogical.layout((row, col))
    # Dynamic outer extents hide this alignment from cute.copy, so use a typed packed view.
    if cutlass.const_expr(count == 1):
        mScaleLogical[(row, col)] = rScale[0]
    elif cutlass.const_expr(count == 2):
        mScalePacked = cute.make_tensor(
            cute.recast_ptr(mScaleLogical.iterator, dtype=cutlass.Uint16),
            cute.make_layout(cute.size(mScaleLogical) // 2),
        )
        rScalePacked = cute.recast_tensor(rScale, dtype=cutlass.Uint16)
        mScalePacked[flat // 2] = rScalePacked[0]
    else:
        assert count == 4, f"unsupported E8M0 scale store count: {count}"
        mScalePacked = cute.make_tensor(
            cute.recast_ptr(mScaleLogical.iterator, dtype=cutlass.Uint32),
            cute.make_layout(cute.size(mScaleLogical) // 4),
        )
        rScalePacked = cute.recast_tensor(rScale, dtype=cutlass.Uint32)
        mScalePacked[flat // 4] = rScalePacked[0]


_NVFP4_GROUP = 16
_NVFP4_DIRECT_QVPT = 8
_NVFP4_DIRECT_HALF = 8


@dsl_user_op
def _cvt_rs_satfinite_e2m1x4_f32_x8(
    v0, v1, v2, v3, v4, v5, v6, v7, rbits0, rbits1, *, loc=None, ip=None
):
    """Pack eight f32 values into one u32 with two Blackwell E2M1x4 SR conversions."""
    args = [
        cutlass.Float32(value).ir_value(loc=loc, ip=ip)
        for value in (v0, v1, v2, v3, v4, v5, v6, v7)
    ]
    args.extend(
        [
            cutlass.Uint32(rbits0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(rbits1).ir_value(loc=loc, ip=ip),
        ]
    )
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "cvt.rs.satfinite.e2m1x4.f32 lo, {$4, $3, $2, $1}, $9;\n\t"
            "cvt.rs.satfinite.e2m1x4.f32 hi, {$8, $7, $6, $5}, $10;\n\t"
            "mov.b32 $0, {lo, hi};\n\t"
            "}",
            "=r,f,f,f,f,f,f,f,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _mxfp8_v2_quantize_stochastic_x32(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    counter_start: cutlass.Uint64,
    philox_k0: cutlass.Uint32,
    philox_k1: cutlass.Uint32,
) -> cute.TensorSSA:
    """Quantize one contiguous 32-value output run with two Philox counters."""
    scaled = values * rcp
    qwords = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Uint32)
    for half in cutlass.range_constexpr(2):
        ctr = counter_start + cutlass.Uint64(half)
        c0 = cutlass.Uint32(ctr & cutlass.Uint64(0xFFFFFFFF))
        c1 = cutlass.Uint32(ctr >> 32)
        zero = cutlass.Uint32(0)
        r0, r1, r2, r3 = _philox_4x32(
            c0, c1, zero, zero, philox_k0, philox_k1
        )
        value = half * 16
        word = half * 4
        qwords[word + 0] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 0],
            scaled[value + 1],
            scaled[value + 2],
            scaled[value + 3],
            r0,
        )
        qwords[word + 1] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 4],
            scaled[value + 5],
            scaled[value + 6],
            scaled[value + 7],
            r1,
        )
        qwords[word + 2] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 8],
            scaled[value + 9],
            scaled[value + 10],
            scaled[value + 11],
            r2,
        )
        qwords[word + 3] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 12],
            scaled[value + 13],
            scaled[value + 14],
            scaled[value + 15],
            r3,
        )
    return cute.recast_tensor(qwords, dtype=cutlass.Float8E4M3FN).load()


@cute.jit
def _quantize_fp4_rtne(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    value_count: cutlass.Constexpr,
) -> cute.TensorSSA:
    """Quantize an x16 or x32 FP32 group into packed E2M1 bytes with RTNE."""
    scaled = values * rcp
    qwords = cute.make_rmem_tensor(
        cute.make_layout(value_count // 8), cutlass.Uint32
    )
    for chunk in cutlass.range_constexpr(value_count // 8):
        offset = chunk * 8
        qwords[chunk] = _cvt_rn_satfinite_e2m1x2_f32_x4(
            scaled[offset + 0],
            scaled[offset + 1],
            scaled[offset + 2],
            scaled[offset + 3],
            scaled[offset + 4],
            scaled[offset + 5],
            scaled[offset + 6],
            scaled[offset + 7],
        )
    return cute.recast_tensor(qwords, dtype=cutlass.Uint8).load()


@cute.jit
def _quantize_fp4_stochastic_x16(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    sr_counter: cutlass.Uint64,
    philox_k0: cutlass.Uint32,
    philox_k1: cutlass.Uint32,
) -> cute.TensorSSA:
    """Quantize 16 scaled FP32 values with one Philox counter and native E2M1 SR."""
    c0 = cutlass.Uint32(sr_counter & cutlass.Uint64(0xFFFFFFFF))
    c1 = cutlass.Uint32(sr_counter >> 32)
    zero = cutlass.Uint32(0)
    r0, r1, r2, r3 = _philox_4x32(
        c0, c1, zero, zero, philox_k0, philox_k1
    )
    qwords = cute.make_rmem_tensor(2, cutlass.Uint32)
    qwords[0] = _cvt_rs_satfinite_e2m1x4_f32_x8(
        values[0] * rcp,
        values[1] * rcp,
        values[2] * rcp,
        values[3] * rcp,
        values[4] * rcp,
        values[5] * rcp,
        values[6] * rcp,
        values[7] * rcp,
        r0,
        r2,
    )
    qwords[1] = _cvt_rs_satfinite_e2m1x4_f32_x8(
        values[8] * rcp,
        values[9] * rcp,
        values[10] * rcp,
        values[11] * rcp,
        values[12] * rcp,
        values[13] * rcp,
        values[14] * rcp,
        values[15] * rcp,
        r1,
        r3,
    )
    return cute.recast_tensor(qwords, dtype=cutlass.Uint8).load()


@cute.jit
def _blockscaled_quantize_group(
    values: cute.TensorSSA,
    outer,
    value_count: cutlass.Constexpr,
    qdata_dtype: cutlass.Constexpr,
    scale_algo: cutlass.Constexpr,
    is_square_scaling: cutlass.Constexpr,
    is_stochastic_qdata_rounding: cutlass.Constexpr,
    sr_counter: cutlass.Uint64 | None,
    philox_k0: cutlass.Uint32 | None,
    philox_k1: cutlass.Uint32 | None,
):
    """Calculate one block scale and quantize the values that share it."""
    if cutlass.const_expr(is_stochastic_qdata_rounding):
        assert sr_counter is not None
        assert philox_k0 is not None
        assert philox_k1 is not None
    else:
        assert sr_counter is None
        assert philox_k0 is None
        assert philox_k1 is None

    amax = cute.math.absf(values).reduce(
        cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
    )
    if cutlass.const_expr(is_square_scaling):
        amax = cute.arch.warp_reduction_max(amax)

    # ScaleAlgo selects only the scale calculation. The surrounding data path is shared.
    if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
        scale, reciprocal = _nvfp4_scale_e4m3(amax, outer, outer)
    elif cutlass.const_expr(qdata_dtype == cutlass.Float4E2M1FN):
        reciprocal, scale = _e8m0_with_max_pos(amax, 6.0)
    else:
        reciprocal, scale = _e8m0(amax)

    if cutlass.const_expr(qdata_dtype == cutlass.Float4E2M1FN):
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            qdata = _quantize_fp4_stochastic_x16(
                values, reciprocal, sr_counter, philox_k0, philox_k1
            )
        else:
            qdata = _quantize_fp4_rtne(values, reciprocal, value_count)
    elif cutlass.const_expr(is_stochastic_qdata_rounding):
        qdata = _mxfp8_v2_quantize_stochastic_x32(
            values, reciprocal, sr_counter, philox_k0, philox_k1
        )
    else:
        qdata = (values * reciprocal).to(cutlass.Float8E4M3FN)
    return qdata, scale


@cute.jit
def _nvfp4_quantize_x16(values, outer):
    """Quantize one 1x16 NVFP4 block and return eight packed bytes plus its E4M3 scale byte."""
    amax = cute.math.absf(values).reduce(
        cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
    )
    inner_e4m3, reciprocal = _nvfp4_scale_e4m3(amax, outer, outer)
    qwords = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Uint32)
    for half in cutlass.range_constexpr(2):
        offset = half * 8
        qwords[half] = _cvt_rn_satfinite_e2m1x2_f32_x4(
            values[offset + 0] * reciprocal,
            values[offset + 1] * reciprocal,
            values[offset + 2] * reciprocal,
            values[offset + 3] * reciprocal,
            values[offset + 4] * reciprocal,
            values[offset + 5] * reciprocal,
            values[offset + 6] * reciprocal,
            values[offset + 7] * reciprocal,
        )
    return cute.recast_tensor(qwords, dtype=cutlass.Uint8).load(), inner_e4m3


@cute.jit
def _nvfp4_quantize_stochastic_x16(values, outer, sr_counter, k0, k1):
    """NVFP4 x16 quantization using one Philox counter and native E2M1x4 SR conversions."""
    amax = cute.math.absf(values).reduce(
        cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
    )
    inner_e4m3, reciprocal = _nvfp4_scale_e4m3(amax, outer, outer)
    c0 = cutlass.Uint32(sr_counter & cutlass.Uint64(0xFFFFFFFF))
    c1 = cutlass.Uint32(sr_counter >> 32)
    zero = cutlass.Uint32(0)
    r0, r1, r2, r3 = _philox_4x32(c0, c1, zero, zero, k0, k1)
    # prng.bits exposes each Philox counter in [r0, r2, r1, r3] order. Feed one word to each
    # consecutive group of four values so this specialization bit-matches the eager gold.
    qwords = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Uint32)
    qwords[0] = _cvt_rs_satfinite_e2m1x4_f32_x8(
        values[0] * reciprocal,
        values[1] * reciprocal,
        values[2] * reciprocal,
        values[3] * reciprocal,
        values[4] * reciprocal,
        values[5] * reciprocal,
        values[6] * reciprocal,
        values[7] * reciprocal,
        r0,
        r2,
    )
    qwords[1] = _cvt_rs_satfinite_e2m1x4_f32_x8(
        values[8] * reciprocal,
        values[9] * reciprocal,
        values[10] * reciprocal,
        values[11] * reciprocal,
        values[12] * reciprocal,
        values[13] * reciprocal,
        values[14] * reciprocal,
        values[15] * reciprocal,
        r1,
        r3,
    )
    return cute.recast_tensor(qwords, dtype=cutlass.Uint8).load(), inner_e4m3


@dsl_user_op
def _cvt_rn_satfinite_e4m3x2_f32_packed(hi, lo, *, loc=None, ip=None):
    """Convert two independent f32 scales to one packed E4M3x2 value."""
    return cutlass.Uint16(
        llvm.inline_asm(
            T.i16(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_e4m3x2_to_f32x2(packed, *, loc=None, ip=None):
    """Decode both bytes of a packed E4M3x2 value with one hardware conversion."""
    f16x2 = cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint16(packed).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.e4m3x2 $0, $1;",
            "=r,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    lo_bits = cutlass.Uint16(f16x2 & cutlass.Uint32(0xFFFF))
    hi_bits = cutlass.Uint16(f16x2 >> cutlass.Uint32(16))
    lo = cutlass.Float16(
        llvm.bitcast(T.f16(), lo_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    hi = cutlass.Float16(
        llvm.bitcast(T.f16(), hi_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    return lo.to(cutlass.Float32), hi.to(cutlass.Float32)


@cute.jit
def _nvfp4_scale_e4m3_fast_x2(amax0, amax1, outer):
    """Generate and decode two inner scales with paired E4M3x2 instructions."""
    local0 = cutlass.min(
        cutlass.max(
            (amax0 * (1.0 / 6.0)) * outer,
            cutlass.Float32(1.0 / (1 << 9)),
        ),
        cutlass.Float32(448.0),
    )
    local1 = cutlass.min(
        cutlass.max(
            (amax1 * (1.0 / 6.0)) * outer,
            cutlass.Float32(1.0 / (1 << 9)),
        ),
        cutlass.Float32(448.0),
    )
    packed = _cvt_rn_satfinite_e4m3x2_f32_packed(local1, local0)
    scale0 = cutlass.Uint8(packed & cutlass.Uint16(0xFF))
    scale1 = cutlass.Uint8(packed >> cutlass.Uint16(8))
    decoded0, decoded1 = _cvt_e4m3x2_to_f32x2(packed)
    reciprocal0 = outer * cute.arch.rcp_approx(decoded0)
    reciprocal1 = outer * cute.arch.rcp_approx(decoded1)
    return scale0, reciprocal0, scale1, reciprocal1


@cute.jit
def _nvfp4_quantize_fast_groups(
    rValues: cute.Tensor,
    rQ: cute.Tensor,
    rScale: cute.Tensor,
    outer,
    counter_start,
    k0,
    k1,
    group_count: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    """Quantize several x16 groups with scale work hoisted ahead of qdata conversion."""
    rValueGroups = cute.tiled_divide(rValues, (_NVFP4_GROUP,))
    rQGroups = cute.tiled_divide(rQ, (_NVFP4_DIRECT_QVPT,))
    rReciprocal = cute.make_rmem_tensor(group_count, cutlass.Float32)
    for pair in cutlass.range_constexpr(group_count // 2):
        group0 = pair * 2
        group1 = group0 + 1
        values0 = rValueGroups[(None, group0)].load()
        values1 = rValueGroups[(None, group1)].load()
        amax0 = cute.math.absf(values0).reduce(
            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
        )
        amax1 = cute.math.absf(values1).reduce(
            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
        )
        scale0, reciprocal0, scale1, reciprocal1 = (
            _nvfp4_scale_e4m3_fast_x2(amax0, amax1, outer)
        )
        rScale[group0] = scale0
        rScale[group1] = scale1
        rReciprocal[group0] = reciprocal0
        rReciprocal[group1] = reciprocal1

    for group in cutlass.range_constexpr(group_count):
        values = rValueGroups[(None, group)].load()
        reciprocal = rReciprocal[group]
        qwords = cute.make_rmem_tensor(2, cutlass.Uint32)
        if cutlass.const_expr(stochastic):
            sr_counter = counter_start + cutlass.Uint64(group)
            c0 = cutlass.Uint32(sr_counter & cutlass.Uint64(0xFFFFFFFF))
            c1 = cutlass.Uint32(sr_counter >> 32)
            zero = cutlass.Uint32(0)
            r0, r1, r2, r3 = _philox_4x32(c0, c1, zero, zero, k0, k1)
            qwords[0] = _cvt_rs_satfinite_e2m1x4_f32_x8(
                values[0] * reciprocal,
                values[1] * reciprocal,
                values[2] * reciprocal,
                values[3] * reciprocal,
                values[4] * reciprocal,
                values[5] * reciprocal,
                values[6] * reciprocal,
                values[7] * reciprocal,
                r0,
                r2,
            )
            qwords[1] = _cvt_rs_satfinite_e2m1x4_f32_x8(
                values[8] * reciprocal,
                values[9] * reciprocal,
                values[10] * reciprocal,
                values[11] * reciprocal,
                values[12] * reciprocal,
                values[13] * reciprocal,
                values[14] * reciprocal,
                values[15] * reciprocal,
                r1,
                r3,
            )
        else:
            for half in cutlass.range_constexpr(2):
                offset = half * 8
                qwords[half] = _cvt_rn_satfinite_e2m1x2_f32_x4(
                    values[offset + 0] * reciprocal,
                    values[offset + 1] * reciprocal,
                    values[offset + 2] * reciprocal,
                    values[offset + 3] * reciprocal,
                    values[offset + 4] * reciprocal,
                    values[offset + 5] * reciprocal,
                    values[offset + 6] * reciprocal,
                    values[offset + 7] * reciprocal,
                )
        rQGroups[(None, group)].store(
            cute.recast_tensor(qwords, dtype=cutlass.Uint8).load()
        )


@cute.jit
def _nvfp4_load_philox_key(mSeed: cute.Tensor):
    """Load one PyTorch Philox key and return its two key words plus counter base."""
    frgKey = cute.make_rmem_tensor(cute.make_layout(2), mSeed.element_type)
    cute.copy(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mSeed.element_type),
        mSeed,
        frgKey,
    )
    key64 = cute.recast_tensor(frgKey, dtype=cutlass.Uint64)
    return (
        cutlass.Uint32(key64[0] & cutlass.Uint64(0xFFFFFFFF)),
        cutlass.Uint32(key64[0] >> 32),
        key64[1],
    )


@cute.jit
def _nvfp4_rht_fwht_x16(
    sInput: cute.Tensor,
    row_start,
    local_col,
    sScaledSigns: cute.Tensor,
):
    """Structured RHT: sign flips followed by a four-stage in-register FWHT."""
    transformed = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.Float32)
    # Fold the exact power-of-two normalization into the signs, and consume the signed inputs
    # directly in the first butterfly stage instead of materializing a separate intermediate.
    for pair in cutlass.range_constexpr(8):
        lo, hi = pair * 2, pair * 2 + 1
        a = sInput[(row_start + lo, local_col)].to(
            cutlass.Float32
        ) * sScaledSigns[lo].to(cutlass.Float32)
        b = sInput[(row_start + hi, local_col)].to(
            cutlass.Float32
        ) * sScaledSigns[hi].to(cutlass.Float32)
        transformed[lo], transformed[hi] = a + b, a - b
    for block in cutlass.range_constexpr(4):
        base = block * 4
        for offset in cutlass.range_constexpr(2):
            lo, hi = base + offset, base + offset + 2
            a, b = transformed[lo], transformed[hi]
            transformed[lo], transformed[hi] = a + b, a - b
    for block in cutlass.range_constexpr(2):
        base = block * 8
        for offset in cutlass.range_constexpr(4):
            lo, hi = base + offset, base + offset + 4
            a, b = transformed[lo], transformed[hi]
            transformed[lo], transformed[hi] = a + b, a - b
    for offset in cutlass.range_constexpr(8):
        a, b = transformed[offset], transformed[offset + 8]
        transformed[offset], transformed[offset + 8] = a + b, a - b

    return transformed.load()


@cute.jit
def _store_swizzled_scale_groups_as_uint(
    mScaleLogical: cute.Tensor,
    rScale: cute.Tensor,
    row,
    scale_col,
    group_count: cutlass.Constexpr,
):
    """Store a thread's adjacent scale bytes using their natural packed width."""
    if cutlass.const_expr(group_count in (1, 2, 4)):
        _e8m0_scale_store_as_uint(
            mScaleLogical, rScale, row, scale_col, group_count
        )
    else:
        assert group_count % 4 == 0
        rScalePacks = cute.tiled_divide(rScale, (4,))
        for pack in cutlass.range_constexpr(group_count // 4):
            _e8m0_scale_store_as_uint(
                mScaleLogical,
                rScalePacks[(None, pack)],
                row,
                scale_col + pack * 4,
                4,
            )
