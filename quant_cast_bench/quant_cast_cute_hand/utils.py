"""Shared helpers for handwritten CuTe DSL quantization kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


def _ceil_div(
    num: int | cutlass.Int32 | cutlass.Int64,
    den: int | cutlass.Int32 | cutlass.Int64 | cutlass.Constexpr,
) -> int | cutlass.Int32 | cutlass.Int64:
    return (num + den - 1) // den


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
