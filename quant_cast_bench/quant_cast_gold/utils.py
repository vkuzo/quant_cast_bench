"""Sub-byte float (fp4 e2m1) conversion and 4-bit packing helpers.

Ported from flexquant v1 nvfp4_utils.py (itself copied from torchao custom_fp_utils.py /
mx_formats/kernels.py). fp4 packs two 4-bit values per byte, stored as
torch.float4_e2m1fn_x2. Independent of flex_tile_map (see quant_cast_gold/recipes.py).
"""

import torch

_EBITS_F32, _MBITS_F32 = 8, 23
_F32_EXP_BIAS = (1 << (_EBITS_F32 - 1)) - 1
_EBITS_F4, _MBITS_F4 = 2, 1


def f32_to_f4_unpacked(x):
    """FP32 -> fp4 e2m1, RNE, saturating. uint8 with bits 4-7 holding the code."""
    ebits, mbits = _EBITS_F4, _MBITS_F4
    exp_bias = (1 << (ebits - 1)) - 1
    max_int = (1 << (ebits + mbits)) - 1
    sign_mask = 1 << (ebits + mbits)
    magic_adder = (1 << (_MBITS_F32 - mbits - 1)) - 1
    max_normal = 2 ** ((1 << ebits) - 1 - exp_bias) * (((1 << (mbits + 1)) - 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (_MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    return (x | sign_lp).to(torch.uint8)


def f4_unpacked_to_f32(x):
    """fp4 e2m1 (uint8, code in bits 0-3) -> FP32. mbits==1 fast path only."""
    ebits, mbits = _EBITS_F4, _MBITS_F4
    sign_mask = 1 << (ebits + mbits)
    exp_bias = (1 << (ebits - 1)) - 1
    mantissa_mask = (1 << mbits) - 1

    sign_lp = x & sign_mask
    x_pos = x ^ sign_lp
    zero_mask = x_pos == 0
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = (exp_biased_lp - exp_bias + _F32_EXP_BIAS).to(torch.int32) << _MBITS_F32
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (_MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    result[zero_mask] = 0
    denormal_exp_biased = 1 - exp_bias + _F32_EXP_BIAS
    result[denormal_mask] = (denormal_exp_biased - mbits) << _MBITS_F32

    sign_f32 = sign_lp.to(torch.int32) << (_MBITS_F32 - mbits + _EBITS_F32 - ebits)
    result = result | sign_f32
    return result.view(torch.float)


def pack_uint4(uint8_data):
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed = uint8_data[::2] | (uint8_data[1::2] << 4)
    return packed.view(*shape[:-1], shape[-1] // 2)


def unpack_uint4(uint8_data):
    shape = uint8_data.shape
    first = (uint8_data & 0b1111).to(torch.uint8)
    second = (uint8_data >> 4).to(torch.uint8)
    return torch.stack([first, second], dim=-1).view(*shape[:-1], shape[-1] * 2)
