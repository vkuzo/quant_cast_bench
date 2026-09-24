"""Sub-byte float (fp4 e2m1) conversion and 4-bit packing helpers.

Ported from flexquant v1 nvfp4_utils.py (itself copied from torchao custom_fp_utils.py /
mx_formats/kernels.py). fp4 packs two 4-bit values per byte, stored as
torch.float4_e2m1fn_x2. Independent of flex_tile_map (see quant_cast_gold/recipes.py).
"""

import torch

_EBITS_F32, _MBITS_F32 = 8, 23
_F32_EXP_BIAS = (1 << (_EBITS_F32 - 1)) - 1
_EBITS_F4, _MBITS_F4 = 2, 1

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_UINT16_MASK = (1 << 16) - 1
_UINT32_MASK = (1 << 32) - 1
_UINT62_MASK = (1 << 62) - 1
_PHILOX_ROUNDS = 7


def _mulhilo_uint32(x, multiplier):
    """Return the high and low halves of a uint32 product using safe int64 operations."""
    x_lo = x & _UINT16_MASK
    x_hi = x >> 16
    multiplier_lo = multiplier & _UINT16_MASK
    multiplier_hi = multiplier >> 16
    product_lo = x_lo * multiplier_lo
    product_cross_0 = x_lo * multiplier_hi
    product_cross_1 = x_hi * multiplier_lo
    product_hi = x_hi * multiplier_hi
    carry = (
        (product_lo >> 16)
        + (product_cross_0 & _UINT16_MASK)
        + (product_cross_1 & _UINT16_MASK)
    )
    lo = (product_lo & _UINT16_MASK) | ((carry & _UINT16_MASK) << 16)
    hi = (
        product_hi
        + (product_cross_0 >> 16)
        + (product_cross_1 >> 16)
        + (carry >> 16)
    ) & _UINT32_MASK
    return hi, lo


def _philox4x32_words(c0, c1, c2, c3, seed, word_count):
    """Run Philox4x32-7 and flatten its four output words per counter."""
    k0 = (seed & _UINT32_MASK).expand_as(c0)
    k1 = ((seed >> 32) & _UINT32_MASK).expand_as(c0)
    for _ in range(_PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo_uint32(c0, _PHILOX_M0)
        hi1, lo1 = _mulhilo_uint32(c2, _PHILOX_M1)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & _UINT32_MASK,
            lo1,
            (hi0 ^ c3 ^ k1) & _UINT32_MASK,
            lo0,
        )
        k0 = (k0 + _PHILOX_W0) & _UINT32_MASK
        k1 = (k1 + _PHILOX_W1) & _UINT32_MASK

    return torch.stack((c0, c1, c2, c3), dim=1).reshape(-1)[:word_count]


def _philox4x32_7_stateless_words(key, word_count):
    """Draw words from a user-managed seed/counter key using Philox4x32-7."""
    key = key.reshape(-1).view(torch.int64)
    counter_count = (word_count + 3) // 4
    counter = key[1] + torch.arange(
        counter_count, dtype=torch.int64, device=key.device
    )
    zeros = torch.zeros_like(counter)
    return _philox4x32_words(
        counter & _UINT32_MASK,
        (counter >> 32) & _UINT32_MASK,
        zeros,
        zeros,
        key[0],
        word_count,
    )


def _philox4x32_7_stateful_words(generator, word_count, device):
    """Draw words using one generator block and tile-invariant logical subsequences."""
    seed, offset_words, intragraph_offset_words = generator.philox_state(4)
    assert int(intragraph_offset_words.item()) % 4 == 0

    # Generator offsets are word-based signed-int64 bit containers. Convert them to a block index
    # without signed overflow: both terms are multiples of four, and the block sum wraps at 2**62.
    offset_words = offset_words.to(device=device)
    offset_blocks = (offset_words >> 2) & _UINT62_MASK
    intragraph_blocks = (
        int(intragraph_offset_words.item()) & ((1 << 64) - 1)
    ) >> 2
    block = (offset_blocks + intragraph_blocks) & _UINT62_MASK

    subsequence_count = (word_count + 3) // 4
    subsequence = torch.arange(
        subsequence_count, dtype=torch.int64, device=device
    )
    return _philox4x32_words(
        (block & _UINT32_MASK).expand_as(subsequence),
        ((block >> 32) & _UINT32_MASK).expand_as(subsequence),
        subsequence & _UINT32_MASK,
        (subsequence >> 32) & _UINT32_MASK,
        seed.to(device=device),
        word_count,
    )


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
