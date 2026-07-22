"""
An example demonstrating deepseek-style quantization of inputs to an
fp8 gemm, plus nvfp4 quantization with two-level (outer + inner) scaling.
"""

import torch

from api import flex_cast_quant_dense
from nvfp4_utils import F4_E2M1_MAX, f32_to_f4_unpacked, pack_uint4

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12
F8E4M3_MAX = FP8_MAX
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


def amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    amax_fp32 = amax.clamp(min=EPS).to(torch.float32)
    return amax_fp32 / FP8_MAX


def cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    reciprocal = 1.0 / scale
    y = tile * reciprocal
    return y.to(torch.float8_e4m3fn)


# nvfp4 two-level scaling callbacks: a per-tensor fp32 outer scale plus a
# per-block e4m3 inner scale.

def nvfp4_inner_amax_to_scale_fn(
    local_amax: torch.Tensor, outer_scale: torch.Tensor
) -> torch.Tensor:
    block_scale_fp32 = local_amax.to(torch.float32) / F4_E2M1_MAX
    scaled = block_scale_fp32 / outer_scale
    return torch.clamp(scaled, min=E4M3_EPS, max=F8E4M3_MAX).to(
        torch.float8_e4m3fn
    )


def nvfp4_outer_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def nvfp4_cast_to_dtype_fn(
    tile: torch.Tensor,
    inner_scale: torch.Tensor,
    outer_scale: torch.Tensor,
) -> torch.Tensor:
    inner_fp32 = inner_scale.to(torch.float32)
    reciprocal = (1.0 / outer_scale) / inner_fp32
    data_scaled = tile.to(torch.float32) * reciprocal
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_unpacked = f32_to_f4_unpacked(data_scaled)
    return pack_uint4(data_unpacked).view(torch.float4_e2m1fn_x2)


def main() -> None:
    M, K, N = 512, 1024, 2048

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # compile is opt-in and required for performance
    flex_cast_quant_dense_c = torch.compile(flex_cast_quant_dense)

    # deepseek style fp8_e4m3 1x128 quant along the K dim for activations
    # Note: torchinductor will generate this kernel from scratch. If
    # torch.compile is enabled on the surrounding program, the kernel
    # will (likely, up to the fuser) be fused into the previous op.
    x_q, x_scale = flex_cast_quant_dense_c(
        x,
        block_size=128,
        dim=-1,
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    print(f"x_q:     shape={tuple(x_q.shape)} dtype={x_q.dtype}")
    print(f"x_scale: shape={tuple(x_scale.shape)} dtype={x_scale.dtype}")

    # deepseek style fp8_e4m3 128x128 quant for weights
    # Note: for this kernel, torchinductor will lower the callbacks 
    # onto a handwritten triton template, flex-attention style 
    # (currently faster than generating from scratch)
    w_q, w_scale = flex_cast_quant_dense_c(
        w,
        block_size=(128, 128),
        dim=(-2, -1),
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        amax_to_scale_fn=amax_to_scale_fn,
        cast_to_dtype_fn=cast_to_dtype_fn,
    )
    print(f"w_q:     shape={tuple(w_q.shape)} dtype={w_q.dtype}")
    print(f"w_scale: shape={tuple(w_scale.shape)} dtype={w_scale.dtype}")

    # TODO(future PR): currently running the example below works in
    # isolation but hangs in dynamo if run after the above examples,
    # need to chase it down, likely a compile bug
    # nvfp4 with two-level scaling: per-tensor fp32 outer scale + per-block
    # e4m3 inner scale. List-typed args carry [inner, outer] in that order.
    # The framework computes the outer scale itself from the input's amax.
    x_q_nvfp4, x_scale_nvfp4 = flex_cast_quant_dense_c(
        x,
        block_size=[16, (-1, -1)],
        dim=[-1, (-2, -1)],
        qdata_dtype=torch.float4_e2m1fn_x2,
        scale_dtype=[torch.float8_e4m3fn, torch.float32],
        amax_to_scale_fn=[
            nvfp4_inner_amax_to_scale_fn,
            nvfp4_outer_amax_to_scale_fn,
        ],
        cast_to_dtype_fn=nvfp4_cast_to_dtype_fn,
    )
    inner_scale, outer_scale = x_scale_nvfp4
    print(
        f"x_q (nvfp4):     shape={tuple(x_q_nvfp4.shape)} dtype={x_q_nvfp4.dtype}"
    )
    print(
        f"x_inner_scale:   shape={tuple(inner_scale.shape)} dtype={inner_scale.dtype}"
    )
    print(
        f"x_outer_scale:   shape={tuple(outer_scale.shape)} dtype={outer_scale.dtype}"
    )

    # lowp gemm here (not shown, since I prototyped this on a B200)
    # TODO(future): hop to an H100 and finish out this example


if __name__ == "__main__":
    main()
