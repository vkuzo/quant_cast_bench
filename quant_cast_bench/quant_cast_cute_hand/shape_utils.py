"""Predefined benchmark shapes for representative model configurations."""


def gpt_oss_120b_m8192_tp8_ep8() -> list[tuple[int, int]]:
    """Return GPT-OSS-120B shapes for M=8192, TP=8, and EP=8."""
    return [
        (8192, 2880),  # self_attn.qkv input; self_attn.o_proj backward dY.
        (8192, 640),  # TP-local self_attn.qkv output and backward dY.
        (8192, 512),  # TP-local self_attn.o_proj input.
        (640, 2880),  # TP-local self_attn.qkv.weight.
        (2880, 512),  # TP-local self_attn.o_proj.weight.
        (92160, 2880),  # EP-local mlp.experts.gate_up_proj.weight (16 experts).
        (46080, 2880),  # EP-local mlp.experts.down_proj.weight (16 experts).
    ]
