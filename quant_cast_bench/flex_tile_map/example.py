"""End-to-end example: a dense (functional, non-gated) MLP whose activation is expressed with
`flex_tile_map`, trained forward + backward under `torch.compile`, and checked against a plain
PyTorch reference MLP that does the same math without flex_tile_map.
"""

import torch

# Importing the package fires the HOP registrations AND auto-installs the mm -> flex_gemm post-grad
# fusion pass (see flex_tile_map/__init__.py -> flex_gemm_to_tile_map_fusion._auto_install).
import quant_cast_bench.flex_tile_map  # noqa: F401
from quant_cast_bench.flex_tile_map.api import flex_tile_map

DIM = 4096


def _activation(acc):
    return acc.relu()


def mlp_flex(x, w1, w2):
    c = torch.mm(x, w1)                # up-projection gemm
    d = flex_tile_map(c, _activation)  # activation (a fusible mm -> flex_tile_map pair)
    return torch.mm(d, w2)             # down-projection gemm


def mlp_reference(x, w1, w2):
    return torch.mm(_activation(torch.mm(x, w1)), w2)


def _sqnr(ref, actual):
    """Signal-to-quantization-noise ratio in dB - higher is better, ~50db is very close"""
    ref, actual = ref.double(), actual.double()
    noise = (ref - actual).pow(2).mean()
    if noise == 0:
        return float("inf")
    return (10 * torch.log10(ref.pow(2).mean() / noise)).item()


def main():
    assert torch.cuda.is_available(), "this example requires a CUDA device"
    torch.manual_seed(0)

    def make_inputs():
        x = torch.randn(DIM, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w1 = torch.randn(DIM, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w2 = torch.randn(DIM, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        return x, w1, w2

    # --- compiled flex_tile_map MLP: forward + backward ---
    x, w1, w2 = make_inputs()
    compiled = torch.compile(mlp_flex, backend="inductor", fullgraph=True)
    out = compiled(x, w1, w2)
    out.sum().backward()  # populates x.grad / w1.grad / w2.grad

    # --- reference MLP (no flex_tile_map): forward + backward on the SAME inputs ---
    rx = x.detach().clone().requires_grad_(True)
    rw1 = w1.detach().clone().requires_grad_(True)
    rw2 = w2.detach().clone().requires_grad_(True)
    ref_out = mlp_reference(rx, rw1, rw2)
    ref_out.sum().backward()

    # --- verify inputs, outputs, and grads match ---
    # inputs are the same tensors up to the detach().clone() above (sanity check they started equal).
    assert torch.equal(x, rx) and torch.equal(w1, rw1) and torch.equal(w2, rw2), "inputs diverged"

    checks = [
        ("output", ref_out, out),
        ("grad_x", rx.grad, x.grad),
        ("grad_w1", rw1.grad, w1.grad),
        ("grad_w2", rw2.grad, w2.grad),
    ]
    # Compare with SQNR, not exact/near-exact equality. Two bf16 effects make elementwise equality
    # the wrong bar: (1) the compiled and eager matmuls use different reduction orders, and (2) the
    # fused flex_gemm epilogue applies the activation in fp32 on the raw accumulator and rounds to
    # bf16 ONCE, whereas the eager reference rounds x@w1 to bf16 first and then applies the activation
    # (a second rounding). For `relu` alone these agree bit-for-bit (relu commutes with rounding), but
    # an activation like `relu(acc) + 1.0` shifts values across the coarser bf16 grid, so ~1% of the
    # hidden activations round differently and propagate through the down-proj matmul. SQNR > 30 dB is
    # the repo's "agrees to bf16 precision" bar (the fused path is in fact the more accurate one).
    print(f"MLP {DIM}x{DIM} bf16, flex_tile_map activation vs reference:")
    for name, ref, actual in checks:
        sqnr = _sqnr(ref, actual)
        assert sqnr > 30.0, f"{name}: SQNR {sqnr:.1f} dB too low"
        print(f"  {name:8s}  SQNR = {sqnr:6.1f} dB  (match)")
    print("all outputs and grads match the reference to bf16 precision.")


if __name__ == "__main__":
    main()
