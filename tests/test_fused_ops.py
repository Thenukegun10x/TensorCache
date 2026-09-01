"""
Unit tests for Fused Triton Kernels.
"""

import pytest
import torch
from tensorcache.fused_ops import (
    quantize_fused_gpu,
    dequantize_fused_gpu,
    dequantize_fused_int4_gpu,
    dequantize_fused_int3_gpu,
    FusedDequantLinear,
    HAS_TRITON,
)
from tensorcache.codec import (
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int4_g32,
    dequantize_int4_g32,
    quantize_int3_g32,
    dequantize_int3_g32,
)


def _is_functional_gpu() -> bool:
    if not torch.cuda.is_available() or not HAS_TRITON:
        return False
    try:
        t = torch.zeros(1, device="cuda:0")
        del t
        return True
    except Exception:
        return False


GPU_AVAILABLE = _is_functional_gpu()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Functional CUDA/ROCm GPU + Triton required for Triton kernels")
def test_fused_dequant_kernel():
    device = "cuda:0"
    x = torch.randn(64, 446, 768, dtype=torch.bfloat16, device=device)
    
    q_int8, scales, shape = quantize_int8_g32(x, group_size=32)
    rec_fused = dequantize_fused_gpu(q_int8, scales, shape, group_size=32)
    
    diff = x.float() - rec_fused.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0
    
    assert rec_fused.shape == x.shape
    assert rec_fused.dtype == torch.bfloat16
    assert rel_rmse < 1.0
    print(f"\n[+] Fused INT8 Dequant Kernel verified! Rel RMSE: {rel_rmse:.4f}%")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Functional CUDA/ROCm GPU + Triton required for Triton kernels")
def test_fused_dequant_int4_kernel():
    device = "cuda:0"
    x = torch.randn(16, 64, 768, dtype=torch.bfloat16, device=device)

    q_packed, scales, shape = quantize_int4_g32(x, group_size=32)
    rec_fused = dequantize_fused_int4_gpu(q_packed, scales, shape, group_size=32)

    diff = x.float() - rec_fused.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0

    assert rec_fused.shape == x.shape
    assert rec_fused.dtype == torch.bfloat16
    assert rel_rmse < 10.0
    print(f"\n[+] Fused INT4 Dequant Kernel verified! Rel RMSE: {rel_rmse:.4f}%")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Functional CUDA/ROCm GPU + Triton required for Triton kernels")
def test_fused_dequant_int3_kernel():
    device = "cuda:0"
    x = torch.randn(16, 64, 768, dtype=torch.bfloat16, device=device)

    q_packed, scales, shape = quantize_int3_g32(x, group_size=32)
    rec_fused = dequantize_fused_int3_gpu(q_packed, scales, shape, group_size=32)

    diff = x.float() - rec_fused.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0

    assert rec_fused.shape == x.shape
    assert rec_fused.dtype == torch.bfloat16
    assert rel_rmse < 25.0
    print(f"\n[+] Fused INT3 Dequant Kernel verified! Rel RMSE: {rel_rmse:.4f}%")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Functional CUDA/ROCm GPU + Triton required for Triton kernels")
def test_fused_dequant_linear():
    device = "cuda:0"
    M, K, N = 128, 768, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    q_int8, scales, shape = quantize_int8_g32(x, group_size=32)
    
    # 1. Standard separate path (Dequant then Linear)
    linear_std = torch.nn.Linear(K, N, bias=True, dtype=torch.bfloat16, device=device)
    x_rec = dequantize_int8_g32(q_int8, scales, shape, group_size=32)
    y_std = linear_std(x_rec)
    
    # 2. Fused Dequant+Linear layer
    fused_linear = FusedDequantLinear(in_features=K, out_features=N, bias=True, group_size=32).to(device)
    fused_linear.weight.data.copy_(linear_std.weight.data)
    fused_linear.bias.data.copy_(linear_std.bias.data)
    
    y_fused = fused_linear(q_int8, scales)
    
    diff = y_std.float() - y_fused.float()
    rel_diff = (torch.norm(diff) / torch.norm(y_std.float())).item() * 100.0
    
    assert y_fused.shape == (M, N)
    assert y_fused.dtype == torch.bfloat16
    assert rel_diff < 0.5  # BF16 dot-product order variations are typically ~0.2%
    print(f"\n[+] Fused Dequant+Linear verified! Output Diff: {rel_diff:.4f}%")


if __name__ == "__main__":
    if GPU_AVAILABLE:
        test_fused_dequant_kernel()
        test_fused_dequant_int4_kernel()
        test_fused_dequant_int3_kernel()
        test_fused_dequant_linear()
        print("\n[+] ALL FUSED KERNEL TESTS PASSED!")
    else:
        print("[-] Skipping: No functional CUDA/ROCm GPU + Triton detected.")
