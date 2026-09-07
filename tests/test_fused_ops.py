"""
Unit tests for Fused Triton Kernels.
"""

import pytest
import torch
import math
from tensorcache.fused_ops import (
    quantize_fused_gpu,
    dequantize_fused_gpu,
    dequantize_fused_int4_gpu,
    dequantize_fused_int3_gpu,
    quantize_fused_wavelet8x_gpu,
    dequantize_fused_wavelet8x_gpu,
    dequantize_sparse_wavelet_gpu,
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


def _natural_test_img(H=64, W=64, seed=42, device="cpu"):
    """Detailed luma + smooth chroma (4:2:0-safe; white RGB noise aliases)."""
    torch.manual_seed(seed)
    raw = torch.randint(0, 256, (H, W), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    lum = torch.nn.functional.avg_pool2d(raw, kernel_size=3, stride=1, padding=1).squeeze()
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=device),
                            torch.linspace(0, 1, W, device=device), indexing="ij")
    tint_r = 24 * torch.sin(2 * math.pi * yy) * torch.cos(2 * math.pi * xx)
    tint_g = 20 * torch.cos(2 * math.pi * (xx + yy))
    tint_b = -18 * torch.sin(2 * math.pi * (xx - yy))
    return torch.stack([lum + tint_r, lum + tint_g, lum + tint_b], dim=-1).clamp(0, 255).byte()


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


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Functional CUDA/ROCm GPU + Triton required for Triton kernels")
def test_fused_wavelet8x_gpu():
    device = "cuda:0"
    H, W, C = 64, 64, 3
    img = _natural_test_img(H, W, device=device)
    
    packed_meta, shape = quantize_fused_wavelet8x_gpu(img, q_scale=1.0)
    rec = dequantize_fused_wavelet8x_gpu(packed_meta, device=device)
    
    assert rec.shape == (H, W, C)
    assert rec.dtype == torch.uint8
    diff = img.float() - rec.float()
    mse = (diff ** 2).mean().item()
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')
    rmse = (math.sqrt(mse) / 255.0) * 100.0
    
    assert psnr > 35.0
    print(f"\n[+] Fused Wavelet 8x GPU Codec verified! PSNR: {psnr:.2f} dB, Rel RMSE: {rmse:.2f}%")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Functional CUDA/ROCm GPU + Triton required for Triton kernels")
def test_sparse_wavelet_decode_gpu():
    """Mega-kernel sparse decode is bit-exact vs the CPU sparse reference
    (adaptive 4:4:4 + 4:2:0 and the static schema), via arenas and batches."""
    from tensorcache.codec import (
        quantize_pixel_wavelet_adaptive, quantize_pixel_wavelet8x,
        dequantize_pixel_wavelet_adaptive, dequantize_pixel_wavelet8x,
        sparse_pack_meta, sparse_unpack_meta, sparse_pack_arena,
    )
    from tensorcache.fused_ops import dequantize_sparse_wavelet_batch_gpu
    device = "cuda:0"
    img = _natural_test_img(96, 96, seed=7)
    cases = []
    for mode in ("ultra", "balanced"):
        m, _ = quantize_pixel_wavelet_adaptive(img, mode=mode)
        cases.append((f"adaptive-{mode}", m, dequantize_pixel_wavelet_adaptive))
    ms, _ = quantize_pixel_wavelet8x(img, q_scale=3.0, chroma420=True)
    cases.append(("static-q3-420", ms, dequantize_pixel_wavelet8x))
    for name, m, cpu_fn in cases:
        sp = sparse_pack_meta(m)
        ref = cpu_fn(sparse_unpack_meta(sp), device="cpu")
        got = dequantize_sparse_wavelet_gpu(sp, device=device)
        assert torch.equal(ref, got.cpu()), name + "-legacy"
        arena = sparse_pack_arena(sp)
        got_a = dequantize_sparse_wavelet_gpu(arena, device=device)
        assert torch.equal(ref, got_a.cpu()), name + "-arena"
        print(f"\n[+] Sparse GPU decode {name}: bit-exact (legacy + arena)")
    # batched: 4x same ultra image, one launch
    m, _ = quantize_pixel_wavelet_adaptive(img, mode="balanced")
    sp = sparse_pack_meta(m)
    ref = dequantize_pixel_wavelet_adaptive(sparse_unpack_meta(sp), device="cpu")
    arenas = [sparse_pack_arena(sp) for _ in range(4)]
    batch = dequantize_sparse_wavelet_batch_gpu(arenas, device=device)
    assert batch.shape == (4, 96, 96, 3)
    for i in range(4):
        assert torch.equal(ref, batch[i].cpu()), f"batch-{i}"
    print("\n[+] Sparse GPU batch-4 decode: bit-exact")


if __name__ == "__main__":
    if GPU_AVAILABLE:
        test_fused_dequant_kernel()
        test_fused_dequant_int4_kernel()
        test_fused_dequant_int3_kernel()
        test_fused_dequant_linear()
        test_fused_wavelet8x_gpu()
        test_sparse_wavelet_decode_gpu()
        print("\n[+] ALL FUSED KERNEL TESTS PASSED!")
    else:
        print("[-] Skipping: No functional CUDA/ROCm GPU + Triton detected.")
