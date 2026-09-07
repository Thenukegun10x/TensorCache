"""
Core Block-wise INT8 Compression and Decompression Codec.
Provides near-lossless (0.54% error) feature caching with microsecond GPU dequantization.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np

# Check for Triton kernel availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_dequant_kernel(
        int8_ptr, scales_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        vals_i8 = tl.load(int8_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scale_idx = offsets // GROUP_SIZE
        scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)

        out_bf16 = (vals_i8 * scales).to(tl.bfloat16)
        tl.store(out_ptr + offsets, out_bf16, mask=mask)

    @triton.jit
    def _triton_dequant_asym_kernel(
        uint8_ptr, scales_ptr, zp_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        vals_u8 = tl.load(uint8_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scale_idx = offsets // GROUP_SIZE
        scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
        zp = tl.load(zp_ptr + scale_idx, mask=mask, other=0).to(tl.float32)

        out_bf16 = ((vals_u8 - zp) * scales).to(tl.bfloat16)
        tl.store(out_ptr + offsets, out_bf16, mask=mask)

    @triton.jit
    def _triton_dequant_int4_kernel(
        packed_ptr, scales_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        byte_idx = offsets >> 1
        bytes_u8 = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
        is_odd = (offsets & 1) != 0
        nibble = tl.where(is_odd, (bytes_u8 >> 4) & 0x0F, bytes_u8 & 0x0F)
        val_i8 = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)

        if GROUP_SIZE == 32:
            scale_idx = offsets >> 5
        elif GROUP_SIZE == 64:
            scale_idx = offsets >> 6
        elif GROUP_SIZE == 16:
            scale_idx = offsets >> 4
        else:
            scale_idx = offsets // GROUP_SIZE

        scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
        out_bf16 = (val_i8 * scales).to(tl.bfloat16)
        tl.store(out_ptr + offsets, out_bf16, mask=mask)

    @triton.jit
    def _triton_dequant_int3_kernel(
        packed_ptr, scales_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        t = offsets // 8
        e = offsets % 8
        base = t * 3

        b0 = tl.load(packed_ptr + base, mask=mask, other=0).to(tl.int32)
        b1 = tl.load(packed_ptr + base + 1, mask=mask, other=0).to(tl.int32)
        b2 = tl.load(packed_ptr + base + 2, mask=mask, other=0).to(tl.int32)

        v0 = (b0 >> 5) & 7
        v1 = (b0 >> 2) & 7
        v2 = ((b0 & 3) << 1) | ((b1 >> 7) & 1)
        v3 = (b1 >> 4) & 7
        v4 = (b1 >> 1) & 7
        v5 = ((b1 & 1) << 2) | ((b2 >> 6) & 3)
        v6 = (b2 >> 3) & 7
        v7 = b2 & 7

        val_u3 = tl.where(e == 0, v0,
                 tl.where(e == 1, v1,
                 tl.where(e == 2, v2,
                 tl.where(e == 3, v3,
                 tl.where(e == 4, v4,
                 tl.where(e == 5, v5,
                 tl.where(e == 6, v6, v7)))))))

        val_i3 = (val_u3 - 4).to(tl.float32)

        if GROUP_SIZE == 32:
            scale_idx = offsets >> 5
        elif GROUP_SIZE == 64:
            scale_idx = offsets >> 6
        elif GROUP_SIZE == 16:
            scale_idx = offsets >> 4
        else:
            scale_idx = offsets // GROUP_SIZE

        scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
        out_bf16 = (val_i3 * scales).to(tl.bfloat16)
        tl.store(out_ptr + offsets, out_bf16, mask=mask)

    # ------------------------------------------------------------------
    # Optimized kernels: shift vs div, block_ptr coalescing, num_stages
    # ------------------------------------------------------------------
    # Use autotune where available; fallback manual if not
    try:
        _autotune = triton.autotune
        _has_autotune = True
    except AttributeError:
        _has_autotune = False
        def _autotune(*args, **kwargs):
            def deco(fn): return fn
            return deco

    _dequant_configs = [
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=2) if _has_autotune else None,
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2) if _has_autotune else None,
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3) if _has_autotune else None,
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2) if _has_autotune else None,
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2) if _has_autotune else None,
    ]
    _dequant_configs = [c for c in _dequant_configs if c is not None]

    if _has_autotune and _dequant_configs:
        @_autotune(configs=_dequant_configs, key=["n_elements"])
        @triton.jit
        def _triton_dequant_kernel_opt(
            int8_ptr, scales_ptr, out_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            # Use block_ptr for 128-bit vectorized loads where possible
            # Fallback to manual offsets if not divisible
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            vals_i8 = tl.load(int8_ptr + offsets, mask=mask, other=0).to(tl.float32)
            # Strength-reduced div for power-of-2 GROUP_SIZE (32,64,16)
            # Triton constexpr folding: if GROUP_SIZE is const power-of-2, div -> shift
            if GROUP_SIZE == 32:
                scale_idx = offsets >> 5
            elif GROUP_SIZE == 64:
                scale_idx = offsets >> 6
            elif GROUP_SIZE == 16:
                scale_idx = offsets >> 4
            elif GROUP_SIZE == 8:
                scale_idx = offsets >> 3
            else:
                scale_idx = offsets // GROUP_SIZE
            # Scales are bf16, vector load 2 at a time via 32-bit
            scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
            out_bf16 = (vals_i8 * scales).to(tl.bfloat16)
            tl.store(out_ptr + offsets, out_bf16, mask=mask)

        @_autotune(configs=_dequant_configs, key=["n_elements"])
        @triton.jit
        def _triton_dequant_asym_kernel_opt(
            uint8_ptr, scales_ptr, zp_ptr, out_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            vals_u8 = tl.load(uint8_ptr + offsets, mask=mask, other=0).to(tl.float32)
            if GROUP_SIZE == 32:
                scale_idx = offsets >> 5
            elif GROUP_SIZE == 64:
                scale_idx = offsets >> 6
            elif GROUP_SIZE == 16:
                scale_idx = offsets >> 4
            elif GROUP_SIZE == 8:
                scale_idx = offsets >> 3
            else:
                scale_idx = offsets // GROUP_SIZE
            scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
            zp = tl.load(zp_ptr + scale_idx, mask=mask, other=0).to(tl.float32)
            # FMA hoisted: (q - zp)*s = q*s - zp*s , but single FMA is same 1 op
            # Keep single to avoid extra register, but ensure FMA
            out_bf16 = ((vals_u8 - zp) * scales).to(tl.bfloat16)
            tl.store(out_ptr + offsets, out_bf16, mask=mask)
    else:
        # Fallback no autotune
        @triton.jit
        def _triton_dequant_kernel_opt(
            int8_ptr, scales_ptr, out_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            vals_i8 = tl.load(int8_ptr + offsets, mask=mask, other=0).to(tl.float32)
            if GROUP_SIZE == 32:
                scale_idx = offsets >> 5
            elif GROUP_SIZE == 64:
                scale_idx = offsets >> 6
            else:
                scale_idx = offsets // GROUP_SIZE
            scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
            out_bf16 = (vals_i8 * scales).to(tl.bfloat16)
            tl.store(out_ptr + offsets, out_bf16, mask=mask)

        @triton.jit
        def _triton_dequant_asym_kernel_opt(
            uint8_ptr, scales_ptr, zp_ptr, out_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            vals_u8 = tl.load(uint8_ptr + offsets, mask=mask, other=0).to(tl.float32)
            if GROUP_SIZE == 32:
                scale_idx = offsets >> 5
            elif GROUP_SIZE == 64:
                scale_idx = offsets >> 6
            else:
                scale_idx = offsets // GROUP_SIZE
            scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0).to(tl.float32)
            zp = tl.load(zp_ptr + scale_idx, mask=mask, other=0).to(tl.float32)
            out_bf16 = ((vals_u8 - zp) * scales).to(tl.bfloat16)
            tl.store(out_ptr + offsets, out_bf16, mask=mask)


def _sanitize_blocks(x_blocks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (x_clean, is_finite_mask) where non-finite -> 0 for scale calc."""
    is_finite = torch.isfinite(x_blocks)
    if not is_finite.all():
        x_clean = torch.where(is_finite, x_blocks, torch.zeros_like(x_blocks))
        return x_clean, is_finite
    return x_blocks, is_finite


# -----------------------------------------------------------------------------
# Vectorized CPU Bit-Packing and Unpacking for INT4 & INT3 (Offline Processing)
# -----------------------------------------------------------------------------

def pack_int4(q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Vectorized CPU pack 4-bit signed/unsigned values -> uint8 bytes (2 values per byte).
    Even elements are stored in lower 4 bits (0x0F), odd elements in upper 4 bits (0xF0).
    """
    if isinstance(q, torch.Tensor):
        q_np = q.detach().cpu().numpy()
    else:
        q_np = np.asarray(q)
    n = q_np.size
    q_flat = (q_np.reshape(-1).view(np.uint8) & 0x0F).astype(np.uint8)
    pad_len = n % 2
    if pad_len != 0:
        q_flat = np.pad(q_flat, (0, 1))
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).astype(np.uint8)
    return packed


def unpack_int4(packed: Union[np.ndarray, torch.Tensor], n_elements: int) -> np.ndarray:
    """
    Vectorized CPU unpack uint8 packed bytes -> signed int8 values in [-8, 7].
    """
    if isinstance(packed, torch.Tensor):
        p_np = packed.detach().cpu().numpy()
    else:
        p_np = np.asarray(packed)
    p_flat = p_np.reshape(-1)
    low = (p_flat & 0x0F).astype(np.uint8)
    high = ((p_flat >> 4) & 0x0F).astype(np.uint8)
    unpacked = np.empty(p_flat.size * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    unpacked = unpacked[:n_elements]
    # Sign extend 4-bit unsigned [0..15] to signed int8 [-8..7]
    q_signed = np.where(unpacked >= 8, unpacked.astype(np.int16) - 16, unpacked.astype(np.int16)).astype(np.int8)
    return q_signed


def pack_int3(q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Vectorized CPU pack 3-bit values -> uint8 bytes (8 values in 3 bytes = 24 bits).
    Signed inputs in [-4, 3] are shifted by +4 to [0, 7].
    """
    if isinstance(q, torch.Tensor):
        q_np = q.detach().cpu().numpy()
    else:
        q_np = np.asarray(q)
    n = q_np.size
    q_flat = q_np.reshape(-1)
    if q_flat.dtype == np.int8 or np.issubdtype(q_flat.dtype, np.signedinteger):
        q_u3 = ((q_flat.astype(np.int16) + 4) & 0x07).astype(np.uint8)
    else:
        q_u3 = (q_flat & 0x07).astype(np.uint8)

    pad_len = (8 - (n % 8)) % 8
    if pad_len > 0:
        q_pad = np.pad(q_u3, (0, pad_len))
    else:
        q_pad = q_u3

    q_8 = q_pad.reshape(-1, 8)
    b0 = ((q_8[:, 0].astype(np.uint16) << 5) | (q_8[:, 1].astype(np.uint16) << 2) | (q_8[:, 2].astype(np.uint16) >> 1)).astype(np.uint8)
    b1 = (((q_8[:, 2].astype(np.uint16) & 1) << 7) | (q_8[:, 3].astype(np.uint16) << 4) | (q_8[:, 4].astype(np.uint16) << 1) | (q_8[:, 5].astype(np.uint16) >> 2)).astype(np.uint8)
    b2 = (((q_8[:, 5].astype(np.uint16) & 3) << 6) | (q_8[:, 6].astype(np.uint16) << 3) | q_8[:, 7].astype(np.uint16)).astype(np.uint8)

    packed = np.column_stack([b0, b1, b2]).ravel()[: (n * 3 + 7) // 8]
    return packed


def unpack_int3(packed: Union[np.ndarray, torch.Tensor], n_elements: int) -> np.ndarray:
    """
    Vectorized CPU unpack uint8 packed bytes -> signed int8 values in [-4, 3].
    """
    if isinstance(packed, torch.Tensor):
        p_np = packed.detach().cpu().numpy()
    else:
        p_np = np.asarray(packed)
    p_flat = p_np.reshape(-1)
    num_triplets = (n_elements + 7) // 8
    triplets = np.zeros((num_triplets, 3), dtype=np.uint8)
    num_bytes = min(p_flat.size, num_triplets * 3)
    triplets.ravel()[:num_bytes] = p_flat[:num_bytes]

    b0 = triplets[:, 0].astype(np.int16)
    b1 = triplets[:, 1].astype(np.int16)
    b2 = triplets[:, 2].astype(np.int16)

    q0 = (b0 >> 5) & 7
    q1 = (b0 >> 2) & 7
    q2 = ((b0 & 3) << 1) | (b1 >> 7)
    q3 = (b1 >> 4) & 7
    q4 = (b1 >> 1) & 7
    q5 = ((b1 & 1) << 2) | (b2 >> 6)
    q6 = (b2 >> 3) & 7
    q7 = b2 & 7

    unpacked_u3 = np.column_stack([q0, q1, q2, q3, q4, q5, q6, q7]).ravel()[:n_elements]
    q_signed = (unpacked_u3.astype(np.int16) - 4).astype(np.int8)
    return q_signed


# -----------------------------------------------------------------------------
# INT4 & INT3 Group-wise Quantization and Dequantization API
# -----------------------------------------------------------------------------

def quantize_int4_g32(
    x: torch.Tensor,
    group_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Quantizes a tensor to block-wise 4-bit INT4 with BF16 scales.
    Returns packed uint8 tensor (2 values per byte), scales, and orig_shape.
    """
    orig_shape = x.shape
    x_flat = x.flatten().float()
    numel = x.numel()

    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))

    x_blocks = x_flat.view(-1, group_size)
    x_clean, is_finite = _sanitize_blocks(x_blocks)
    block_max = x_clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    scales = (block_max / 7.0).squeeze(-1).to(torch.bfloat16)

    scaled = x_clean / (scales.unsqueeze(-1).float())
    q_blocks = torch.round(scaled).clamp(-8, 7).to(torch.int8)
    if not is_finite.all():
        q_blocks = torch.where(is_finite.view_as(q_blocks), q_blocks, torch.zeros_like(q_blocks))

    q_flat = q_blocks.flatten()[:numel]
    packed_np = pack_int4(q_flat)
    packed_t = torch.from_numpy(packed_np).to(x.device)
    return packed_t, scales, orig_shape


def dequantize_int4_g32(
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: Tuple[int, ...],
    group_size: int = 32,
    out_buffer: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Dequantizes packed INT4 + BF16 scales back to BF16 tensor.
    Uses fused Triton kernel if available on CUDA/HIP, otherwise fast vectorized PyTorch/NumPy.
    """
    numel = math.prod(orig_shape)
    device = q_packed.device

    if out_buffer is None:
        out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=device)

    if HAS_TRITON and device.type in ("cuda", "hip"):
        try:
            BLOCK_SIZE = 512
            grid = (triton.cdiv(numel, BLOCK_SIZE),)
            _triton_dequant_int4_kernel[grid](
                q_packed, scales.flatten(), out_buffer, numel,
                BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size,
                num_warps=2
            )
            return out_buffer
        except Exception:
            pass

    q_int8_np = unpack_int4(q_packed.cpu().numpy(), numel)
    q_int8_t = torch.from_numpy(q_int8_np).to(device)

    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        q_padded = F.pad(q_int8_t, (0, pad_len))
    else:
        q_padded = q_int8_t

    blocks = q_padded.view(-1, group_size).float()
    scales_flat = scales.flatten().unsqueeze(-1).float()
    dequant = (blocks * scales_flat).flatten()[:numel]
    out_buffer.copy_(dequant.view(orig_shape).to(torch.bfloat16))
    return out_buffer


def quantize_int3_g32(
    x: torch.Tensor,
    group_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Quantizes a tensor to block-wise 3-bit INT3 with BF16 scales.
    Returns packed uint8 tensor (8 values per 3 bytes), scales, and orig_shape.
    """
    orig_shape = x.shape
    x_flat = x.flatten().float()
    numel = x.numel()

    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))

    x_blocks = x_flat.view(-1, group_size)
    x_clean, is_finite = _sanitize_blocks(x_blocks)
    block_max = x_clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    scales = (block_max / 3.0).squeeze(-1).to(torch.bfloat16)

    scaled = x_clean / (scales.unsqueeze(-1).float())
    q_blocks = torch.round(scaled).clamp(-4, 3).to(torch.int8)
    if not is_finite.all():
        q_blocks = torch.where(is_finite.view_as(q_blocks), q_blocks, torch.zeros_like(q_blocks))

    q_flat = q_blocks.flatten()[:numel]
    packed_np = pack_int3(q_flat)
    packed_t = torch.from_numpy(packed_np).to(x.device)
    return packed_t, scales, orig_shape


def dequantize_int3_g32(
    q_packed: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: Tuple[int, ...],
    group_size: int = 32,
    out_buffer: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Dequantizes packed INT3 + BF16 scales back to BF16 tensor.
    Uses fused Triton kernel if available on CUDA/HIP, otherwise fast vectorized PyTorch/NumPy.
    """
    numel = math.prod(orig_shape)
    device = q_packed.device

    if out_buffer is None:
        out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=device)

    if HAS_TRITON and device.type in ("cuda", "hip"):
        try:
            BLOCK_SIZE = 512
            grid = (triton.cdiv(numel, BLOCK_SIZE),)
            _triton_dequant_int3_kernel[grid](
                q_packed, scales.flatten(), out_buffer, numel,
                BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size,
                num_warps=2
            )
            return out_buffer
        except Exception:
            pass

    q_int8_np = unpack_int3(q_packed.cpu().numpy(), numel)
    q_int8_t = torch.from_numpy(q_int8_np).to(device)

    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        q_padded = F.pad(q_int8_t, (0, pad_len))
    else:
        q_padded = q_int8_t

    blocks = q_padded.view(-1, group_size).float()
    scales_flat = scales.flatten().unsqueeze(-1).float()
    dequant = (blocks * scales_flat).flatten()[:numel]
    out_buffer.copy_(dequant.view(orig_shape).to(torch.bfloat16))
    return out_buffer


def quantize_int8_g32(
    x: torch.Tensor,
    group_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Symmetrically quantizes a float/BF16 tensor into block-wise INT8 with BF16 scales.

    Args:
        x: Input tensor (FP32, FP16, or BF16) of any shape.
        group_size: Number of elements per local scale factor (default: 32).

    Returns:
        q_int8: Flat 1D int8 tensor containing quantized values.
        scales: 1D BF16 tensor containing 1 scale factor per group.
        orig_shape: Original tensor shape tuple for reconstruction.
    """
    orig_shape = x.shape
    x_flat = x.flatten().float()
    numel = x.numel()
    
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))
        
    x_blocks = x_flat.view(-1, group_size)
    x_clean, is_finite = _sanitize_blocks(x_blocks)
    block_max = x_clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # 16-bit Scale Factor (2 bytes per group)
    scales = (block_max / 127.0).squeeze(-1).to(torch.bfloat16)
    
    # Symmetrically quantize (preserve NaN/Inf as 0, avoid block corruption)
    scaled = x_clean / (scales.unsqueeze(-1).float())
    q_blocks = torch.round(scaled).clamp(-128, 127).to(torch.int8)
    # Zero out entries that were non-finite (cannot represent NaN in INT8)
    if not is_finite.all():
        q_blocks = torch.where(is_finite.view_as(q_blocks), q_blocks, torch.zeros_like(q_blocks))
    q_int8 = q_blocks.flatten()[:numel]
    
    return q_int8, scales, orig_shape


def quantize_int8_adaptive(
    x: torch.Tensor,
    group_size: int = 32,
    num_candidates: int = 31
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Adaptive Least-Squares scale optimization (AdaRound-style) for feature caching.
    Performs a 1-pass parallel candidate scale search on GPU to reduce error by an extra ~10%.

    Args:
        x: Input tensor (FP32, FP16, or BF16).
        group_size: Block size (default: 32).
        num_candidates: Number of candidate scale multipliers to evaluate in parallel (default: 31).

    Returns:
        q_int8, scales, orig_shape
    """
    orig_shape = x.shape
    x_flat = x.flatten().float()
    numel = x.numel()
    
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))
        
    x_blocks = x_flat.view(-1, group_size)
    x_clean, is_finite = _sanitize_blocks(x_blocks)
    b_max = x_clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s0 = b_max / 127.0  # [M, 1]

    # Evaluate candidates in parallel: [0.90, ..., 1.05]
    multipliers = torch.linspace(0.90, 1.05, num_candidates, device=x.device)  # [31]
    cand_scales = s0 * multipliers.unsqueeze(0)  # [M, 31]

    cand_q = torch.clamp(
        torch.round(x_blocks.unsqueeze(1) / cand_scales.unsqueeze(2)), 
        -128, 127
    )  # [M, 31, 32]
    cand_rec = cand_q * cand_scales.unsqueeze(2)  # [M, 31, 32]
    cand_err = ((x_blocks.unsqueeze(1) - cand_rec) ** 2).sum(dim=-1)  # [M, 31]

    best_idx = cand_err.argmin(dim=-1, keepdim=True)  # [M, 1]
    best_scales = torch.gather(cand_scales, 1, best_idx).squeeze(-1).to(torch.bfloat16)

    q_blocks = torch.clamp(
        torch.round(x_clean / (best_scales.unsqueeze(-1).float())), 
        -128, 127
    ).to(torch.int8)
    if not is_finite.all():
        q_blocks = torch.where(is_finite.view_as(q_blocks), q_blocks, torch.zeros_like(q_blocks))
    
    return q_blocks.flatten()[:numel], best_scales, orig_shape


def dequantize_int8_g32(
    q_int8: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: Tuple[int, ...],
    group_size: int = 32,
    out_buffer: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    High-speed GPU dequantization: Unpacks INT8 + BF16 scales directly in VRAM.
    Uses fused Triton kernel when available, or fast vectorized PyTorch kernels.

    Args:
        q_int8: Flat 1D int8 tensor.
        scales: 1D BF16 scales tensor.
        orig_shape: Original shape tuple.
        group_size: Group size (default: 32).
        out_buffer: Optional pre-allocated destination tensor to eliminate allocator overhead.

    Returns:
        Decompressed BF16 tensor reshaped to orig_shape.
    """
    numel = q_int8.numel()
    device = q_int8.device
    
    if out_buffer is None:
        out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=device)

    # Use fused Triton kernel on GPU if available and CUDA/HIP enabled
    if HAS_TRITON and device.type in ("cuda", "hip"):
        # Try optimized kernel (autotuned if available)
        try:
            if '_triton_dequant_kernel_opt' in globals():
                # Check if autotune is active (kernel has .configs)
                is_autotuned = hasattr(_triton_dequant_kernel_opt, 'configs') or '_has_autotune' in globals() and globals().get('_has_autotune')
                if is_autotuned:
                    grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
                    _triton_dequant_kernel_opt[grid](
                        q_int8, scales.flatten(), out_buffer, numel,
                        GROUP_SIZE=group_size,
                    )
                else:
                    BLOCK_SIZE = 1024
                    grid = (triton.cdiv(numel, BLOCK_SIZE),)
                    _triton_dequant_kernel_opt[grid](
                        q_int8, scales.flatten(), out_buffer, numel,
                        BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size,
                    )
                return out_buffer
        except Exception:
            pass
        # Fallback to original fixed 512 kernel
        BLOCK_SIZE = 512
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _triton_dequant_kernel[grid](
            q_int8, scales.flatten(), out_buffer, numel,
            BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size,
            num_warps=2
        )
        return out_buffer
    
    # Vectorized fast PyTorch fallback - flatten scales for batched [B,seq,dim]
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        q_int8_padded = F.pad(q_int8, (0, pad_len))
    else:
        q_int8_padded = q_int8
        
    blocks = q_int8_padded.view(-1, group_size).float()
    scales_flat = scales.flatten()
    dequant = blocks * scales_flat.unsqueeze(-1).float()
    out_buffer.copy_(dequant.flatten()[:numel].view(orig_shape).to(torch.bfloat16))
    return out_buffer


AMO_BQ_PRESETS = {
    # (num_candidates, lo, hi, description)
    "fast":      (16, 0.95, 1.05, "0.49% @ 6.9ms, 5.4M, fastest, -12% vs sym"),
    "balanced":  (32, 0.95, 1.05, "0.478% @ 13ms, sweet spot, -13.7% vs sym"),
    "accurate":  (48, 0.95, 1.10, "0.473% @ 49ms, best G32 accuracy, -14.6% vs sym"),
    "max":       (64, 0.85, 1.10, "0.468% @ 59ms, diminishing returns"),
    # no-search baseline for reference (not a preset, handled via amo_lo=hi=1.0,N=1)
}

def _resolve_amo_preset(mode: Optional[str], num_candidates: Optional[int], lo: Optional[float], hi: Optional[float]):
    if mode is None:
        return num_candidates, lo, hi
    if mode not in AMO_BQ_PRESETS:
        raise ValueError(f"Unknown amo_mode {mode!r}, choose from {list(AMO_BQ_PRESETS.keys())}")
    preset_N, preset_lo, preset_hi, _ = AMO_BQ_PRESETS[mode]
    return preset_N if num_candidates is None else num_candidates, \
           preset_lo if lo is None else lo, \
           preset_hi if hi is None else hi


def quantize_int8_amo_bq(
    x: torch.Tensor,
    group_size: int = 32,
    num_candidates: int = 32,
    lo: float = 0.95,
    hi: float = 1.05,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    Asymmetric MSE-Optimal Block Quantization (AMO-BQ).

    Per-block min-max + zero-point with parallel candidate clipping search.
    For each block [G] we evaluate ``num_candidates`` scale multipliers
    ``m in [lo, hi]``:

        s_c = (b_max - b_min)/255 * m
        zp_c = clamp(round(-b_min / s_c), 0, 255)
        q_c = clamp(round(x / s_c + zp_c), 0, 255)
        rec_c = (q_c - zp_c) * s_c

    Picks ``m`` minimizing ``||x - rec_c||^2`` per block.  Chunked over
    blocks to cap VRAM (~8192 blocks / chunk).

    Storage: 1B data + 2B BF16 scale + 1B zp per 32-elem = 1.09375 B/elem
             (1.83x vs BF16, +0.03125 vs sym 1.0625).

    Args:
        x: Input tensor FP32/BF16 any shape.
        group_size: Block size (default 32).
        num_candidates: Candidates in [lo,hi] (default 48, ignored if mode set).
        lo, hi: Multiplier range (default 0.85-1.10, ignored if mode set).
        mode: Preset "fast" (16,0.95-1.05), "balanced" (32,0.95-1.05),
              "accurate" (48,0.95-1.10), "max" (64,0.85-1.10). Overrides N/lo/hi if set.

    Returns:
        q_uint8: Flat 1D uint8 tensor.
        scales: 1D BF16 tensor [num_blocks]
        zero_points: 1D uint8 tensor [num_blocks]
        orig_shape: Original shape tuple.
    """
    # Resolve preset if mode given
    if mode is not None:
        num_candidates, lo, hi = _resolve_amo_preset(mode, None, None, None)

    # Try fused Triton single-pass quant for GPU (offline but much faster)
    # Skip fused for NaN/Inf (Triton min/max would propagate NaN)
    if HAS_TRITON and x.is_cuda and x.device.type in ("cuda", "hip") and torch.isfinite(x).all():
        try:
            from .fused_ops import quantize_amo_fused_gpu
            return quantize_amo_fused_gpu(x, group_size=group_size, mode=mode, num_candidates=num_candidates, lo=lo, hi=hi)
        except Exception:
            pass  # fall back to PyTorch chunked

    orig_shape = x.shape
    x_flat = x.flatten().float()
    numel = x.numel()
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        x_flat = F.pad(x_flat, (0, pad_len))
    x_blocks = x_flat.view(-1, group_size)  # [M,G]
    num_blocks = x_blocks.shape[0]

    x_clean, is_finite = _sanitize_blocks(x_blocks)
    # Use clean for stats; keep original for error calc but mask NaNs
    b_min = x_clean.amin(dim=-1, keepdim=True)  # [M,1]
    b_max = x_clean.amax(dim=-1, keepdim=True)  # [M,1]
    # If block all non-finite, set range 1e-8
    b_range = (b_max - b_min).clamp(min=1e-8)  # [M,1]
    b_range = torch.where(torch.isfinite(b_range), b_range, torch.ones_like(b_range)*1e-8)
    s0 = b_range / 255.0  # [M,1]
    s0 = torch.where(torch.isfinite(s0), s0, torch.ones_like(s0)* (1e-8/255.0))

    multipliers = torch.linspace(lo, hi, num_candidates, device=x.device, dtype=torch.float32)  # [C]

    # Chunked search to avoid OOM (M up to ~170k for 5M elems)
    chunk = 8192
    best_scales = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    best_zps = torch.empty(num_blocks, device=x.device, dtype=torch.float32)

    for start in range(0, num_blocks, chunk):
        end = min(start + chunk, num_blocks)
        xb_c = x_blocks[start:end]  # [Bc,G] original (with NaN)
        xb_clean_c = x_clean[start:end]  # [Bc,G] sanitized
        is_finite_c = is_finite[start:end]  # [Bc,G]
        s0_c = s0[start:end]  # [Bc,1]
        b_min_c = b_min[start:end]  # [Bc,1]

        # [Bc,C,1]
        cand_scales = s0_c.unsqueeze(1) * multipliers.view(1, -1, 1)
        # zp per candidate: [Bc,C,1]
        cand_zps = torch.clamp(torch.round(-b_min_c.unsqueeze(1) / cand_scales), 0, 255)

        # Quant candidates: [Bc,C,G] using clean
        cand_q = torch.clamp(torch.round(xb_clean_c.unsqueeze(1) / cand_scales + cand_zps), 0, 255)
        cand_rec = (cand_q - cand_zps) * cand_scales  # [Bc,C,G]
        # Mask out non-finite positions for error (0 diff)
        diff = torch.where(is_finite_c.unsqueeze(1), xb_clean_c.unsqueeze(1) - cand_rec, torch.zeros_like(cand_rec))
        cand_err = (diff ** 2).sum(dim=-1)  # [Bc,C]

        best_idx = cand_err.argmin(dim=-1)  # [Bc]
        arange = torch.arange(end - start, device=x.device)
        best_scales[start:end] = cand_scales[arange, best_idx].squeeze(-1)
        best_zps[start:end] = cand_zps[arange, best_idx].squeeze(-1)

    # Final quant with best params (use clean, then mask)
    best_scales_f = best_scales.unsqueeze(-1)  # [M,1]
    best_zps_f = best_zps.unsqueeze(-1)  # [M,1]
    q_blocks = torch.clamp(torch.round(x_clean / best_scales_f + best_zps_f), 0, 255).to(torch.uint8)
    # For non-finite, set q = zp so rec = 0
    if not is_finite.all():
        q_blocks = torch.where(is_finite.view_as(q_blocks), q_blocks, best_zps_f.expand_as(q_blocks).to(torch.uint8))
    q_flat = q_blocks.flatten()[:numel]

    return q_flat, best_scales.to(torch.bfloat16), best_zps.to(torch.uint8), orig_shape


def dequantize_int8_amo_bq(
    q_uint8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    orig_shape: Tuple[int, ...],
    group_size: int = 32,
    out_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize AMO-BQ: rec = (q - zp) * scale.

    Args:
        q_uint8: Flat 1D uint8 tensor.
        scales: 1D BF16 scales [num_blocks].
        zero_points: 1D uint8 zp [num_blocks].
        orig_shape: Original shape.
        group_size: Group size.
        out_buffer: Optional pre-allocated BF16 output.

    Returns:
        BF16 tensor of orig_shape.
    """
    numel = q_uint8.numel()
    device = q_uint8.device
    if out_buffer is None:
        out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=device)

    if HAS_TRITON and device.type in ("cuda", "hip"):
        # Try optimized asym kernel
        try:
            if '_triton_dequant_asym_kernel_opt' in globals():
                is_autotuned = hasattr(_triton_dequant_asym_kernel_opt, 'configs') or '_has_autotune' in globals() and globals().get('_has_autotune')
                if is_autotuned:
                    grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
                    _triton_dequant_asym_kernel_opt[grid](
                        q_uint8, scales.flatten(), zero_points.flatten(), out_buffer, numel,
                        GROUP_SIZE=group_size,
                    )
                else:
                    BLOCK_SIZE = 1024
                    grid = (triton.cdiv(numel, BLOCK_SIZE),)
                    _triton_dequant_asym_kernel_opt[grid](
                        q_uint8, scales.flatten(), zero_points.flatten(), out_buffer, numel,
                        BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size,
                    )
                return out_buffer
        except Exception:
            pass
        BLOCK_SIZE = 512
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _triton_dequant_asym_kernel[grid](
            q_uint8, scales.flatten(), zero_points.flatten(), out_buffer, numel,
            BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size,
            num_warps=2
        )
        return out_buffer

    # PyTorch fallback - flatten scales/zp for batched inputs [B,seq,dim] -> flat
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        q_padded = F.pad(q_uint8, (0, pad_len))
    else:
        q_padded = q_uint8
    blocks = q_padded.view(-1, group_size).float()  # [M,G]
    scales_flat = scales.flatten()
    zp_flat = zero_points.flatten()
    zp_f = zp_flat.unsqueeze(-1).float()  # [M,1]
    sc_f = scales_flat.unsqueeze(-1).float()  # [M,1]
    dequant = (blocks - zp_f) * sc_f
    out_buffer.copy_(dequant.flatten()[:numel].view(orig_shape).to(torch.bfloat16))
    return out_buffer


class BlockwiseInt8Codec:
    """
    High-level codec interface for Block-wise INT8 compression.
    Modes:
      - default (sym G32): quantize_int8_g32
      - adaptive: quantize_int8_adaptive
      - amo_bq: quantize_int8_amo_bq (asym + MSE-optimal)
        presets via amo_mode: "fast" (16,0.95-1.05), "balanced" (32,0.95-1.05),
        "accurate" (48,0.95-1.10), "max" (64,0.85-1.10)
    """
    def __init__(self, group_size: int = 32, adaptive: bool = False, amo_bq: bool = False,
                 amo_lo: float = 0.95, amo_hi: float = 1.05, amo_candidates: int = 32,
                 amo_mode: Optional[str] = None):
        if group_size <=0 or group_size & (group_size-1) and group_size not in (24,48):
            # warn but allow; power-of-two recommended for Triton
            pass
        if amo_bq and amo_mode and amo_mode not in AMO_BQ_PRESETS:
            raise ValueError(f"amo_mode {amo_mode!r} unknown, choose {list(AMO_BQ_PRESETS.keys())}")
        if adaptive and amo_bq:
            raise ValueError("Choose one: adaptive or amo_bq, not both")
        self.group_size = group_size
        self.adaptive = adaptive
        self.amo_bq = amo_bq
        self.amo_mode = amo_mode
        # Resolve preset eagerly for inspectability
        if amo_mode is not None:
            n, lo, hi = _resolve_amo_preset(amo_mode, None, None, None)
            self.amo_candidates = n
            self.amo_lo = lo
            self.amo_hi = hi
        else:
            self.amo_lo = amo_lo
            self.amo_hi = amo_hi
            self.amo_candidates = amo_candidates

    def __repr__(self):
        if self.amo_bq:
            bpe = 1+3/self.group_size
            return f"BlockwiseInt8Codec(G={self.group_size}, amo_bq={self.amo_mode or f'{self.amo_candidates},{self.amo_lo}-{self.amo_hi}'} {bpe:.4f}B 1:{2/bpe:.2f}x)"
        if self.adaptive:
            return f"BlockwiseInt8Codec(G={self.group_size}, adaptive 0.90-1.05 1:{2/(1+2/self.group_size):.2f}x)"
        return f"BlockwiseInt8Codec(G={self.group_size}, sym 1:{2/(1+2/self.group_size):.2f}x)"

    def quantize(self, x: torch.Tensor):
        if self.amo_bq:
            return quantize_int8_amo_bq(
                x, group_size=self.group_size,
                num_candidates=self.amo_candidates, lo=self.amo_lo, hi=self.amo_hi,
                mode=self.amo_mode
            )
        if self.adaptive:
            return quantize_int8_adaptive(x, group_size=self.group_size)
        return quantize_int8_g32(x, group_size=self.group_size)

    def dequantize(
        self,
        q_int8: torch.Tensor,
        scales: torch.Tensor,
        orig_shape: Tuple[int, ...],
        zero_points: Optional[torch.Tensor] = None,
        out_buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if zero_points is not None:
            return dequantize_int8_amo_bq(
                q_int8, scales, zero_points, orig_shape,
                group_size=self.group_size, out_buffer=out_buffer
            )
        # Auto-detect AMO-BQ by dtype: zp is uint8, q is uint8 for AMO-BQ
        # Fall back to symmetric path for backward compat
        return dequantize_int8_g32(
            q_int8, scales, orig_shape, group_size=self.group_size, out_buffer=out_buffer
        )


# =============================================================================
# 8x GPU WAVELET CODEC (JPEG-XS Style CDF 5/3 Lifting + RCT)
# =============================================================================

def rct_forward(rgb: torch.Tensor) -> torch.Tensor:
    """
    Reversible Color Transform (RCT) RGB -> YCbCr (Integer shift-add).
    Y = (R + 2*G + B) >> 2
    Cb = B - G
    Cr = R - G
    """
    r, g, b = rgb[..., 0].to(torch.int32), rgb[..., 1].to(torch.int32), rgb[..., 2].to(torch.int32)
    y = (r + 2 * g + b) >> 2
    cb = b - g
    cr = r - g
    return torch.stack([y, cb, cr], dim=-1)


def rct_inverse(yuv: torch.Tensor) -> torch.Tensor:
    """
    Inverse Reversible Color Transform (RCT) YCbCr -> RGB.
    G = Y - ((Cb + Cr) >> 2)
    R = Cr + G
    B = Cb + G
    """
    y, cb, cr = yuv[..., 0].to(torch.int32), yuv[..., 1].to(torch.int32), yuv[..., 2].to(torch.int32)
    g = y - ((cb + cr) >> 2)
    r = cr + g
    b = cb + g
    return torch.stack([r, g, b], dim=-1).clamp(0, 255).to(torch.uint8)


def dwt_53_1d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D LeGall 5/3 (CDF 5/3) Wavelet Forward Lifting.

    Pair sums are truncated to the output length, so odd-length signals work
    (odd[j] pairs with even[j], even[j+1]).
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    No = odd.shape[-1]
    Ne = even.shape[-1]
    even_pad = F.pad(even, (0, 1), mode='replicate')
    d = odd - ((even_pad[..., :-1] + even_pad[..., 1:]) >> 1)[..., :No]
    d_pad = F.pad(d, (1, 1), mode='replicate')
    s = even + ((d_pad[..., :-1] + d_pad[..., 1:] + 2) >> 2)[..., :Ne]
    return s, d


def idwt_53_1d(s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """1D LeGall 5/3 (CDF 5/3) Wavelet Inverse Lifting. Mirrors forward."""
    Ns, Nd = s.shape[-1], d.shape[-1]
    d_pad = F.pad(d, (1, 1), mode='replicate')
    even = s - ((d_pad[..., :-1] + d_pad[..., 1:] + 2) >> 2)[..., :Ns]
    even_pad = F.pad(even, (0, 1), mode='replicate')
    odd = d + ((even_pad[..., :-1] + even_pad[..., 1:]) >> 1)[..., :Nd]
    out = torch.empty(s.shape[:-1] + (s.shape[-1] + d.shape[-1],), dtype=s.dtype, device=s.device)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return out


def _dwt_53_2d_step_batched(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched 2D 5/3 DWT for [B, H, W] -> 4x [B, H/2, W/2]."""
    s_r, d_r = dwt_53_1d(x)  # [B, H, W/2]
    # Column lift via transposed row lift
    LL_T, LH_T = dwt_53_1d(s_r.transpose(-2, -1))  # [B, W/2, H/2]
    HL_T, HH_T = dwt_53_1d(d_r.transpose(-2, -1))
    LL = LL_T.transpose(-2, -1)
    LH = LH_T.transpose(-2, -1)
    HL = HL_T.transpose(-2, -1)
    HH = HH_T.transpose(-2, -1)
    return LL, LH, HL, HH


def _idwt_53_2d_step_batched(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
    """Batched 2D 5/3 IDWT for [B, Hs, Ws] -> [B, H, W] where H=2*Hs, W=2*Ws."""
    # Column lift via transposed row lift
    s_r = idwt_53_1d(LL.transpose(-2, -1), LH.transpose(-2, -1)).transpose(-2, -1)
    d_r = idwt_53_1d(HL.transpose(-2, -1), HH.transpose(-2, -1)).transpose(-2, -1)
    return idwt_53_1d(s_r, d_r)


def _downsample2(x: torch.Tensor) -> torch.Tensor:
    """2x box downsample [B, H, W] int32 -> [B, H/2, W/2] (H, W even)."""
    return ((x[:, 0::2, 0::2] + x[:, 0::2, 1::2]
             + x[:, 1::2, 0::2] + x[:, 1::2, 1::2] + 2) >> 2)


def _upsample2(x: torch.Tensor) -> torch.Tensor:
    """2x nearest upsample [B, H, W] -> [B, 2H, 2W].

    Measured against a grid-matched tent ((3a+b)/4 and cell-copy variants) on
    COCO chroma: nearest wins by ~1dB — subsampled chroma here is
    high-frequency dominated (already JPEG-mangled), so neighbor averaging
    adds error. Also the cheapest option. Trivially bit-exact CPU/CUDA.
    """
    return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


def quantize_pixel_wavelet8x(
    img: torch.Tensor | np.ndarray,
    q_scale: float = 3.0,
    chroma420: bool = True,
) -> Tuple[dict, Tuple[int, int, int]]:
    """
    Quantize an RGB image tensor or array using 4-Level Dyadic Wavelet Lifting (JPEG-XS Style).
    chroma420=True (default): luma 4 DWT levels at full res, chroma 3 levels
      at half res (planes named by spatial equivalence; only L1 is luma-only).
    chroma420=False: full-res 4:4:4 chroma (all 3 channels have L1).
    Batched GPU path: channels in parallel via _dwt_53_2d_step_batched.
    """
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img).to(torch.int32)
    else:
        t = img.to(torch.int32)

    H, W, C = t.shape
    # Pad to multiple of 16 for 4 luma levels of DWT (chroma half-res => mult of 8, 3 levels)
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode='replicate').permute(1, 2, 0)

    # RCT -> luma full-res + chroma full or half res
    # GPU on-the-fly: if t is on cuda, all ops run as CUDA kernels
    yuv = rct_forward(t)  # [Hp, Wp, 3]
    yuv_b = yuv.permute(2, 0, 1).contiguous()  # [3, Hp, Wp] int32
    Y = yuv_b[0:1]  # [1, Hp, Wp]
    Cc = _downsample2(yuv_b[1:3]) if chroma420 else yuv_b[1:3]  # [2, Hc, Wc] or [2, Hp, Wp]

    # Luma 4-level DWT; chroma 3-level (420) or 4-level (444) DWT
    YL1, YLH1, YHL1, YHH1 = _dwt_53_2d_step_batched(Y)
    YL2, YLH2, YHL2, YHH2 = _dwt_53_2d_step_batched(YL1)
    YL3, YLH3, YHL3, YHH3 = _dwt_53_2d_step_batched(YL2)
    YL4, YLH4, YHL4, YHH4 = _dwt_53_2d_step_batched(YL3)
    if chroma420:
        CL2, CLH2, CHL2, CHH2 = _dwt_53_2d_step_batched(Cc)
        CL3, CLH3, CHL3, CHH3 = _dwt_53_2d_step_batched(CL2)
        CL4, CLH4, CHL4, CHH4 = _dwt_53_2d_step_batched(CL3)
    else:
        CL1, CLH1, CHL1, CHH1 = _dwt_53_2d_step_batched(Cc)
        CL2, CLH2, CHL2, CHH2 = _dwt_53_2d_step_batched(CL1)
        CL3, CLH3, CHL3, CHH3 = _dwt_53_2d_step_batched(CL2)
        CL4, CLH4, CHL4, CHH4 = _dwt_53_2d_step_batched(CL3)

    # Per-level deadzone steps [3]: luma c_f=1.0, chroma c_f=1.8 at the
    # spatially equivalent level (chroma L2 <-> luma L2, etc.)
    dev = yuv_b.device
    q_l4 = torch.tensor([max(1, int(round(q_scale*0.5*1.0))), max(1, int(round(q_scale*0.5*1.8))), max(1, int(round(q_scale*0.5*1.8)))], device=dev)
    q_l3 = torch.tensor([max(1, int(round(q_scale*1.0*1.0))), max(1, int(round(q_scale*1.0*1.8))), max(1, int(round(q_scale*1.0*1.8)))], device=dev)
    q_l2 = torch.tensor([max(1, int(round(q_scale*2.0*1.0))), max(1, int(round(q_scale*2.0*1.8))), max(1, int(round(q_scale*2.0*1.8)))], device=dev)
    q_l1 = torch.tensor([max(1, int(round(q_scale*4.0*1.0))), max(1, int(round(q_scale*4.0*1.8))), max(1, int(round(q_scale*4.0*1.8)))], device=dev)

    # Stack levels across channels (shapes match by spatial-equivalence naming)
    def stk(*planes):
        return torch.cat(planes, dim=0)
    LL4 = stk(YL4, CL4)  # [3, H4, W4]
    LH4, HL4, HH4 = stk(YLH4, CLH4), stk(YHL4, CHL4), stk(YHH4, CHH4)
    LH3, HL3, HH3 = stk(YLH3, CLH3), stk(YHL3, CHL3), stk(YHH3, CHH3)
    LH2, HL2, HH2 = stk(YLH2, CLH2), stk(YHL2, CHL2), stk(YHH2, CHH2)
    if not chroma420:
        LH1, HL1, HH1 = stk(YLH1, CLH1), stk(YHL1, CHL1), stk(YHH1, CHH1)

    # Quantize batched: [3, H, W] / [3,1,1] -> int8
    LL4_q = LL4.to(torch.int16)  # keep per-plane int16

    LH4_q = torch.div(LH4, q_l4.view(3,1,1), rounding_mode='trunc').to(torch.int8)
    HL4_q = torch.div(HL4, q_l4.view(3,1,1), rounding_mode='trunc').to(torch.int8)
    HH4_q = torch.div(HH4, (q_l4*2).view(3,1,1), rounding_mode='trunc').to(torch.int8)

    LH3_q = torch.div(LH3, q_l3.view(3,1,1), rounding_mode='trunc').to(torch.int8)
    HL3_q = torch.div(HL3, q_l3.view(3,1,1), rounding_mode='trunc').to(torch.int8)
    HH3_q = torch.div(HH3, (q_l3*2).view(3,1,1), rounding_mode='trunc').to(torch.int8)

    LH2_q = torch.div(LH2, q_l2.view(3,1,1), rounding_mode='trunc').to(torch.int8)
    HL2_q = torch.div(HL2, q_l2.view(3,1,1), rounding_mode='trunc').to(torch.int8)
    HH2_q = torch.div(HH2, (q_l2*2).view(3,1,1), rounding_mode='trunc').to(torch.int8)

    if chroma420:
        # L1 luma-only; zero chroma HH at its finest level (L2)
        LH1_q = torch.div(YLH1, int(q_l1[0]), rounding_mode='trunc').to(torch.int8)
        HL1_q = torch.div(YHL1, int(q_l1[0]), rounding_mode='trunc').to(torch.int8)
        HH1_q = torch.div(YHH1, int(q_l1[0])*2, rounding_mode='trunc').to(torch.int8)
        HH2_q[1:] = 0
    else:
        LH1_q = torch.div(LH1, q_l1.view(3,1,1), rounding_mode='trunc').to(torch.int8)
        HL1_q = torch.div(HL1, q_l1.view(3,1,1), rounding_mode='trunc').to(torch.int8)
        HH1_q = torch.div(HH1, (q_l1*2).view(3,1,1), rounding_mode='trunc').to(torch.int8)
        HH1_q[1:] = 0

    # Unbatch to list of dicts (420 chroma dicts have no L1 key)
    encoded_channels = []
    for c in range(3):
        d = {
            'LL4': LL4_q[c],
            'L4': (LH4_q[c], HL4_q[c], HH4_q[c], int(q_l4[c].item())),
            'L3': (LH3_q[c], HL3_q[c], HH3_q[c], int(q_l3[c].item())),
            'L2': (LH2_q[c], HL2_q[c], HH2_q[c], int(q_l2[c].item())),
        }
        if c == 0 or not chroma420:
            d['L1'] = (LH1_q[c if not chroma420 else 0], HL1_q[c if not chroma420 else 0],
                       HH1_q[c if not chroma420 else 0], int(q_l1[c].item()))
        encoded_channels.append(d)

    return {
        'channels': encoded_channels,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'orig_shape': (H, W, C),
        'chroma420': chroma420,
    }, (H, W, C)


def _dequant_static_plane(q: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Deadzone-aware reconstruction: bin centroid instead of bin edge.

    Quantize truncates toward zero (deadzone 2*step), so reconstruct at
    k*step + sign(k)*step//2. Zero bin is unaffected. Must match bit-exactly
    in all decode paths (CPU stacks + Triton fused).
    """
    qi = q.to(torch.int32)
    return qi * step + torch.sign(qi) * (step // 2)


def _chroma_is_subsampled(packed_meta: dict) -> bool:
    """4:2:0 iff chroma dicts lack finest-level keys (4:4:4 keeps full-res chroma).

    Handles both schemas: static level keys ('L1') and adaptive plane keys
    ('LH1'/'HL1'/'HH1').
    """
    ch = packed_meta['channels']
    if len(ch) <= 1:
        return True
    c1 = ch[1]
    if 'L1' in c1 or 'LH1' in c1 or 'HL1' in c1 or 'HH1' in c1:
        return False
    return True


def _wavelet_batched_stacks(packed_meta: dict, dev: torch.device):
    """Helper: build batched [3, H, W] stacks for each level to reduce kernel launches 3x."""
    channels_data = packed_meta['channels']
    # LL4 [3, H4, W4] int32
    LL4 = torch.stack([c['LL4'].to(dev).to(torch.int32) for c in channels_data], dim=0)
    # Level 4
    q4 = torch.tensor([c['L4'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
    LH4 = _dequant_static_plane(torch.stack([c['L4'][0] for c in channels_data], dim=0).to(dev), q4)
    HL4 = _dequant_static_plane(torch.stack([c['L4'][1] for c in channels_data], dim=0).to(dev), q4)
    HH4 = _dequant_static_plane(torch.stack([c['L4'][2] for c in channels_data], dim=0).to(dev), q4 * 2)
    # Level 3
    q3 = torch.tensor([c['L3'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
    LH3 = _dequant_static_plane(torch.stack([c['L3'][0] for c in channels_data], dim=0).to(dev), q3)
    HL3 = _dequant_static_plane(torch.stack([c['L3'][1] for c in channels_data], dim=0).to(dev), q3)
    HH3 = _dequant_static_plane(torch.stack([c['L3'][2] for c in channels_data], dim=0).to(dev), q3 * 2)
    # Level 2
    q2 = torch.tensor([c['L2'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
    LH2 = _dequant_static_plane(torch.stack([c['L2'][0] for c in channels_data], dim=0).to(dev), q2)
    HL2 = _dequant_static_plane(torch.stack([c['L2'][1] for c in channels_data], dim=0).to(dev), q2)
    HH2 = _dequant_static_plane(torch.stack([c['L2'][2] for c in channels_data], dim=0).to(dev), q2 * 2)
    # Level 1: all 3 channels in 4:4:4 mode, luma-only under 4:2:0
    if _chroma_is_subsampled(packed_meta):
        yc = channels_data[0]
        q1 = torch.tensor([yc['L1'][3]], device=dev, dtype=torch.int32).view(1, 1, 1)
        LH1 = _dequant_static_plane(torch.stack([yc['L1'][0]], dim=0).to(dev), q1)
        HL1 = _dequant_static_plane(torch.stack([yc['L1'][1]], dim=0).to(dev), q1)
        HH1 = _dequant_static_plane(torch.stack([yc['L1'][2]], dim=0).to(dev), q1 * 2)
    else:
        q1 = torch.tensor([c['L1'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
        LH1 = _dequant_static_plane(torch.stack([c['L1'][0] for c in channels_data], dim=0).to(dev), q1)
        HL1 = _dequant_static_plane(torch.stack([c['L1'][1] for c in channels_data], dim=0).to(dev), q1)
        HH1 = _dequant_static_plane(torch.stack([c['L1'][2] for c in channels_data], dim=0).to(dev), q1 * 2)
    return LL4, (LH4, HL4, HH4), (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)


def dequantize_pixel_wavelet8x(
    packed_meta: dict,
    device: str | torch.device = "cpu",
    out_buffer: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Dequantize 4-Level Dyadic Wavelet bitstream into RGB uint8 image tensor [H, W, 3].
    GPU path: fused Triton (RCT + row/col lifting) when available, else batched PyTorch.
    Zero intermediate VRAM when out_buffer provided; otherwise allocates minimal.
    Supports both static (global q) and adaptive (per-block 4b) packed formats.
    """
    dev = torch.device(device)
    # Adaptive format detection - dispatch to adaptive decoder
    if packed_meta.get('adaptive', False):
        return dequantize_pixel_wavelet_adaptive(packed_meta, device=device, out_buffer=out_buffer)
    # Triton fused path - 8x fewer launches, coalesced 128-bit loads, shift vs div
    if HAS_TRITON and dev.type in ("cuda", "hip"):
        try:
            from .fused_ops import dequantize_fused_wavelet8x_gpu
            return dequantize_fused_wavelet8x_gpu(packed_meta, device=device, out_buffer=out_buffer)
        except Exception:
            pass

    channels_data = packed_meta['channels']
    H, W, C = packed_meta['orig_shape']

    # Batched PyTorch path: 4:4:4 -> full 4-stage chain on [3, ...];
    # 4:2:0 -> luma 4-stage chain + chroma 3-stage chain + 2x upsample
    LL4, (LH4, HL4, HH4), (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = _wavelet_batched_stacks(packed_meta, dev)

    if not _chroma_is_subsampled(packed_meta):
        rec_ll3 = _idwt_53_2d_step_batched(LL4, LH4, HL4, HH4)
        rec_ll2 = _idwt_53_2d_step_batched(rec_ll3, LH3, HL3, HH3)
        rec_ll1 = _idwt_53_2d_step_batched(rec_ll2, LH2, HL2, HH2)
        rec_yuv_batched = _idwt_53_2d_step_batched(rec_ll1, LH1, HL1, HH1)  # [3, Hp, Wp]
    else:
        # LL4/L4/L3/L2 stacks are [3, ...] (Y + subsampled C); L1 is [1, ...] (Y only)
        rec_ll3 = _idwt_53_2d_step_batched(LL4, LH4, HL4, HH4)  # [3, Hp/8]
        rec_ll2_y = _idwt_53_2d_step_batched(rec_ll3[0:1], LH3[0:1], HL3[0:1], HH3[0:1])
        rec_ll2_c = _idwt_53_2d_step_batched(rec_ll3[1:3], LH3[1:3], HL3[1:3], HH3[1:3])
        rec_ll1_y = _idwt_53_2d_step_batched(rec_ll2_y, LH2[0:1], HL2[0:1], HH2[0:1])
        rec_c_half = _idwt_53_2d_step_batched(rec_ll2_c, LH2[1:3], HL2[1:3], HH2[1:3])  # [2, Hc, Wc]
        rec_y = _idwt_53_2d_step_batched(rec_ll1_y, LH1, HL1, HH1)  # [1, Hp, Wp]
        rec_yuv_batched = torch.cat([rec_y, _upsample2(rec_c_half)], dim=0)  # [3, Hp, Wp]

    # RCT inverse: [3, Hp, Wp] -> [Hp, Wp, 3]
    rec_yuv = rec_yuv_batched.permute(1, 2, 0)  # [Hp, Wp, 3]
    rec_rgb_full = rct_inverse(rec_yuv)
    rec_rgb = rec_rgb_full[:H, :W, :]

    if out_buffer is not None:
        out_buffer.copy_(rec_rgb)
        return out_buffer
    return rec_rgb


# =============================================================================
# TUNABLE ADAPTIVE WAVELET (XS-native RDO, 4b idx, GPU/CPU)
# =============================================================================
# Default 16-entry codebook for m, 4b per block (0.125b/elem). Quant step = base_q * m.
# All 16 nibble values used (was 8 of 16) at zero extra bit cost.
# Base q per level: q_l4=0.5*q_scale*c_f, q_l3=1.0*, q_l2=2*, q_l1=4*
ADAPTIVE_CODEBOOK = torch.tensor(
    [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375,
     1.5, 1.625, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], dtype=torch.float32)
# Measured synthesis energy gains (pixel_SSE / coeff_SSE) through float 5/3
# IDWT (no rounding noise), normalized to LH1=1.0 (see benchmarks/xs_gains.py).
# LH/HL pairs identical (transform is exactly symmetric).
# RDO multiplies by RCT channel energy: Y=3.0, Cb/Cr=0.69 (traced integer ops).
SUBBAND_GAINS = {
    "LH4": 30.165, "HL4": 30.165, "HH4": 8.589,
    "LH3": 7.907, "HL3": 7.907, "HH3": 2.333,
    "LH2": 2.351, "HL2": 2.351, "HH2": 0.788,
    "LH1": 1.000, "HL1": 1.000, "HH1": 0.479,
}
RCT_GAIN_Y = 3.0
RCT_GAIN_C = 0.69
# Tunable presets: q_scale, lamb  (lamb trades D vs R in RDO)
# Measured on 16x COCO val 336^2 with sparse bitstream (PSNR dB / actual ratio):
WAVELET_ADAPTIVE_PRESETS = {
    "ultra":      (1.0, 1.0),   # ~44.4dB 3.22x highest fidelity (4:4:4)
    "high":       (3.0, 5.0),   # ~39.8dB 5.57x balanced high quality (4:4:4)
    "balanced":   (3.0, 10.0),  # ~36.8dB 7.80x default (4:2:0)
    "compress":   (5.0, 20.0),  # ~35.3dB 9.76x high compress (4:2:0)
    "ultra_comp": (8.0, 50.0),  # ~33.0dB 13.6x max compress (4:2:0)
}

def _pack_4b(idx: torch.Tensor) -> torch.Tensor:
    """Pack M uint8 idx (0-15) into ceil(M/2) uint8 bytes (low nibble first)."""
    M = idx.numel()
    if M % 2 == 1:
        idx = F.pad(idx, (0, 1))
    idx = idx.view(-1, 2)
    packed = (idx[:, 1].to(torch.uint8) << 4) | idx[:, 0].to(torch.uint8)
    return packed

def _unpack_4b(packed: torch.Tensor, M: int) -> torch.Tensor:
    """Unpack 4b packed bytes to M uint8 idx."""
    # packed [(M+1)//2]
    # expand
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    # interleave
    unpacked = torch.empty(M, dtype=torch.uint8, device=packed.device)
    # even indices -> low, odd -> high
    # Use vectorized
    # packed has ceil(M/2) entries, each gives 2 idx
    # Create [ceil,2] then flatten
    # For odd M, last high is pad
    tmp = torch.stack([low, high], dim=1).view(-1)[:M]
    return tmp

def _quant_adaptive_plane(
    coeff: torch.Tensor,  # [H,W] int32
    base_q: int,
    codebook: torch.Tensor,  # [C] float
    lamb: float,
    G: int = 32,
    gain: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-block RDO for one plane. Returns (q_int8 [H,W], idx_packed uint8 [(M+1)//2], rec_int32 [H,W]).
    GPU/CPU agnostic - runs on coeff.device.
    Rate uses the true sparse-bitstream price: 1 bit for an all-zero block
    (occupancy only), else 1 occ + 1 mode + min(32 flat, 8+4k hier) mask bits
    + 4 idx bits + 8 per nonzero coefficient (k = nonempty nibbles).
    Distortion is weighted by the subband synthesis gain (pixel_SSE/coeff_SSE).
    """
    H, W = coeff.shape
    flat = coeff.flatten().float()
    N = flat.numel()
    pad = (G - N % G) % G
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, G)  # [M,G]
    M = blocks.shape[0]
    C = codebook.numel()
    # cand steps [C]
    cand_steps = base_q * codebook.to(blocks.device).float()  # [C]
    steps = cand_steps.view(1, -1, 1)  # [1,C,1]
    # Vectorized RDO: cand_q [M,C,G], cand_rec [M,C,G]
    cand_q = torch.round(blocks.unsqueeze(1) / steps).clamp(-128, 127)
    cand_rec = cand_q * steps
    D = ((blocks.unsqueeze(1) - cand_rec) ** 2).sum(-1) * gain  # [M,C]
    nz = (cand_q != 0)  # [M,C,G] bool (reused for nnz + nibble counts)
    nnz = nz.sum(-1).float()  # [M,C]
    if G == 32:
        # True hierarchical-mask price: 1 occ bit; empty blocks stop there.
        # Occupied: +1 mode bit + min(32 flat, 8+4k hier) mask bits + 4 idx + 8/coeff,
        # k = #nonempty nibbles (hier wins iff k<=5).
        k = nz.view(M, C, 8, 4).any(-1).sum(-1).float()  # [M,C]
        maskbits = torch.where(k <= 5, 8.0 + 4.0 * k, torch.full_like(k, 32.0))
        R = torch.where(nnz == 0, torch.ones_like(nnz), 6.0 + maskbits + 8.0 * nnz)
    else:
        R = torch.where(nnz == 0, torch.ones_like(nnz), 37.0 + 8.0 * nnz)
    cost = D + lamb * R
    best = cost.argmin(-1)  # [M]
    best_steps = cand_steps[best]  # [M]
    q_blocks = torch.round(blocks / best_steps.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    # TCQ-lite refinement: per-coefficient choice among {q0-1, q0, q0+1}.
    # Exact (not approximate): given the block step, rate is additive per
    # coefficient (8b iff nonzero), so independent per-coeff argmin is optimal.
    # No trellis/state needed. Block step (idx) unchanged -> format untouched.
    s = best_steps.unsqueeze(-1)  # [M,1]
    q0 = q_blocks.to(torch.int16)
    qc = torch.stack([q0 - 1, q0, q0 + 1], dim=-1).clamp(-128, 127)  # [M,G,3]
    dc = gain * (blocks.unsqueeze(-1).float() - qc.float() * s.unsqueeze(-1).float()) ** 2
    rc = lamb * 8.0 * (qc != 0).float()
    q_blocks = qc.gather(-1, (dc + rc).argmin(-1, keepdim=True)).squeeze(-1).to(torch.int8)
    rec_blocks = q_blocks.float() * best_steps.unsqueeze(-1)
    rec = rec_blocks.view(-1)[:N].view(H, W).to(torch.int32)
    q_plane = q_blocks.view(-1)[:N].view(H, W).to(torch.int8)
    idx = best.to(torch.uint8)  # [M] 0-15
    idx_packed = _pack_4b(idx)
    return q_plane, idx_packed, rec

def _dequant_adaptive_plane(
    q_plane: torch.Tensor,  # [H,W] int8
    idx_packed: torch.Tensor,  # [(M+1)//2] uint8
    base_q: int,
    codebook: torch.Tensor,
    G: int = 32,
) -> torch.Tensor:
    """Dequant one plane: q * base_q * codebook[idx]. Supports packed 4b idx."""
    H, W = q_plane.shape
    N = H * W
    M = (N + G - 1) // G
    idx = _unpack_4b(idx_packed, M)  # [M]
    # Expand idx per element
    # block_id per element
    # Use repeat_interleave
    idx_expanded = torch.repeat_interleave(idx, G)[:N].view(H, W)
    steps = base_q * codebook[idx_expanded.long()].to(q_plane.device).float()
    rec = q_plane.to(torch.float32) * steps
    return rec.to(torch.int32)

def quantize_pixel_wavelet_adaptive(
    img: torch.Tensor | np.ndarray,
    q_scale: float = 3.0,
    lamb: float = 5.0,
    G: int = 32,
    codebook: torch.Tensor | None = None,
    mode: str | None = None,
    chroma420: bool | None = None,
) -> tuple[dict, tuple[int, int, int]]:
    """
    Tunable adaptive JPEG-XS wavelet: per-block G=32 RDO D+lamb*R with 4b codebook.
    Lower lamb/q_scale -> lower error (higher fidelity), higher -> higher compression.
    Presets: ultra/high/balanced/compress/ultra_comp (see WAVELET_ADAPTIVE_PRESETS).
    chroma420: 4:2:0 subsampled chroma (None = auto: 4:4:4 for ultra/high, 4:2:0 below).

    Args:
        q_scale: base deadzone scale (0.5-8.0, default 3.0). Smaller = finer.
        lamb: RDO tradeoff (0.1-50, default 5.0). Smaller = favor PSNR, larger = favor bits.
        G: block size (32 default, 16/64 also tunable but 32 is Pareto).
        codebook: [C] float m values (default 16-entry [0.5,3.0] -> 4b).
        mode: preset name overrides q_scale/lamb.
        chroma420: subsample chroma 2x (None = auto by mode).

    Returns packed_meta with adaptive=True, plus orig_shape. Use dequantize_pixel_wavelet_adaptive.
    """
    if mode is not None:
        if mode not in WAVELET_ADAPTIVE_PRESETS:
            raise ValueError(f"Unknown mode {mode}, choose from {list(WAVELET_ADAPTIVE_PRESETS)}")
        q_scale, lamb = WAVELET_ADAPTIVE_PRESETS[mode]
    if chroma420 is None:
        chroma420 = False if mode in ("ultra", "high") else True
    if codebook is None:
        codebook = ADAPTIVE_CODEBOOK
    codebook = codebook.to(torch.float32)

    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img).to(torch.int32)
    else:
        t = img.to(torch.int32)

    H, W, C = t.shape
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode='replicate').permute(1, 2, 0)

    # RCT -> luma full-res + chroma full (444) or half res (420)
    yuv = rct_forward(t)  # [Hp,Wp,3]
    yuv_b = yuv.permute(2, 0, 1).contiguous()  # [3,Hp,Wp]
    Y = yuv_b[0:1]  # [1, Hp, Wp]
    Cc = _downsample2(yuv_b[1:3]) if chroma420 else yuv_b[1:3]

    # Luma 4-level DWT; chroma 3-level (420) or 4-level (444) DWT
    YL1, YLH1, YHL1, YHH1 = _dwt_53_2d_step_batched(Y)
    YL2, YLH2, YHL2, YHH2 = _dwt_53_2d_step_batched(YL1)
    YL3, YLH3, YHL3, YHH3 = _dwt_53_2d_step_batched(YL2)
    YL4, YLH4, YHL4, YHH4 = _dwt_53_2d_step_batched(YL3)
    if chroma420:
        CL2, CLH2, CHL2, CHH2 = _dwt_53_2d_step_batched(Cc)
        CL3, CLH3, CHL3, CHH3 = _dwt_53_2d_step_batched(CL2)
        CL4, CLH4, CHL4, CHH4 = _dwt_53_2d_step_batched(CL3)
    else:
        CL1, CLH1, CHL1, CHH1 = _dwt_53_2d_step_batched(Cc)
        CL2, CLH2, CHL2, CHH2 = _dwt_53_2d_step_batched(CL1)
        CL3, CLH3, CHL3, CHH3 = _dwt_53_2d_step_batched(CL2)
        CL4, CLH4, CHL4, CHH4 = _dwt_53_2d_step_batched(CL3)

    # Per-level base_q per channel [3] (chroma at spatially equivalent level x1.8).
    # NOTE: bases stay fine-grained (0.5/1/2/4); cross-level allocation is the
    # RDO's job via SUBBAND_GAINS x RCT channel gains, not the base scale.
    # (Static-path equal-slope rescaling must NOT leak in here: it would lift
    # the finest available step and cap top-end fidelity.)
    dev = yuv_b.device
    def make_qs(scale, c_f_y=1.0, c_f_c=1.8):
        return torch.tensor([max(1, int(round(scale*c_f_y))), max(1, int(round(scale*c_f_c))), max(1, int(round(scale*c_f_c)))], device=dev)
    q_l4 = make_qs(q_scale*0.5)
    q_l3 = make_qs(q_scale*1.0)
    q_l2 = make_qs(q_scale*2.0)
    q_l1 = make_qs(q_scale*4.0)

    # Stack shared levels across channels (shapes match by spatial naming)
    def stk(*planes):
        return torch.cat(planes, dim=0)
    LL4 = stk(YL4, CL4)  # [3, H4, W4]
    LH4, HL4, HH4 = stk(YLH4, CLH4), stk(YHL4, CHL4), stk(YHH4, CHH4)
    LH3, HL3, HH3 = stk(YLH3, CLH3), stk(YHL3, CHL3), stk(YHH3, CHH3)
    LH2, HL2, HH2 = stk(YLH2, CLH2), stk(YHL2, CHL2), stk(YHH2, CHH2)
    # L1 luma-only
    LL4_q = LL4.to(torch.int16)

    planes_y = [
        (YLH4[0], q_l4, "LH4"), (YHL4[0], q_l4, "HL4"), (YHH4[0], q_l4*2, "HH4"),
        (YLH3[0], q_l3, "LH3"), (YHL3[0], q_l3, "HL3"), (YHH3[0], q_l3*2, "HH3"),
        (YLH2[0], q_l2, "LH2"), (YHL2[0], q_l2, "HL2"), (YHH2[0], q_l2*2, "HH2"),
        (YLH1[0], q_l1, "LH1"), (YHL1[0], q_l1, "HL1"), (YHH1[0], q_l1*2, "HH1"),
    ]
    # Chroma planes share the stacked tensors; per-channel slice taken in loop.
    # (stacked [3,...] with Y at [0]; chroma loop uses entries 1, 2)
    planes_c = [
        (LH4, q_l4, "LH4"), (HL4, q_l4, "HL4"), (HH4, q_l4*2, "HH4"),
        (LH3, q_l3, "LH3"), (HL3, q_l3, "HL3"), (HH3, q_l3*2, "HH3"),
        (LH2, q_l2, "LH2"), (HL2, q_l2, "HL2"), (HH2, q_l2*2, "HH2"),
    ]
    if not chroma420:
        # 4:4:4 chroma also gets L1 (full-res), stacked for uniform [c] indexing
        LH1c, HL1c, HH1c = stk(YLH1, CLH1), stk(YHL1, CHL1), stk(YHH1, CHH1)
        planes_c += [(LH1c, q_l1, "LH1"), (HL1c, q_l1, "HL1"), (HH1c, q_l1*2, "HH1")]

    adaptive_channels = []  # per channel dict
    # Luma: 12 planes RDO. Chroma: 9 (420) or 12 (444) planes RDO.
    zero_hh = "HH2" if chroma420 else "HH1"  # finest chroma level
    for c in range(3):
        chan_dict = {'LL4': LL4_q[c]}
        plist = planes_y if c == 0 else planes_c
        for (coeff_all, base_q_all, name) in plist:
            coeff = coeff_all if c == 0 else coeff_all[c]  # [H,W]
            bq = int(base_q_all[0].item()) if c == 0 else int(base_q_all[c].item())
            # Zero chroma HH at its finest level
            if name == zero_hh and c > 0:
                # Zero plane - store zeros and packed zero idx with correct M
                q_zero = torch.zeros_like(coeff, dtype=torch.int8)
                Hc, Wc = coeff.shape
                N = Hc * Wc
                M = (N + G - 1) // G
                idx_packed = torch.zeros((M + 1) // 2, dtype=torch.uint8, device=dev)
                chan_dict[name] = (q_zero, idx_packed, bq)
                continue
            q_packed, idx_packed, rec = _quant_adaptive_plane(
                coeff, bq, codebook, lamb, G,
                gain=SUBBAND_GAINS[name] * (RCT_GAIN_Y if c == 0 else RCT_GAIN_C))
            chan_dict[name] = (q_packed, idx_packed, bq)
        adaptive_channels.append(chan_dict)

    # Build compact packed_meta
    # To keep backward compat with dequant, we store channels as list of dicts with keys per plane
    # plus meta
    # For dequant we need to know G, codebook, lamb, q_scale
    return {
        'channels': adaptive_channels,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'orig_shape': (H, W, C),
        'adaptive': True,
        'G': G,
        'q_scale': q_scale,
        'lamb': lamb,
        'codebook': codebook.cpu(),
        'mode': mode,
        'chroma420': chroma420,
    }, (H, W, C)

def dequantize_pixel_wavelet_adaptive(
    packed_meta: dict,
    device: str | torch.device = "cpu",
    out_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize adaptive wavelet. Tunable via packed_meta lamb/q_scale (lower = fidelity).
    GPU path uses fused Triton adaptive dequant + IDWT when available, else batched PyTorch.
    """
    dev = torch.device(device)
    # Try Triton fused path
    if HAS_TRITON and dev.type in ("cuda", "hip"):
        try:
            from .fused_ops import dequantize_fused_wavelet_adaptive_gpu
            return dequantize_fused_wavelet_adaptive_gpu(packed_meta, device=device, out_buffer=out_buffer)
        except Exception:
            pass

    H, W, C = packed_meta['orig_shape']
    G = packed_meta.get('G', 32)
    codebook = packed_meta.get('codebook', ADAPTIVE_CODEBOOK).to(dev).float()
    channels = packed_meta['channels']

    # Reconstruct per channel planes (chroma dicts lack L1 keys under 4:2:0)
    rec_planes = {}  # name -> [k, H, W] int32 (k=3 shared levels, k=1 luma L1)
    LL4_stack = torch.stack([ch['LL4'].to(dev).to(torch.int32) for ch in channels], dim=0)  # [3,H4,W4]
    rec_planes['LL4'] = LL4_stack

    plane_names = ["LH4","HL4","HH4","LH3","HL3","HH3","LH2","HL2","HH2","LH1","HL1","HH1"]
    for name in plane_names:
        rec_list = []
        for c in range(3):
            ch = channels[c]
            if name not in ch:
                continue  # chroma has no L1
            q_plane, idx_packed, bq = ch[name]
            q_plane = q_plane.to(dev)
            idx_packed = idx_packed.to(dev)
            rec = _dequant_adaptive_plane(q_plane, idx_packed, bq, codebook, G)
            rec_list.append(rec)
        rec_planes[name] = torch.stack(rec_list, dim=0)

    # 4:4:4 -> full chain; 4:2:0 -> luma 4-stage + chroma 3-stage + upsample
    if not _chroma_is_subsampled(packed_meta):
        rec_ll3 = _idwt_53_2d_step_batched(rec_planes['LL4'], rec_planes['LH4'], rec_planes['HL4'], rec_planes['HH4'])
        rec_ll2 = _idwt_53_2d_step_batched(rec_ll3, rec_planes['LH3'], rec_planes['HL3'], rec_planes['HH3'])
        rec_ll1 = _idwt_53_2d_step_batched(rec_ll2, rec_planes['LH2'], rec_planes['HL2'], rec_planes['HH2'])
        rec_yuv_b = _idwt_53_2d_step_batched(rec_ll1, rec_planes['LH1'], rec_planes['HL1'], rec_planes['HH1'])
    else:
        rec_ll3 = _idwt_53_2d_step_batched(rec_planes['LL4'], rec_planes['LH4'], rec_planes['HL4'], rec_planes['HH4'])
        rec_ll2_y = _idwt_53_2d_step_batched(rec_ll3[0:1], rec_planes['LH3'][0:1], rec_planes['HL3'][0:1], rec_planes['HH3'][0:1])
        rec_ll2_c = _idwt_53_2d_step_batched(rec_ll3[1:3], rec_planes['LH3'][1:3], rec_planes['HL3'][1:3], rec_planes['HH3'][1:3])
        rec_ll1_y = _idwt_53_2d_step_batched(rec_ll2_y, rec_planes['LH2'][0:1], rec_planes['HL2'][0:1], rec_planes['HH2'][0:1])
        rec_c_half = _idwt_53_2d_step_batched(rec_ll2_c, rec_planes['LH2'][1:3], rec_planes['HL2'][1:3], rec_planes['HH2'][1:3])
        rec_y = _idwt_53_2d_step_batched(rec_ll1_y, rec_planes['LH1'], rec_planes['HL1'], rec_planes['HH1'])
        rec_yuv_b = torch.cat([rec_y, _upsample2(rec_c_half)], dim=0)

    rec_yuv = rec_yuv_b.permute(1, 2, 0)
    rec_rgb_full = rct_inverse(rec_yuv)
    rec_rgb = rec_rgb_full[:H, :W, :]

    if out_buffer is not None:
        out_buffer.copy_(rec_rgb)
        return out_buffer
    return rec_rgb


# =============================================================================
# SPARSE BYTE-ALIGNED BITSTREAM (CPU reference pack/unpack for measurement)
# -----------------------------------------------------------------------------
# Per quantized int8 plane [h, w], row-major, blocked at G (default 32):
#   mask  uint8 [M, 4]   bit i of block b = (elem b*G+i != 0), LSB first
#   vals  int8  [nnz]    nonzero values in row-major order
# LL4 stays raw int16; adaptive 4b idx planes stay as-is (already packed).
# Unpack rebuilds the dense packed_meta schema, so every existing CPU/GPU
# dequantizer works unchanged (proves the format plugs into the pipeline).
# =============================================================================

def sparse_pack_plane(
    q_plane: torch.Tensor,
    G: int = 32,
) -> dict:
    """Pack one int8 plane. Empty blocks cost 1 occ bit. Occupied blocks use
    flat 32b masks or hierarchical nibble masks (8 presence bits + 4b per
    nonempty nibble), whichever is smaller (hier wins iff <=5 of 8 nibbles
    nonempty: 8+4k<=28 < 32). A mode bit per occupied block selects.

    Returns dict(mask [Mf,4] flat masks, hflags [Mh] presence bytes,
    hnib packed nonempty nibbles, mode [(Mo+7)//8] over occupied blocks,
    occ [(M+7)//8], vals int8 [nnz], shape (h, w), M).
    vals stay shared/in-order, so unpack needs no extra indexing.
    """
    if G != 32:
        raise ValueError(f"sparse bitstream requires G=32, got {G}")
    h, w = q_plane.shape
    flat = q_plane.reshape(-1).to(torch.int8)
    N = flat.numel()
    pad = (G - N % G) % G
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, G)
    M = blocks.shape[0]
    nz = blocks != 0  # [M, G] bool
    w8 = (1 << torch.arange(8, device=blocks.device)).to(torch.uint8)
    # occupancy: 1 bit per block, LSB first
    occ = nz.any(-1).to(torch.uint8)  # [M]
    occ_pad = (8 - M % 8) % 8
    if occ_pad:
        occ = F.pad(occ, (0, occ_pad))
    occ_bytes = (occ.view(-1, 8) * w8).sum(-1).to(torch.uint8)
    # masks + vals for occupied blocks only
    occ_b = occ[:M].bool()
    occ_blocks = blocks[occ_b]  # [Mo, G]
    Mo = occ_blocks.shape[0]
    # flat-vs-hier per occupied block (hier wins iff <=5 nonempty nibbles)
    onz = occ_blocks != 0  # [Mo, G]
    nib_any = onz.view(Mo, 8, 4).any(-1) if Mo else torch.zeros(0, 8, dtype=torch.bool)
    k = nib_any.sum(-1)  # [Mo]
    use_hier = k <= 5
    mode_bits = use_hier.to(torch.uint8)
    mp = (8 - Mo % 8) % 8
    mode_bytes = ((F.pad(mode_bits, (0, mp)).view(-1, 8) * w8).sum(-1).to(torch.uint8)
                  if Mo else torch.zeros(0, dtype=torch.uint8))
    fb = occ_blocks[~use_hier]  # [Mf, G]
    fbnz = (fb != 0).view(-1, 4, 8).to(torch.uint8) if fb.numel() else fb.view(0, 4, 8)
    flat_masks = (fbnz * w8).sum(-1).to(torch.uint8) if fb.numel() else \
        torch.zeros(0, 4, dtype=torch.uint8)
    hb = occ_blocks[use_hier]  # [Mh, G]
    if hb.numel():
        hnib_any = (hb != 0).view(-1, 8, 4).any(-1)  # [Mh, 8]
        hflags = (hnib_any.to(torch.uint8) * w8).sum(-1).to(torch.uint8)  # [Mh]
        nib4 = ((hb != 0).view(-1, 8, 4).to(torch.uint8)
                * torch.tensor([1, 2, 4, 8], device=blocks.device).to(torch.uint8)).sum(-1)
        hnib_vals = nib4.reshape(-1)[hnib_any.reshape(-1)].to(torch.uint8)
        hnib = _pack_4b(hnib_vals) if hnib_vals.numel() else torch.zeros(0, dtype=torch.uint8)
    else:
        hflags = torch.zeros(0, dtype=torch.uint8)
        hnib = torch.zeros(0, dtype=torch.uint8)
    vals = occ_blocks[onz].to(torch.int8)
    return {"mask": flat_masks, "hflags": hflags, "hnib": hnib, "mode": mode_bytes,
            "occ": occ_bytes, "vals": vals, "shape": (h, w), "M": M}


def sparse_unpack_plane(
    packed: dict,
    G: int = 32,
) -> torch.Tensor:
    """Inverse of sparse_pack_plane -> int8 plane [h, w]."""
    if G != 32:
        raise ValueError(f"sparse bitstream requires G=32, got {G}")
    mask, vals, occ = packed["mask"], packed["vals"], packed["occ"]
    hflags, hnib, mode_b = packed["hflags"], packed["hnib"], packed["mode"]
    h, w = tuple(packed["shape"])
    M = int(packed["M"])
    N = h * w
    ar = torch.arange(8, device=occ.device)
    w8 = (1 << ar).to(torch.uint8)
    occ_bits = ((occ.unsqueeze(-1).to(torch.int16) >> ar) & 1).reshape(-1)[:M].bool()
    Mo = int(occ_bits.sum().item())
    mode = (((mode_b.unsqueeze(-1).to(torch.int16) >> ar) & 1).reshape(-1)[:Mo].bool()) \
        if Mo else torch.zeros(0, dtype=torch.bool, device=occ.device)
    # mode expanded to full block grid (flat decoding below is fully vectorized)
    mode_full = torch.zeros(M, dtype=torch.bool, device=occ.device)
    if Mo:
        mode_full[occ_bits] = mode
    flat_sel = occ_bits & ~mode_full
    full_mask = torch.zeros((M, 4), dtype=torch.uint8, device=mask.device)
    if flat_sel.any():
        full_mask[flat_sel] = mask.to(torch.uint8)
    hier_sel = torch.where(occ_bits & mode_full)[0]  # [Mh] block ids
    Mh = hier_sel.numel()
    if Mh:
        # total hier nibbles = sum of presence popcounts
        k = (((hflags.unsqueeze(-1).to(torch.int16) >> ar) & 1).sum(-1).to(torch.long))  # [Mh]
        npop = int(k.sum().item())
        hnib_flat = _unpack_4b(hnib, npop) if npop else \
            torch.zeros(0, dtype=torch.uint8, device=occ.device)
        # scatter nibbles into [Mh, 8] presence slots, then expand to mask bits
        flagbits = (((hflags.unsqueeze(-1).to(torch.int16) >> ar) & 1).bool())  # [Mh,8]
        brow = torch.repeat_interleave(torch.arange(Mh, device=occ.device), k)
        slot = torch.arange(8, device=occ.device).expand(Mh, 8)[flagbits]
        nibpat = torch.zeros((Mh, 8), dtype=torch.uint8, device=occ.device)
        if npop:
            nibpat[brow, slot] = hnib_flat.to(torch.uint8)
        ar4 = torch.arange(4, device=occ.device)
        hbits = (((nibpat.unsqueeze(-1).to(torch.int16) >> ar4) & 1)
                 .reshape(Mh, 32).to(torch.uint8))  # [Mh,32]
        full_mask[hier_sel] = (hbits.view(Mh, 4, 8) * w8).sum(-1).to(torch.uint8)
    bits = ((full_mask.unsqueeze(-1).to(torch.int16) >> ar) & 1)
    nz_full = bits.reshape(-1).bool()  # [M*G], pad region always zero
    out = torch.zeros(M * G, dtype=torch.int8, device=mask.device)
    out[nz_full] = vals.to(torch.int8)
    return out[:N].view(h, w)


def sparse_pack_meta(packed_meta: dict, G: int = 32) -> dict:
    """Pack a static or adaptive wavelet packed_meta into a sparse bitstream dict."""
    adaptive = bool(packed_meta.get("adaptive", False))
    sparse_channels = []
    for ch in packed_meta["channels"]:
        entry: dict = {"LL4": ch["LL4"].to(torch.int16).cpu()}
        if adaptive:
            for name, v in ch.items():
                if name == "LL4":
                    continue
                q_plane, idx_packed, bq = v
                p = sparse_pack_plane(q_plane.cpu().to(torch.int8), G)
                p["idx_packed"] = idx_packed.cpu()
                p["bq"] = int(bq)
                entry[name] = p
        else:
            for lvl in ("L4", "L3", "L2", "L1"):
                if lvl not in ch:
                    continue  # chroma has no L1 under 4:2:0
                LH, HL, HH, q = ch[lvl]
                planes = {}
                for pname, plane in (("LH", LH), ("HL", HL), ("HH", HH)):
                    planes[pname] = sparse_pack_plane(plane.cpu().to(torch.int8), G)
                entry[lvl] = {"planes": planes, "q": int(q)}
        sparse_channels.append(entry)
    out = {
        "adaptive": adaptive,
        "G": G,
        "orig_shape": packed_meta["orig_shape"],
        "pad_h": packed_meta["pad_h"],
        "pad_w": packed_meta["pad_w"],
        "channels": sparse_channels,
    }
    if adaptive:
        out["codebook"] = packed_meta["codebook"].cpu()
        out["q_scale"] = packed_meta.get("q_scale")
        out["lamb"] = packed_meta.get("lamb")
        out["mode"] = packed_meta.get("mode")
    return out


def sparse_unpack_meta(sparse: dict) -> dict:
    """Rebuild dense packed_meta from a sparse dict (schema the dequantizers expect)."""
    G = sparse.get("G", 32)
    dev = torch.device("cpu")
    channels = []
    if sparse["adaptive"]:
        for ch in sparse["channels"]:
            d = {"LL4": ch["LL4"].to(dev)}
            for name, p in ch.items():
                if name == "LL4":
                    continue
                q = sparse_unpack_plane(p, G)
                d[name] = (q, p["idx_packed"].to(dev), int(p["bq"]))
            channels.append(d)
    else:
        for ch in sparse["channels"]:
            d = {"LL4": ch["LL4"].to(dev)}
            for lvl in ("L4", "L3", "L2", "L1"):
                if lvl not in ch:
                    continue  # chroma has no L1 under 4:2:0
                e = ch[lvl]
                planes = tuple(
                    sparse_unpack_plane(e["planes"][pn], G)
                    for pn in ("LH", "HL", "HH")
                )
                d[lvl] = planes + (int(e["q"]),)
            channels.append(d)
    out = {
        "adaptive": sparse["adaptive"],
        "orig_shape": tuple(sparse["orig_shape"]),
        "pad_h": int(sparse["pad_h"]),
        "pad_w": int(sparse["pad_w"]),
        "channels": channels,
    }
    if sparse["adaptive"]:
        out["G"] = G
        out["codebook"] = sparse["codebook"]
        out["q_scale"] = sparse.get("q_scale")
        out["lamb"] = sparse.get("lamb")
        out["mode"] = sparse.get("mode")
    return out


def _popcount_u8(x: torch.Tensor) -> torch.Tensor:
    """Vectorized popcount for uint8/int32 tensor -> int32 counts. GPU-safe, no sync."""
    ar8 = torch.arange(8, device=x.device)
    return (((x.to(torch.int32).unsqueeze(-1) >> ar8) & 1).sum(-1).to(torch.int32))


def sparse_unpack_plane_gpu(
    packed: dict,
    G: int = 32,
) -> torch.Tensor:
    """Sync-free GPU port of sparse_unpack_plane -> int8 plane [h, w].

    Identical output to sparse_unpack_plane, but every data-dependent count
    is replaced by clamped cumsum ranks + where-masks, so no .item() sync
    ever fires. All blob tensors must already live on the target device
    (caller moves them H2D); compute never leaves it.
    """
    if G != 32:
        raise ValueError(f"sparse bitstream requires G=32, got {G}")
    dev = packed["occ"].device
    mask, vals, occ = packed["mask"], packed["vals"], packed["occ"]
    hflags, hnib, mode_b = packed["hflags"], packed["hnib"], packed["mode"]
    h, w = tuple(packed["shape"])
    M = int(packed["M"])
    N = h * w
    ar8 = torch.arange(8, device=dev)
    ar32 = torch.arange(32, device=dev)
    occ_bits = (((occ.to(torch.int16).unsqueeze(-1) >> ar8) & 1)
                .reshape(-1)[:M].bool())
    # mode bits over the full block grid via occupancy rank (no Mo sync)
    m_all = (((mode_b.to(torch.int16).unsqueeze(-1) >> ar8) & 1)
             .reshape(-1).bool()) if mode_b.numel() else occ_bits[:0]
    if m_all.numel() < M:
        m_all = torch.cat([m_all, torch.zeros(M - m_all.numel(),
                                              dtype=torch.bool, device=dev)])
    else:
        m_all = m_all[:M]
    rank = (occ_bits.cumsum(0) - 1).clamp(min=0)
    mode_full = torch.where(occ_bits, m_all[rank], torch.zeros((), dtype=torch.bool, device=dev))
    flat_sel = occ_bits & ~mode_full
    hier_sel = occ_bits & mode_full
    # per-block 32-bit occupancy words; flat and hier selections are disjoint
    mask32 = torch.zeros(M, dtype=torch.int32, device=dev)
    if mask.numel():
        fm = mask.to(torch.int32)  # [Mf, 4], Mf == flat_sel.count by construction
        mask32[flat_sel] = (fm[:, 0] | (fm[:, 1] << 8)
                            | (fm[:, 2] << 16) | (fm[:, 3] << 24))
    hf = hflags if hflags.numel() else torch.zeros(1, dtype=torch.uint8, device=dev)
    hrank = (hier_sel.cumsum(0) - 1).clamp(min=0, max=hf.numel() - 1)
    hfb = hf[hrank].to(torch.int32)  # presence byte per block (masked unless hier)
    k_all = torch.where(hier_sel, _popcount_u8(hfb), torch.zeros((), dtype=torch.int32, device=dev))
    off_all = (k_all.cumsum(0) - k_all).to(torch.long)
    if hnib.numel():
        pairs_all = torch.stack([hnib & 0xF, (hnib >> 4) & 0xF], dim=1).view(-1)
    else:
        pairs_all = torch.zeros(1, dtype=torch.uint8, device=dev)
    plast = pairs_all.numel() - 1
    for s in range(8):
        bit = hier_sel & ((hfb >> s) & 1).bool()
        intra = _popcount_u8(hfb & ((1 << s) - 1))
        pos = (off_all + intra.to(torch.long)).clamp(max=plast)
        nib = pairs_all[pos].to(torch.int32)
        mask32 |= torch.where(bit, (nib & 0xF) << (4 * s),
                              torch.zeros((), dtype=torch.int32, device=dev))
    nz_full = (((mask32.unsqueeze(-1) >> ar32) & 1).bool()).reshape(-1)
    out = torch.zeros(M * G, dtype=torch.int8, device=dev)
    out[nz_full] = vals.to(torch.int8)
    return out[:N].view(h, w)


def sparse_unpack_meta_gpu(sparse: dict, device: str | torch.device) -> dict:
    """Move sparse blobs H2D and unpack every plane on-device (no syncs).

    Returns the dense packed_meta schema on `device`, ready for any GPU
    dequantizer. Pure function of the sparse dict (caller keeps ownership).
    """
    dev = torch.device(device)
    mv = lambda t: t.to(dev, non_blocking=True)
    G = sparse.get("G", 32)
    channels = []
    if sparse["adaptive"]:
        for ch in sparse["channels"]:
            d = {"LL4": mv(ch["LL4"])}
            for name, p in ch.items():
                if name == "LL4":
                    continue
                q = sparse_unpack_plane_gpu(
                    {"mask": mv(p["mask"]), "hflags": mv(p["hflags"]),
                     "hnib": mv(p["hnib"]), "mode": mv(p["mode"]),
                     "occ": mv(p["occ"]), "vals": mv(p["vals"]),
                     "shape": tuple(p["shape"]), "M": int(p["M"])}, G)
                d[name] = (q, mv(p["idx_packed"]), int(p["bq"]))
            channels.append(d)
    else:
        for ch in sparse["channels"]:
            d = {"LL4": mv(ch["LL4"])}
            for lvl in ("L4", "L3", "L2", "L1"):
                if lvl not in ch:
                    continue  # chroma has no L1 under 4:2:0
                e = ch[lvl]
                planes = tuple(
                    sparse_unpack_plane_gpu(
                        {"mask": mv(e["planes"][pn]["mask"]),
                         "hflags": mv(e["planes"][pn]["hflags"]),
                         "hnib": mv(e["planes"][pn]["hnib"]),
                         "mode": mv(e["planes"][pn]["mode"]),
                         "occ": mv(e["planes"][pn]["occ"]),
                         "vals": mv(e["planes"][pn]["vals"]),
                         "shape": tuple(e["planes"][pn]["shape"]),
                         "M": int(e["planes"][pn]["M"])}, G)
                    for pn in ("LH", "HL", "HH")
                )
                d[lvl] = planes + (int(e["q"]),)
            channels.append(d)
    out = {
        "adaptive": sparse["adaptive"],
        "orig_shape": tuple(sparse["orig_shape"]),
        "pad_h": int(sparse["pad_h"]),
        "pad_w": int(sparse["pad_w"]),
        "channels": channels,
    }
    if sparse["adaptive"]:
        out["G"] = G
        out["codebook"] = sparse["codebook"].to(dev, non_blocking=True)
        out["q_scale"] = sparse.get("q_scale")
        out["lamb"] = sparse.get("lamb")
        out["mode"] = sparse.get("mode")
    return out


def sparse_pack_arena(sparse: dict) -> dict:
    """Repackage a sparse bitstream dict into the arena stored format.

    The arena IS the decode-ready layout: one concatenated u8 stream blob,
    one i8 vals blob, a [P,8] int32 meta table of per-plane stream offsets,
    and raw int16 LL4 — so the decode-side plan is ~zero CPU (concat + H2D).
    Pure CPU torch, no syncs (all lengths are tensor metadata).

    meta cols: 0 occ, 1 mode, 2 fmask, 3 hflags, 4 vals, 5 idx, 6 param, 7 hnib.
    inv entries: (name, channel, h, w, M); static names normalized ('LH4').
    """
    adaptive = bool(sparse.get("adaptive", False))
    inv = []  # fixed plane order: channel-major
    for c, ch in enumerate(sparse["channels"]):
        if adaptive:
            names = [n for n in ("LH4", "HL4", "HH4", "LH3", "HL3", "HH3",
                                 "LH2", "HL2", "HH2", "LH1", "HL1", "HH1") if n in ch]
            for n in names:
                p = ch[n]
                h, w = tuple(p["shape"])
                inv.append((n, c, h, w, int(p["M"])))
        else:
            for lvl in ("L4", "L3", "L2", "L1"):
                if lvl not in ch:
                    continue
                for pn in ("LH", "HL", "HH"):
                    p = ch[lvl]["planes"][pn]
                    h, w = tuple(p["shape"])
                    inv.append((pn + lvl[1:], c, h, w, int(p["M"])))
    P = len(inv)
    meta = torch.zeros((P, 8), dtype=torch.int32)
    au8, ai8 = [], []
    ll4_parts = [ch["LL4"].to(torch.int16).reshape(-1) for ch in sparse["channels"]]
    ll4_shapes = [tuple(ch["LL4"].shape) for ch in sparse["channels"]]
    ou = oi = 0
    for i, (name, c, h, w, M) in enumerate(inv):
        ch = sparse["channels"][c]
        if adaptive:
            op, ip, param = ch[name], ch[name]["idx_packed"], int(ch[name]["bq"])
        else:
            lvl, pn = "L" + name[2:], name[:2]
            e = ch[lvl]
            op, ip = e["planes"][pn], None
            param = int(e["q"]) * (2 if pn == "HH" else 1)
        occ = op["occ"].reshape(-1)
        mode = op["mode"].reshape(-1)
        fmask = op["mask"].reshape(-1)
        hfl = op["hflags"].reshape(-1)
        hnib = op["hnib"].reshape(-1)
        idx = ip.reshape(-1) if ip is not None else torch.zeros(0, dtype=torch.uint8)
        vals = op["vals"].reshape(-1)
        meta[i, 0] = ou
        meta[i, 1] = ou + occ.numel()
        meta[i, 2] = ou + occ.numel() + mode.numel()
        meta[i, 3] = ou + occ.numel() + mode.numel() + fmask.numel()
        meta[i, 7] = ou + occ.numel() + mode.numel() + fmask.numel() + hfl.numel()
        meta[i, 4] = oi
        meta[i, 6] = param
        au8 += [occ, mode, fmask, hfl, hnib, idx]
        ai8 += [vals]
        ou = ou + occ.numel() + mode.numel() + fmask.numel() + hfl.numel() + hnib.numel() + idx.numel()
        oi += vals.numel()
        meta[i, 5] = ou - idx.numel()  # idx base
    out = {
        "format": "xs-arena-v1",
        "adaptive": adaptive,
        "orig_shape": tuple(sparse["orig_shape"]),
        "pad_h": int(sparse["pad_h"]),
        "pad_w": int(sparse["pad_w"]),
        "inv": inv,
        "B": sum(e[4] for e in inv),
        "P": P,
        "arena_u8": torch.cat(au8, dim=0) if au8 else torch.zeros(0, dtype=torch.uint8),
        "arena_i8": torch.cat(ai8, dim=0).to(torch.int8) if ai8 else torch.zeros(0, dtype=torch.int8),
        "meta": meta,
        "ll4": torch.cat(ll4_parts, dim=0) if ll4_parts else torch.zeros(0, dtype=torch.int16),
        "ll4_shapes": ll4_shapes,
    }
    if adaptive:
        out["codebook"] = sparse["codebook"].to(torch.float32).reshape(-1)
        out["q_scale"] = sparse.get("q_scale")
        out["lamb"] = sparse.get("lamb")
        out["mode"] = sparse.get("mode")
    return out


def arena_nbytes(arena: dict) -> int:
    """Actual stored bytes of an arena dict (tensors only)."""
    return (arena["arena_u8"].nelement() + arena["arena_i8"].nelement()
            + arena["meta"].nelement() * 4 + arena["ll4"].nelement() * 2)


def _plane_nbytes(p: dict) -> int:
    return (p["mask"].nelement() + p["hflags"].nelement() + p["hnib"].nelement()
            + p["mode"].nelement() + p["vals"].nelement() + p["occ"].nelement())


def sparse_nbytes(sparse: dict) -> int:
    """Actual stored bytes of a sparse dict (tensors only; header is tens of bytes).

    Accepts arena dicts too (detected via the 'format' marker).
    """
    if sparse.get("format") == "xs-arena-v1":
        return arena_nbytes(sparse)
    total = 0
    for ch in sparse["channels"]:
        total += ch["LL4"].nelement() * 2
        if sparse["adaptive"]:
            for name, p in ch.items():
                if name == "LL4":
                    continue
                total += _plane_nbytes(p) + p["idx_packed"].nelement()
        else:
            for lvl in ("L4", "L3", "L2", "L1"):
                if lvl not in ch:
                    continue  # chroma has no L1 under 4:2:0
                for pn in ("LH", "HL", "HH"):
                    total += _plane_nbytes(ch[lvl]["planes"][pn])
    return total

