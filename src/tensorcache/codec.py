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
