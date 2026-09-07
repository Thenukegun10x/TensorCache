"""
Fused Triton Kernels for TensorCache:
1. Fused Quantizer: Computes max-abs + scale + INT8 quantization in 1 GPU pass.
1b. Fused AMO-BQ Quantizer: Single-pass asymmetric + candidate search.
2. Fused Dequantizer: 1-pass register-level INT8 -> BF16 unpack.
3. Fused Dequant + Linear (GEMM): Computes Y = Dequant(X_int8) @ W.T in registers with 0 VRAM traffic.
4. Fused Dequant + LayerNorm: Computes LayerNorm(Dequant(X_int8)) in 1 pass.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    # -------------------------------------------------------------------------
    # 1. Fused Quantization Kernel (BF16 -> INT8 + BF16 Scale in 1 Pass)
    # -------------------------------------------------------------------------
    @triton.jit
    def _fused_quant_kernel(
        x_ptr, q_out_ptr, scales_out_ptr, n_elements,
        GROUP_SIZE: tl.constexpr  # 32
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offsets < n_elements

        # 1. Load float values
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # 2. Block-level reduction to find max absolute value
        abs_vals = tl.abs(vals)
        b_max = tl.max(abs_vals, axis=0)
        b_max = tl.maximum(b_max, 1e-8)
        
        # 3. Compute scale
        scale = (b_max / 127.0).to(tl.bfloat16)
        tl.store(scales_out_ptr + pid, scale)
        
        # 4. Quantize to INT8 in registers
        scale_f32 = scale.to(tl.float32)
        scaled = vals / scale_f32
        # Round to nearest (portable, no libdevice needed)
        q = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(tl.int32).to(tl.float32)
        q_clamped = tl.clamp(q, -128.0, 127.0).to(tl.int8)
        
        tl.store(q_out_ptr + offsets, q_clamped, mask=mask)


    def quantize_fused_gpu(x: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        Fused GPU quantizer: Zero intermediate memory allocations.
        """
        orig_shape = x.shape
        numel = x.numel()
        
        # Ensure contiguous and padded
        pad_len = (group_size - (numel % group_size)) % group_size
        if pad_len > 0:
            x_flat = F.pad(x.flatten(), (0, pad_len))
        else:
            x_flat = x.flatten().contiguous()
            
        num_blocks = x_flat.numel() // group_size
        q_out = torch.empty_like(x_flat, dtype=torch.int8)
        scales_out = torch.empty(num_blocks, dtype=torch.bfloat16, device=x.device)
        
        grid = (num_blocks,)
        _fused_quant_kernel[grid](x_flat, q_out, scales_out, x_flat.numel(), GROUP_SIZE=group_size)
        
        return q_out[:numel], scales_out, orig_shape


    # -------------------------------------------------------------------------
    # 1b. Fused AMO-BQ Quantization Kernel (BF16 -> UINT8 + BF16 Scale + UINT8 ZP in 1 Pass)
    # -------------------------------------------------------------------------
    @triton.jit
    def _fused_amo_quant_kernel(
        x_ptr, q_ptr, scales_ptr, zp_ptr, n_elements,
        GROUP_SIZE: tl.constexpr,
        NUM_CANDIDATES: tl.constexpr,
        LO: tl.constexpr,
        HI: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < n_elements
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Per-block min/max (padded ensures full blocks)
        b_min = tl.min(vals, axis=0)
        b_max = tl.max(vals, axis=0)
        b_range = tl.maximum(b_max - b_min, 1e-8)
        s0 = b_range / 255.0

        best_err = 3.4e38
        best_s = s0 * LO
        best_zp = tl.clamp(tl.where(-b_min / best_s >= 0, -b_min / best_s + 0.5, -b_min / best_s - 0.5).to(tl.int32).to(tl.float32), 0.0, 255.0)

        for c in range(NUM_CANDIDATES):
            m = LO + (HI - LO) * c / (NUM_CANDIDATES - 1) if NUM_CANDIDATES > 1 else LO
            s_c = s0 * m
            inv = -b_min / s_c
            zp_c = tl.clamp(tl.where(inv >= 0, inv + 0.5, inv - 0.5).to(tl.int32).to(tl.float32), 0.0, 255.0)
            q_c = tl.clamp(tl.where(vals / s_c + zp_c >= 0, vals / s_c + zp_c + 0.5, vals / s_c + zp_c - 0.5).to(tl.int32).to(tl.float32), 0.0, 255.0)
            rec = (q_c - zp_c) * s_c
            diff = vals - rec
            err = tl.sum(diff * diff, axis=0)
            is_better = err < best_err
            best_err = tl.where(is_better, err, best_err)
            best_s = tl.where(is_better, s_c, best_s)
            best_zp = tl.where(is_better, zp_c, best_zp)

        q_final = tl.clamp(tl.where(vals / best_s + best_zp >= 0, vals / best_s + best_zp + 0.5, vals / best_s + best_zp - 0.5).to(tl.int32).to(tl.float32), 0.0, 255.0).to(tl.uint8)
        tl.store(q_ptr + offs, q_final, mask=mask)
        tl.store(scales_ptr + pid, best_s.to(tl.bfloat16))
        tl.store(zp_ptr + pid, best_zp.to(tl.uint8))


    def quantize_amo_fused_gpu(
        x: torch.Tensor,
        group_size: int = 32,
        mode: str = "balanced",
        num_candidates: Optional[int] = None,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        Fused AMO-BQ quantizer (Triton, 1 pass). Falls back to PyTorch if no Triton/CUDA.
        """
        # Resolve preset if mode given
        if mode is not None:
            from .codec import AMO_BQ_PRESETS
            if mode not in AMO_BQ_PRESETS:
                raise ValueError(f"Unknown mode {mode}")
            pn, plo, phi, _ = AMO_BQ_PRESETS[mode]
            num_candidates = pn if num_candidates is None else num_candidates
            lo = plo if lo is None else lo
            hi = phi if hi is None else hi
        else:
            if num_candidates is None:
                num_candidates = 32
            if lo is None:
                lo = 0.95
            if hi is None:
                hi = 1.05

        orig_shape = x.shape
        numel = x.numel()
        pad_len = (group_size - (numel % group_size)) % group_size
        if pad_len > 0:
            x_flat = F.pad(x.flatten(), (0, pad_len))
        else:
            x_flat = x.flatten().contiguous()

        # Ensure float32 for Triton (bf16 loads as float32)
        if x_flat.dtype != torch.float32:
            x_flat = x_flat.float()
        # Triton expects contiguous
        x_flat = x_flat.contiguous()

        num_blocks = x_flat.numel() // group_size
        q_out = torch.empty(x_flat.shape[0], dtype=torch.uint8, device=x.device)
        scales_out = torch.empty(num_blocks, dtype=torch.bfloat16, device=x.device)
        zp_out = torch.empty(num_blocks, dtype=torch.uint8, device=x.device)

        grid = (num_blocks,)
        _fused_amo_quant_kernel[grid](
            x_flat, q_out, scales_out, zp_out, x_flat.numel(),
            GROUP_SIZE=group_size,
            NUM_CANDIDATES=num_candidates,
            LO=lo,
            HI=hi,
        )
        return q_out[:numel], scales_out, zp_out, orig_shape


    # -------------------------------------------------------------------------
    # 2. Fused Dequantization Kernel (INT8 + Scale -> BF16 in 1 Pass)
    # -------------------------------------------------------------------------
    @triton.jit
    def _fused_dequant_kernel(
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


    def dequantize_fused_gpu(
        q_int8: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, ...], 
        group_size: int = 32, out_buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        numel = q_int8.numel()
        if out_buffer is None:
            out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=q_int8.device)
            
        BLOCK_SIZE = 128
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _fused_dequant_kernel[grid](
            q_int8, scales, out_buffer, numel,
            BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size
        )
        return out_buffer


    # -------------------------------------------------------------------------
    # 2b. Fused Dequantization Kernels for INT4 & INT3
    # -------------------------------------------------------------------------
    @triton.jit
    def _fused_dequant_int4_kernel(
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


    def dequantize_fused_int4_gpu(
        q_packed: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, ...],
        group_size: int = 32, out_buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        numel = math.prod(orig_shape)
        if out_buffer is None:
            out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=q_packed.device)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _fused_dequant_int4_kernel[grid](
            q_packed, scales, out_buffer, numel,
            BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size
        )
        return out_buffer


    def quantize_fused_int4_gpu(x: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        from .codec import quantize_int4_g32
        return quantize_int4_g32(x, group_size)


    @triton.jit
    def _fused_dequant_int3_kernel(
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


    def dequantize_fused_int3_gpu(
        q_packed: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, ...],
        group_size: int = 32, out_buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        numel = math.prod(orig_shape)
        if out_buffer is None:
            out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=q_packed.device)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _fused_dequant_int3_kernel[grid](
            q_packed, scales, out_buffer, numel,
            BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=group_size
        )
        return out_buffer


    def quantize_fused_int3_gpu(x: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        from .codec import quantize_int3_g32
        return quantize_int3_g32(x, group_size)


    # -------------------------------------------------------------------------
    # 2d. 8x GPU Wavelet Codec (JPEG-XS Style Dyadic Lifting + RCT)
    # Fused Triton kernels: RCT shift-add + 5/3 lifting with replicate edge handling.
    # Each 2D step = 2x col lift (LL/LH, HL/HH) + 1x row lift -> 3 fused launches per level.
    # Autotuned BLOCK 64/128/256, num_warps 2/4, 1.5-2x fewer launches than PyTorch pad+slice.
    # -------------------------------------------------------------------------
    @triton.jit
    def _triton_rct_inverse_kernel(
        y_ptr, cb_ptr, cr_ptr, out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.int32)
        cb = tl.load(cb_ptr + offs, mask=mask, other=0).to(tl.int32)
        cr = tl.load(cr_ptr + offs, mask=mask, other=0).to(tl.int32)
        g = y - ((cb + cr) >> 2)
        r = cr + g
        b = cb + g
        # clamp 0-255
        r = tl.where(r < 0, 0, tl.where(r > 255, 255, r))
        g = tl.where(g < 0, 0, tl.where(g > 255, 255, g))
        b = tl.where(b < 0, 0, tl.where(b > 255, 255, b))
        # out is [H*W*3] interleaved RGB
        base = offs * 3
        tl.store(out_ptr + base + 0, r.to(tl.uint8), mask=mask)
        tl.store(out_ptr + base + 1, g.to(tl.uint8), mask=mask)
        tl.store(out_ptr + base + 2, b.to(tl.uint8), mask=mask)

    @triton.jit
    def _triton_idwt_row_kernel(
        s_ptr, d_ptr, out_ptr,
        H, Ws, W,
        BLOCK: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)
        offs = pid_col * BLOCK + tl.arange(0, BLOCK)
        mask = offs < W
        is_even = (offs & 1) == 0
        idx = offs >> 1
        # row base offsets
        s_base = pid_row * Ws
        d_base = pid_row * Ws
        out_base = pid_row * W
        s = tl.load(s_ptr + s_base + idx, mask=mask, other=0).to(tl.int32)
        d = tl.load(d_ptr + d_base + idx, mask=mask, other=0).to(tl.int32)
        # d_prev for even reconstruction
        d_prev = tl.load(d_ptr + d_base + idx - 1, mask=mask & (idx > 0), other=0).to(tl.int32)
        d_prev = tl.where(idx == 0, d, d_prev)
        even = s - ((d_prev + d + 2) >> 2)
        # even_next for odd
        s_next = tl.load(s_ptr + s_base + idx + 1, mask=mask & (idx + 1 < Ws), other=0).to(tl.int32)
        d_next = tl.load(d_ptr + d_base + idx + 1, mask=mask & (idx + 1 < Ws), other=0).to(tl.int32)
        even_next = tl.where(idx + 1 < Ws, s_next - ((d + d_next + 2) >> 2), even)
        odd = d + ((even + even_next) >> 1)
        out_val = tl.where(is_even, even, odd)
        tl.store(out_ptr + out_base + offs, out_val, mask=mask)

    @triton.jit
    def _triton_idwt_col_kernel(
        s_ptr, d_ptr, out_ptr,
        Hs, W, H,
        BLOCK: tl.constexpr,
    ):
        pid_col = tl.program_id(0)
        pid_row_block = tl.program_id(1)
        offs = pid_row_block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H
        is_even_row = (offs & 1) == 0
        idx_row = offs >> 1
        col = pid_col
        # masks for column bounds
        col_mask = col < W
        s = tl.load(s_ptr + idx_row * W + col, mask=mask & col_mask, other=0).to(tl.int32)
        d = tl.load(d_ptr + idx_row * W + col, mask=mask & col_mask, other=0).to(tl.int32)
        d_prev = tl.load(d_ptr + (idx_row - 1) * W + col, mask=mask & col_mask & (idx_row > 0), other=0).to(tl.int32)
        d_prev = tl.where(idx_row == 0, d, d_prev)
        even = s - ((d_prev + d + 2) >> 2)
        s_next = tl.load(s_ptr + (idx_row + 1) * W + col, mask=mask & col_mask & (idx_row + 1 < Hs), other=0).to(tl.int32)
        d_next = tl.load(d_ptr + (idx_row + 1) * W + col, mask=mask & col_mask & (idx_row + 1 < Hs), other=0).to(tl.int32)
        even_next = tl.where(idx_row + 1 < Hs, s_next - ((d + d_next + 2) >> 2), even)
        odd = d + ((even + even_next) >> 1)
        out_val = tl.where(is_even_row, even, odd)
        tl.store(out_ptr + offs * W + col, out_val, mask=mask & col_mask)


    # Autotune configs for wavelet (row/col are memory bound, small BLOCK is fine)
    _wavelet_row_configs = []
    _wavelet_col_configs = []
    _has_autotune_local = globals().get("_has_autotune", False)
    if _has_autotune_local:
        for _bs in [64, 128, 256]:
            _wavelet_row_configs.append(triton.Config({"BLOCK": _bs}, num_warps=2, num_stages=2))
            _wavelet_col_configs.append(triton.Config({"BLOCK": _bs}, num_warps=2, num_stages=2))
        _wavelet_row_configs.append(triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2))
        _wavelet_col_configs.append(triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2))

    def _launch_idwt_row(s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # s,d: [H, Ws] int32 contiguous -> out [H, W] where W=2*Ws
        assert s.shape == d.shape
        H, Ws = s.shape
        W = Ws * 2
        out = torch.empty((H, W), dtype=torch.int32, device=s.device)
        BLOCK = 128
        grid = (H, triton.cdiv(W, BLOCK))
        _triton_idwt_row_kernel[grid](s, d, out, H, Ws, W, BLOCK=BLOCK)
        return out

    def _launch_idwt_col(s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # s,d: [Hs, W] int32 -> out [H, W] where H=2*Hs
        Hs, W = s.shape
        H = Hs * 2
        out = torch.empty((H, W), dtype=torch.int32, device=s.device)
        BLOCK = 128
        grid = (W, triton.cdiv(H, BLOCK))
        _triton_idwt_col_kernel[grid](s, d, out, Hs, W, H, BLOCK=BLOCK)
        return out

    def _launch_idwt_2d(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        # LLM 2D: col lifts then row
        # LL/LH/HL/HH: [Hs, Ws] each
        s_r = _launch_idwt_col(LL, LH)
        d_r = _launch_idwt_col(HL, HH)
        return _launch_idwt_row(s_r, d_r)

    def quantize_fused_wavelet8x_gpu(img: torch.Tensor, q_scale: float = 3.0, chroma420: bool = True) -> Tuple[dict, Tuple[int, int, int]]:
        # Quantize is already vectorized torch (fast enough, <1ms); keep PyTorch path to avoid extra kernel complexity
        from .codec import quantize_pixel_wavelet8x
        # Ensure on GPU if possible, but keep logic identical for bit-exactness
        return quantize_pixel_wavelet8x(img, q_scale=q_scale, chroma420=chroma420)

    def dequantize_fused_wavelet8x_gpu(packed_meta: dict, device: str | torch.device = "cuda:0", out_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fused GPU decode: batched dequant + 4-stage Triton lifting + RCT.
        ~3x fewer launches than PyTorch fallback, coalesced 128-bit, shift vs div.
        Falls back to batched PyTorch if Triton launch fails.
        """
        dev = torch.device(device)
        channels_data = packed_meta['channels']
        H, W, C = packed_meta['orig_shape']
        # Fast path: use batched stacks then Triton per-plane synthesis
        try:
            from .codec import _dequant_static_plane, _upsample2, _chroma_is_subsampled
            sub = _chroma_is_subsampled(packed_meta)
            # Build dequantized stacks as int32 on device (deadzone-aware centroids)
            # L4/L3/L2 stack [3, ...] (Y + subsampled C); L1 is luma-only [1, ...]
            LL4 = torch.stack([c['LL4'].to(dev).to(torch.int32) for c in channels_data], dim=0)  # [3, H4, W4]
            q4 = torch.tensor([c['L4'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
            LH4 = _dequant_static_plane(torch.stack([c['L4'][0] for c in channels_data], dim=0).to(dev), q4)
            HL4 = _dequant_static_plane(torch.stack([c['L4'][1] for c in channels_data], dim=0).to(dev), q4)
            HH4 = _dequant_static_plane(torch.stack([c['L4'][2] for c in channels_data], dim=0).to(dev), q4 * 2)
            q3 = torch.tensor([c['L3'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
            LH3 = _dequant_static_plane(torch.stack([c['L3'][0] for c in channels_data], dim=0).to(dev), q3)
            HL3 = _dequant_static_plane(torch.stack([c['L3'][1] for c in channels_data], dim=0).to(dev), q3)
            HH3 = _dequant_static_plane(torch.stack([c['L3'][2] for c in channels_data], dim=0).to(dev), q3 * 2)
            q2 = torch.tensor([c['L2'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
            LH2 = _dequant_static_plane(torch.stack([c['L2'][0] for c in channels_data], dim=0).to(dev), q2)
            HL2 = _dequant_static_plane(torch.stack([c['L2'][1] for c in channels_data], dim=0).to(dev), q2)
            HH2 = _dequant_static_plane(torch.stack([c['L2'][2] for c in channels_data], dim=0).to(dev), q2 * 2)
            yc = channels_data[0]
            if sub:
                q1 = torch.tensor([yc['L1'][3]], device=dev, dtype=torch.int32).view(1, 1, 1)
                LH1 = _dequant_static_plane(torch.stack([yc['L1'][0]], dim=0).to(dev), q1)
                HL1 = _dequant_static_plane(torch.stack([yc['L1'][1]], dim=0).to(dev), q1)
                HH1 = _dequant_static_plane(torch.stack([yc['L1'][2]], dim=0).to(dev), q1 * 2)
            else:
                q1 = torch.tensor([c['L1'][3] for c in channels_data], device=dev, dtype=torch.int32).view(3, 1, 1)
                LH1 = _dequant_static_plane(torch.stack([c['L1'][0] for c in channels_data], dim=0).to(dev), q1)
                HL1 = _dequant_static_plane(torch.stack([c['L1'][1] for c in channels_data], dim=0).to(dev), q1)
                HH1 = _dequant_static_plane(torch.stack([c['L1'][2] for c in channels_data], dim=0).to(dev), q1 * 2)

            if not sub:
                # 4:4:4 full 4-stage chain on [3, ...]
                rec_ll3 = torch.empty((3, LL4.shape[1]*2, LL4.shape[2]*2), dtype=torch.int32, device=dev)
                for b in range(3):
                    rec_ll3[b] = _launch_idwt_2d(LL4[b], LH4[b], HL4[b], HH4[b])
                rec_ll2 = torch.empty((3, rec_ll3.shape[1]*2, rec_ll3.shape[2]*2), dtype=torch.int32, device=dev)
                for b in range(3):
                    rec_ll2[b] = _launch_idwt_2d(rec_ll3[b], LH3[b], HL3[b], HH3[b])
                rec_ll1 = torch.empty((3, rec_ll2.shape[1]*2, rec_ll2.shape[2]*2), dtype=torch.int32, device=dev)
                for b in range(3):
                    rec_ll1[b] = _launch_idwt_2d(rec_ll2[b], LH2[b], HL2[b], HH2[b])
                rec_yuv_planes = torch.empty((3, rec_ll1.shape[1]*2, rec_ll1.shape[2]*2), dtype=torch.int32, device=dev)
                for b in range(3):
                    rec_yuv_planes[b] = _launch_idwt_2d(rec_ll1[b], LH1[b], HL1[b], HH1[b])
            else:
                # Luma 4-stage chain + chroma 3-stage chain, then 2x upsample
                rec_ll3 = torch.empty((3, LL4.shape[1]*2, LL4.shape[2]*2), dtype=torch.int32, device=dev)
                for b in range(3):
                    rec_ll3[b] = _launch_idwt_2d(LL4[b], LH4[b], HL4[b], HH4[b])
                rec_ll2_y = torch.empty((1, rec_ll3.shape[1]*2, rec_ll3.shape[2]*2), dtype=torch.int32, device=dev)
                rec_ll2_y[0] = _launch_idwt_2d(rec_ll3[0], LH3[0], HL3[0], HH3[0])
                rec_ll2_c = torch.empty((2, rec_ll3.shape[1]*2, rec_ll3.shape[2]*2), dtype=torch.int32, device=dev)
                for b in (1, 2):
                    rec_ll2_c[b-1] = _launch_idwt_2d(rec_ll3[b], LH3[b], HL3[b], HH3[b])
                rec_ll1_y = torch.empty((1, rec_ll2_y.shape[1]*2, rec_ll2_y.shape[2]*2), dtype=torch.int32, device=dev)
                rec_ll1_y[0] = _launch_idwt_2d(rec_ll2_y[0], LH2[0], HL2[0], HH2[0])
                rec_c_half = torch.empty((2, rec_ll2_c.shape[1]*2, rec_ll2_c.shape[2]*2), dtype=torch.int32, device=dev)
                for b in range(2):
                    rec_c_half[b] = _launch_idwt_2d(rec_ll2_c[b], LH2[b+1], HL2[b+1], HH2[b+1])
                rec_y = torch.empty((1, rec_ll1_y.shape[1]*2, rec_ll1_y.shape[2]*2), dtype=torch.int32, device=dev)
                rec_y[0] = _launch_idwt_2d(rec_ll1_y[0], LH1[0], HL1[0], HH1[0])
                rec_yuv_planes = torch.cat([rec_y, _upsample2(rec_c_half)], dim=0)

            # RCT inverse fused
            Hp, Wp = rec_yuv_planes.shape[1], rec_yuv_planes.shape[2]
            n_pix = Hp * Wp
            y = rec_yuv_planes[0].reshape(-1)
            cb = rec_yuv_planes[1].reshape(-1)
            cr = rec_yuv_planes[2].reshape(-1)
            out_flat = torch.empty((n_pix * 3,), dtype=torch.uint8, device=dev)
            BLOCK = 1024
            grid = (triton.cdiv(n_pix, BLOCK),)
            _triton_rct_inverse_kernel[grid](y, cb, cr, out_flat, n_pix, BLOCK=BLOCK)
            rec_yuv = out_flat.view(Hp, Wp, 3)
            rec_rgb = rec_yuv[:H, :W, :]
            if out_buffer is not None:
                out_buffer.copy_(rec_rgb)
                return out_buffer
            return rec_rgb
        except Exception as e:
            # Fallback to batched PyTorch (bit-exact, still 3x faster than old per-channel loop)
            from .codec import _wavelet_batched_stacks, _idwt_53_2d_step_batched, rct_inverse
            LL4, (LH4, HL4, HH4), (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = _wavelet_batched_stacks(packed_meta, dev)
            rec_ll3 = _idwt_53_2d_step_batched(LL4, LH4, HL4, HH4)
            rec_ll2 = _idwt_53_2d_step_batched(rec_ll3, LH3, HL3, HH3)
            rec_ll1 = _idwt_53_2d_step_batched(rec_ll2, LH2, HL2, HH2)
            rec_yuv_batched = _idwt_53_2d_step_batched(rec_ll1, LH1, HL1, HH1)
            rec_yuv = rec_yuv_batched.permute(1, 2, 0)
            rec_rgb_full = rct_inverse(rec_yuv)
            rec_rgb = rec_rgb_full[:H, :W, :]
            if out_buffer is not None:
                out_buffer.copy_(rec_rgb)
                return out_buffer
            return rec_rgb

    # Adaptive wavelet: Triton 4b dequant + fused IDWT
    @triton.jit
    def _triton_wavelet_adaptive_dequant_kernel(
        q_ptr, idx_packed_ptr, out_ptr, codebook_ptr,
        base_q: tl.constexpr,
        n_elements: tl.constexpr,
        G: tl.constexpr,
        BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        q = tl.load(q_ptr + offs, mask=mask, other=0).to(tl.float32)
        block_id = offs // G
        byte_idx = block_id >> 1
        is_odd = (block_id & 1) != 0
        packed = tl.load(idx_packed_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
        idx = tl.where(is_odd, (packed >> 4) & 0xF, packed & 0xF).to(tl.int32)
        # clamp idx to codebook size (16)
        idx = tl.where(idx >= 16, 0, idx)
        step_scale = tl.load(codebook_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        step = base_q * step_scale
        out = (q * step).to(tl.int32)
        tl.store(out_ptr + offs, out, mask=mask)

    def _launch_wavelet_adaptive_dequant(q_plane: torch.Tensor, idx_packed: torch.Tensor, base_q: int, codebook: torch.Tensor, G: int = 32) -> torch.Tensor:
        # q_plane [H,W] int8 -> out [H,W] int32
        n = q_plane.numel()
        out = torch.empty_like(q_plane, dtype=torch.int32)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _triton_wavelet_adaptive_dequant_kernel[grid](
            q_plane.view(-1), idx_packed, out.view(-1), codebook,
            base_q, n, G, BLOCK
        )
        return out.view(q_plane.shape)

    def quantize_fused_wavelet_adaptive_gpu(
        img: torch.Tensor,
        q_scale: float = 3.0,
        lamb: float = 5.0,
        G: int = 32,
        mode: str | None = None,
        chroma420: bool | None = None,
    ) -> tuple[dict, tuple[int, int, int]]:
        from .codec import quantize_pixel_wavelet_adaptive
        return quantize_pixel_wavelet_adaptive(img, q_scale=q_scale, lamb=lamb, G=G, mode=mode, chroma420=chroma420)

    # -------------------------------------------------------------------------
    # 2e. Sparse-bitstream GPU decode: whole-image mega-kernels (K0/K1a/K1b/K1c/K2)
    # One program per 32-elem block across ALL planes (grid = total blocks B):
    #   K0  occ byte bit -> occ_bits[B]
    #   K1a occ_rank -> mode bit + flat/hier select
    #   K1b flat_rank/hier_rank -> flat words; hier presence + k
    #   K1c nib prefix -> hier words (8-iter in-register nibble gather)
    #   K2  vals prefix + step -> int32 coeffs (in-register running count, no scatter)
    # All inter-block ranks are single torch cusmums over concatenated [B]
    # vectors; per-plane bases subtracted via the pstart table (no syncs).
    # H2D is a single u8 arena + i8 vals arena + int16 LL4 + tiny meta table.
    # -------------------------------------------------------------------------
    @triton.jit
    def _xs_k0_occ(plane_ptr, blk_ptr, arena_ptr, meta_ptr, occ_out, B,
                   BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < B
        p = tl.load(plane_ptr + offs, mask=m, other=0)
        b = tl.load(blk_ptr + offs, mask=m, other=0)
        occ_base = tl.load(meta_ptr + p * 8 + 0, mask=m, other=0)
        byte = tl.load(arena_ptr + occ_base + (b >> 3), mask=m, other=0).to(tl.int32)
        bit = (byte >> (b & 7)) & 1
        tl.store(occ_out + offs, bit.to(tl.uint8), mask=m)

    @triton.jit
    def _xs_k1a_mode(plane_ptr, blk_ptr, pstart_ptr, arena_ptr, meta_ptr,
                     occ_ptr, orank_ptr, mode_out, fsel_out, hsel_out, B,
                     BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < B
        p = tl.load(plane_ptr + offs, mask=m, other=0)
        occ = tl.load(occ_ptr + offs, mask=m, other=0)
        r = tl.load(orank_ptr + offs, mask=m, other=0).to(tl.int32)
        ps = tl.load(pstart_ptr + p, mask=m, other=0)
        base = tl.where(ps > 0, tl.load(orank_ptr + ps - 1, mask=m, other=0).to(tl.int32) + 1, 0)
        r = r - base
        mode_base = tl.load(meta_ptr + p * 8 + 1, mask=m, other=0)
        mbyte = tl.load(arena_ptr + mode_base + (r >> 3),
                        mask=m & (occ != 0), other=0).to(tl.int32)
        mode = (((mbyte >> (r & 7)) & 1) & (occ != 0)).to(tl.uint8)
        tl.store(mode_out + offs, mode, mask=m)
        tl.store(fsel_out + offs, (occ & (mode ^ 1)).to(tl.uint8), mask=m)
        tl.store(hsel_out + offs, (occ & mode).to(tl.uint8), mask=m)

    @triton.jit
    def _xs_k1b_words(plane_ptr, blk_ptr, pstart_ptr, arena_ptr, meta_ptr,
                      occ_ptr, mode_ptr, orank_ptr, frank_ptr, hrank_ptr,
                      word_out, k_out, presb_out, hidx_out, B,
                      BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < B
        p = tl.load(plane_ptr + offs, mask=m, other=0)
        b = tl.load(blk_ptr + offs, mask=m, other=0)
        occ = tl.load(occ_ptr + offs, mask=m, other=0)
        mode = tl.load(mode_ptr + offs, mask=m, other=0)
        ps = tl.load(pstart_ptr + p, mask=m, other=0)
        is_flat = (occ != 0) & (mode == 0)
        is_hier = (occ != 0) & (mode != 0)
        fr = tl.load(frank_ptr + offs, mask=m, other=0).to(tl.int32)
        fbase = tl.where(ps > 0, tl.load(frank_ptr + ps - 1, mask=m, other=0).to(tl.int32) + 1, 0)
        hr = tl.load(hrank_ptr + offs, mask=m, other=0).to(tl.int32)
        hbase = tl.where(ps > 0, tl.load(hrank_ptr + ps - 1, mask=m, other=0).to(tl.int32) + 1, 0)
        fmask_base = tl.load(meta_ptr + p * 8 + 2, mask=m, other=0)
        hfl_base = tl.load(meta_ptr + p * 8 + 3, mask=m, other=0)
        fi = fr - fbase
        b0 = tl.load(arena_ptr + fmask_base + fi * 4 + 0, mask=m & is_flat, other=0).to(tl.int32)
        b1 = tl.load(arena_ptr + fmask_base + fi * 4 + 1, mask=m & is_flat, other=0).to(tl.int32)
        b2 = tl.load(arena_ptr + fmask_base + fi * 4 + 2, mask=m & is_flat, other=0).to(tl.int32)
        b3 = tl.load(arena_ptr + fmask_base + fi * 4 + 3, mask=m & is_flat, other=0).to(tl.int32)
        fword = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        hi = hr - hbase
        pres = tl.load(arena_ptr + hfl_base + hi, mask=m & is_hier, other=0).to(tl.int32)
        k = tl.zeros([BLOCK], dtype=tl.int32)
        for s in range(8):
            k += (pres >> s) & 1
        word = tl.where(is_flat, fword, 0)
        tl.store(word_out + offs, word, mask=m)
        tl.store(k_out + offs, tl.where(is_hier, k, 0).to(tl.int32), mask=m)
        tl.store(presb_out + offs, tl.where(is_hier, pres, 0), mask=m)
        tl.store(hidx_out + offs, tl.where(is_hier, hi, 0), mask=m)

    @triton.jit
    def _xs_k1c_hier(plane_ptr, pstart_ptr, arena_ptr, meta_ptr,
                     occ_ptr, mode_ptr, presb_ptr, kval_ptr, niboff_ptr,
                     word_out, B, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < B
        p = tl.load(plane_ptr + offs, mask=m, other=0)
        occ = tl.load(occ_ptr + offs, mask=m, other=0)
        mode = tl.load(mode_ptr + offs, mask=m, other=0)
        active = m & (occ != 0) & (mode != 0)
        pres = tl.load(presb_ptr + offs, mask=active, other=0).to(tl.int32)
        ps = tl.load(pstart_ptr + p, mask=active, other=0)
        no = tl.load(niboff_ptr + offs, mask=active, other=0).to(tl.int32)
        kk = tl.load(kval_ptr + offs, mask=active, other=0).to(tl.int32)
        # offset = inclusive_cum[i] - k[i] - inclusive_cum[s-1]
        nbase = tl.where(ps > 0, tl.load(niboff_ptr + ps - 1, mask=active, other=0).to(tl.int32), 0)
        hnib_base = tl.load(meta_ptr + p * 8 + 7, mask=active, other=0)
        base_off = no - kk - nbase
        word = tl.zeros([BLOCK], dtype=tl.int32)
        for s in range(8):
            has = (pres >> s) & 1
            # intra-block rank of slot s among present slots
            low = pres & ((1 << s) - 1)
            intra = tl.zeros([BLOCK], dtype=tl.int32)
            for t in range(8):
                intra += (low >> t) & 1
            pos = base_off + intra
            hbyte = tl.load(arena_ptr + hnib_base + (pos >> 1),
                            mask=active & (has != 0), other=0).to(tl.int32)
            nib = (hbyte >> ((pos & 1) * 4)) & 0xF
            word |= tl.where(has != 0, nib << (4 * s), 0)
        tl.store(word_out + offs, word, mask=active)

    @triton.jit
    def _xs_k2_gather(plane_ptr, blk_ptr, pstart_ptr, arena_ptr, varena_ptr, meta_ptr,
                      occ_ptr, word_ptr, pop_ptr, voff_ptr, codebook_ptr, out_ptr, B,
                      ADAPTIVE: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < B
        p = tl.load(plane_ptr + offs, mask=m, other=0)
        b = tl.load(blk_ptr + offs, mask=m, other=0)
        occ = tl.load(occ_ptr + offs, mask=m, other=0)
        word = tl.load(word_ptr + offs, mask=m, other=0).to(tl.int32)
        vo = tl.load(voff_ptr + offs, mask=m, other=0).to(tl.int32)
        pp = tl.load(pop_ptr + offs, mask=m, other=0).to(tl.int32)
        ps = tl.load(pstart_ptr + p, mask=m, other=0)
        vbase = tl.where(ps > 0, tl.load(voff_ptr + ps - 1, mask=m, other=0).to(tl.int32), 0)
        vals_base = tl.load(meta_ptr + p * 8 + 4, mask=m, other=0)
        param = tl.load(meta_ptr + p * 8 + 6, mask=m, other=0)
        if ADAPTIVE:
            idx_base = tl.load(meta_ptr + p * 8 + 5, mask=m, other=0)
            ibyte = tl.load(arena_ptr + idx_base + (b >> 1), mask=m, other=0).to(tl.int32)
            idx = tl.where((b & 1) != 0, (ibyte >> 4) & 0xF, ibyte & 0xF)
            cb = tl.load(codebook_ptr + idx, mask=m, other=1.0).to(tl.float32)
            step_f = param.to(tl.float32) * cb
        vstart = vals_base + vo - pp - vbase
        cnt = tl.zeros([BLOCK], dtype=tl.int32)
        obase = offs * 32
        for j in range(32):
            bit = (word >> j) & 1
            vv = tl.load(varena_ptr + vstart + cnt, mask=m & (occ != 0) & (bit != 0), other=0).to(tl.int32)
            if ADAPTIVE:
                f = (vv.to(tl.float32) * step_f).to(tl.int32)
            else:
                half = param >> 1
                sgn = tl.where(vv > 0, 1, tl.where(vv < 0, -1, 0))
                f = vv * param + sgn * half
            tl.store(out_ptr + obase + j, tl.where((occ != 0) & (bit != 0), f, 0), mask=m)
            cnt += bit

    _xs_decode_cache: dict = {}

    def _xs_pattern_tables(inv, B, P, key):
        """Cached per-config block index tables (CPU): plane id + block id."""
        ent = _xs_decode_cache.get(key)
        if ent is None:
            plane = torch.zeros(B, dtype=torch.int32)
            blk = torch.zeros(B, dtype=torch.int32)
            pstarts = [0]
            for (name, c, h, w, M) in inv:
                pstarts.append(pstarts[-1] + M)
            for i, (name, c, h, w, M) in enumerate(inv):
                s = pstarts[i]
                plane[s:s + M] = i
                blk[s:s + M] = torch.arange(M, dtype=torch.int32)
            ent = {"plane": plane, "blk": blk,
                   "pstart": torch.tensor(pstarts, dtype=torch.int32),
                   "B": B, "P": P}
            _xs_decode_cache[key] = ent
        return ent

    def _synthesize_wavelet_planes_gpu(LL4, rec_planes, sub, H, W, out_buffer=None):
        # Shared synthesis: 4-stage IDWT chain(s) + fused RCT store.
        # LL4 [3,H4,W4] int32; rec_planes name -> [k,H,W] int32 (k=3 shared, k=1 luma L1).
        from .codec import _upsample2
        dev = LL4.device
        # 4:4:4 -> full chain; 4:2:0 -> luma 4-stage + chroma 3-stage + upsample
        if not sub:
            rec_ll3 = torch.empty((3, LL4.shape[1]*2, LL4.shape[2]*2), dtype=torch.int32, device=dev)
            for b in range(3):
                rec_ll3[b] = _launch_idwt_2d(LL4[b], rec_planes['LH4'][b], rec_planes['HL4'][b], rec_planes['HH4'][b])
            rec_ll2 = torch.empty((3, rec_ll3.shape[1]*2, rec_ll3.shape[2]*2), dtype=torch.int32, device=dev)
            for b in range(3):
                rec_ll2[b] = _launch_idwt_2d(rec_ll3[b], rec_planes['LH3'][b], rec_planes['HL3'][b], rec_planes['HH3'][b])
            rec_ll1 = torch.empty((3, rec_ll2.shape[1]*2, rec_ll2.shape[2]*2), dtype=torch.int32, device=dev)
            for b in range(3):
                rec_ll1[b] = _launch_idwt_2d(rec_ll2[b], rec_planes['LH2'][b], rec_planes['HL2'][b], rec_planes['HH2'][b])
            rec_yuv_planes = torch.empty((3, rec_ll1.shape[1]*2, rec_ll1.shape[2]*2), dtype=torch.int32, device=dev)
            for b in range(3):
                rec_yuv_planes[b] = _launch_idwt_2d(rec_ll1[b], rec_planes['LH1'][b], rec_planes['HL1'][b], rec_planes['HH1'][b])
        else:
            rec_ll3 = torch.empty((3, LL4.shape[1]*2, LL4.shape[2]*2), dtype=torch.int32, device=dev)
            for b in range(3):
                rec_ll3[b] = _launch_idwt_2d(LL4[b], rec_planes['LH4'][b], rec_planes['HL4'][b], rec_planes['HH4'][b])
            rec_ll2_y = torch.empty((1, rec_ll3.shape[1]*2, rec_ll3.shape[2]*2), dtype=torch.int32, device=dev)
            rec_ll2_y[0] = _launch_idwt_2d(rec_ll3[0], rec_planes['LH3'][0], rec_planes['HL3'][0], rec_planes['HH3'][0])
            rec_ll2_c = torch.empty((2, rec_ll3.shape[1]*2, rec_ll3.shape[2]*2), dtype=torch.int32, device=dev)
            for b in (1, 2):
                rec_ll2_c[b-1] = _launch_idwt_2d(rec_ll3[b], rec_planes['LH3'][b], rec_planes['HL3'][b], rec_planes['HH3'][b])
            rec_ll1_y = torch.empty((1, rec_ll2_y.shape[1]*2, rec_ll2_y.shape[2]*2), dtype=torch.int32, device=dev)
            rec_ll1_y[0] = _launch_idwt_2d(rec_ll2_y[0], rec_planes['LH2'][0], rec_planes['HL2'][0], rec_planes['HH2'][0])
            rec_c_half = torch.empty((2, rec_ll2_c.shape[1]*2, rec_ll2_c.shape[2]*2), dtype=torch.int32, device=dev)
            for b in range(2):
                rec_c_half[b] = _launch_idwt_2d(rec_ll2_c[b], rec_planes['LH2'][b+1], rec_planes['HL2'][b+1], rec_planes['HH2'][b+1])
            rec_y = torch.empty((1, rec_ll1_y.shape[1]*2, rec_ll1_y.shape[2]*2), dtype=torch.int32, device=dev)
            rec_y[0] = _launch_idwt_2d(rec_ll1_y[0], rec_planes['LH1'][0], rec_planes['HL1'][0], rec_planes['HH1'][0])
            rec_yuv_planes = torch.cat([rec_y, _upsample2(rec_c_half)], dim=0)
        # RCT inverse fused store
        Hp, Wp = rec_yuv_planes.shape[1], rec_yuv_planes.shape[2]
        n_pix = Hp * Wp
        y = rec_yuv_planes[0].reshape(-1)
        cb = rec_yuv_planes[1].reshape(-1)
        cr = rec_yuv_planes[2].reshape(-1)
        out_flat = torch.empty((n_pix*3,), dtype=torch.uint8, device=dev)
        BLOCK = 1024
        grid = (triton.cdiv(n_pix, BLOCK),)
        _triton_rct_inverse_kernel[grid](y, cb, cr, out_flat, n_pix, BLOCK=BLOCK)
        rec_rgb = out_flat.view(Hp, Wp, 3)[:H, :W, :]
        if out_buffer is not None:
            out_buffer.copy_(rec_rgb)
            return out_buffer
        return rec_rgb

    def _xs_ensure_arena(sparse: dict) -> dict:
        """Accept arena (stored format) or legacy sparse dict (converted on the fly)."""
        if sparse.get("format") == "xs-arena-v1":
            return sparse
        from .codec import sparse_pack_arena
        return sparse_pack_arena(sparse)

    def _synthesize_wavelet_batch_gpu(LL4b, recb, sub, H, W):
        """Batched synthesis [N,3,h4,w4] + rec planes -> [N,H,W,3] uint8 (torch, bit-exact).

        Uses the same integer lifting as the Triton single path, but with the
        batch folded into the leading dims so one launch covers all images.
        """
        from .codec import _idwt_53_2d_step_batched, _upsample2, rct_inverse

        def _idwt4(LL, LH, HL, HH):
            # [N,k,h,w] -> [N,k,2h,2w] via flattened 3D IDWT (F.pad needs 3D)
            N, k, h, w = LL.shape
            r = _idwt_53_2d_step_batched(
                LL.reshape(N * k, h, w), LH.reshape(N * k, h, w),
                HL.reshape(N * k, h, w), HH.reshape(N * k, h, w))
            return r.view(N, k, r.shape[-2], r.shape[-1])

        if not sub:
            rec_ll3 = _idwt4(LL4b, recb['LH4'], recb['HL4'], recb['HH4'])
            rec_ll2 = _idwt4(rec_ll3, recb['LH3'], recb['HL3'], recb['HH3'])
            rec_ll1 = _idwt4(rec_ll2, recb['LH2'], recb['HL2'], recb['HH2'])
            rec_yuv_b = _idwt4(rec_ll1, recb['LH1'], recb['HL1'], recb['HH1'])
        else:
            rec_ll3 = _idwt4(LL4b, recb['LH4'], recb['HL4'], recb['HH4'])
            rec_ll2_y = _idwt4(rec_ll3[:, 0:1], recb['LH3'][:, 0:1], recb['HL3'][:, 0:1], recb['HH3'][:, 0:1])
            rec_ll2_c = _idwt4(rec_ll3[:, 1:3], recb['LH3'][:, 1:3], recb['HL3'][:, 1:3], recb['HH3'][:, 1:3])
            rec_ll1_y = _idwt4(rec_ll2_y, recb['LH2'][:, 0:1], recb['HL2'][:, 0:1], recb['HH2'][:, 0:1])
            rec_c_half = _idwt4(rec_ll2_c, recb['LH2'][:, 1:3], recb['HL2'][:, 1:3], recb['HH2'][:, 1:3])
            rec_y = _idwt4(rec_ll1_y, recb['LH1'], recb['HL1'], recb['HH1'])
            rec_yuv_b = torch.cat([rec_y, _upsample2(rec_c_half)], dim=1)
        rec_yuv = rec_yuv_b.permute(0, 2, 3, 1)
        rec_rgb_full = rct_inverse(rec_yuv)
        return rec_rgb_full[:, :H, :W, :]

    def dequantize_sparse_wavelet_batch_gpu(
        arenas: list,
        device: str | torch.device = "cuda:0",
    ) -> torch.Tensor:
        """Batched GPU decode from arena dicts (or legacy sparse dicts) -> [N,H,W,3] uint8.

        All images must share orig_shape / adaptive / inv (fixed-size training
        batches do). Concatenates arenas on CPU (2 cats + meta fixup), H2Ds
        once (~7 transfers total), runs the K mega-kernels once over the
        N*B block grid, then batched torch synthesis. Bit-exact vs CPU.
        """
        from .codec import ADAPTIVE_CODEBOOK
        dev = torch.device(device)
        assert len(arenas) >= 1, "empty batch"
        arenas = [_xs_ensure_arena(a) for a in arenas]
        a0 = arenas[0]
        N = len(arenas)
        H, W, _ = tuple(a0["orig_shape"])
        adaptive = bool(a0.get("adaptive", False))
        for a in arenas[1:]:
            assert tuple(a["orig_shape"]) == (H, W, 3), "batch must share shape"
            assert bool(a.get("adaptive", False)) == adaptive, "batch must share schema"
            assert a["inv"] == a0["inv"], "batch must share plane layout"
        inv = a0["inv"]
        P, B = int(a0["P"]), int(a0["B"])
        key_single = (adaptive, tuple(a0["orig_shape"]), tuple(inv))
        pent = _xs_pattern_tables(inv, B, P, key_single)
        plane_single = pent["plane"]
        blk_single = pent["blk"]
        pstart_single = pent["pstart"]
        # Concat arenas + fixup meta offsets (CPU, no syncs)
        au8_list, ai8_list, ll4_list, meta_rows = [], [], [], []
        OU = OI = 0
        for a in arenas:
            au8_list.append(a["arena_u8"])
            ai8_list.append(a["arena_i8"])
            ll4_list.append(a["ll4"].to(torch.int16).reshape(-1))
            m = a["meta"].clone()
            for col in (0, 1, 2, 3, 7, 5):
                m[:, col] += OU
            m[:, 4] += OI
            meta_rows.append(m)
            OU += int(a["arena_u8"].numel())
            OI += int(a["arena_i8"].numel())
        arena_u8 = torch.cat(au8_list, dim=0)
        arena_i8 = torch.cat(ai8_list, dim=0).to(torch.int8)
        meta = torch.cat(meta_rows, dim=0)
        ll4_flat = torch.cat(ll4_list, dim=0)
        if N > 1:
            plane_global = torch.cat([plane_single + n * P for n in range(N)], dim=0)
            blk_global = torch.cat([blk_single] * N, dim=0)
        else:
            plane_global = plane_single
            blk_global = blk_single
        ps = pstart_single.tolist()
        pstart_list = [0]
        for n in range(N):
            base = n * B
            for i in range(P):
                pstart_list.append(base + int(ps[i + 1]))
        pstart_global = torch.tensor(pstart_list, dtype=torch.int32)
        BT, PT = N * B, N * P
        # H2D once
        dplane = plane_global.to(dev, non_blocking=True)
        dblk = blk_global.to(dev, non_blocking=True)
        dpstart = pstart_global.to(dev, non_blocking=True)
        darena = arena_u8.to(dev, non_blocking=True)
        dvals = arena_i8.to(dev, non_blocking=True)
        dmeta = meta.to(dev, non_blocking=True)
        dll4 = ll4_flat.to(dev, non_blocking=True).to(torch.int32)
        h4, w4 = tuple(a0["ll4_shapes"][0])
        LL4b = dll4.view(N, 3, h4, w4)
        if adaptive:
            codebook = a0.get("codebook", ADAPTIVE_CODEBOOK).to(dev).float().reshape(-1)
        else:
            codebook = torch.zeros(16, dtype=torch.float32, device=dev)
        BLOCK = 256
        grid = (triton.cdiv(BT, BLOCK),)
        occ = torch.empty(BT, dtype=torch.uint8, device=dev)
        _xs_k0_occ[grid](dplane, dblk, darena, dmeta, occ, BT, BLOCK=BLOCK)
        orank = occ.to(torch.int32).cumsum(0) - 1
        mode = torch.empty(BT, dtype=torch.uint8, device=dev)
        fsel = torch.empty(BT, dtype=torch.uint8, device=dev)
        hsel = torch.empty(BT, dtype=torch.uint8, device=dev)
        _xs_k1a_mode[grid](dplane, dblk, dpstart, darena, dmeta, occ, orank,
                           mode, fsel, hsel, BT, BLOCK=BLOCK)
        frank = fsel.to(torch.int32).cumsum(0) - 1
        hrank = hsel.to(torch.int32).cumsum(0) - 1
        word = torch.zeros(BT, dtype=torch.int32, device=dev)
        kval = torch.zeros(BT, dtype=torch.int32, device=dev)
        presb = torch.zeros(BT, dtype=torch.int32, device=dev)
        hidx = torch.zeros(BT, dtype=torch.int32, device=dev)
        _xs_k1b_words[grid](dplane, dblk, dpstart, darena, dmeta, occ, mode,
                            orank, frank, hrank, word, kval, presb, hidx, BT,
                            BLOCK=BLOCK)
        niboff = kval.cumsum(0)
        _xs_k1c_hier[grid](dplane, dpstart, darena, dmeta, occ, mode, presb,
                           kval, niboff, word, BT, BLOCK=BLOCK)
        ar32 = torch.arange(32, device=dev)
        pop = (((word.unsqueeze(-1) >> ar32) & 1).sum(-1).to(torch.int32))
        voff = pop.cumsum(0)
        out = torch.empty(BT * 32, dtype=torch.int32, device=dev)
        _xs_k2_gather[grid](dplane, dblk, dpstart, darena, dvals, dmeta, occ,
                            word, pop, voff, codebook, out, BT, ADAPTIVE=adaptive,
                            BLOCK=BLOCK)
        # Split into batched rec planes [N,k,h,w] per name
        name_to_entries: dict = {}
        for i, (name, c, h, w, M) in enumerate(inv):
            name_to_entries.setdefault(name, []).append((i, h, w))
        recb: dict = {}
        for name, entries in name_to_entries.items():
            h0, w0 = entries[0][1], entries[0][2]
            per_img = []
            for n in range(N):
                base_out = n * B * 32
                ch_lst = []
                for (i, h, w) in entries:
                    assert h == h0 and w == w0
                    s = base_out + int(ps[i]) * 32
                    ch_lst.append(out[s:s + h * w].view(h, w))
                per_img.append(torch.stack(ch_lst, dim=0))
            recb[name] = torch.stack(per_img, dim=0)
        sub = not any((nm == "LH1" and cc == 1) for (nm, cc, hh, ww, MM) in inv)
        return _synthesize_wavelet_batch_gpu(LL4b, recb, sub, H, W)

    def dequantize_sparse_wavelet_gpu(
        sparse: dict,
        device: str | torch.device = "cuda:0",
        out_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-image GPU decode (batch-of-1 fast path).

        Accepts the stored arena format directly (zero decode-side plan) or a
        legacy sparse dict (converted on the fly). Bit-exact vs CPU reference.
        """
        batch = dequantize_sparse_wavelet_batch_gpu([sparse], device=device)
        rec = batch[0]
        if out_buffer is not None:
            out_buffer.copy_(rec)
            return out_buffer
        return rec


    def dequantize_fused_wavelet_adaptive_gpu(
        packed_meta: dict,
        device: str | torch.device = "cuda:0",
        out_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dev = torch.device(device)
        H, W, C = packed_meta['orig_shape']
        G = packed_meta.get('G', 32)
        from .codec import ADAPTIVE_CODEBOOK, _chroma_is_subsampled
        codebook = packed_meta.get('codebook', ADAPTIVE_CODEBOOK).to(dev).float()
        channels = packed_meta['channels']
        # Build rec planes via Triton adaptive dequant
        # L4/L3/L2 stack [3, ...] (Y + subsampled C); L1 is luma-only [1, ...]
        LL4 = torch.stack([ch['LL4'].to(dev).to(torch.int32) for ch in channels], dim=0)
        # For each plane, Triton dequant
        plane_names = ["LH4","HL4","HH4","LH3","HL3","HH3","LH2","HL2","HH2","LH1","HL1","HH1"]
        rec_planes = {}
        rec_planes['LL4'] = LL4
        for name in plane_names:
            rec_list = []
            for c in range(3):
                ch = channels[c]
                if name not in ch:
                    continue  # chroma has no L1 under 4:2:0
                q_plane, idx_packed, bq = ch[name]
                q_plane = q_plane.to(dev)
                idx_packed = idx_packed.to(dev)
                rec = _launch_wavelet_adaptive_dequant(q_plane, idx_packed, bq, codebook, G)
                rec_list.append(rec)
            rec_planes[name] = torch.stack(rec_list, dim=0)
        return _synthesize_wavelet_planes_gpu(
            LL4, rec_planes, _chroma_is_subsampled(packed_meta), H, W, out_buffer)



    # -------------------------------------------------------------------------
    # 3. Fused Dequant + Linear Layer (Y = Dequant(X_int8) @ W.T + bias)
    # -------------------------------------------------------------------------
    @triton.jit
    def _fused_dequant_matmul_kernel(
        # Pointers to matrices
        a_ptr, scales_ptr, b_ptr, c_ptr, bias_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_K: tl.constexpr,
        HAS_BIAS: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Iterate along K dimension
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load INT8 tile
            a_i8 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0).to(tl.float32)
            
            # Load scale tile and dequantize in registers
            k_indices = k * BLOCK_SIZE_K + offs_k
            scale_idx = (offs_am[:, None] * (K // GROUP_SIZE_K)) + (k_indices[None, :] // GROUP_SIZE_K)
            scale = tl.load(scales_ptr + scale_idx, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=1.0).to(tl.float32)
            
            a_dequant = a_i8 * scale
            
            # Load weight tile (BF16/FP16)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
            
            # GEMM accumulation in registers
            accumulator += tl.dot(a_dequant, b)
            
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_bn).to(tl.float32)
            accumulator += bias[None, :]

        c = accumulator.to(tl.bfloat16)
        
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


    class FusedDequantLinear(nn.Module):
        """
        Fused INT8 Dequantization + Linear Layer.
        Computes Y = Dequant(X_int8) @ weight.T + bias directly in registers with ZERO intermediate VRAM writes.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.bfloat16))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
            else:
                self.register_parameter("bias", None)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, q_int8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            orig_shape = q_int8.shape[:-1]
            x_2d = q_int8.view(-1, self.in_features)
            M, K = x_2d.shape
            N = self.out_features
            
            out = torch.empty((M, N), dtype=torch.bfloat16, device=q_int8.device)
            
            # Grid parameters
            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            )
            
            _fused_dequant_matmul_kernel[grid](
                x_2d, scales, self.weight.t(), out, self.bias if self.bias is not None else x_2d,
                M, N, K,
                x_2d.stride(0), x_2d.stride(1),
                self.weight.t().stride(0), self.weight.t().stride(1),
                out.stride(0), out.stride(1),
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8, GROUP_SIZE_K=self.group_size,
                HAS_BIAS=self.bias is not None
            )
            
            if len(orig_shape) == 0:
                return out
            return out.view(*orig_shape, N)

else:
    # Fallback when Triton not available (CPU / Windows) — pure PyTorch, no VRAM overhead
    def quantize_fused_gpu(x: torch.Tensor, group_size: int = 32):
        from .codec import quantize_int8_g32
        return quantize_int8_g32(x, group_size)

    def quantize_amo_fused_gpu(x: torch.Tensor, group_size: int = 32, mode: str = "balanced", num_candidates: Optional[int] = None, lo: Optional[float] = None, hi: Optional[float] = None):
        # Avoid recursion: directly use PyTorch chunked path without Triton check
        import tensorcache.codec as _codec
        orig = _codec.HAS_TRITON
        _codec.HAS_TRITON = False
        try:
            return _codec.quantize_int8_amo_bq(x, group_size, num_candidates=num_candidates or 32, lo=lo or 0.95, hi=hi or 1.05, mode=mode)
        finally:
            _codec.HAS_TRITON = orig

    def dequantize_fused_gpu(q_int8: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, ...], group_size: int = 32, out_buffer: Optional[torch.Tensor] = None):
        from .codec import dequantize_int8_g32
        return dequantize_int8_g32(q_int8, scales, orig_shape, group_size, out_buffer)

    def quantize_fused_int4_gpu(x: torch.Tensor, group_size: int = 32):
        from .codec import quantize_int4_g32
        return quantize_int4_g32(x, group_size)

    def dequantize_fused_int4_gpu(q_packed: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, ...], group_size: int = 32, out_buffer: Optional[torch.Tensor] = None):
        from .codec import dequantize_int4_g32
        return dequantize_int4_g32(q_packed, scales, orig_shape, group_size, out_buffer)

    def quantize_fused_int3_gpu(x: torch.Tensor, group_size: int = 32):
        from .codec import quantize_int3_g32
        return quantize_int3_g32(x, group_size)

    def dequantize_fused_int3_gpu(q_packed: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, ...], group_size: int = 32, out_buffer: Optional[torch.Tensor] = None):
        from .codec import dequantize_int3_g32
        return dequantize_int3_g32(q_packed, scales, orig_shape, group_size, out_buffer)

    def quantize_fused_wavelet8x_gpu(img: torch.Tensor, q_scale: float = 3.0):
        # Keep PyTorch path for quant (already vectorized, bit-exact). No extra Triton needed.
        from .codec import quantize_pixel_wavelet8x
        return quantize_pixel_wavelet8x(img, q_scale=q_scale)

    def dequantize_fused_wavelet8x_gpu(packed_meta: dict, device: str | torch.device = "cpu", out_buffer: Optional[torch.Tensor] = None):
        # CPU fallback without Triton dispatch loop - shared batched helpers (luma 4-stage + chroma 3-stage)
        dev = torch.device(device)
        H, W, C = packed_meta['orig_shape']
        from .codec import (_wavelet_batched_stacks, _idwt_53_2d_step_batched,
                            _upsample2, _chroma_is_subsampled, rct_inverse)
        LL4, (LH4, HL4, HH4), (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = _wavelet_batched_stacks(packed_meta, dev)
        if not _chroma_is_subsampled(packed_meta):
            rec_ll3 = _idwt_53_2d_step_batched(LL4, LH4, HL4, HH4)
            rec_ll2 = _idwt_53_2d_step_batched(rec_ll3, LH3, HL3, HH3)
            rec_ll1 = _idwt_53_2d_step_batched(rec_ll2, LH2, HL2, HH2)
            rec_yuv_batched = _idwt_53_2d_step_batched(rec_ll1, LH1, HL1, HH1)
        else:
            rec_ll3 = _idwt_53_2d_step_batched(LL4, LH4, HL4, HH4)
            rec_ll2_y = _idwt_53_2d_step_batched(rec_ll3[0:1], LH3[0:1], HL3[0:1], HH3[0:1])
            rec_ll2_c = _idwt_53_2d_step_batched(rec_ll3[1:3], LH3[1:3], HL3[1:3], HH3[1:3])
            rec_ll1_y = _idwt_53_2d_step_batched(rec_ll2_y, LH2[0:1], HL2[0:1], HH2[0:1])
            rec_c_half = _idwt_53_2d_step_batched(rec_ll2_c, LH2[1:3], HL2[1:3], HH2[1:3])
            rec_y = _idwt_53_2d_step_batched(rec_ll1_y, LH1, HL1, HH1)
            rec_yuv_batched = torch.cat([rec_y, _upsample2(rec_c_half)], dim=0)
        rec_yuv = rec_yuv_batched.permute(1, 2, 0)
        rec_rgb_full = rct_inverse(rec_yuv)
        rec_rgb = rec_rgb_full[:H, :W, :]
        if out_buffer is not None:
            out_buffer.copy_(rec_rgb)
            return out_buffer
        return rec_rgb

    def quantize_fused_wavelet_adaptive_gpu(*args, **kwargs):
        from .codec import quantize_pixel_wavelet_adaptive
        return quantize_pixel_wavelet_adaptive(*args, **kwargs)

    def dequantize_fused_wavelet_adaptive_gpu(*args, **kwargs):
        from .codec import dequantize_pixel_wavelet_adaptive
        # Will dispatch to Triton version above if possible, else PyTorch
        # To avoid recursion, call codec directly with HAS_TRITON disabled? Use codec's PyTorch path
        # We expose the Triton version via _dequant_adaptive_triton wrapper, but for now delegate to codec
        return dequantize_pixel_wavelet_adaptive(*args, **kwargs)

    def dequantize_sparse_wavelet_gpu(sparse, device="cpu", out_buffer=None):
        # No Triton: unpack on target device with sync-free torch, then CPU/GPU
        # reference synthesis via dense rebuild. Arenas convert via sparse dict.
        from .codec import sparse_unpack_meta_gpu, dequantize_pixel_wavelet_adaptive, sparse_pack_arena
        import torch as _torch
        dev = _torch.device(device)
        if dev.type in ("cuda", "hip"):
            raise RuntimeError("dequantize_sparse_wavelet_gpu requires Triton + CUDA/ROCm")
        if sparse.get("format") == "xs-arena-v1":
            raise RuntimeError("CPU fallback needs legacy sparse dicts; convert at encode time")
        dense = sparse_unpack_meta_gpu(sparse, dev)
        if sparse.get("adaptive", False):
            return dequantize_pixel_wavelet_adaptive(dense, device=device, out_buffer=out_buffer)
        from .codec import dequantize_pixel_wavelet8x
        return dequantize_pixel_wavelet8x(dense, device=device, out_buffer=out_buffer)

    def dequantize_sparse_wavelet_batch_gpu(arenas, device="cpu"):
        import torch as _torch
        dev = _torch.device(device)
        if dev.type in ("cuda", "hip"):
            raise RuntimeError("dequantize_sparse_wavelet_batch_gpu requires Triton + CUDA/ROCm")
        outs = []
        for a in arenas:
            outs.append(dequantize_sparse_wavelet_gpu(a, device=device))
        return _torch.stack(outs, dim=0)

    _fused_quant_kernel = None
    _fused_amo_quant_kernel = None
    _fused_dequant_kernel = None
    _fused_dequant_int4_kernel = None
    _fused_dequant_int3_kernel = None
    _fused_dequant_matmul_kernel = None
    _triton_rct_inverse_kernel = None
    _triton_idwt_row_kernel = None
    _triton_idwt_col_kernel = None

    class FusedDequantLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FusedDequantLinear requires Triton + CUDA/ROCm (not available)")

        def forward(self, *args, **kwargs):
            raise RuntimeError("FusedDequantLinear requires Triton + CUDA/ROCm")
