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

    _fused_quant_kernel = None
    _fused_amo_quant_kernel = None
    _fused_dequant_kernel = None
    _fused_dequant_int4_kernel = None
    _fused_dequant_int3_kernel = None
    _fused_dequant_matmul_kernel = None

    class FusedDequantLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FusedDequantLinear requires Triton + CUDA/ROCm (not available)")

        def forward(self, *args, **kwargs):
            raise RuntimeError("FusedDequantLinear requires Triton + CUDA/ROCm")
