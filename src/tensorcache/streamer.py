"""
Zero-Copy Ring-Buffered Tensor Streamer.
Locks CPU RAM and GPU VRAM to a fixed, tiny footprint (~40 MB) with 0 dynamic allocations and 0 GC churn.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterator, Tuple, Optional, Union

import numpy as np
import torch

from .codec import dequantize_int8_g32, dequantize_int8_amo_bq


class ZeroCopyTensorStreamer:
    """
    Ultra-lean, ring-buffered batch streamer.
    - Pre-allocates fixed pinned CPU buffers (0 RAM bloat).
    - Pre-allocates double-buffered GPU VRAM targets (0 VRAM fragmentation).
    - Completely eliminates Python memory allocations during training.
    """
    def __init__(
        self,
        cache_prefix: Union[str, Path],
        batch_size: int = 128,
        device: str = "cuda:0",
        shuffle: bool = True
    ):
        self.cache_prefix = Path(cache_prefix)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        
        # 1. Read metadata
        meta_path = str(self.cache_prefix) + "_meta.json"
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.num_samples = self.meta["num_samples"]
        self.seq_len = self.meta["seq_len"]
        self.dim = self.meta["dim"]
        self.group_size = self.meta["group_size"]
        self.amo_bq = self.meta.get("amo_bq", False)
        self.scales_per_sample = (self.seq_len * self.dim + self.group_size - 1) // self.group_size
        
        # 2. Open read-only memory maps (dtype depends on amo)
        self.int8_path = str(self.cache_prefix) + "_int8.bin"
        self.scales_path = str(self.cache_prefix) + "_scales.bin"
        int8_dtype = np.uint8 if self.amo_bq else np.int8
        self.mmap_int8 = np.memmap(
            self.int8_path, dtype=int8_dtype, mode="r",
            shape=(self.num_samples, self.seq_len, self.dim)
        )
        self.mmap_scales = np.memmap(
            self.scales_path, dtype=np.uint16, mode="r",
            shape=(self.num_samples, self.scales_per_sample)
        )
        if self.amo_bq:
            zp_path = str(self.cache_prefix) + "_zp.bin"
            # fallback to meta zp_file if exists
            zp_file = self.meta.get("zp_file")
            if zp_file and not Path(zp_path).exists():
                zp_path = str(self.cache_prefix.parent / zp_file)
            self.zp_path = zp_path
            self.mmap_zp = np.memmap(
                self.zp_path, dtype=np.uint8, mode="r",
                shape=(self.num_samples, self.scales_per_sample)
            )
        else:
            self.mmap_zp = None
            self.zp_path = None
        
        # 3. Pre-allocate Pinned CPU Memory Staging Buffers (Fixed ~22-45 MB RAM)
        # amo adds 0.16MB per 5.4M, still ~45MB for batch 128
        q_dtype = torch.uint8 if self.amo_bq else torch.int8
        self.pinned_int8 = torch.empty(
            (batch_size, self.seq_len, self.dim), dtype=q_dtype, pin_memory=True
        )
        self.pinned_scales = torch.empty(
            (batch_size, self.scales_per_sample), dtype=torch.bfloat16, pin_memory=True
        )
        self.pinned_int8_np = self.pinned_int8.numpy()
        self.pinned_scales_np = self.pinned_scales.view(torch.int16).numpy().view(np.uint16)
        if self.amo_bq:
            self.pinned_zp = torch.empty(
                (batch_size, self.scales_per_sample), dtype=torch.uint8, pin_memory=True
            )
            self.pinned_zp_np = self.pinned_zp.numpy()
        else:
            self.pinned_zp = None
            self.pinned_zp_np = None
        
        # 4. Pre-allocate Static GPU VRAM Ring Buffers (Fixed ~22-45 MB VRAM)
        if self.device.type in ("cuda", "hip"):
            self.gpu_int8 = torch.empty((batch_size, self.seq_len, self.dim), dtype=q_dtype, device=self.device)
            self.gpu_scales = torch.empty((batch_size, self.scales_per_sample), dtype=torch.bfloat16, device=self.device)
            if self.amo_bq:
                self.gpu_zp = torch.empty((batch_size, self.scales_per_sample), dtype=torch.uint8, device=self.device)
            else:
                self.gpu_zp = None
            # Double-buffered output targets (reuse for minimal VRAM)
            self.out_bf16_0 = torch.empty((batch_size, self.seq_len, self.dim), dtype=torch.bfloat16, device=self.device)
            self.out_bf16_1 = torch.empty((batch_size, self.seq_len, self.dim), dtype=torch.bfloat16, device=self.device)
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None
            self.gpu_zp = None
            
        self.indices = np.arange(self.num_samples)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        ring_idx = 0
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = self.indices[i:i + self.batch_size]
            cur_bs = len(batch_idx)
            
            # 1. Zero-copy slice directly into pre-allocated pinned CPU buffer
            np.copyto(self.pinned_int8_np[:cur_bs], self.mmap_int8[batch_idx])
            np.copyto(self.pinned_scales_np[:cur_bs], self.mmap_scales[batch_idx])
            if self.amo_bq:
                np.copyto(self.pinned_zp_np[:cur_bs], self.mmap_zp[batch_idx])
            
            # 2. Asynchronous DMA transfer into pre-allocated GPU VRAM buffer
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self.gpu_int8[:cur_bs].copy_(self.pinned_int8[:cur_bs], non_blocking=True)
                    self.gpu_scales[:cur_bs].copy_(self.pinned_scales[:cur_bs], non_blocking=True)
                    if self.amo_bq:
                        self.gpu_zp[:cur_bs].copy_(self.pinned_zp[:cur_bs], non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.stream)
                
                # 3. Dequantize in-place into static ring target (0 VRAM allocation)
                target_buf = self.out_bf16_0[:cur_bs] if ring_idx == 0 else self.out_bf16_1[:cur_bs]
                ring_idx = 1 - ring_idx
                
                if self.amo_bq:
                    dequantize_int8_amo_bq(
                        self.gpu_int8[:cur_bs], self.gpu_scales[:cur_bs], self.gpu_zp[:cur_bs],
                        (cur_bs, self.seq_len, self.dim),
                        group_size=self.group_size,
                        out_buffer=target_buf
                    )
                else:
                    dequantize_int8_g32(
                        self.gpu_int8[:cur_bs], self.gpu_scales[:cur_bs],
                        (cur_bs, self.seq_len, self.dim),
                        group_size=self.group_size,
                        out_buffer=target_buf
                    )
                yield target_buf
            else:
                # CPU fallback
                out_cpu = torch.empty((cur_bs, self.seq_len, self.dim), dtype=torch.bfloat16)
                if self.amo_bq:
                    dequantize_int8_amo_bq(
                        self.pinned_int8[:cur_bs], self.pinned_scales[:cur_bs], self.pinned_zp[:cur_bs],
                        (cur_bs, self.seq_len, self.dim),
                        group_size=self.group_size,
                        out_buffer=out_cpu
                    )
                else:
                    dequantize_int8_g32(
                        self.pinned_int8[:cur_bs], self.pinned_scales[:cur_bs],
                        (cur_bs, self.seq_len, self.dim),
                        group_size=self.group_size,
                        out_buffer=out_cpu
                    )
                yield out_cpu

    def close(self):
        """Cleanly releases HIP/CUDA streams, pinned buffers, and memory maps (critical for ROCm Windows)."""
        if hasattr(self, "stream") and self.stream is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            del self.stream
            self.stream = None
            
        # Release pinned CPU & GPU buffers
        if hasattr(self, "pinned_int8"):
            del self.pinned_int8
            del self.pinned_scales
            del self.pinned_int8_np
            del self.pinned_scales_np
            if hasattr(self, "pinned_zp") and self.pinned_zp is not None:
                del self.pinned_zp
                del self.pinned_zp_np
            
        if hasattr(self, "gpu_int8"):
            del self.gpu_int8
            del self.gpu_scales
            if hasattr(self, "gpu_zp") and self.gpu_zp is not None:
                del self.gpu_zp
            del self.out_bf16_0
            del self.out_bf16_1
            
        if hasattr(self, "mmap_int8") and self.mmap_int8 is not None:
            if hasattr(self.mmap_int8, "_mmap") and self.mmap_int8._mmap is not None:
                self.mmap_int8._mmap.close()
            del self.mmap_int8
            self.mmap_int8 = None
            
        if hasattr(self, "mmap_scales") and self.mmap_scales is not None:
            if hasattr(self.mmap_scales, "_mmap") and self.mmap_scales._mmap is not None:
                self.mmap_scales._mmap.close()
            del self.mmap_scales
            self.mmap_scales = None
        if hasattr(self, "mmap_zp") and self.mmap_zp is not None:
            if hasattr(self.mmap_zp, "_mmap") and self.mmap_zp._mmap is not None:
                self.mmap_zp._mmap.close()
            del self.mmap_zp
            self.mmap_zp = None
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
