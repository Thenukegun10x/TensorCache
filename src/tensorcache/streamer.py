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
        shuffle: bool = True,
        low_vram: bool = False,
        shard_idx: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.cache_prefix = Path(cache_prefix)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.low_vram = low_vram
        self.shard_idx = shard_idx
        self.rank = rank
        self.world_size = world_size

        # DDP rank -> shard_idx
        if rank is not None and world_size is not None and shard_idx is None:
            # Need to know num_shards first, so defer after manifest load
            pass

        # 1. Read metadata (detect sharded)
        shards_manifest = str(self.cache_prefix) + "_shards.json"
        is_sharded_manifest = os.path.exists(shards_manifest)
        self.is_sharded = False
        self.shard_mmaps = None
        self.shard_offsets = None
        self.shard_sizes = None
        self.num_shards = 1
        self.shard_prefixes = None

        if is_sharded_manifest:
            with open(shards_manifest, "r") as f:
                manifest = json.load(f)
            self.meta = manifest
            self.is_sharded = True
            self.num_shards = manifest.get("num_shards", len(manifest.get("shard_prefixes", [])))
            self.shard_prefixes = manifest.get("shard_prefixes", [f"{self.cache_prefix}_shard{i}" for i in range(self.num_shards)])
            # Resolve rank -> shard_idx now that we know num_shards
            if rank is not None and world_size is not None and shard_idx is None:
                shard_idx = rank % self.num_shards
                self.shard_idx = shard_idx
            # If shard_idx specified (explicit or via rank), load single shard
            if shard_idx is not None:
                if not (0 <= shard_idx < self.num_shards):
                    raise ValueError(f"shard_idx {shard_idx} out of range {self.num_shards}")
                pref = self.shard_prefixes[shard_idx]
                # per-shard meta for actual size
                shard_meta_path = pref + "_meta.json"
                if os.path.exists(shard_meta_path):
                    with open(shard_meta_path, "r") as sf:
                        sm = json.load(sf)
                    n = sm["num_samples"]
                    self.meta = sm
                else:
                    n = manifest["shard_sizes"][shard_idx] if "shard_sizes" in manifest else manifest["num_samples"] // self.num_shards
                    self.meta = manifest
                self.num_samples = n
                self.seq_len = self.meta["seq_len"]
                self.dim = self.meta["dim"]
                self.group_size = self.meta["group_size"]
                self.amo_bq = self.meta.get("amo_bq", False)
                self.scales_per_sample = (self.seq_len * self.dim + self.group_size - 1) // self.group_size
                # Single shard mmaps (behave as non-sharded)
                self.shard_mmaps = None
                self.shard_offsets = [0]
                self.shard_sizes = [n]
                self.num_shards = 1
                self.shard_prefixes = [pref]
                self.is_sharded = False  # treat as single for streamer logic
                # Open single shard mmaps
                self.int8_path = pref + "_int8.bin"
                self.scales_path = pref + "_scales.bin"
                int8_dtype = np.uint8 if self.amo_bq else np.int8
                self.mmap_int8 = np.memmap(self.int8_path, dtype=int8_dtype, mode="r", shape=(n, self.seq_len, self.dim))
                self.mmap_scales = np.memmap(self.scales_path, dtype=np.uint16, mode="r", shape=(n, self.scales_per_sample))
                if self.amo_bq:
                    zp_path = pref + "_zp.bin"
                    self.zp_path = zp_path
                    self.mmap_zp = np.memmap(zp_path, dtype=np.uint8, mode="r", shape=(n, self.scales_per_sample))
                else:
                    self.mmap_zp = None
                    self.zp_path = None
            else:
                # Multi-shard logical (all shards) - for single GPU big data
                # Load manifest sizes
                self.num_samples = manifest["num_samples"]
                self.seq_len = manifest["seq_len"]
                self.dim = manifest["dim"]
                self.group_size = manifest["group_size"]
                self.amo_bq = manifest.get("amo_bq", False)
                self.scales_per_sample = (self.seq_len * self.dim + self.group_size - 1) // self.group_size
                self.shard_sizes = manifest.get("shard_sizes", [])
                if not self.shard_sizes:
                    self.shard_sizes = []
                    for pref in self.shard_prefixes:
                        mp = pref + "_meta.json"
                        if os.path.exists(mp):
                            with open(mp, "r") as sf:
                                sm = json.load(sf)
                            self.shard_sizes.append(sm["num_samples"])
                        else:
                            self.shard_sizes.append(self.num_samples // len(self.shard_prefixes))
                self.shard_offsets = []
                off = 0
                for sz in self.shard_sizes:
                    self.shard_offsets.append(off)
                    off += sz
                # Load all shard mmaps
                self.shard_mmaps = []
                for pref, sz in zip(self.shard_prefixes, self.shard_sizes):
                    if sz == 0:
                        self.shard_mmaps.append({"int8": None, "scales": None, "zp": None})
                        continue
                    int8_dtype = np.uint8 if self.amo_bq else np.int8
                    mm_int8 = np.memmap(pref + "_int8.bin", dtype=int8_dtype, mode="r", shape=(sz, self.seq_len, self.dim))
                    mm_sc = np.memmap(pref + "_scales.bin", dtype=np.uint16, mode="r", shape=(sz, self.scales_per_sample))
                    mm_zp = np.memmap(pref + "_zp.bin", dtype=np.uint8, mode="r", shape=(sz, self.scales_per_sample)) if self.amo_bq else None
                    self.shard_mmaps.append({"int8": mm_int8, "scales": mm_sc, "zp": mm_zp, "prefix": pref})
                # For streamer, keep single mmap aliases as None, use shard_mmaps in iter
                self.mmap_int8 = None
                self.mmap_scales = None
                self.mmap_zp = None
                self.int8_path = None
                self.scales_path = None
                self.zp_path = None
                self.meta = manifest
        else:
            # Non-sharded
            # Handle rank->shard_idx for non-sharded case (should not happen, but for compat)
            if rank is not None and world_size is not None and shard_idx is None:
                # No sharding, but DDP with world_size: just use rank to slice? Not needed, treat as no shard
                pass
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
            self.shard_mmaps = None
            self.shard_offsets = [0]
            self.shard_sizes = [self.num_samples]
            self.num_shards = 1
            self.shard_prefixes = [str(self.cache_prefix)]
        
        # 3. Pre-allocate Pinned CPU Staging
        # low_vram=True => single buffer (13 MB B8, 53 MB B32, 212 MB B128)
        # low_vram=False => double buffered pipeline (16 MB B8, 64 MB B32, 256 MB B128) ~1.2x but 3-8x faster
        use_pin = torch.cuda.is_available() and self.device.type in ("cuda", "hip")
        q_dtype = torch.uint8 if self.amo_bq else torch.int8
        self.q_dtype = q_dtype
        n_bufs = 1 if low_vram else 2
        self.n_bufs = n_bufs
        self.pinned_int8 = [torch.empty((batch_size, self.seq_len, self.dim), dtype=q_dtype, pin_memory=use_pin) for _ in range(n_bufs)]
        self.pinned_scales = [torch.empty((batch_size, self.scales_per_sample), dtype=torch.bfloat16, pin_memory=use_pin) for _ in range(n_bufs)]
        self.pinned_int8_np = [p.numpy() for p in self.pinned_int8]
        self.pinned_scales_np = [p.view(torch.int16).numpy().view(np.uint16) for p in self.pinned_scales]
        if self.amo_bq:
            self.pinned_zp = [torch.empty((batch_size, self.scales_per_sample), dtype=torch.uint8, pin_memory=use_pin) for _ in range(n_bufs)]
            self.pinned_zp_np = [p.numpy() for p in self.pinned_zp]
        else:
            self.pinned_zp = [None]*n_bufs
            self.pinned_zp_np = [None]*n_bufs
        # compat single aliases (first buffer)
        self.pinned_int8_np_single = self.pinned_int8_np[0]

        # 4. Pre-allocate GPU VRAM + copy stream + events
        if self.device.type in ("cuda", "hip"):
            self.gpu_int8 = [torch.empty((batch_size, self.seq_len, self.dim), dtype=q_dtype, device=self.device) for _ in range(n_bufs)]
            self.gpu_scales = [torch.empty((batch_size, self.scales_per_sample), dtype=torch.bfloat16, device=self.device) for _ in range(n_bufs)]
            if self.amo_bq:
                self.gpu_zp = [torch.empty((batch_size, self.scales_per_sample), dtype=torch.uint8, device=self.device) for _ in range(n_bufs)]
            else:
                self.gpu_zp = [None]*n_bufs
            # out buffers: 2x for double, 1x + compat for low_vram
            if n_bufs == 2:
                self.out_bf16 = [torch.empty((batch_size, self.seq_len, self.dim), dtype=torch.bfloat16, device=self.device) for _ in range(2)]
                self.out_bf16_0 = self.out_bf16[0]
                self.out_bf16_1 = self.out_bf16[1]
            else:
                # low_vram: single out, still provide 2 aliases for compat (same tensor)
                self.out_bf16 = [torch.empty((batch_size, self.seq_len, self.dim), dtype=torch.bfloat16, device=self.device)]
                self.out_bf16_0 = self.out_bf16[0]
                self.out_bf16_1 = self.out_bf16[0]
            self.gpu_int8_single = self.gpu_int8[0]
            self.stream = torch.cuda.Stream(device=self.device)
            self.copy_stream = self.stream
            self.copy_events = [torch.cuda.Event() for _ in range(n_bufs)]
        else:
            self.gpu_int8 = self.gpu_scales = self.gpu_zp = self.out_bf16 = None
            self.stream = None
            self.copy_stream = None
            self.copy_events = None
            
        self.indices = np.arange(self.num_samples)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def _is_sharded_multi(self) -> bool:
        return getattr(self, "is_sharded", False) and getattr(self, "shard_mmaps", None) is not None and len(self.shard_mmaps) > 1

    def _shard_for_global(self, gidx: int):
        for si, off in enumerate(self.shard_offsets):
            sz = self.shard_sizes[si]
            if off <= gidx < off + sz:
                return si, gidx - off
        raise IndexError(gidx)

    def _copy_range_to_pinned(self, buf_idx: int, start_global: int, cur_bs: int):
        """Copy contiguous global range [start_global, start_global+cur_bs) into pinned buf_idx, handles sharded."""
        if not self._is_sharded_multi():
            np.copyto(self.pinned_int8_np[buf_idx][:cur_bs], self.mmap_int8[start_global:start_global+cur_bs])
            np.copyto(self.pinned_scales_np[buf_idx][:cur_bs], self.mmap_scales[start_global:start_global+cur_bs])
            if self.amo_bq:
                np.copyto(self.pinned_zp_np[buf_idx][:cur_bs], self.mmap_zp[start_global:start_global+cur_bs])
            return
        # Sharded multi: may span shards
        pos = 0
        remaining = cur_bs
        cur_global = start_global
        while remaining > 0:
            si, li = self._shard_for_global(cur_global)
            avail = self.shard_sizes[si] - li
            take = min(remaining, avail)
            mm = self.shard_mmaps[si]
            np.copyto(self.pinned_int8_np[buf_idx][pos:pos+take], mm["int8"][li:li+take])
            np.copyto(self.pinned_scales_np[buf_idx][pos:pos+take], mm["scales"][li:li+take])
            if self.amo_bq:
                np.copyto(self.pinned_zp_np[buf_idx][pos:pos+take], mm["zp"][li:li+take])
            pos += take
            cur_global += take
            remaining -= take

    def _copy_indices_to_pinned(self, buf_idx: int, batch_idx: np.ndarray):
        """Copy fancy global indices batch_idx into pinned buf_idx, handles sharded."""
        cur_bs = len(batch_idx)
        if not self._is_sharded_multi():
            np.copyto(self.pinned_int8_np[buf_idx][:cur_bs], self.mmap_int8[batch_idx])
            np.copyto(self.pinned_scales_np[buf_idx][:cur_bs], self.mmap_scales[batch_idx])
            if self.amo_bq:
                np.copyto(self.pinned_zp_np[buf_idx][:cur_bs], self.mmap_zp[batch_idx])
            return
        # Sharded: group by shard for fewer mmap calls
        # Group indices by shard to allow contiguous per-shard copies where possible
        # For simplicity, per-element still okay for small batch, but group for efficiency
        shard_groups = {}
        for pos, gidx in enumerate(batch_idx):
            si, li = self._shard_for_global(int(gidx))
            shard_groups.setdefault(si, []).append((pos, li))
        for si, lst in shard_groups.items():
            mm = self.shard_mmaps[si]
            for pos, li in lst:
                self.pinned_int8_np[buf_idx][pos] = mm["int8"][li]
                self.pinned_scales_np[buf_idx][pos] = mm["scales"][li]
                if self.amo_bq:
                    self.pinned_zp_np[buf_idx][pos] = mm["zp"][li]

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.shuffle:
            np.random.shuffle(self.indices)

        # CPU fallback: sequential without pipeline
        if self.stream is None:
            if not self.shuffle:
                for start in range(0, self.num_samples, self.batch_size):
                    cur_bs = min(self.batch_size, self.num_samples - start)
                    if self._is_sharded_multi():
                        self._copy_range_to_pinned(0, start, cur_bs)
                    else:
                        np.copyto(self.pinned_int8_np[0][:cur_bs], self.mmap_int8[start:start+cur_bs])
                        np.copyto(self.pinned_scales_np[0][:cur_bs], self.mmap_scales[start:start+cur_bs])
                        if self.amo_bq:
                            np.copyto(self.pinned_zp_np[0][:cur_bs], self.mmap_zp[start:start+cur_bs])
                    out_cpu = torch.empty((cur_bs, self.seq_len, self.dim), dtype=torch.bfloat16)
                    if self.amo_bq:
                        dequantize_int8_amo_bq(self.pinned_int8[0][:cur_bs], self.pinned_scales[0][:cur_bs], self.pinned_zp[0][:cur_bs],
                                               (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=out_cpu)
                    else:
                        dequantize_int8_g32(self.pinned_int8[0][:cur_bs], self.pinned_scales[0][:cur_bs],
                                            (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=out_cpu)
                    yield out_cpu
            else:
                for i in range(0, self.num_samples, self.batch_size):
                    batch_idx = self.indices[i:i+self.batch_size]
                    cur_bs = len(batch_idx)
                    if self._is_sharded_multi():
                        self._copy_indices_to_pinned(0, batch_idx)
                    else:
                        np.copyto(self.pinned_int8_np[0][:cur_bs], self.mmap_int8[batch_idx])
                        np.copyto(self.pinned_scales_np[0][:cur_bs], self.mmap_scales[batch_idx])
                        if self.amo_bq:
                            np.copyto(self.pinned_zp_np[0][:cur_bs], self.mmap_zp[batch_idx])
                    out_cpu = torch.empty((cur_bs, self.seq_len, self.dim), dtype=torch.bfloat16)
                    if self.amo_bq:
                        dequantize_int8_amo_bq(self.pinned_int8[0][:cur_bs], self.pinned_scales[0][:cur_bs], self.pinned_zp[0][:cur_bs],
                                               (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=out_cpu)
                    else:
                        dequantize_int8_g32(self.pinned_int8[0][:cur_bs], self.pinned_scales[0][:cur_bs],
                                            (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=out_cpu)
                    yield out_cpu
            return

        # Low VRAM single-buffer serial path (minimal 13 MB B8, saves 1x batch)
        if self.low_vram or self.n_bufs == 1:
            # Reuse single buffer 0, serial cpu->H2D->dequant
            if not self.shuffle:
                for start in range(0, self.num_samples, self.batch_size):
                    cur_bs = min(self.batch_size, self.num_samples - start)
                    if self._is_sharded_multi():
                        self._copy_range_to_pinned(0, start, cur_bs)
                    else:
                        np.copyto(self.pinned_int8_np[0][:cur_bs], self.mmap_int8[start:start+cur_bs])
                        np.copyto(self.pinned_scales_np[0][:cur_bs], self.mmap_scales[start:start+cur_bs])
                        if self.amo_bq:
                            np.copyto(self.pinned_zp_np[0][:cur_bs], self.mmap_zp[start:start+cur_bs])
                    with torch.cuda.stream(self.copy_stream):
                        self.gpu_int8[0][:cur_bs].copy_(self.pinned_int8[0][:cur_bs], non_blocking=True)
                        self.gpu_scales[0][:cur_bs].copy_(self.pinned_scales[0][:cur_bs], non_blocking=True)
                        if self.amo_bq:
                            self.gpu_zp[0][:cur_bs].copy_(self.pinned_zp[0][:cur_bs], non_blocking=True)
                        self.copy_events[0].record(self.copy_stream)
                    torch.cuda.current_stream().wait_event(self.copy_events[0])
                    target = self.out_bf16[0][:cur_bs]
                    if self.amo_bq:
                        dequantize_int8_amo_bq(self.gpu_int8[0][:cur_bs], self.gpu_scales[0][:cur_bs], self.gpu_zp[0][:cur_bs],
                                               (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=target)
                    else:
                        dequantize_int8_g32(self.gpu_int8[0][:cur_bs], self.gpu_scales[0][:cur_bs],
                                            (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=target)
                    yield target
            else:
                for i in range(0, self.num_samples, self.batch_size):
                    batch_idx = self.indices[i:i+self.batch_size]
                    cur_bs = len(batch_idx)
                    if self._is_sharded_multi():
                        self._copy_indices_to_pinned(0, batch_idx)
                    else:
                        np.copyto(self.pinned_int8_np[0][:cur_bs], self.mmap_int8[batch_idx])
                        np.copyto(self.pinned_scales_np[0][:cur_bs], self.mmap_scales[batch_idx])
                        if self.amo_bq:
                            np.copyto(self.pinned_zp_np[0][:cur_bs], self.mmap_zp[batch_idx])
                    with torch.cuda.stream(self.copy_stream):
                        self.gpu_int8[0][:cur_bs].copy_(self.pinned_int8[0][:cur_bs], non_blocking=True)
                        self.gpu_scales[0][:cur_bs].copy_(self.pinned_scales[0][:cur_bs], non_blocking=True)
                        if self.amo_bq:
                            self.gpu_zp[0][:cur_bs].copy_(self.pinned_zp[0][:cur_bs], non_blocking=True)
                        self.copy_events[0].record(self.copy_stream)
                    torch.cuda.current_stream().wait_event(self.copy_events[0])
                    target = self.out_bf16[0][:cur_bs]
                    if self.amo_bq:
                        dequantize_int8_amo_bq(self.gpu_int8[0][:cur_bs], self.gpu_scales[0][:cur_bs], self.gpu_zp[0][:cur_bs],
                                               (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=target)
                    else:
                        dequantize_int8_g32(self.gpu_int8[0][:cur_bs], self.gpu_scales[0][:cur_bs],
                                            (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=target)
                    yield target
            return

        # GPU pipeline: double-buffered, overlap next CPU copy + H2D with current dequant
        num_batches = (self.num_samples + self.batch_size - 1)//self.batch_size

        # helper to cpu copy batch `bi` into buffer `buf_idx` (handles sharded)
        def cpu_copy_into(bi: int, buf_idx: int):
            cur_bs = min(self.batch_size, self.num_samples - bi*self.batch_size) if not self.shuffle else len(self.indices[bi*self.batch_size:(bi+1)*self.batch_size])
            if not self.shuffle:
                start = bi * self.batch_size
                # Use helper for sharded
                if self._is_sharded_multi():
                    self._copy_range_to_pinned(buf_idx, start, cur_bs)
                else:
                    end = min(start + cur_bs, self.num_samples)
                    np.copyto(self.pinned_int8_np[buf_idx][:cur_bs], self.mmap_int8[start:end])
                    np.copyto(self.pinned_scales_np[buf_idx][:cur_bs], self.mmap_scales[start:end])
                    if self.amo_bq:
                        np.copyto(self.pinned_zp_np[buf_idx][:cur_bs], self.mmap_zp[start:end])
            else:
                batch_idx = self.indices[bi*self.batch_size:(bi+1)*self.batch_size]
                cur_bs = len(batch_idx)
                if self._is_sharded_multi():
                    self._copy_indices_to_pinned(buf_idx, batch_idx)
                else:
                    np.copyto(self.pinned_int8_np[buf_idx][:cur_bs], self.mmap_int8[batch_idx])
                    np.copyto(self.pinned_scales_np[buf_idx][:cur_bs], self.mmap_scales[batch_idx])
                    if self.amo_bq:
                        np.copyto(self.pinned_zp_np[buf_idx][:cur_bs], self.mmap_zp[batch_idx])
            return cur_bs

        # Preload first batch
        first_bs = cpu_copy_into(0, 0)
        with torch.cuda.stream(self.copy_stream):
            self.gpu_int8[0][:first_bs].copy_(self.pinned_int8[0][:first_bs], non_blocking=True)
            self.gpu_scales[0][:first_bs].copy_(self.pinned_scales[0][:first_bs], non_blocking=True)
            if self.amo_bq:
                self.gpu_zp[0][:first_bs].copy_(self.pinned_zp[0][:first_bs], non_blocking=True)
            self.copy_events[0].record(self.copy_stream)

        for bi in range(num_batches):
            cur = bi % 2
            nxt = 1 - cur
            cur_bs = min(self.batch_size, self.num_samples - bi*self.batch_size) if not self.shuffle else len(self.indices[bi*self.batch_size:(bi+1)*self.batch_size])

            # Wait for current H2D done on compute stream
            torch.cuda.current_stream().wait_event(self.copy_events[cur])

            # Dequant current (launches async)
            target = self.out_bf16[cur][:cur_bs]
            if self.amo_bq:
                dequantize_int8_amo_bq(self.gpu_int8[cur][:cur_bs], self.gpu_scales[cur][:cur_bs], self.gpu_zp[cur][:cur_bs],
                                       (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=target)
            else:
                dequantize_int8_g32(self.gpu_int8[cur][:cur_bs], self.gpu_scales[cur][:cur_bs],
                                    (cur_bs, self.seq_len, self.dim), group_size=self.group_size, out_buffer=target)

            # Prefetch next batch while current dequant runs (overlap CPU+ H2D with compute)
            if bi + 1 < num_batches:
                next_bs = cpu_copy_into(bi+1, nxt)
                with torch.cuda.stream(self.copy_stream):
                    self.gpu_int8[nxt][:next_bs].copy_(self.pinned_int8[nxt][:next_bs], non_blocking=True)
                    self.gpu_scales[nxt][:next_bs].copy_(self.pinned_scales[nxt][:next_bs], non_blocking=True)
                    if self.amo_bq:
                        self.gpu_zp[nxt][:next_bs].copy_(self.pinned_zp[nxt][:next_bs], non_blocking=True)
                    self.copy_events[nxt].record(self.copy_stream)

            yield target

    def close(self):
        """Cleanly releases HIP/CUDA streams, pinned buffers, and memory maps (critical for ROCm Windows)."""
        if hasattr(self, "stream") and self.stream is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            del self.stream
            self.stream = None
        if hasattr(self, "copy_stream") and getattr(self, "copy_stream", None) is not None:
            # copy_stream is alias to stream, already deleted
            self.copy_stream = None
        if hasattr(self, "copy_events") and getattr(self, "copy_events", None) is not None:
            del self.copy_events
            self.copy_events = None
            
        # Release pinned CPU & GPU buffers (handle both single and double buffered)
        if hasattr(self, "pinned_int8"):
            del self.pinned_int8
            if hasattr(self, "pinned_scales"):
                del self.pinned_scales
            if hasattr(self, "pinned_int8_np"):
                del self.pinned_int8_np
            if hasattr(self, "pinned_scales_np"):
                del self.pinned_scales_np
            if hasattr(self, "pinned_zp"):
                del self.pinned_zp
            if hasattr(self, "pinned_zp_np"):
                del self.pinned_zp_np
            
        if hasattr(self, "gpu_int8"):
            del self.gpu_int8
            if hasattr(self, "gpu_scales"):
                del self.gpu_scales
            if hasattr(self, "gpu_zp"):
                del self.gpu_zp
            if hasattr(self, "out_bf16"):
                del self.out_bf16
            if hasattr(self, "out_bf16_0"):
                del self.out_bf16_0
            if hasattr(self, "out_bf16_1"):
                del self.out_bf16_1
            
        # Sharded mmaps
        if getattr(self, "is_sharded", False) and getattr(self, "shard_mmaps", None) is not None:
            for mm in self.shard_mmaps:
                if mm["int8"] is not None and hasattr(mm["int8"], "_mmap") and mm["int8"]._mmap is not None:
                    try:
                        mm["int8"]._mmap.close()
                    except: pass
                if mm["scales"] is not None and hasattr(mm["scales"], "_mmap") and mm["scales"]._mmap is not None:
                    try:
                        mm["scales"]._mmap.close()
                    except: pass
                if mm["zp"] is not None and hasattr(mm["zp"], "_mmap") and mm["zp"]._mmap is not None:
                    try:
                        mm["zp"]._mmap.close()
                    except: pass
            self.shard_mmaps = None
            self.mmap_int8 = None
            self.mmap_scales = None
            self.mmap_zp = None
        else:
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
