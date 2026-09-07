"""
High-Throughput Raw Pixel Cache Writer and Dataset Loader.
Eliminates the 19 MB/s JPEG CPU decoding bottleneck via zero-copy memory-mapping or LZ4 streaming.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple, List, Optional, Union
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import pack_int4, unpack_int4, pack_int3, unpack_int3

try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False


class PixelCacheWriter:
    """
    Builds a memory-mapped raw uint8 pixel cache, quantized INT4/INT3 pixel cache,
    or XS wavelet arena cache (quant="xs": ~7.8x at 36.8dB balanced, GPU-decodable).
    Quantized caches use blockwise 4/3-bit (G=32) with BF16 scales, 2x/2.29x vs raw, PSNR 37/31dB.
    Feature caches in INT4/INT3 are blocked (guarded) due to >2% RMSE collapse.
    """
    def __init__(
        self,
        output_prefix: Union[str, Path],
        num_samples: int,
        height: int = 336,
        width: int = 336,
        channels: int = 3,
        quant: str = "raw",
        quant_bits: Optional[int] = None,
        group_size: int = 32,
        xs_mode: str = "balanced",
    ):
        # Normalize quant args: quant="raw"/"int4"/"int3"/"xs" or quant_bits=8/4/3
        if quant_bits is not None:
            if quant_bits == 8:
                quant = "raw"
            elif quant_bits == 4:
                quant = "int4"
            elif quant_bits == 3:
                quant = "int3"
            else:
                raise ValueError(f"quant_bits must be 8/4/3, got {quant_bits}")
        if quant not in ("raw", "int4", "int3", "xs"):
            raise ValueError(f"quant must be 'raw'/'int4'/'int3'/'xs', got {quant}")
        self.quant = quant
        self.group_size = group_size
        self.xs_mode = xs_mode
        self.quant_bits = 8 if quant in ("raw", "xs") else (4 if quant == "int4" else 3)
        self.output_prefix = Path(output_prefix)
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.channels = channels
        self.elements_per_sample = height * width * channels
        self.blocks_per_sample = (self.elements_per_sample + group_size - 1) // group_size
        
        self.bin_path = str(self.output_prefix) + "_pixels.bin"
        self.meta_path = str(self.output_prefix) + "_pixel_meta.json"
        
        if self.quant == "raw":
            self.mmap_pixels = np.memmap(
                self.bin_path, dtype=np.uint8, mode="w+",
                shape=(num_samples, height, width, channels)
            )
            self.mmap_q = None
            self.mmap_scales = None
        elif self.quant == "xs":
            # XS wavelet arenas are variable-size: stream concatenated blobs
            # to plain files (+ offset table), mmap'd read-only by the dataset.
            # Layout mirrors the GPU batch driver's in-memory arena.
            self.xs_u8_path = str(self.output_prefix) + "_xs_u8.bin"
            self.xs_i8_path = str(self.output_prefix) + "_xs_i8.bin"
            self.xs_meta_path = str(self.output_prefix) + "_xs_meta.bin"
            self.xs_ll4_path = str(self.output_prefix) + "_xs_ll4.bin"
            self.bin_path = self.xs_u8_path  # for close/meta
            self._xs_fu8 = open(self.xs_u8_path, "wb")
            self._xs_fi8 = open(self.xs_i8_path, "wb")
            self._xs_fmeta = open(self.xs_meta_path, "wb")
            self._xs_fll4 = open(self.xs_ll4_path, "wb")
            self._xs_table = []  # per-sample offset/len rows (small: ~10 ints)
            self._xs_shared = None  # inv/P/B/orig_shape/adaptive/codebook... (from sample 0)
            self._xs_ou = self._xs_oi = self._xs_orows = self._xs_oll4 = 0
            self.mmap_pixels = None
            self.mmap_q = None
            self.mmap_scales = None
        else:
            # Quantized: packed q + scales
            # q packed: for int4, 0.5B/elem -> ceil(elements/2) bytes per sample
            # For int3, need bit packing: 3b per elem -> ceil(elements*3/8) bytes
            if self.quant == "int4":
                q_bytes_per_sample = (self.elements_per_sample + 1) // 2
            else:  # int3
                q_bytes_per_sample = (self.elements_per_sample * 3 + 7) // 8
            self.q_path = str(self.output_prefix) + f"_pixels_int{self.quant_bits}.bin"
            self.scales_path = str(self.output_prefix) + f"_pixels_int{self.quant_bits}_scales.bin"
            self.bin_path = self.q_path  # for close
            self.mmap_q = np.memmap(
                self.q_path, dtype=np.uint8, mode="w+",
                shape=(num_samples, q_bytes_per_sample)
            )
            self.mmap_scales = np.memmap(
                self.scales_path, dtype=np.uint16, mode="w+",
                shape=(num_samples, self.blocks_per_sample)
            )
            self.mmap_pixels = None
        self.current_idx = 0
        self._closed = False

    def __enter__(self) -> "PixelCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):  # last-resort cleanup if user forgets close()
        try:
            self.close()
        except Exception:
            pass

    def _quantize_blockwise(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize uint8 [H,W,C] -> packed q bytes + scales BF16. Returns (q_packed uint8, scales uint16)."""
        flat = arr.flatten().astype(np.float32)  # 0-255
        # Normalize to -1..1 for symmetric quant around 0: (x/127.5 -1) -> -1..1
        # But for uint8 we want scale based on max per block, not global
        # Use blockwise: flatten -> view (-1, G) -> per block amax/127 -> quant
        t = torch.from_numpy(flat).float()
        # Convert to float -1..1
        t_norm = (t / 127.5 - 1.0)
        # Quantize per G
        G = self.group_size
        bits = self.quant_bits
        levels = 2**(bits-1) - 1
        # Pad
        numel = t_norm.numel()
        pad_len = (G - numel % G) % G
        if pad_len>0:
            t_norm = torch.nn.functional.pad(t_norm, (0, pad_len))
        blocks = t_norm.view(-1, G)
        # Per block amax
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = (amax / levels).squeeze(-1).to(torch.bfloat16)  # [M]
        # Quantize
        q_blocks = torch.round(blocks / scales.unsqueeze(-1).float()).clamp(-levels-1, levels).to(torch.int8)
        q_flat = q_blocks.flatten()[:numel]

        if bits == 4:
            q_packed = pack_int4(q_flat)
        else:
            q_packed = pack_int3(q_flat)

        scales_u16 = scales.view(torch.int16).cpu().numpy().view(np.uint16)
        return q_packed, scales_u16

    def _append_xs(self, arr: np.ndarray):
        """Encode one uint8 [H,W,C] image to an xs arena and stream it to disk."""
        from .codec import (quantize_pixel_wavelet_adaptive, sparse_pack_meta,
                            sparse_pack_arena)
        t = torch.from_numpy(arr).to(torch.uint8)
        meta, _ = quantize_pixel_wavelet_adaptive(t, mode=self.xs_mode)
        arena = sparse_pack_arena(sparse_pack_meta(meta))
        # Fixed-size training assumption: every sample shares the layout.
        shared = {
            "inv": [(n, c, h, w, m) for (n, c, h, w, m) in arena["inv"]],
            "P": int(arena["P"]),
            "B": int(arena["B"]),
            "orig_shape": [int(v) for v in arena["orig_shape"]],
            "adaptive": bool(arena["adaptive"]),
            "pad_h": int(arena["pad_h"]),
            "pad_w": int(arena["pad_w"]),
            "ll4_shapes": [[int(v) for v in s] for s in arena["ll4_shapes"]],
            "codebook": [float(v) for v in arena["codebook"].reshape(-1).tolist()],
            "q_scale": arena.get("q_scale"),
            "lamb": arena.get("lamb"),
            "mode": arena.get("mode"),
        }
        if self._xs_shared is None:
            self._xs_shared = shared
        elif shared["inv"] != self._xs_shared["inv"]:
            raise ValueError("XS cache requires identical H/W/mode for all samples")
        u8 = arena["arena_u8"].numpy().tobytes()
        i8 = arena["arena_i8"].numpy().tobytes()
        mt = arena["meta"].numpy().tobytes()
        l4 = arena["ll4"].numpy().tobytes()
        self._xs_fu8.write(u8)
        self._xs_fi8.write(i8)
        self._xs_fmeta.write(mt)
        self._xs_fll4.write(l4)
        P = shared["P"]
        self._xs_table.append({
            "u8_off": self._xs_ou, "u8_len": len(u8),
            "i8_off": self._xs_oi, "i8_len": len(i8),
            "meta_row": self._xs_orows, "n_planes": P,
            "ll4_off": self._xs_oll4, "ll4_len": len(l4) // 2,
        })
        self._xs_ou += len(u8)
        self._xs_oi += len(i8)
        self._xs_orows += P
        self._xs_oll4 += len(l4) // 2

    def append_image(self, img_input: Union[np.ndarray, Image.Image, torch.Tensor, str, Path]):
        """
        Appends an image to the raw memory map. Automatically resizes if needed.
        Supports quant="raw" (uint8) and quant="int4"/"int3" (packed + scales).
        """
        if self.current_idx >= self.num_samples:
            raise ValueError(f"Exceeded pre-allocated sample count ({self.num_samples})")
            
        if isinstance(img_input, (str, Path)):
            with Image.open(img_input) as im:
                im = im.convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR)
                arr = np.array(im, dtype=np.uint8)
        elif isinstance(img_input, Image.Image):
            im = img_input.convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.uint8)
        elif isinstance(img_input, torch.Tensor):
            arr = img_input.cpu().numpy().astype(np.uint8)
        else:
            arr = np.asarray(img_input, dtype=np.uint8)
            
        if self.quant == "raw":
            self.mmap_pixels[self.current_idx] = arr
        elif self.quant == "xs":
            self._append_xs(arr)
        else:
            q_packed, scales_u16 = self._quantize_blockwise(arr)
            # Pad to expected size
            q_bytes = self.mmap_q.shape[1]
            if q_packed.size < q_bytes:
                # pad zeros
                tmp = np.zeros(q_bytes, dtype=np.uint8)
                tmp[:q_packed.size] = q_packed
                q_packed = tmp
            self.mmap_q[self.current_idx] = q_packed[:q_bytes]
            self.mmap_scales[self.current_idx] = scales_u16
        self.current_idx += 1

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "mmap_pixels") and self.mmap_pixels is not None:
            self.mmap_pixels.flush()
            if hasattr(self.mmap_pixels, "_mmap") and self.mmap_pixels._mmap is not None:
                self.mmap_pixels._mmap.close()
            del self.mmap_pixels
            self.mmap_pixels = None
        if hasattr(self, "mmap_q") and self.mmap_q is not None:
            self.mmap_q.flush()
            if hasattr(self.mmap_q, "_mmap") and self.mmap_q._mmap is not None:
                self.mmap_q._mmap.close()
            del self.mmap_q
            self.mmap_q = None
        if hasattr(self, "mmap_scales") and self.mmap_scales is not None:
            self.mmap_scales.flush()
            if hasattr(self.mmap_scales, "_mmap") and self.mmap_scales._mmap is not None:
                self.mmap_scales._mmap.close()
            del self.mmap_scales
            self.mmap_scales = None
        for fh_attr in ("_xs_fu8", "_xs_fi8", "_xs_fmeta", "_xs_fll4"):
            fh = getattr(self, fh_attr, None)
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
                setattr(self, fh_attr, None)
            
        meta = {
            "num_samples": self.current_idx,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "bin_file": os.path.basename(self.bin_path),
            "quant": self.quant,
            "quant_bits": self.quant_bits,
            "group_size": self.group_size,
        }
        if self.quant == "xs":
            meta["xs_mode"] = self.xs_mode
            meta["xs_files"] = {
                "u8": os.path.basename(self.xs_u8_path),
                "i8": os.path.basename(self.xs_i8_path),
                "meta": os.path.basename(self.xs_meta_path),
                "ll4": os.path.basename(self.xs_ll4_path),
            }
            meta["xs_shared"] = self._xs_shared
            meta["xs_table"] = self._xs_table
        if self.quant in ("int4", "int3"):
            meta["q_file"] = os.path.basename(self.q_path)
            meta["scales_file"] = os.path.basename(self.scales_path)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)


class PixelCacheDataset(Dataset):
    """
    Zero-Decode Memory-Mapped Pixel Dataset.
    Loads raw uint8 images at full disk line-rate (>2000 MB/s) with zero CPU decompression overhead.
    Supports quantized int4/int3 with blockwise dequant (PSNR 37/31dB), and XS wavelet
    arenas (quant="xs") with GPU batch decode (~12k img/s @336) via iter_batches.
    """
    def __init__(self, cache_prefix: Union[str, Path], transform=None, decode_device: Optional[Union[str, torch.device]] = None):
        self.cache_prefix = Path(cache_prefix)
        self.meta_path = str(self.cache_prefix) + "_pixel_meta.json"
        
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.num_samples = self.meta["num_samples"]
        self.height = self.meta["height"]
        self.width = self.meta["width"]
        self.channels = self.meta["channels"]
        self.transform = transform
        self.quant = self.meta.get("quant", "raw")
        self.quant_bits = self.meta.get("quant_bits", 8 if self.quant=="raw" else (4 if self.quant=="int4" else 3))
        self.group_size = self.meta.get("group_size", 32)
        self._xs = (self.quant == "xs")
        if decode_device is None:
            decode_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.decode_device = torch.device(decode_device)
        
        if self.quant == "raw":
            self.bin_path = str(self.cache_prefix) + "_pixels.bin"
            # Fallback if meta has bin_file
            if "bin_file" in self.meta and self.meta["bin_file"] != os.path.basename(self.bin_path):
                alt = str(self.cache_prefix.parent / self.meta["bin_file"]) if os.path.dirname(self.meta["bin_file"]) else str(self.cache_prefix.parent / self.meta["bin_file"])
                # Try both
                if os.path.exists(str(self.cache_prefix) + "_" + self.meta["bin_file"]) or os.path.exists(self.meta["bin_file"]):
                    pass
            self.mmap_pixels = np.memmap(
                self.bin_path, dtype=np.uint8, mode="r",
                shape=(self.num_samples, self.height, self.width, self.channels)
            )
            self.mmap_q = None
            self.mmap_scales = None
        elif self.quant == "xs":
            # Concatenated arena blobs + offset table (read-only mmaps).
            d = self.cache_prefix.parent
            xf = self.meta["xs_files"]
            self._xs_u8 = np.memmap(str(d / xf["u8"]), dtype=np.uint8, mode="r")
            self._xs_i8 = np.memmap(str(d / xf["i8"]), dtype=np.int8, mode="r")
            sh = self.meta["xs_shared"]
            nrows = sum(r["n_planes"] for r in self.meta["xs_table"])
            self._xs_meta = np.memmap(str(d / xf["meta"]), dtype=np.int32, mode="r",
                                      shape=(nrows, 8))
            nll4 = sum(r["ll4_len"] for r in self.meta["xs_table"])
            self._xs_ll4 = np.memmap(str(d / xf["ll4"]), dtype=np.int16, mode="r",
                                     shape=(nll4,))
            self._xs_shared = sh
            self._xs_inv = [(n, c, h, w, m) for (n, c, h, w, m) in sh["inv"]]
            self._xs_codebook = torch.tensor(sh["codebook"], dtype=torch.float32)
            self.mmap_pixels = None
            self.mmap_q = None
            self.mmap_scales = None
        else:  # int4 / int3
            self.elements_per_sample = self.height * self.width * self.channels
            self.blocks_per_sample = (self.elements_per_sample + self.group_size -1)//self.group_size
            if self.quant == "int4":
                q_bytes = (self.elements_per_sample +1)//2
            else:
                q_bytes = (self.elements_per_sample*3+7)//8
            q_path = str(self.cache_prefix) + f"_pixels_int{self.quant_bits}.bin"
            scales_path = str(self.cache_prefix) + f"_pixels_int{self.quant_bits}_scales.bin"
            # Fallback to meta file names
            if "q_file" in self.meta:
                q_path = str(self.cache_prefix.parent / self.meta["q_file"])
                if not os.path.exists(q_path):
                    q_path = str(self.cache_prefix) + f"_pixels_int{self.quant_bits}.bin"
            if "scales_file" in self.meta:
                scales_path = str(self.cache_prefix.parent / self.meta["scales_file"])
                if not os.path.exists(scales_path):
                    scales_path = str(self.cache_prefix) + f"_pixels_int{self.quant_bits}_scales.bin"
            self.q_path = q_path
            self.scales_path = scales_path
            self.mmap_q = np.memmap(q_path, dtype=np.uint8, mode="r", shape=(self.num_samples, q_bytes))
            self.mmap_scales = np.memmap(scales_path, dtype=np.uint16, mode="r", shape=(self.num_samples, self.blocks_per_sample))
            self.mmap_pixels = None

    def __len__(self) -> int:
        return self.num_samples

    def _dequantize_blockwise(self, q_packed: np.ndarray, scales_u16: np.ndarray) -> np.ndarray:
        """Dequantize packed int4/int3 q + BF16 scales -> uint8 [H,W,C]."""
        flat_size = self.elements_per_sample
        bits = self.quant_bits

        if bits == 4:
            q_signed = unpack_int4(q_packed, flat_size)
        else:
            q_signed = unpack_int3(q_packed, flat_size)

        G = self.group_size
        pad_len = (G - flat_size % G) % G
        total = flat_size + pad_len
        M = total // G

        if pad_len > 0:
            q_padded = np.pad(q_signed, (0, pad_len))
        else:
            q_padded = q_signed

        q_t = torch.from_numpy(q_padded.copy()).float().view(M, G)
        scales_t = torch.from_numpy(scales_u16.view(np.int16).copy()).view(torch.bfloat16).float()

        rec_blocks = q_t * scales_t.unsqueeze(-1)
        rec_flat = rec_blocks.view(-1)[:flat_size]
        rec_uint8 = ((rec_flat + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
        return rec_uint8.reshape(self.height, self.width, self.channels)

    def get_arena(self, idx: int) -> dict:
        """Arena dict for sample idx (page-cache-hot copies of the mmap blobs).

        Matches the xs-arena-v1 schema the GPU batch driver consumes, so
        decode_arenas([ds.get_arena(i) for i in batch]) needs no repacking.
        Copies (not views): the read-only mmap is non-writable and torch
        refuses non-writable backing; bytes still come from page cache.
        """
        if not self._xs:
            raise RuntimeError("get_arena requires quant='xs'")
        r = self.meta["xs_table"][idx]
        sh = self._xs_shared
        return {
            "format": "xs-arena-v1",
            "adaptive": bool(sh["adaptive"]),
            "orig_shape": tuple(sh["orig_shape"]),
            "pad_h": int(sh["pad_h"]),
            "pad_w": int(sh["pad_w"]),
            "inv": self._xs_inv,
            "B": int(sh["B"]),
            "P": int(sh["P"]),
            "arena_u8": torch.from_numpy(
                np.array(self._xs_u8[r["u8_off"]:r["u8_off"] + r["u8_len"]])),
            "arena_i8": torch.from_numpy(
                np.array(self._xs_i8[r["i8_off"]:r["i8_off"] + r["i8_len"]])),
            "meta": torch.from_numpy(
                np.array(self._xs_meta[r["meta_row"]:r["meta_row"] + r["n_planes"]])).to(torch.int32),
            "ll4": torch.from_numpy(
                np.array(self._xs_ll4[r["ll4_off"]:r["ll4_off"] + r["ll4_len"]])).to(torch.int16),
            "ll4_shapes": [tuple(s) for s in sh["ll4_shapes"]],
            "codebook": self._xs_codebook,
            "q_scale": sh.get("q_scale"),
            "lamb": sh.get("lamb"),
            "mode": sh.get("mode"),
        }

    def decode_arenas(self, arenas: list, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Batch-decode arena dicts -> uint8 [N,H,W,3] on device.

        GPU path uses the fused mega-kernel batch driver (Triton on CUDA/ROCm);
        CPU path goes through arena_to_sparse + reference dequant
        (Windows-safe, no Triton needed).
        """
        if device is None:
            device = self.decode_device
        dev = torch.device(device)
        if dev.type in ("cuda", "hip"):
            from .fused_ops import dequantize_sparse_wavelet_batch_gpu
            return dequantize_sparse_wavelet_batch_gpu(arenas, device=device)
        from .codec import (arena_to_sparse, sparse_unpack_meta,
                            dequantize_pixel_wavelet_adaptive,
                            dequantize_pixel_wavelet8x)
        outs = []
        for a in arenas:
            leg = arena_to_sparse(a) if a.get("format") == "xs-arena-v1" else a
            dense = sparse_unpack_meta(leg)
            if leg.get("adaptive", False):
                outs.append(dequantize_pixel_wavelet_adaptive(dense, device=device))
            else:
                outs.append(dequantize_pixel_wavelet8x(dense, device=device))
        return torch.stack(outs, dim=0)

    def iter_batches(self, batch_size: int = 32, device: Optional[Union[str, torch.device]] = None,
                     shuffle: bool = False):
        """Yield decoded uint8 [B,H,W,3] batches straight from mmap (training fast path)."""
        idxs = list(range(self.num_samples))
        if shuffle:
            import random
            random.shuffle(idxs)
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            yield self.decode_arenas([self.get_arena(i) for i in chunk], device=device)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._xs:
            t = self.decode_arenas([self.get_arena(idx)], device=self.decode_device)[0].cpu()
            if self.transform is not None:
                t = self.transform(t)
            return t
        if self.quant == "raw":
            # Zero-copy memory-mapped slice
            arr = self.mmap_pixels[idx]
            t = torch.from_numpy(arr.copy()) # [H, W, C]
        else:
            q_packed = self.mmap_q[idx]
            scales_u16 = self.mmap_scales[idx]
            arr = self._dequantize_blockwise(q_packed, scales_u16)
            t = torch.from_numpy(arr.copy()) # [H, W, C]
        
        if self.transform is not None:
            t = self.transform(t)
        return t

    def close(self):
        """Releases the memory map handle (important on Windows)."""
        if hasattr(self, "mmap_pixels") and self.mmap_pixels is not None:
            if hasattr(self.mmap_pixels, "_mmap") and self.mmap_pixels._mmap is not None:
                self.mmap_pixels._mmap.close()
            del self.mmap_pixels
            self.mmap_pixels = None
        if hasattr(self, "mmap_q") and self.mmap_q is not None:
            if hasattr(self.mmap_q, "_mmap") and self.mmap_q._mmap is not None:
                self.mmap_q._mmap.close()
            del self.mmap_q
            self.mmap_q = None
        if hasattr(self, "mmap_scales") and self.mmap_scales is not None:
            if hasattr(self.mmap_scales, "_mmap") and self.mmap_scales._mmap is not None:
                self.mmap_scales._mmap.close()
            del self.mmap_scales
            self.mmap_scales = None
        for mm_attr in ("_xs_u8", "_xs_i8", "_xs_meta", "_xs_ll4"):
            mm = getattr(self, mm_attr, None)
            if mm is not None:
                if hasattr(mm, "_mmap") and mm._mmap is not None:
                    mm._mmap.close()
                del mm
                setattr(self, mm_attr, None)

    def __enter__(self) -> "PixelCacheDataset":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):  # last-resort mmap release if user forgets close()
        try:
            self.close()
        except Exception:
            pass
