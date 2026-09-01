"""
Tailored INT8 compressor for feature caches.
Exploits per-block low entropy: constant runs + small bit-width for boring images.

Per 32-elem block (G=32) of int8 q:
- If all 32 same -> RLE 2B (value + count) vs 32B => 16x
- Else if max_abs < 8 -> 4b pack (4 bits per elem) 16B vs 32B => 2x
- Else 8b raw 32B

Stored as:
- offsets: uint32 per block (prefix sum of compressed sizes) for O(1) random access
- bitwidths: uint8 per block (0=RLE, 4, 8)
- data: packed bytes flat

GPU decompress: one threadblock per block, load offset/len/bitwidth, unpack via shift/mask, no branch mispredict.

Tailored for int8: 256 alphabet small, runs common on boring images, small magnitudes common on Gaussian features.
High BW: decompress = shift/mask + table lookup, fused with dequant `q*scale` in one kernel.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Tuple

def compress_int8_blocks(q_int8: torch.Tensor, group_size: int = 32) -> Tuple[bytes, torch.Tensor, torch.Tensor, int]:
    """
    CPU compress int8 flat q. Returns (data_bytes, offsets, bitwidths, num_blocks)
    q_int8: flat 1D int8 tensor (e.g., from quantize_int8_g32)
    Vectorized per-block analysis for speed (no Python loop for min/max).
    """
    numel = q_int8.numel()
    pad_len = (group_size - numel % group_size) % group_size
    if pad_len>0:
        q_padded = torch.nn.functional.pad(q_int8, (0, pad_len))
    else:
        q_padded = q_int8
    blocks = q_padded.view(-1, group_size)  # [M,32]
    M = blocks.shape[0]
    # Vectorized per-block min/max/range
    mn = blocks.min(dim=-1).values  # [M] int8
    mx = blocks.max(dim=-1).values
    rng = (mx.to(torch.int16) - mn.to(torch.int16)).to(torch.int16)  # avoid overflow
    # First pass: 0=RLE, 4=range<16, else need outlier check
    # For range>=16, check outliers for 4b+1/2
    # Outlier = count of (q - mn >15)
    # Vectorized outlier count: for each block, count >15
    # Use broadcasting: (blocks - mn.unsqueeze(-1) >15).sum(-1)
    # But blocks is int8, mn is int8, need int16
    diff = blocks.to(torch.int16) - mn.to(torch.int16).unsqueeze(-1)  # [M,32] 0..255
    outliers = (diff > 15).sum(dim=-1)  # [M] 0..32
    # For rng>=16, if outliers <=2 and remaining range <16, use 4b+outlier
    # Check remaining range after removing outliers: need to find max of non-outliers
    # For outlier blocks, remaining max is second max if outlier is max, but for simplicity check if outliers <=2
    # We'll use: if outliers<=2, then we can store as 4b+outliers
    # Determine bitwidths:
    # 0 if rng==0
    # 4 if rng<16
    # 5 if rng>=16 and outliers==1
    # 6 if rng>=16 and outliers==2
    # else 8
    # For outlier case, need to ensure remaining range <16: but if outliers are the large values, remaining max may be < mn+15?
    # Since diff>15 for outliers, the remaining max diff <=15, so remaining range <16 by definition if outliers are exactly those >15
    # So if outliers<=2, the remaining 30-31 values are within [mn, mn+15] => range <16 for remaining
    bitwidths = torch.where(rng == 0, torch.tensor(0, dtype=torch.uint8, device=blocks.device),
                torch.where(rng < 16, torch.tensor(4, dtype=torch.uint8, device=blocks.device),
                torch.where(outliers == 1, torch.tensor(5, dtype=torch.uint8, device=blocks.device),
                torch.where(outliers == 2, torch.tensor(6, dtype=torch.uint8, device=blocks.device),
                            torch.tensor(8, dtype=torch.uint8, device=blocks.device)))))
    bases = mn.to(torch.int8)  # for 0,4,5,6
    sizes = torch.where(bitwidths==0, torch.tensor(1, dtype=torch.int32, device=blocks.device),
            torch.where(bitwidths==4, torch.tensor(17, dtype=torch.int32, device=blocks.device),
            torch.where(bitwidths==5, torch.tensor(19, dtype=torch.int32, device=blocks.device),  # 1 base +16 packed +1 pos +1 val
            torch.where(bitwidths==6, torch.tensor(21, dtype=torch.int32, device=blocks.device),  # 1+16+2*2
                        torch.tensor(32, dtype=torch.int32, device=blocks.device)))))
    # offsets prefix sum (vectorized cumsum)
    offsets = torch.empty(M, dtype=torch.int32, device=blocks.device)
    # Use cumsum: offsets[i] = sum_{j<i} sizes[j]
    if M>0:
        cumsum = torch.cumsum(sizes, dim=0)
        offsets[0]=0
        if M>1:
            offsets[1:] = cumsum[:-1]
        total = int(cumsum[-1].item())
    else:
        total=0
    # Pack data
    data = bytearray(total)
    if blocks.device.type == 'cpu':
        blocks_np = blocks.numpy().view(np.uint8)  # [M,32] uint8
        bitwidths_np = bitwidths.numpy()
        offsets_np = offsets.numpy()
        bases_np = bases.numpy().view(np.uint8)
        data_np = np.frombuffer(data, dtype=np.uint8)
        # Precompute diff for 4b/5/6 cases
        # For 4b, need diff = q - base
        # For outlier, need to find outlier positions
        for i in range(M):
            bw = int(bitwidths_np[i])
            off = int(offsets_np[i])
            if bw == 0:
                data_np[off] = bases_np[i]
            elif bw == 4:
                base = int(bases[i].item())
                data_np[off] = base & 0xFF
                b = blocks[i]
                diff = (b.to(torch.int16) - base).to(torch.uint8).numpy()
                packed = (diff[1::2].astype(np.uint8) << 4) | diff[0::2].astype(np.uint8)
                data_np[off+1:off+17] = packed
            elif bw == 5 or bw == 6:
                base = int(bases[i].item())
                data_np[off] = base & 0xFF
                b = blocks[i]
                # Find outliers: where (q - base >15)
                # Use numpy
                b_np = b.numpy().view(np.int8).astype(np.int16)  # int8 to int16
                base16 = int(base)
                diff_np = b_np - base16  # 0..255
                outlier_mask = diff_np > 15
                outlier_idx = np.where(outlier_mask)[0]  # 1 or 2
                # Pack non-outliers as 4b: need to pack 32 values but outliers will be stored separately
                # For simplicity, pack all 32 as 4b with outliers' nibbles as 0, then store outlier pos+val separately
                # But for decompress we need to know which positions are outliers to replace
                # Instead, pack all as 4b with outliers' diff truncated to 0 (will be overwritten)
                # We'll store packed for all 32 as if outliers were 0, then store outlier corrections
                # For now, pack diff clamped to 0..15
                diff_clamped = np.clip(diff_np, 0, 15).astype(np.uint8)
                packed = (diff_clamped[1::2].astype(np.uint8) << 4) | diff_clamped[0::2].astype(np.uint8)
                data_np[off+1:off+17] = packed
                # Store outliers after packed: pos 1B + val 1B per outlier
                # For 5: 1 outlier, for 6: 2 outliers
                # Outlier pos as uint8 0..31, val as int8
                base_off = off+17
                for k, idx in enumerate(outlier_idx[:2]):  # at most 2
                    data_np[base_off + k*2] = int(idx) & 0xFF
                    data_np[base_off + k*2 + 1] = int(b[int(idx)].item()) & 0xFF
            else:  # 8
                data_np[off:off+32] = blocks_np[i]
    else:
        for i in range(M):
            bw = bitwidths[i].item()
            off = offsets[i].item()
            b = blocks[i]
            if bw == 0:
                data[off] = int(bases[i].item()) & 0xFF
            elif bw == 4:
                base = int(bases[i].item())
                data[off] = base & 0xFF
                for j in range(0, 32, 2):
                    v0 = int(b[j].item()) - base
                    v1 = int(b[j+1].item()) - base
                    data[off + 1 + j//2] = ((v1 & 0xF) << 4) | (v0 & 0xF)
            elif bw == 5 or bw == 6:
                base = int(bases[i].item())
                data[off] = base & 0xFF
                # Find outliers
                outlier_idx = []
                for j in range(32):
                    if int(b[j].item()) - base > 15:
                        outlier_idx.append(j)
                        if len(outlier_idx) >= (1 if bw==5 else 2):
                            break
                # Pack all as 4b with outliers clamped
                for j in range(0, 32, 2):
                    v0 = int(b[j].item()) - base
                    v1 = int(b[j+1].item()) - base
                    v0c = 0 if v0>15 else v0
                    v1c = 0 if v1>15 else v1
                    data[off + 1 + j//2] = ((v1c & 0xF) << 4) | (v0c & 0xF)
                base_off = off+17
                for k, idx in enumerate(outlier_idx):
                    data[base_off + k*2] = idx & 0xFF
                    data[base_off + k*2 + 1] = int(b[idx].item()) & 0xFF
            else:
                for j in range(32):
                    data[off + j] = int(b[j].item()) & 0xFF
    return bytes(data), offsets.cpu(), bitwidths.cpu(), M

def decompress_int8_blocks(data: bytes, offsets: torch.Tensor, bitwidths: torch.Tensor, group_size: int=32, orig_numel: int=None, device: str='cpu') -> torch.Tensor:
    """
    Decompress to flat int8 tensor. If orig_numel given, trim padding.
    CPU vectorized for 8b memcpy, Python loop only for RLE/4b (rare).
    Supports 0=RLE,4=4b,5=4b+1 outlier,6=4b+2 outliers,8=raw,7=Huffman (future)
    """
    M = offsets.numel()
    total_needed = M*group_size
    out = torch.empty(total_needed, dtype=torch.int8)
    if offsets.device.type == 'cpu' and isinstance(data, (bytes, bytearray)):
        import numpy as np
        out_np = out.numpy().view(np.int8)
        data_np = np.frombuffer(data, dtype=np.uint8)
        bitwidths_np = bitwidths.numpy()
        offsets_np = offsets.numpy()
        for i in range(M):
            bw = int(bitwidths_np[i])
            off = int(offsets_np[i])
            base_idx = i*group_size
            if bw == 0:
                val = int(data_np[off])
                if val >= 128:
                    val -= 256
                out_np[base_idx:base_idx+group_size] = val
            elif bw == 4:
                base = int(data_np[off])
                if base >= 128:
                    base -= 256
                packed = data_np[off+1:off+17]
                low = packed & 0xF
                high = (packed >> 4) & 0xF
                vals = np.empty(32, dtype=np.uint8)
                vals[0::2] = low
                vals[1::2] = high
                out_np[base_idx:base_idx+group_size] = (base + vals.astype(np.int16)).astype(np.int8)
            elif bw == 5:
                base = int(data_np[off])
                if base >= 128:
                    base -= 256
                packed = data_np[off+1:off+17]
                low = packed & 0xF
                high = (packed >> 4) & 0xF
                vals = np.empty(32, dtype=np.uint8)
                vals[0::2] = low
                vals[1::2] = high
                out_np[base_idx:base_idx+group_size] = (base + vals.astype(np.int16)).astype(np.int8)
                # outlier correction
                pos = int(data_np[off+17])
                val = int(data_np[off+18])
                if val >= 128:
                    val -= 256
                out_np[base_idx + pos] = val
            elif bw == 6:
                base = int(data_np[off])
                if base >= 128:
                    base -= 256
                packed = data_np[off+1:off+17]
                low = packed & 0xF
                high = (packed >> 4) & 0xF
                vals = np.empty(32, dtype=np.uint8)
                vals[0::2] = low
                vals[1::2] = high
                out_np[base_idx:base_idx+group_size] = (base + vals.astype(np.int16)).astype(np.int8)
                pos1 = int(data_np[off+17]); val1 = int(data_np[off+18])
                pos2 = int(data_np[off+19]); val2 = int(data_np[off+20])
                if val1 >= 128:
                    val1 -= 256
                if val2 >= 128:
                    val2 -= 256
                out_np[base_idx + pos1] = val1
                out_np[base_idx + pos2] = val2
            elif bw == 7:
                # Huffman placeholder: not implemented, fallback to 8b
                out_np[base_idx:base_idx+group_size] = data_np[off:off+32].view(np.int8)
            else:  # 8
                out_np[base_idx:base_idx+group_size] = data_np[off:off+32].view(np.int8)
    else:
        for i in range(M):
            bw = bitwidths[i].item()
            off = offsets[i].item()
            if bw == 0:
                val = data[off]
                if val >= 128:
                    val -= 256
                out[i*group_size:(i+1)*group_size] = val
            elif bw == 4:
                base = data[off]
                if base >= 128:
                    base -= 256
                for j in range(0, group_size, 2):
                    byte = data[off + 1 + j//2]
                    v0 = byte & 0xF
                    v1 = (byte >> 4) & 0xF
                    out[i*group_size + j] = base + v0
                    out[i*group_size + j + 1] = base + v1
            elif bw == 5:
                base = data[off]
                if base >= 128:
                    base -= 256
                for j in range(0, group_size, 2):
                    byte = data[off + 1 + j//2]
                    v0 = byte & 0xF
                    v1 = (byte >> 4) & 0xF
                    out[i*group_size + j] = base + v0
                    out[i*group_size + j + 1] = base + v1
                pos = data[off+17]; val = data[off+18]
                if val >= 128:
                    val -= 256
                out[i*group_size + pos] = val
            elif bw == 6:
                base = data[off]
                if base >= 128:
                    base -= 256
                for j in range(0, group_size, 2):
                    byte = data[off + 1 + j//2]
                    v0 = byte & 0xF
                    v1 = (byte >> 4) & 0xF
                    out[i*group_size + j] = base + v0
                    out[i*group_size + j + 1] = base + v1
                pos1 = data[off+17]; val1 = data[off+18]
                pos2 = data[off+19]; val2 = data[off+20]
                if val1 >= 128:
                    val1 -= 256
                if val2 >= 128:
                    val2 -= 256
                out[i*group_size + pos1] = val1
                out[i*group_size + pos2] = val2
            elif bw == 7:
                for j in range(group_size):
                    val = data[off + j]
                    if val >= 128:
                        val -= 256
                    out[i*group_size + j] = val
            else:
                for j in range(group_size):
                    val = data[off + j]
                    if val >= 128:
                        val -= 256
                    out[i*group_size + j] = val
    if orig_numel is not None:
        out = out[:orig_numel]
    return out

# GPU fused decompress+dequant - portable Triton, no vendor intrinsics
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

if HAS_TRITON:
    @triton.jit
    def _fused_decompress_dequant_kernel(
        data_ptr, offsets_ptr, bitwidths_ptr, scales_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
    ):
        """
        Fused RLE/4b(+outlier)/Huffman decompress + dequant: one program per GROUP_SIZE (32) block.
        No vendor intrinsics, portable tl.load/store.
        Scales are BF16 per block.
        Data: RLE 1B, 4b 17B (1B base+16B), 4b+1 outlier 19B (+1pos+1val), 4b+2 21B, 8b 32B, Huffman 7 not yet
        """
        pid = tl.program_id(axis=0)
        block_idx = pid
        bw = tl.load(bitwidths_ptr + block_idx)
        scale = tl.load(scales_ptr + block_idx).to(tl.float32)
        off = tl.load(offsets_ptr + block_idx)
        offs = tl.arange(0, GROUP_SIZE)
        # 8b
        q8 = tl.load(data_ptr + off + offs, mask=offs < GROUP_SIZE, other=0).to(tl.int16)
        q8 = tl.where(q8 >= 128, q8 - 256, q8)
        # 4b
        base4 = tl.load(data_ptr + off, mask=(bw==4) | (bw==5) | (bw==6), other=0).to(tl.int16)
        base4 = tl.where(base4 >= 128, base4 - 256, base4)
        packed_idx = off + 1 + offs // 2
        packed = tl.load(data_ptr + packed_idx, mask=((bw==4) | (bw==5) | (bw==6)) & (offs < GROUP_SIZE), other=0).to(tl.int16)
        shift = (offs % 2) * 4
        nibble = (packed >> shift) & 0xF
        q4 = base4 + nibble
        # 0
        base0 = tl.load(data_ptr + off, mask=bw==0, other=0).to(tl.int16)
        base0 = tl.where(base0 >= 128, base0 - 256, base0)
        q0 = base0
        # 5: 4b +1 outlier at off+17 pos, off+18 val
        # Need to handle outlier correction for 5 and 6
        # For 5: outlier pos at off+17, val at off+18
        # For 6: pos1 at off+17,val1 at off+18,pos2 at off+19,val2 at off+20
        # We need to check if offs == outlier pos then use outlier val else q4
        # Load outlier info
        pos1 = tl.load(data_ptr + off + 17, mask=bw==5, other=255).to(tl.int32)
        val1 = tl.load(data_ptr + off + 18, mask=bw==5, other=0).to(tl.int16)
        val1 = tl.where(val1 >= 128, val1 - 256, val1)
        pos2_1 = tl.load(data_ptr + off + 17, mask=bw==6, other=255).to(tl.int32)
        val2_1 = tl.load(data_ptr + off + 18, mask=bw==6, other=0).to(tl.int16)
        val2_1 = tl.where(val2_1 >= 128, val2_1 - 256, val2_1)
        pos2_2 = tl.load(data_ptr + off + 19, mask=bw==6, other=255).to(tl.int32)
        val2_2 = tl.load(data_ptr + off + 20, mask=bw==6, other=0).to(tl.int16)
        val2_2 = tl.where(val2_2 >= 128, val2_2 - 256, val2_2)
        # Select q for 5/6
        q5 = tl.where(offs == pos1, val1, q4)
        q6 = tl.where(offs == pos2_1, val2_1, tl.where(offs == pos2_2, val2_2, q4))
        # Final select
        # bw 0,4,5,6,8 (7 Huffman fallback to 8)
        q = tl.where(bw == 0, q0,
            tl.where(bw == 4, q4,
            tl.where(bw == 5, q5,
            tl.where(bw == 6, q6, q8))))
        out_val = (q.to(tl.float32) * scale).to(tl.bfloat16)
        out_offs = block_idx * GROUP_SIZE + offs
        mask = out_offs < n_elements
        tl.store(out_ptr + out_offs, out_val, mask=mask)

    def dequantize_fused_gpu(data: bytes, offsets: torch.Tensor, bitwidths: torch.Tensor, scales: torch.Tensor, orig_shape, group_size=32, out_buffer=None):
        """GPU fused decompress+dequant for high BW. Uses Triton kernel, falls back to CPU if needed."""
        numel = 1
        for d in orig_shape:
            numel *= d
        device = scales.device
        if out_buffer is None:
            out_buffer = torch.empty(orig_shape, dtype=torch.bfloat16, device=device)
        # Fast path: if all 8b (no compression) or fallback, just plain dequant (no decompress overhead)
        # Check if bitwidths all 8 (common for diverse)
        try:
            # Use CPU bitwidths check (small tensor, cheap)
            if torch.all(bitwidths == 8):
                # Data is raw q, no decompress needed
                if isinstance(data, (bytes, bytearray)):
                    # data is raw q bytes, convert to tensor
                    import numpy as np
                    q = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy().view(np.int8)).to(device)
                    # Trim to numel if padded
                    if q.numel() > numel:
                        q = q[:numel]
                elif isinstance(data, torch.Tensor):
                    q = data.to(device)
                    if q.numel() > numel:
                        q = q[:numel]
                else:
                    raise TypeError
                from .codec import dequantize_int8_g32
                return dequantize_int8_g32(q, scales, orig_shape, group_size, out_buffer)
        except Exception:
            pass
        # Otherwise need decompress
        try:
            if isinstance(data, (bytes, bytearray)):
                data_np = np.frombuffer(data, dtype=np.uint8)
                data_gpu = torch.from_numpy(data_np.copy()).to(device)
            elif isinstance(data, torch.Tensor):
                data_gpu = data.to(device)
            else:
                raise TypeError("data must be bytes or tensor")
            offsets_gpu = offsets.to(device)
            bitwidths_gpu = bitwidths.to(device)
            scales_gpu = scales.to(device)
            M = (numel + group_size -1)//group_size
            grid = (M,)
            _fused_decompress_dequant_kernel[grid](
                data_gpu, offsets_gpu, bitwidths_gpu, scales_gpu, out_buffer,
                numel,
                BLOCK_SIZE=group_size, GROUP_SIZE=group_size
            )
            return out_buffer
        except Exception as e:
            q = decompress_int8_blocks(data if isinstance(data, (bytes, bytearray)) else data.cpu().numpy().tobytes() if isinstance(data, torch.Tensor) else data, offsets, bitwidths, group_size, numel)
            q = q.to(device)
            from .codec import dequantize_int8_g32
            return dequantize_int8_g32(q, scales, orig_shape, group_size, out_buffer)
else:
    def dequantize_fused_gpu(*args, **kwargs):
        raise RuntimeError("Triton not available for fused GPU decompress")

def bench_ratio(q: torch.Tensor, group_size=32):
    data, offs, bws, M = compress_int8_blocks(q, group_size)
    orig = q.numel() * 1
    comp_data = len(data)
    # Real overhead: offsets 4B + bitwidth 1B per block, but if all 8b we fallback to raw (no overhead)
    comp_total = comp_data + offs.numel()*4 + bws.numel()*1
    # Fallback to raw if expansion
    if comp_total > orig:
        # Use raw
        comp_data = orig
        comp_total = orig
        ratio_ideal = 1.0
        ratio_real = 1.0
        cnt_rle = cnt_4b = cnt_8b = 0
        # Recompute counts for reporting (still from original)
        cnt_rle = (bws==0).sum().item()
        cnt_4b = (bws==4).sum().item()
        cnt_8b = (bws==8).sum().item()
        return {
            'orig': orig,
            'comp_data': comp_data,
            'comp_total': comp_total,
            'ratio_ideal': ratio_ideal,
            'ratio_real': ratio_real,
            'cnt_rle': cnt_rle,
            'cnt_4b': cnt_4b,
            'cnt_8b': cnt_8b,
            'M': M,
            'fallback': True,
        }
    ideal_comp = comp_data
    ratio_ideal = orig / ideal_comp if ideal_comp>0 else 0
    ratio_real = orig / comp_total if comp_total>0 else 0
    cnt_rle = (bws==0).sum().item()
    cnt_4b = (bws==4).sum().item()
    cnt_8b = (bws==8).sum().item()
    return {
        'orig': orig,
        'comp_data': ideal_comp,
        'comp_total': comp_total,
        'ratio_ideal': ratio_ideal,
        'ratio_real': ratio_real,
        'cnt_rle': cnt_rle,
        'cnt_4b': cnt_4b,
        'cnt_8b': cnt_8b,
        'M': M,
        'fallback': False,
    }
