"""Research experiment: true data-parallel entropy coding for GPUs (Triton).

rANS/tANS are serial *within* a stream: symbol[n+1] needs state[n]. This
experiment instead codes `i8` vals with Rice(k) at block granularity:

- Encode of every symbol is independent given k (shifts/masks only, no
  division, no tables): q = u >> k, r = u & ((1<<k)-1), code = q ones +
  stop-zero + k remainder bits. Block output offsets come from one
  `torch.cumsum` over per-block bit sums (parallel prefix sum).
- One Triton program per block (default 128 symbols): packs its symbols
  into registers with scalar bit ops, flushes 32-bit words with
  `atomic_or` (only shared edge words collide). Thousands of programs run
  concurrently; no cross-block dependency.
- u8 mask sections are bitmap-like (rANS only got 1.27x) and are left raw
  in v1 — this experiment targets the i8 vals section, which dominates
  bytes (~120KB vs ~23KB) and where rANS got ~2.05x.

Layout (per section): [u32 n][u8 k][u8 logB][u32 nblocks][nblocks x u32 len]
  + concatenated block payloads. Decode needs only the header + payload.

Not wired into PixelCache paths; compares ratio + throughput vs rans on
real XS arenas. Run: `.venv/bin/python TensorCache/benchmarks/exp_triton_rice.py`
"""

from __future__ import annotations

import time

import numpy as np
import torch
import triton
import triton.language as tl


def zigzag_encode(v: torch.Tensor) -> torch.Tensor:
    return ((v << 1) ^ (v >> 7)).to(torch.int32) & 0xFFFF


def zigzag_decode(u: torch.Tensor) -> torch.Tensor:
    return (((u >> 1).to(torch.int32) ^ -((u & 1).to(torch.int32))).to(torch.int8))


def rice_len(u: torch.Tensor, k: int) -> torch.Tensor:
    return ((u >> k) + 1 + k).to(torch.int32)


def pick_k(u: torch.Tensor) -> int:
    """Cheapest k in 0..4 by exact total-bit cost (one vectorized pass)."""
    best, best_bits = 0, None
    for k in range(5):
        bits = int(rice_len(u, k).sum().item())
        if best_bits is None or bits < best_bits:
            best, best_bits = k, bits
    return best


@triton.jit
def _rice_pack_kernel(
    u_ptr, off_ptr, out_ptr,
    K: tl.constexpr, BLOCK: tl.constexpr,
):
    """One program per block: pack BLOCK zigzag symbols at bit offset off.

    out_ptr is int32 words; shared edge words merge via atomic_or.
    All loops are scalar while-loops (unary length is data-dependent).
    """
    pid = tl.program_id(0)
    base = pid * BLOCK
    off = tl.load(off_ptr + pid).to(tl.int32)
    w = off // 32
    b = off % 32
    cur = 0
    for i in range(BLOCK):
        u = tl.load(u_ptr + base + i).to(tl.int32)
        q = u >> K
        r = u & ((1 << K) - 1)
        # unary run: q ones
        rem = q
        while rem > 0:
            space = 32 - b
            run = tl.minimum(rem, space)
            mask = ((tl.full((), 1, tl.int64) << run.to(tl.int64)) - 1).to(tl.int32)
            cur |= mask << b
            b += run
            rem -= run
            if b == 32:
                tl.atomic_or(out_ptr + w, cur)
                w += 1
                b = 0
                cur = 0
        # stop zero: single 0 bit
        b += 1
        if b == 32:
            tl.atomic_or(out_ptr + w, cur)
            w += 1
            b = 0
            cur = 0
        # remainder, LSB first, in chunks
        remb = K
        rv = r
        while remb > 0:
            space = 32 - b
            take = tl.minimum(remb, space)
            mask = ((tl.full((), 1, tl.int64) << take.to(tl.int64)) - 1).to(tl.int32)
            cur |= (rv & mask) << b
            rv >>= take
            b += take
            remb -= take
            if b == 32:
                tl.atomic_or(out_ptr + w, cur)
                w += 1
                b = 0
                cur = 0
    tl.atomic_or(out_ptr + w, cur)


@triton.jit
def _rice_unpack_kernel(
    out_ptr, off_ptr, len_ptr, u_ptr,
    K: tl.constexpr, BLOCK: tl.constexpr,
):
    """One program per block: sequential bit-reader, writes its symbols."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    pos = tl.load(off_ptr + pid).to(tl.int32)
    n = tl.load(len_ptr + pid).to(tl.int32)
    for i in range(BLOCK):
        if i < n:
            q = 0
            word = tl.load(out_ptr + (pos // 32))
            bit = (word >> (pos % 32)) & 1
            pos += 1
            while bit == 1:
                q += 1
                word = tl.load(out_ptr + (pos // 32))
                bit = (word >> (pos % 32)) & 1
                pos += 1
            r = 0
            rb = 0
            while rb < K:
                word = tl.load(out_ptr + (pos // 32))
                bit = (word >> (pos % 32)) & 1
                pos += 1
                r |= bit << rb
                rb += 1
            tl.store(u_ptr + base + i, (q << K) | r)


def rice_encode_triton(vals_i8: torch.Tensor, block: int = 128):
    """i8 CUDA tensor -> (blob words CUDA, header dict). Block-parallel."""
    assert vals_i8.is_cuda and vals_i8.dtype == torch.int8
    n = vals_i8.numel()
    nb = (n + block - 1) // block
    pad = nb * block - n
    if pad:
        vals_i8 = torch.cat([vals_i8.reshape(-1),
                             torch.zeros(pad, dtype=torch.int8, device=vals_i8.device)])
    else:
        vals_i8 = vals_i8.reshape(-1).contiguous()
    u = zigzag_encode(vals_i8.to(torch.int32)).contiguous()
    k = pick_k(u)
    lens = rice_len(u, k).to(torch.int64)  # bits per symbol (padded incl.)
    blk_id = torch.arange(nb * block, device=u.device) // block
    blk_bits = torch.zeros(nb, dtype=torch.int64, device=u.device).scatter_add_(0, blk_id, lens)
    blk_off = torch.cat([torch.zeros(1, dtype=torch.int64, device=u.device),
                         blk_bits.cumsum(0)[:-1]]).to(torch.int32)
    total_words = int((blk_bits.sum().item() + 31) // 32)
    out = torch.zeros(total_words + 1, dtype=torch.int32, device=u.device)
    _rice_pack_kernel[(nb,)](u, blk_off, out, K=k, BLOCK=block)
    torch.cuda.synchronize()
    # real per-block symbol counts (last block may be short)
    counts = torch.full((nb,), block, dtype=torch.int32, device=u.device)
    if pad:
        counts[-1] = block - pad
    return out[:total_words], {
        "n": n, "k": k, "block": block, "nblocks": nb,
        "blk_off": blk_off.cpu(),
        "blk_len": counts.cpu(),
    }


def rice_decode_triton(blob: torch.Tensor, hdr: dict) -> torch.Tensor:
    assert blob.is_cuda
    nb, block, k, n = hdr["nblocks"], hdr["block"], hdr["k"], hdr["n"]
    total = nb * block
    out = torch.empty(total, dtype=torch.int32, device=blob.device)
    _rice_unpack_kernel[(nb,)](
        blob, hdr["blk_off"].to(blob.device), hdr["blk_len"].to(blob.device),
        out, K=k, BLOCK=block)
    torch.cuda.synchronize()
    return zigzag_decode(out[:n])


def main():
    from tensorcache import codec as C
    from tensorcache import rans as R

    torch.manual_seed(0)
    dev = "cuda:0"
    print("== parallel-Rice vs rANS on real XS i8 sections ==")
    for scale, name in [(8, "smooth"), (30, "mid"), (80, "textured")]:
        img = (torch.randn(336, 336, 3) * scale + 128).clamp(0, 255).to(torch.uint8)
        meta, _ = C.quantize_pixel_wavelet_adaptive(img, mode="balanced")
        arena = C.sparse_pack_arena(C.sparse_pack_meta(meta))
        v = arena["arena_i8"].reshape(-1).clone()
        raw = v.numpy().tobytes()

        # rANS baseline (CPU numba, warmed)
        bb = R.encode(raw)
        for _ in range(5):
            R.decode(bb)
        t0 = time.perf_counter()
        for _ in range(20):
            R.decode(bb)
        t1 = time.perf_counter()
        rans_ms = (t1 - t0) / 20 * 1e3

        # Triton Rice (GPU)
        vc = v.to(dev)
        blob, hdr = rice_encode_triton(vc)  # warm (includes JIT)
        blob, hdr = rice_encode_triton(vc)
        back = rice_decode_triton(blob, hdr)
        assert torch.equal(back.cpu(), v.cpu()), f"roundtrip mismatch on {name}"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            rice_decode_triton(blob, hdr)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rice_ms = (t1 - t0) / 20 * 1e3

        # stored size incl. tiny header (~16B + 4B/block)
        stored = len(blob.cpu().numpy().tobytes()) + 16 + 4 * hdr["nblocks"]
        print(f"{name}: n={len(raw)} rANS {len(raw)/len(bb):.2f}x/{rans_ms:.2f}ms "
              f"vs Rice(k={hdr['k']}) {len(raw)/stored:.2f}x/{rice_ms:.3f}ms "
              f"({rans_ms/max(rice_ms,1e-9):.1f}x decode speedup)")


if __name__ == "__main__":
    main()
