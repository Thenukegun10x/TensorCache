"""Shannon bound vs what the coders actually achieve, per XS section kind.

For each section (u8, i8, ll4 planes) on real COCO arenas:
  - global order-0 Shannon entropy  H = -sum p log2 p   (bits/symbol)
  - per-block entropy (B=256)       H(X|block)          (the adaptive bound)
  - rANS stored bytes (one table per section = global order-0)
  - dict-tANS stored bytes (8 shared tables, per-block selection)
and reports coding efficiency (ideal / actual).

Note the per-block figure is the *empirical* entropy of a small sample, so it
is biased low and is an optimistic floor, not an achievable rate.

Usage: .venv/bin/python TensorCache/benchmarks/entropy_vs_tans.py [ncal] [neval]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tensorcache import codec as C
from tensorcache import rans as R
from tensorcache import tans as T
from tensorcache.tans_dict import collect_block_hists, learn_dictionary

DATA = Path("TensorCache/data/coco_val")
H = W = 336
B = 256
KINDS = ("u8", "i8", "ll4_lo", "ll4_hi")


def sections(path: Path) -> dict:
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR),
                       dtype=np.uint8)
    t = torch.from_numpy(arr).to(torch.uint8)
    meta, _ = C.quantize_pixel_wavelet_adaptive(t, mode="balanced")
    a = C.sparse_pack_arena(C.sparse_pack_meta(meta))
    l4 = a["ll4"].numpy().astype(np.int16).tobytes()
    return {"u8": a["arena_u8"].numpy().tobytes(),
            "i8": a["arena_i8"].numpy().tobytes(),
            "ll4_lo": l4[0::2], "ll4_hi": l4[1::2]}


def shannon_bits(data: bytes) -> float:
    a = np.frombuffer(data, dtype=np.uint8)
    if a.size == 0:
        return 0.0
    c = np.bincount(a, minlength=256).astype(np.float64)
    p = c[c > 0] / a.size
    return float(-(p * np.log2(p)).sum())


def block_entropy_bits(data: bytes, block: int) -> float:
    """Sum over blocks of n_b * H(block) / n  (empirical, biased low)."""
    a = np.frombuffer(data, dtype=np.uint8)
    n = a.size
    if n == 0:
        return 0.0
    total = 0.0
    for s in range(0, n, block):
        blk = a[s:s + block]
        c = np.bincount(blk, minlength=256).astype(np.float64)
        p = c[c > 0] / blk.size
        total += blk.size * float(-(p * np.log2(p)).sum())
    return total / n


def main():
    ncal = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    neval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    files = sorted(DATA.glob("*.jpg"))
    cal = [sections(f) for f in files[400:400 + ncal]]
    ev = [sections(f) for f in files[:neval]]
    tabs = {k: learn_dictionary(collect_block_hists([s[k] for s in cal], B),
                                K=8, seed=0) for k in KINDS}
    print(f"COCO balanced 336, dict from {ncal} disjoint imgs, eval {neval} imgs")
    print(f"{'kind':8} {'n sym':>9} {'H bits':>7} {'H|blk':>7} {'rANS b':>7} "
          f"{'tANS b':>7} {'rANS eff':>9} {'tANS eff':>9} {'tANS vs H':>10}")

    tot = {k: 0.0 for k in ("raw", "h", "hb", "rans", "tans")}
    for kind in KINDS:
        blob = b"".join(s[kind] for s in ev)
        n = len(blob)
        h = shannon_bits(blob)
        hb = block_entropy_bits(blob, B)
        rans_b = sum(len(R.encode(s[kind])) for s in ev)
        tans_b = sum(len(T.encode(s[kind], block=B, tables=tabs[kind],
                                  embed_tables=False)) for s in ev)
        # stored bits per symbol (excluding the fixed cache table, amortized)
        rp = rans_b * 8 / n
        tp = tans_b * 8 / n
        print(f"{kind:8} {n:>9} {h:>7.3f} {hb:>7.3f} {rp:>7.3f} {tp:>7.3f} "
              f"{h/rp*100:>8.1f}% {h/tp*100:>8.1f}% {(h-tp):>+9.3f}")
        for k in tot:
            tot[k] += {"raw": n, "h": h * n / 8, "hb": hb * n / 8,
                       "rans": rans_b, "tans": tans_b}[k]
    print()
    print(f"{'TOTAL':8} raw {tot['raw']/1024:9.1f} KB")
    print(f"{'':8} Shannon ideal  {tot['h']/1024:9.1f} KB  "
          f"({tot['h']*8/tot['raw']:.3f} bits/sym)")
    print(f"{'':8} block-cond H   {tot['hb']/1024:9.1f} KB  "
          f"({tot['hb']*8/tot['raw']:.3f} bits/sym, optimistic)")
    print(f"{'':8} rANS actual    {tot['rans']/1024:9.1f} KB  "
          f"({tot['rans']*8/tot['raw']:.3f} bits/sym, "
          f"{tot['h']/tot['rans']*100:.1f}% of Shannon)")
    print(f"{'':8} tANS actual    {tot['tans']/1024:9.1f} KB  "
          f"({tot['tans']*8/tot['raw']:.3f} bits/sym, "
          f"{tot['h']/tot['tans']*100:.1f}% of Shannon)")
    print(f"\ntANS redundancy over Shannon: "
          f"{(tot['tans']-tot['h'])/tot['h']*100:+.2f}% "
          f"({(tot['tans']-tot['h'])/1024:.1f} KB on this eval set)")
    print(f"tANS closes {100*(tot['rans']-tot['tans'])/(tot['rans']-tot['h']):.0f}% "
          f"of the rANS-over-Shannon gap")


if __name__ == "__main__":
    main()
