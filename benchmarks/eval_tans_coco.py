"""Does the dictionary win hold across real images? Disjoint COCO eval.

Learns dictionaries on NCAL images, evaluates on a disjoint NEVAL slice,
and reports the FULL v2 section mix (u8 + i8 + ll4 byte-planar + raw meta)
against both raw RGB and the production rANS path — per image, so we see
the spread and the head-to-head win rate, not just an aggregate.

Usage: .venv/bin/python TensorCache/benchmarks/eval_tans_coco.py [ncal] [neval]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tensorcache import codec as C
from tensorcache import rans as R
from tensorcache import tans as T
from tensorcache.tans_dict import collect_block_hists, learn_dictionary, save_tables

DATA = Path("TensorCache/data/coco_val")
DICT_PATH = Path("TensorCache/research/tans_dicts_eval.json")
H = W = 336
K = 8
MODE = "balanced"


def sections(path: Path) -> dict:
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR),
                       dtype=np.uint8)
    t = torch.from_numpy(arr).to(torch.uint8)
    meta, _ = C.quantize_pixel_wavelet_adaptive(t, mode=MODE)
    a = C.sparse_pack_arena(C.sparse_pack_meta(meta))
    ll4 = a["ll4"].numpy().tobytes()
    return {"u8": a["arena_u8"].numpy().tobytes(),
            "i8": a["arena_i8"].numpy().tobytes(),
            "meta": a["meta"].numpy().tobytes(),
            "ll4_lo": ll4[0::2], "ll4_hi": ll4[1::2]}


def learn(dicts_raw: list, kinds):
    out = {}
    for kind in kinds:
        Hh = collect_block_hists([d[kind] for d in dicts_raw], T.B_DEFAULT)
        out[kind] = learn_dictionary(Hh, K=K, seed=0)
    return out


def rans_total(sec: dict) -> int:
    m0, b0 = R.pack_section(sec["u8"])
    m1, b1 = R.pack_section(sec["i8"])
    # v2 ll4 treatment: byte-planar, each plane its own rANS decision
    a, x0 = R.pack_section(sec["ll4_lo"])
    b, x1 = R.pack_section(sec["ll4_hi"])
    return len(b0) + len(b1) + len(x0) + len(x1) + len(sec["meta"])


def tans_total(sec: dict, tabs: dict) -> int:
    total = 0
    for kind in ("u8", "i8", "ll4_lo", "ll4_hi"):
        m, blob = T.pack_section_tans(sec[kind], tables=tabs[kind],
                                      embed_tables=False)
        total += len(blob)
    return total + len(sec["meta"])  # meta stays raw in both paths


def main():
    ncal = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    neval = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    files = sorted(DATA.glob("*.jpg"))
    assert len(files) >= ncal + neval
    cal_f, ev_f = files[:ncal], files[ncal:ncal + neval]
    print(f"cal {ncal} | eval {neval} | mode {MODE} | K={K} | disjoint")

    t0 = time.perf_counter()
    cal, ev = [], []
    for i, f in enumerate(cal_f):
        cal.append(sections(f))
        if (i + 1) % 100 == 0:
            print(f"  cal arenas {i+1}/{ncal} ({time.perf_counter()-t0:.0f}s)")
    for i, f in enumerate(ev_f):
        ev.append(sections(f))
    print(f"built {ncal+neval} arenas in {time.perf_counter()-t0:.0f}s")

    t0 = time.perf_counter()
    tabs = learn(cal, ("u8", "i8", "ll4_lo", "ll4_hi"))
    save_tables({"R": 12, "K": K, "kinds": tabs}, DICT_PATH)
    print(f"learned dicts in {time.perf_counter()-t0:.1f}s "
          f"({DICT_PATH.stat().st_size} B, once per cache)")

    raw_rgb = H * W * 3
    # Dictionary cost is paid ONCE per cache; charge a per-image share so the
    # comparison is honest at this cache size (at 5000 images it is ~0.04%).
    dict_bytes = DICT_PATH.stat().st_size  # upper bound (JSON, all 256 freqs)
    share = dict_bytes / len(ev)
    print(f"dictionary: {dict_bytes} B once, charged {share:.0f} B/img at n={len(ev)}")
    rows = []
    for sec in ev:
        rr = rans_total(sec)
        tt = tans_total(sec, tabs) + share
        rows.append((rr, tt))
    rr = np.array([r[0] for r in rows], dtype=np.float64)
    tt = np.array([r[1] for r in rows], dtype=np.float64)
    ratio_rans = raw_rgb / rr
    ratio_tans = raw_rgb / tt
    win = (tt < rr)
    print(f"\n== per-image stored bytes vs raw RGB ({raw_rgb} B), n={len(ev)} ==")
    print(f"rANS          ratio  mean {ratio_rans.mean():.3f}x  "
          f"p5 {np.percentile(ratio_rans,5):.3f}  p50 {np.percentile(ratio_rans,50):.3f}  "
          f"p95 {np.percentile(ratio_rans,95):.3f}  min {ratio_rans.min():.3f}")
    print(f"dict-tANS     ratio  mean {ratio_tans.mean():.3f}x  "
          f"p5 {np.percentile(ratio_tans,5):.3f}  p50 {np.percentile(ratio_tans,50):.3f}  "
          f"p95 {np.percentile(ratio_tans,95):.3f}  min {ratio_tans.min():.3f}")
    delta = (rr - tt) / rr * 100
    print(f"\ntANS vs rANS: mean {delta.mean():+.2f}%  "
          f"p5 {np.percentile(delta,5):+.2f}%  p50 {np.percentile(delta,50):+.2f}%  "
          f"p95 {np.percentile(delta,95):+.2f}%  (positive = tANS smaller)")
    print(f"images where dict-tANS beats rANS: {win.sum()}/{len(ev)} "
          f"({win.mean()*100:.1f}%)")
    print(f"worst regression: {delta.min():+.2f}%   best: {delta.max():+.2f}%")
    # aggregate (byte-weighted, what a cache actually stores)
    agg_rans = raw_rgb * len(ev) / rr.sum()
    agg_tans = raw_rgb * len(ev) / tt.sum()
    print(f"\naggregate (byte-weighted): rANS {agg_rans:.3f}x  dict-tANS {agg_tans:.3f}x  "
          f"({(rr.sum()-tt.sum())/rr.sum()*100:+.2f}%)")
    # how much of the win is the sections rANS was bad at
    for kind in ("u8", "i8", "ll4_lo", "ll4_hi"):
        gr = sum(len(R.pack_section(s[kind])[1]) for s in ev)
        gt = sum(len(T.pack_section_tans(s[kind], tables=tabs[kind],
                                         embed_tables=False)[1]) for s in ev)
        raw = sum(len(s[kind]) for s in ev)
        print(f"  {kind:7} raw {raw/1024:8.1f}KB  rANS {raw/gr:.3f}x  "
              f"tANS {raw/gt:.3f}x  ({(gr-gt)/gr*100:+.1f}% bytes)")


if __name__ == "__main__":
    main()
