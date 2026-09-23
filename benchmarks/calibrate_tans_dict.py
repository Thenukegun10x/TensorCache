"""Calibrate tANS dictionaries on COCO-336 arenas, evaluate on held-out images.

Split: calibration = first NCAL images of data/coco_val (sorted),
eval = disjoint later slice. Learns {u8, i8} x K tables, saves them to
research/tans_dicts.json, then reports rANS vs single-table tANS vs
dictionary-tANS (referenced mode: selectors in-section, tables amortized
once per cache in meta) per section kind.

Also reports: in-sample (cal) vs eval cross-entropy gap, K sweep, and
per-block vs per-section table selection.

Usage: .venv/bin/python TensorCache/benchmarks/calibrate_tans_dict.py [ncal] [neval]
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
from tensorcache.tans_dict import (
    codelengths,
    collect_block_hists,
    learn_dictionary,
    save_tables,
)

DATA = Path("TensorCache/data/coco_val")
DICT_PATH = Path("TensorCache/research/tans_dicts.json")
H = W = 336


def arena_sections(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
    t = torch.from_numpy(arr).to(torch.uint8)
    meta, _ = C.quantize_pixel_wavelet_adaptive(t, mode="balanced")
    arena = C.sparse_pack_arena(C.sparse_pack_meta(meta))
    return {
        "u8": arena["arena_u8"].numpy().tobytes(),
        "i8": arena["arena_i8"].numpy().tobytes(),
    }


def xent_gap(sections: list, tables: list, R: int = 12) -> float:
    """Mean per-symbol cross-entropy gap: best-dict-table vs own histogram.

    Positive = dictionary costs this many extra bits/symbol vs a table
    learned on the data itself (generalization + quantization gap).
    """
    L = np.stack([codelengths(f, R) for f in tables], axis=1)
    tot_bits = tot_syms = 0.0
    for raw in sections:
        h = np.bincount(np.frombuffer(raw, dtype=np.uint8),
                        minlength=256).astype(np.float64)
        own = R - np.log2(np.where(h > 0, h / h.sum() * (1 << R), 1))
        own_bits = float((h * own).sum())
        best_bits = float(np.min(h @ L))
        tot_bits += best_bits - own_bits
        tot_syms += h.sum()
    return tot_bits / tot_syms


def main():
    ncal = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    neval = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    files = sorted(DATA.glob("*.jpg"))
    assert len(files) >= ncal + neval, "not enough COCO images"
    cal_files, eval_files = files[:ncal], files[ncal:ncal + neval]
    print(f"calibration: {len(cal_files)} imgs, eval: {len(eval_files)} imgs (disjoint)")

    t0 = time.perf_counter()
    cal = {"u8": [], "i8": []}
    for f in cal_files:
        for kind, raw in arena_sections(f).items():
            cal[kind].append(raw)
    eval_secs = {"u8": [], "i8": []}
    for f in eval_files:
        for kind, raw in arena_sections(f).items():
            eval_secs[kind].append(raw)
    print(f"arena build: {time.perf_counter()-t0:.1f}s")

    K = 8
    tables = {"R": 12, "K": K, "kinds": {}}
    for kind in ("u8", "i8"):
        Hh = collect_block_hists(cal[kind], T.B_DEFAULT)
        print(f"learn {kind}: {Hh.shape[0]} blocks")
        tables["kinds"][kind] = learn_dictionary(Hh, K=K, seed=0)
    save_tables(tables, DICT_PATH)
    dict_json = len(DICT_PATH.read_bytes())
    print(f"saved -> {DICT_PATH} ({dict_json} bytes, amortized over cache)")

    for kind in ("u8", "i8"):
        tabs = tables["kinds"][kind]
        print(f"== {kind}: xent gap cal {xent_gap(cal[kind][:12], tabs):+.3f} "
              f"vs eval {xent_gap(eval_secs[kind], tabs):+.3f} bits/sym ==")
        for K2 in (4, 16):
            t2 = learn_dictionary(collect_block_hists(cal[kind], T.B_DEFAULT),
                                  K=K2, seed=0)
            print(f"   K={K2}: eval gap {xent_gap(eval_secs[kind], t2):+.3f} bits/sym")

    # size eval: referenced mode + amortized table share (COCO-5000 scale)
    n_cache_sections = 5000  # tables live once per cache in meta
    print(f"{'kind':4} {'raw':>10} {'rANS':>10} {'tANS-1':>10} "
          f"{'sec-sel':>10} {'blk-sel':>10}")
    grand = {"raw": 0, "rans": 0, "single": 0, "sec": 0, "blk": 0}
    for kind in ("u8", "i8"):
        tabs = tables["kinds"][kind]
        share = dict_json / 2 / n_cache_sections  # half the JSON per kind, ~5000 sections
        t = {"raw": 0, "rans": 0, "single": 0, "sec": 0, "blk": 0}
        for raw in eval_secs[kind]:
            rb = R.encode(raw)
            sb = T.encode(raw)
            # per-section selection: best whole-section table, referenced
            sec_best = min(len(T.encode(raw, tables=[f], embed_tables=False))
                           for f in tabs) + 1  # +1 selector byte (xs row)
            # per-block selection, referenced
            blk = len(T.encode(raw, tables=tabs, embed_tables=False))
            assert R.decode(rb) == raw == T.decode(sb)
            assert T.decode(T.encode(raw, tables=tabs, embed_tables=False),
                            tables=tabs) == raw
            t["raw"] += len(raw)
            t["rans"] += len(rb)
            t["single"] += len(sb)
            t["sec"] += sec_best + share
            t["blk"] += blk + share
        print(f"{kind:4} {t['raw']:>10} {t['raw']/t['rans']:>9.3f}x "
              f"{t['raw']/t['single']:>9.3f}x {t['raw']/t['sec']:>9.3f}x "
              f"{t['raw']/t['blk']:>9.3f}x")
        for k in grand:
            grand[k] += t[k]
    print(f"total: rANS {grand['raw']/grand['rans']:.3f}x, "
          f"tANS-1 {grand['raw']/grand['single']:.3f}x, "
          f"sec-sel {grand['raw']/grand['sec']:.3f}x, "
          f"blk-sel {grand['raw']/grand['blk']:.3f}x")
    print(f"blk-sel vs rANS: {(grand['rans']-grand['blk'])/grand['rans']*100:+.2f}% "
          f"(+ = dict smaller)")


if __name__ == "__main__":
    main()
