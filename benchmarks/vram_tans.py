"""Peak VRAM: holding a dataset compressed (tANS) vs decoded (arenas).

The VRAM-resident story: if entropy runs on the GPU, the compressed blobs
can live on-device, so the whole dataset fits in a fraction of the space
the decoded arenas would take. Measures real allocations on the device and
extrapolates how many images fit in a VRAM budget.

Usage: .venv/bin/python TensorCache/benchmarks/vram_tans.py [nimg] [budget_mb]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tensorcache import codec as C
from tensorcache import tans as T
from tensorcache.tans_dict import (
    collect_block_hists, dictionary_to_meta, learn_dictionary,
)

DATA = Path("TensorCache/data/coco_val")
H = W = 336
DEV = "cuda:0"
B = 256
KINDS = ("u8", "i8", "ll4_lo", "ll4_hi")


def load_images(files):
    arrs = []
    for f in files:
        with Image.open(f) as im:
            arrs.append(np.array(im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR),
                                 dtype=np.uint8))
    return torch.from_numpy(np.stack(arrs)).to(torch.uint8)


def build_arenas(imgs):
    return C.sparse_pack_arena_batched(
        C.sparse_pack_meta_batched(
            C.quantize_pixel_wavelet_adaptive_batched(imgs, mode="balanced")))


def section_bytes(a):
    l4 = a["ll4"].numpy().astype(np.int16).tobytes()
    return {"u8": a["arena_u8"].numpy().tobytes(), "i8": a["arena_i8"].numpy().tobytes(),
            "ll4_lo": l4[0::2], "ll4_hi": l4[1::2]}


def main():
    nimg = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    budget_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    torch.cuda.set_device(DEV)
    files = sorted(DATA.glob("*.jpg"))
    print(f"VRAM: {nimg} COCO images, budget {budget_mb} MB (4050 = 6144 MB)")

    cal = build_arenas(load_images(files[400:520]))
    tabs = {k: learn_dictionary(collect_block_hists(
        [section_bytes(a)[k] for a in cal], B), K=8, seed=0) for k in KINDS}
    meta = dictionary_to_meta(tabs, R=12, K=8, B=B)
    meta_bytes = len(json.dumps(meta))

    arenas = build_arenas(load_images(files[:nimg]))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # (a) decoded arenas resident: these are what a GPU-resident *decoded*
    # cache would have to hold (u8 + i8 + ll4 + meta).
    base = torch.cuda.memory_allocated()
    keep_arenas = []
    for a in arenas:
        keep_arenas.append({k: v.to(DEV) if torch.is_tensor(v) else v
                            for k, v in a.items()})
    torch.cuda.synchronize()
    arena_vram = torch.cuda.memory_allocated() - base
    arena_img = arena_vram / nimg

    # (b) compressed resident: encode once, keep the blobs + tables on device.
    enc = {k: [] for k in KINDS}
    for a in arenas:
        sec = section_bytes(a)
        for k in KINDS:
            enc[k].append(T.encode(sec[k], block=B, tables=tabs[k],
                                   embed_tables=False))
    del keep_arenas
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    keep_blobs = []
    for k in KINDS:
        for b in enc[k]:
            keep_blobs.append(torch.frombuffer(b, dtype=torch.uint8).to(DEV))
    table_vram = sum(1 << 12 for _ in tabs)  # decode tables resident
    torch.cuda.synchronize()
    blob_vram = torch.cuda.memory_allocated() - base
    comp_img = blob_vram / nimg

    n_arena = int(budget_mb * 1e6 / arena_img)
    n_comp = int(budget_mb * 1e6 / comp_img)
    print(f"\nper-image resident VRAM:")
    print(f"  decoded arenas   : {arena_img/1024:7.1f} KB/img")
    print(f"  tANS blobs       : {comp_img/1024:7.1f} KB/img "
          f"({comp_img/arena_img-1:+.0%} vs decoded)")
    print(f"  + dictionary     : {meta_bytes/1024:7.1f} KB once "
          f"(negligible per image at scale)")
    print(f"\nimages that fit in a {budget_mb} MB budget:")
    print(f"  decoded arenas   : {n_arena:>7,}  (~{n_arena/nimg:.0f}x this eval set)")
    print(f"  tANS blobs       : {n_comp:>7,}  ({n_comp/max(n_arena,1):.2f}x more)")
    print(f"  COCO-5000 decoded: {5000*arena_img/1e6:.0f} MB "
          f"vs compressed {5000*comp_img/1e6:.0f} MB "
          f"(saves {(5000*arena_img-5000*comp_img)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
