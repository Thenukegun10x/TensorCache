"""M1 speed: Triton block-tANS decode vs numba CPU rANS/tANS (loader-shaped).

The timed loop contains ONLY kernel launches (meta tensors and payload are
assembled before timing), so the numbers are decode throughput, not Python.
Sweeps block size B (parallelism vs ratio) since block-tANS decode is
serial-per-block and needs many resident streams to fill the GPU.

Usage: .venv/bin/python TensorCache/benchmarks/bench_tans_gpu.py [nimg] [reps]
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
from tensorcache.tans_dict import load_tables

DATA = Path("TensorCache/data/coco_val")
DICT_PATH = Path("TensorCache/research/tans_dicts.json")
H = W = 336
DEV = "cuda:0"


def arena_i8(path: Path) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
    t = torch.from_numpy(arr).to(torch.uint8)
    meta, _ = C.quantize_pixel_wavelet_adaptive(t, mode="balanced")
    return C.sparse_pack_arena(C.sparse_pack_meta(meta))["arena_i8"].numpy().tobytes()


def build_gpu_batch(raws, blobs, tabs, B, NS):
    """Precompute per-table-group meta tensors + concatenated payload (GPU).

    Returns (payload_cat, groups, out, total, nblocks_used) where groups is
    a list of (meta_cuda [M,5] int32, sym, nb, base) per selector value that
    occurs. Static across timing reps.
    """
    parts = [T.split_section(b, tables=tabs) for b in blobs]
    codecs = [T._Codec(f, NS.bit_length() - 1) for f in tabs]
    total = sum(len(r) for r in raws)
    out = torch.empty(total, dtype=torch.uint8, device=DEV)
    pay_parts, rows_by_sel, poff = [], {}, 0
    ooff = 0
    for sp, raw in zip(parts, raws):
        nb, Bc = sp["nblocks"], sp["B"]
        sels = sp["sels"]
        bl = np.array([(x + 7) // 8 for x in sp["bit_lens"]])
        goff = np.zeros(nb, dtype=np.int64)
        acc = 0
        for g in np.argsort(np.asarray(sels), kind="stable"):
            goff[int(g)] = acc
            acc += bl[int(g)]
        for b in range(nb):
            n_b = min(Bc, sp["n"] - b * Bc)
            rows_by_sel.setdefault(int(sels[b]), []).append(
                [poff + int(goff[b]), sp["bit_lens"][b], sp["finals"][b],
                 n_b, ooff])
            ooff += n_b
        pay_parts.append(np.frombuffer(sp["payload"], dtype=np.uint8))
        poff += len(sp["payload"])
    payload_cat = torch.from_numpy(np.concatenate(pay_parts)).to(DEV)
    groups = []
    for g, rows in sorted(rows_by_sel.items()):
        meta = torch.tensor(np.array(rows, dtype=np.int64), dtype=torch.int32,
                            device=DEV)
        c = codecs[g]
        groups.append((
            meta,
            torch.from_numpy(c.dsym_np).to(DEV),
            torch.from_numpy(c.dnb_np).to(DEV),
            torch.from_numpy(c.dbase_np.astype(np.int32)).to(DEV),
        ))
    return payload_cat, groups, out, total, sum(len(r) for r in rows_by_sel.values())


def main():
    nimg = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    assert T.HAS_TRITON and T.HAS_NUMBA, "need Triton + numba"
    assert torch.cuda.is_available()
    torch.cuda.set_device(DEV)
    files = sorted(DATA.glob("*.jpg"))[200:200 + nimg]
    raws = [arena_i8(f) for f in files]
    tabs = load_tables(DICT_PATH)["kinds"]["i8"]
    NS = 1 << 12
    total_out = sum(len(r) for r in raws)
    print(f"sections: {nimg} COCO i8, {total_out/1e6:.2f} MB decoded output")

    # CPU baselines (1 core)
    T.encode(raws[0][:4096], tables=tabs, embed_tables=False)
    rb = [R.encode(r) for r in raws]
    tb = [T.encode(r, tables=tabs, embed_tables=False) for r in raws]
    t0 = time.perf_counter()
    for _ in range(5):
        for b in rb:
            R.decode(b)
    tr = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        for b in tb:
            T.decode(b, tables=tabs)
    tt = (time.perf_counter() - t0) / 5
    print(f"cpu-numba rANS: {tr*1e3:7.2f} ms/batch  {total_out/tr/1e9:5.2f} GB/s (1 core)")
    print(f"cpu-numba tANS: {tt*1e3:7.2f} ms/batch  {total_out/tt/1e9:5.2f} GB/s (1 core)")

    print(f"{'B':>7} {'blocks':>8} {'stored KB':>10} {'ratio':>7} "
          f"{'rANS r':>7} {'gpu ms':>8} {'GB/s':>7} {'vs rANS':>8}")
    ratio_raw = sum(len(r) for r in raws)
    ratio_rans = sum(len(b) for b in rb)
    rans_ratio = ratio_raw / ratio_rans
    print(f"{'rANS':>7} {sum(len(r) for r in raws)*0:>8} "
          f"{ratio_rans/1024:>10.1f} {rans_ratio:>6.3f}x {'-':>7} "
          f"{tr*1e3:>8.3f} {total_out/tr/1e9:>6.2f} {'1.00x':>8}   (cpu 1 core)")
    for B in (64, 128, 256, 512, 1024, 2048):
        blobs = [T.encode(r, block=B, tables=tabs, embed_tables=False) for r in raws]
        stored = sum(len(b) for b in blobs)
        payload_cat, groups, out, total, nblocks = build_gpu_batch(
            raws, blobs, tabs, B, NS)
        # Sustained warmup: this laptop GPU idles at ~210 MHz and only boosts
        # under load, so short bursts measure clock ramp, not throughput.
        # ~1s of kernels per config before timing.
        t_warm = time.perf_counter()
        while time.perf_counter() - t_warm < 1.0:
            for meta, sym, nb_, base in groups:
                T.tans_unpack_group_gpu(payload_cat, meta, sym, nb_, base, out, NS, B)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            for meta, sym, nb_, base in groups:
                T.tans_unpack_group_gpu(payload_cat, meta, sym, nb_, base, out, NS, B)
        torch.cuda.synchronize()
        tg = (time.perf_counter() - t0) / reps
        back = out.cpu().numpy().tobytes()
        assert back == b"".join(raws), f"GPU mismatch at B={B}"
        print(f"{B:>7} {nblocks:>8} {stored/1024:>10.1f} {ratio_raw/stored:>6.3f}x "
              f"{rans_ratio:>7.3f} {tg*1e3:>8.3f} {total_out/tg/1e9:>6.2f} "
              f"{tr/tg:>7.2f}x")


if __name__ == "__main__":
    main()
