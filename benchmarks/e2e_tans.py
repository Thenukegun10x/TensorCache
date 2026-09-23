"""End-to-end: images -> wavelet -> tANS (encode) -> GPU tANS decode ->
GPU wavelet decode -> pixels. Verifies pixel-exactness and times both
halves against the untampered arena decode.

Encode side uses the production batched GPU wavelet encoder + CPU numba
tANS (the Triton encoder is validated but not wired in yet). Decode side
is fully GPU: grouped tANS kernels feeding the existing XS batch driver.

Usage: .venv/bin/python TensorCache/benchmarks/e2e_tans.py [nimg]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tensorcache import codec as C
from tensorcache import tans as T
from tensorcache.fused_ops import dequantize_sparse_wavelet_batch_gpu
from tensorcache.tans_dict import collect_block_hists, learn_dictionary

DATA = Path("TensorCache/data/coco_val")
H = W = 336
DEV = "cuda:0"
NS = 1 << 12
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
    metas = C.quantize_pixel_wavelet_adaptive_batched(imgs, mode="balanced")
    return C.sparse_pack_arena_batched(C.sparse_pack_meta_batched(metas))


def section_bytes(a):
    l4 = a["ll4"].numpy().astype(np.int16).tobytes()
    return {"u8": a["arena_u8"].numpy().tobytes(),
            "i8": a["arena_i8"].numpy().tobytes(),
            "ll4_lo": l4[0::2], "ll4_hi": l4[1::2]}


def encode_blobs(arenas, tabs):
    enc = {k: [] for k in KINDS}
    for a in arenas:
        sec = section_bytes(a)
        for k in KINDS:
            enc[k].append(T.encode(sec[k], block=B, tables=tabs[k],
                                   embed_tables=False))
    return enc


def build_plan(blobs, tables, device):
    """One-time decode plan for a section kind: payload + per-table launches.

    Mirrors what a loader does at cache open (parse blobs, group blocks by
    table, upload payload/meta, allocate output). The timed path then only
    launches kernels — no Python parsing, no H2D.
    """
    parts = [T.split_section(b, tables=tables) for b in blobs]
    codecs = [T._Codec(f, 12) for f in tables]
    total = sum(sp["n"] for sp in parts)
    out = torch.empty(total, dtype=torch.uint8, device=device)
    pay_list, rows_by_g, poff, ooff = [], {}, 0, 0
    for sp in parts:
        nb, sels = sp["nblocks"], sp["sels"]
        bl = np.array([(x + 7) // 8 for x in sp["bit_lens"]])
        goff = np.zeros(nb, dtype=np.int64)
        acc = 0
        for g in np.argsort(np.asarray(sels), kind="stable"):
            goff[int(g)] = acc
            acc += bl[int(g)]
        for b in range(nb):
            n_b = min(sp["B"], sp["n"] - b * sp["B"])
            rows_by_g.setdefault(int(sels[b]), []).append(
                [poff + int(goff[b]), sp["bit_lens"][b], sp["finals"][b], n_b, ooff])
            ooff += n_b
        pay_list.append(np.frombuffer(sp["payload"], dtype=np.uint8))
        poff += len(sp["payload"])
    payload = torch.from_numpy(np.concatenate(pay_list)).to(device)
    launches = []
    for g, rows in sorted(rows_by_g.items()):
        meta = torch.tensor(np.array(rows, dtype=np.int64), dtype=torch.int32,
                            device=device)
        if meta.shape[0] % 32:  # pre-pad so the timed path allocates nothing
            pad = 32 - meta.shape[0] % 32
            extra = torch.zeros((pad, 5), dtype=torch.int32, device=device)
            extra[:, 2] = NS
            meta = torch.cat([meta, extra], dim=0)
        c = codecs[g]
        launches.append((meta, torch.from_numpy(c.dsym_np).to(device),
                         torch.from_numpy(c.dnb_np).to(device),
                         torch.from_numpy(c.dbase_np.astype(np.int32)).to(device)))
    return {"payload": payload, "out": out, "launches": launches, "n": total}


def run_plan(plan, block, device):
    """Timed path: only kernel launches, no allocation or H2D."""
    out = plan["out"]
    for meta, sym, nb_, base in plan["launches"]:
        T._tans_unpack_kernel[(meta.shape[0] // 32,)](
            plan["payload"], meta, sym, nb_, base, out,
            NS=NS, BMAX=block, PAY_N=len(plan["payload"]), num_warps=1)
    return out


def main():
    nimg = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    global B
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    torch.cuda.set_device(DEV)
    files = sorted(DATA.glob("*.jpg"))
    print(f"e2e: {nimg} COCO images, block B={B}, mode balanced")

    # Calibration dicts from a disjoint slice (as a real cache build would).
    cal_arenas = build_arenas(load_images(files[400:520]))
    tabs = {}
    for k in KINDS:
        secs = [section_bytes(a)[k] for a in cal_arenas]
        tabs[k] = learn_dictionary(collect_block_hists(secs, B), K=8, seed=0)
    print(f"dicts learned from 120 disjoint images")

    imgs = load_images(files[:nimg])
    raw_rgb = nimg * H * W * 3

    # ---- encode ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    arenas = build_arenas(imgs)
    torch.cuda.synchronize()
    t_wave_enc = time.perf_counter() - t0
    # warm the numba encoder (JIT is ~1-2 s and would otherwise land in the
    # timed region); one throwaway image is enough
    encode_blobs(arenas[:1], tabs)
    t0 = time.perf_counter()
    enc = encode_blobs(arenas, tabs)
    t_tans_enc = time.perf_counter() - t0
    stored = sum(len(b) for k in enc for b in enc[k])
    stored += sum(a["meta"].numel() * 4 for a in arenas)
    print(f"encode: wavelet+pack {t_wave_enc/nimg*1e3:7.1f} ms/img | "
          f"tANS {t_tans_enc/nimg*1e3:5.2f} ms/img")
    print(f"stored {stored/1e6:.2f} MB vs raw RGB {raw_rgb/1e6:.2f} MB "
          f"= {raw_rgb/stored:.2f}x")

    ref = dequantize_sparse_wavelet_batch_gpu(arenas, device=DEV)
    torch.cuda.synchronize()

    plans = {k: build_plan(enc[k], tabs[k], DEV) for k in KINDS}

    def decode_full():
        dec = {k: run_plan(plans[k], B, DEV) for k in KINDS}
        u8, i8, lo, hi = dec["u8"], dec["i8"], dec["ll4_lo"], dec["ll4_hi"]
        rebuilt, ou, oi, ol = [], 0, 0, 0
        for a in arenas:
            nu = a["arena_u8"].numel()
            ni = a["arena_i8"].numel()
            nl = a["ll4"].numel()
            b = dict(a)
            b["arena_u8"] = u8[ou:ou + nu]
            b["arena_i8"] = i8[oi:oi + ni].view(torch.int8)
            b["ll4"] = torch.stack([lo[ol:ol + nl], hi[ol:ol + nl]],
                                   dim=-1).view(torch.int16)
            rebuilt.append(b)
            ou += nu
            oi += ni
            ol += nl
        return dequantize_sparse_wavelet_batch_gpu(rebuilt, device=DEV)

    out = decode_full()
    torch.cuda.synchronize()
    ok = torch.equal(out.cpu(), ref.cpu())
    print(f"pixel-exact vs untampered GPU decode: {ok}")
    assert ok, "e2e roundtrip is not pixel-exact"

    # ---- timing (sustained warmup: laptop GPU idles at ~210 MHz) ----
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < 1.5:
        decode_full()
        dequantize_sparse_wavelet_batch_gpu(arenas, device=DEV)
    torch.cuda.synchronize()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        for k in KINDS:
            run_plan(plans[k], B, DEV)
    torch.cuda.synchronize()
    t_tans = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        decode_full()
    torch.cuda.synchronize()
    t_e2e = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        dequantize_sparse_wavelet_batch_gpu(arenas, device=DEV)
    torch.cuda.synchronize()
    t_wave = (time.perf_counter() - t0) / reps
    print(f"\ndecode throughput ({nimg} imgs/batch, steady state, B={B}):")
    print(f"  wavelet only (untampered arena): {t_wave*1e3:7.2f} ms  "
          f"{nimg/t_wave:8.0f} img/s")
    print(f"  tANS kernels only              : {t_tans*1e3:7.2f} ms  "
          f"{nimg/t_tans:8.0f} img/s")
    print(f"  tANS + wavelet (e2e)           : {t_e2e*1e3:7.2f} ms  "
          f"{nimg/t_e2e:8.0f} img/s")
    print(f"  entropy overhead               : {(t_e2e-t_wave)*1e3:7.2f} ms  "
          f"({(t_e2e-t_wave)/t_wave*100:+.0f}%)")
    print(f"  compressed bytes fed to GPU    : {stored/nimg/1024:.1f} KB/img "
          f"(vs {H*W*3/1024:.0f} KB/img raw)")
    arena_bytes = sum(a["arena_u8"].numel() + a["arena_i8"].numel()
                      + a["ll4"].numel() * 2 + a["meta"].numel() * 4
                      for a in arenas) / nimg
    print(f"  decoded arena (old H2D payload): {arena_bytes/1024:.1f} KB/img")
    print(f"  H2D traffic reduction          : "
          f"{1 - (stored/nimg)/arena_bytes:.0%}")


if __name__ == "__main__":
    main()
