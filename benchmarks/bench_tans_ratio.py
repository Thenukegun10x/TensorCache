"""M0 ratio check: block-tANS (tans.py) vs rANS (rans.py) on real XS sections.

Pure-Python reference timings included for calibration (numba/Triton come
later); the exit criterion is RATIO: within ±1% of rANS on mid/textured,
better-or-equal on smooth.
"""

import time

import torch

from tensorcache import codec as C
from tensorcache import rans as R
from tensorcache import tans as T


def bench_section(name, raw: bytes):
    rb = R.encode(raw)
    t0 = time.perf_counter()
    tb = T.encode(raw)
    t1 = time.perf_counter()
    assert R.decode(rb) == raw and T.decode(tb) == raw
    print(f"{name}: len={len(raw):>7} rANS {len(raw)/len(rb):.3f}x "
          f"vs tANS {len(raw)/len(tb):.3f}x "
          f"(delta {(len(tb)-len(rb))/len(rb)*100:+.2f}%, "
          f"tans_enc {(t1-t0)*1e3:.0f}ms)")


def main():
    torch.manual_seed(0)
    for scale, name in [(8, "smooth"), (30, "mid"), (80, "textured")]:
        img = (torch.randn(336, 336, 3) * scale + 128).clamp(0, 255).to(torch.uint8)
        meta, _ = C.quantize_pixel_wavelet_adaptive(img, mode="balanced")
        arena = C.sparse_pack_arena(C.sparse_pack_meta(meta))
        print(f"== {name} ==")
        bench_section("u8 ", arena["arena_u8"].numpy().tobytes())
        bench_section("i8 ", arena["arena_i8"].numpy().tobytes())
        bench_section("ll4", arena["ll4"].numpy().tobytes())


if __name__ == "__main__":
    main()
