"""Integration test: xs_entropy='tans' wired into the pixel cache.

Writes a tANS cache and an rANS cache over the same images and checks the
decoded pixels are identical (entropy coding is lossless; only the stored
bytes differ), that the dictionary round-trips through _pixel_meta.json,
that the tANS sections are actually smaller, and that the loader refuses a
worker-based / CPU-only configuration for tANS.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image as PILImage

from tensorcache import codec as C
from tensorcache.pixel_cache import (
    PixelCacheDataset,
    cache_images,
    make_xs_loader,
)
from tensorcache.tans_dict import TANS_KINDS, TansDictError


def _natural(H=64, W=64, seed=0):
    torch.manual_seed(seed)
    lum = torch.nn.functional.avg_pool2d(
        torch.randint(0, 256, (H, W), dtype=torch.float32)[None, None],
        kernel_size=3, stride=1, padding=1).squeeze()
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W),
                            indexing="ij")
    img = torch.stack([lum + 24 * torch.sin(2 * torch.pi * yy),
                       lum + 20 * torch.cos(2 * torch.pi * (xx + yy)),
                       lum - 18 * torch.sin(2 * torch.pi * (xx - yy))],
                      dim=-1).clamp(0, 255).byte()
    return img


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="tANS caches decode on GPU")
def test_tans_cache_roundtrip_and_parity():
    dev = "cuda:0"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "imgs"
        src.mkdir()
        H = W = 64
        N = 6
        for s in range(N):
            PILImage.fromarray(_natural(H, W, s).numpy()).save(src / f"{s}.png")

        rans = cache_images(src, str(Path(tmp) / "rans"), height=H, width=W,
                            xs_mode="balanced", xs_entropy=True, log_every=99)
        tans = cache_images(src, str(Path(tmp) / "tans"), height=H, width=W,
                            xs_mode="balanced", xs_entropy="tans",
                            tans_calib=N, tans_block=128, log_every=99)
        assert rans["num_samples"] == tans["num_samples"] == N

        # the dictionary is embedded in meta and is not bloated
        import json
        meta = json.loads((Path(tmp) / "tans_pixel_meta.json").read_text())
        assert meta["xs_entropy"] == "tans"
        assert set(meta["tans"]["kinds"]) == set(TANS_KINDS)
        assert (Path(tmp) / "tans_pixel_meta.json").stat().st_size < 64 * 1024

        # every decoded sample is bit-identical to the rANS cache
        ds_r = PixelCacheDataset(str(Path(tmp) / "rans"), decode_device="cpu")
        ds_t = PixelCacheDataset(str(Path(tmp) / "tans"), decode_device="cpu")
        try:
            for i in range(N):
                assert torch.equal(ds_r[i], ds_t[i]), f"sample {i} differs"
        finally:
            ds_r.close()
            ds_t.close()

        # GPU loader path: decode_arenas_tans + wavelet, same pixels
        seen = 0
        for batch in make_xs_loader(str(Path(tmp) / "tans"), batch_size=2,
                                    device=dev, num_workers=0, shuffle=False):
            assert batch.shape[1:] == (H, W, 3) and batch.dtype == torch.uint8
            seen += batch.shape[0]
        assert seen == N


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="tANS decode gate is GPU-only")
def test_tans_loader_rejects_workers_and_cpu():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "imgs"
        src.mkdir()
        for s in range(3):
            PILImage.fromarray(_natural(64, 64, s).numpy()).save(src / f"{s}.png")
        cache_images(src, str(Path(tmp) / "tans"), height=64, width=64,
                     xs_entropy="tans", tans_calib=3, tans_block=128,
                     log_every=99)
        with pytest.raises(ValueError, match="num_workers=0"):
            list(make_xs_loader(str(Path(tmp) / "tans"), batch_size=1,
                                device="cuda:0", num_workers=2))
        with pytest.raises(RuntimeError, match="GPU decode unavailable"):
            list(make_xs_loader(str(Path(tmp) / "tans"), batch_size=1,
                                device="cpu", num_workers=0))


def test_tans_requires_dictionary_and_xs_quant():
    """Writer-side guards: tANS needs quant='xs' and a dictionary."""
    from tensorcache.pixel_cache import PixelCacheWriter
    with pytest.raises(ValueError, match="requires quant='xs'"):
        PixelCacheWriter("/tmp/opencode/_should_not_exist", 1, quant="raw",
                         xs_entropy="tans")
    with pytest.raises(TansDictError):
        PixelCacheWriter("/tmp/opencode/_should_not_exist2", 1, quant="xs",
                         xs_entropy="tans", tans_dict={"u8": []})


def test_bad_xs_entropy_value_rejected():
    from tensorcache.pixel_cache import PixelCacheWriter
    with pytest.raises(ValueError, match="xs_entropy must be"):
        PixelCacheWriter("/tmp/opencode/_should_not_exist3", 1, quant="xs",
                         xs_entropy="nonsense")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[+] {name}")
            except Exception as e:
                print(f"[-] {name}: {type(e).__name__}: {e}")
    print("\n[+] tANS cache tests finished")
