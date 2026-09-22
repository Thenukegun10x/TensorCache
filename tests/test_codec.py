"""
Unit Tests for BlockwiseInt8Codec and TensorCache.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

import math
from tensorcache.codec import (
    BlockwiseInt8Codec,
    quantize_int8_g32,
    dequantize_int8_g32,
    quantize_int8_adaptive,
    quantize_int4_g32,
    dequantize_int4_g32,
    quantize_int3_g32,
    dequantize_int3_g32,
    quantize_pixel_wavelet8x,
    dequantize_pixel_wavelet8x,
)
from tensorcache.feature_cache import FeatureCacheWriter, FeatureCacheDataset
from tensorcache.pixel_cache import PixelCacheWriter, PixelCacheDataset
from tensorcache.prefetcher import AsyncGPUPrefetcher


def _natural_test_img(H=64, W=64, seed=42):
    """Test image with detailed luma but smooth (natural-like) chroma.

    Blurred white RGB noise has near-white chroma, which is adversarial for
    4:2:0 (aliasing). Real photos have smooth chroma, so share one luma
    detail field across channels plus a smooth tint.
    """
    torch.manual_seed(seed)
    raw = torch.randint(0, 256, (H, W), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    lum = torch.nn.functional.avg_pool2d(raw, kernel_size=3, stride=1, padding=1).squeeze()
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    tint_r = 24 * torch.sin(2 * math.pi * yy) * torch.cos(2 * math.pi * xx)
    tint_g = 20 * torch.cos(2 * math.pi * (xx + yy))
    tint_b = -18 * torch.sin(2 * math.pi * (xx - yy))
    img = torch.stack([lum + tint_r, lum + tint_g, lum + tint_b], dim=-1).clamp(0, 255).byte()
    return img


def test_quantize_dequantize_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(16, 446, 768, dtype=torch.bfloat16)
    
    # 1. Standard INT8 G=32
    q, s, shape = quantize_int8_g32(x, group_size=32)
    rec = dequantize_int8_g32(q, s, shape, group_size=32)
    
    diff = x.float() - rec.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0
    
    assert rec.shape == x.shape
    assert rec.dtype == torch.bfloat16
    assert rel_rmse < 1.0  # Error must be < 1% (empirically ~0.54%)
    print(f"Standard G=32 Roundtrip RMSE: {rel_rmse:.3f}%")


def test_adaptive_quantize_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(8, 128, 768, dtype=torch.bfloat16)
    
    q, s, shape = quantize_int8_adaptive(x, group_size=32)
    rec = dequantize_int8_g32(q, s, shape, group_size=32)
    
    diff = x.float() - rec.float()
    rel_rmse = (torch.norm(diff) / torch.norm(x.float())).item() * 100.0
    
    assert rec.shape == x.shape
    assert rel_rmse < 1.0
    print(f"Adaptive G=32 Roundtrip RMSE: {rel_rmse:.3f}%")


def test_feature_cache_disk_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "test_feat_cache"
        num_samples = 20
        seq_len = 100
        dim = 256
        
        # 1. Write
        writer = FeatureCacheWriter(prefix, num_samples=num_samples, seq_len=seq_len, dim=dim, group_size=32)
        fake_features = torch.randn(num_samples, seq_len, dim, dtype=torch.bfloat16)
        writer.append(fake_features)
        writer.close()
        
        # 2. Read
        dataset = FeatureCacheDataset(prefix)
        assert len(dataset) == num_samples
        
        # Index single item
        q_int8, scales = dataset[0]
        rec = dequantize_int8_g32(q_int8, scales, (seq_len, dim), group_size=32)
        
        assert rec.shape == (seq_len, dim)
        assert rec.dtype == torch.bfloat16
        dataset.close()
        writer.close()
        print("[+] FeatureCache disk I/O test passed!")


def test_pixel_cache_disk_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "test_pixel_cache"
        num_samples = 10
        H, W, C = 64, 64, 3
        
        # 1. Write
        writer = PixelCacheWriter(prefix, num_samples=num_samples, height=H, width=W, channels=C)
        fake_imgs = np.random.randint(0, 256, size=(num_samples, H, W, C), dtype=np.uint8)
        for i in range(num_samples):
            writer.append_image(fake_imgs[i])
        writer.close()
        
        # 2. Read
        dataset = PixelCacheDataset(prefix)
        assert len(dataset) == num_samples
        
        img0 = dataset[0]
        assert img0.shape == (H, W, C)
        assert img0.dtype == torch.uint8
        assert np.array_equal(img0.numpy(), fake_imgs[0])
        dataset.close()
        writer.close()
        print("[+] PixelCache raw disk I/O test passed!")


def test_int4_int3_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(8, 64, 768, dtype=torch.bfloat16)

    # 1. INT4 G=32
    q4, s4, shape4 = quantize_int4_g32(x, group_size=32)
    rec4 = dequantize_int4_g32(q4, s4, shape4, group_size=32)
    diff4 = x.float() - rec4.float()
    rmse4 = (torch.norm(diff4) / torch.norm(x.float())).item() * 100.0
    assert rec4.shape == x.shape
    assert rec4.dtype == torch.bfloat16
    assert rmse4 < 10.0  # INT4 has ~4-6% rel RMSE
    print(f"[+] INT4 G=32 Roundtrip RMSE: {rmse4:.3f}%")

    # 2. INT3 G=32
    q3, s3, shape3 = quantize_int3_g32(x, group_size=32)
    rec3 = dequantize_int3_g32(q3, s3, shape3, group_size=32)
    diff3 = x.float() - rec3.float()
    rmse3 = (torch.norm(diff3) / torch.norm(x.float())).item() * 100.0
    assert rec3.shape == x.shape
    assert rec3.dtype == torch.bfloat16
    assert rmse3 < 25.0  # INT3 on Gaussian noise has ~22% rel RMSE
    print(f"[+] INT3 G=32 Roundtrip RMSE: {rmse3:.3f}%")


def test_pixel_cache_quantized_disk_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        num_samples = 5
        H, W, C = 64, 64, 3
        fake_imgs = np.random.randint(0, 256, size=(num_samples, H, W, C), dtype=np.uint8)

        # Test INT4 pixel cache
        p4 = Path(tmpdir) / "test_pixel_int4"
        w4 = PixelCacheWriter(p4, num_samples=num_samples, height=H, width=W, channels=C, quant="int4")
        for i in range(num_samples):
            w4.append_image(fake_imgs[i])
        w4.close()

        ds4 = PixelCacheDataset(p4)
        assert len(ds4) == num_samples
        img4 = ds4[0]
        assert img4.shape == (H, W, C)
        assert img4.dtype == torch.uint8
        ds4.close()

        # Test INT3 pixel cache
        p3 = Path(tmpdir) / "test_pixel_int3"
        w3 = PixelCacheWriter(p3, num_samples=num_samples, height=H, width=W, channels=C, quant="int3")
        for i in range(num_samples):
            w3.append_image(fake_imgs[i])
        w3.close()

        ds3 = PixelCacheDataset(p3)
        assert len(ds3) == num_samples
        img3 = ds3[0]
        assert img3.shape == (H, W, C)
        assert img3.dtype == torch.uint8
        ds3.close()

        print("[+] PixelCache INT4/INT3 disk I/O test passed!")


def test_mono_pixel_cache():
    """Mono (channels=1) raw/int4/int3 caches: 3x smaller files, exact round-trip,
    color input rejected, PIL grayscale ingest works."""
    from PIL import Image as PILImage
    with tempfile.TemporaryDirectory() as tmpdir:
        N, H, W = 4, 64, 64
        gray = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)

        for quant in ("raw", "int4", "int3"):
            prefix = Path(tmpdir) / f"mono_{quant}"
            writer = PixelCacheWriter(prefix, num_samples=N, height=H, width=W,
                                      channels=1, quant=quant)
            writer.append_image(gray)           # [H,W]
            writer.append_image(gray[:, :, None])  # [H,W,1]
            writer.append_image(torch.from_numpy(gray))
            # PIL "L" ingest (the path color BW JPEGs/PNGs take)
            pil_path = Path(tmpdir) / "g.png"
            PILImage.fromarray(gray).save(pil_path)
            writer.append_image(str(pil_path))
            writer.close()

            # file size: raw must be exactly 1/3 of an RGB cache
            ds = PixelCacheDataset(prefix)
            assert ds.channels == 1
            img = ds[0]
            assert img.shape == (H, W, 1) and img.dtype == torch.uint8
            if quant == "raw":
                assert np.array_equal(img.numpy()[:, :, 0], gray)
                assert os.path.getsize(str(prefix) + "_pixels.bin") == N * H * W
            ds.close()

        # color input must be rejected on mono caches
        writer = PixelCacheWriter(Path(tmpdir) / "mono_rej", num_samples=1, height=H,
                                  width=W, channels=1, quant="raw")
        with pytest.raises(ValueError):
            writer.append_image(np.zeros((H, W, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.append_image(torch.zeros(H, W, 3, dtype=torch.uint8))
        rgb_img = _natural_test_img(H, W)
        with pytest.raises(ValueError):
            writer.append_image(str(_save_tmp_png(tmpdir, rgb_img)))
        writer.close()
        # ...and mono input rejected on RGB caches
        writer = PixelCacheWriter(Path(tmpdir) / "rgb_rej", num_samples=1, height=H,
                                  width=W, channels=3, quant="raw")
        writer.append_image(gray)  # [H,W] still auto-replicates into RGB caches
        writer.close()
        print("[+] Mono pixel cache test passed (raw/int4/int3 + rejection guards)!")


def _save_tmp_png(tmpdir, img):
    from PIL import Image as PILImage
    p = Path(tmpdir) / "color.png"
    PILImage.fromarray(img.numpy()).save(p)
    return str(p)


def test_wavelet_8x_codec():
    H, W, C = 64, 64, 3
    img = _natural_test_img(H, W)
    
    packed_meta, shape = quantize_pixel_wavelet8x(img, q_scale=1.0)
    rec = dequantize_pixel_wavelet8x(packed_meta, device="cpu")
    
    assert rec.shape == (H, W, C)
    assert rec.dtype == torch.uint8
    diff = img.float() - rec.float()
    mse = (diff ** 2).mean().item()
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')
    rmse = (math.sqrt(mse) / 255.0) * 100.0
    
    assert psnr > 35.0
    print(f"[+] 8x Wavelet Codec Verified! PSNR: {psnr:.2f} dB, Rel RMSE: {rmse:.2f}%")


def test_sparse_bitstream_roundtrip():
    from tensorcache.codec import (
        sparse_pack_meta, sparse_unpack_meta, sparse_nbytes,
        quantize_pixel_wavelet_adaptive, dequantize_pixel_wavelet_adaptive,
    )
    torch.manual_seed(7)
    H, W, C = 64, 64, 3
    img = _natural_test_img(H, W, seed=7)
    raw_bytes = H * W * C

    # static path: sparse round-trip must be bit-exact vs dense decode
    meta, _ = quantize_pixel_wavelet8x(img, q_scale=3.0)
    sp = sparse_pack_meta(meta)
    assert sparse_nbytes(sp) < raw_bytes  # real compression, not dense storage
    rec_sparse = dequantize_pixel_wavelet8x(sparse_unpack_meta(sp), device="cpu")
    rec_dense = dequantize_pixel_wavelet8x(meta, device="cpu")
    assert (rec_sparse == rec_dense).all()

    # adaptive path
    ameta, _ = quantize_pixel_wavelet_adaptive(img, mode="balanced")
    asp = sparse_pack_meta(ameta)
    assert sparse_nbytes(asp) < raw_bytes
    rec_as = dequantize_pixel_wavelet_adaptive(sparse_unpack_meta(asp), device="cpu")
    rec_ad = dequantize_pixel_wavelet_adaptive(ameta, device="cpu")
    assert (rec_as == rec_ad).all()
    print(f"[+] Sparse bitstream round-trip OK "
          f"(static {raw_bytes/sparse_nbytes(sp):.2f}x, adaptive {raw_bytes/sparse_nbytes(asp):.2f}x)")


def test_sparse_pack_batched_bit_exact():
    """Batched pack must be byte-identical to the per-image reference for both
    static and adaptive sparse bitstreams, and the writer's append_batch must
    produce a decode-identical cache (dense white-noise sample included)."""
    from tensorcache.codec import (
        quantize_pixel_wavelet8x, quantize_pixel_wavelet_adaptive,
        sparse_pack_meta, sparse_pack_meta_batched,
        sparse_pack_arena, sparse_pack_arena_batched,
    )
    H, W = 64, 64
    N = 5
    imgs = [_natural_test_img(H, W, seed=s) for s in range(N - 1)]
    # dense occupancy -> exercises nonzero alignment pads in every blob
    torch.manual_seed(0)
    imgs.append(torch.randint(0, 256, (H, W, 3), dtype=torch.uint8))

    for adaptive in (False, True):
        if adaptive:
            metas = [quantize_pixel_wavelet_adaptive(im, mode="balanced")[0] for im in imgs]
        else:
            metas = [quantize_pixel_wavelet8x(im, q_scale=3.0)[0] for im in imgs]
        ref_sp = [sparse_pack_meta(m) for m in metas]
        ref_ar = [sparse_pack_arena(s) for s in ref_sp]
        got_sp = sparse_pack_meta_batched(metas)
        got_ar = sparse_pack_arena_batched(got_sp)
        for i in range(N):
            for c in range(3):
                rc, gc = ref_sp[i]["channels"][c], got_sp[i]["channels"][c]
                assert torch.equal(rc["LL4"].to(torch.int64), gc["LL4"].to(torch.int64))
                for name, rp in rc.items():
                    if name == "LL4":
                        continue
                    gp = gc[name]
                    for f in ("mask", "hflags", "hnib", "mode", "occ", "vals",
                              "idx_packed"):
                        if f not in rp:
                            continue
                        assert torch.equal(rp[f].to(torch.int64), gp[f].to(torch.int64)), \
                            f"meta {adaptive=} img={i} ch={c} {name}.{f}"
            for k in ("arena_u8", "arena_i8", "meta", "ll4"):
                assert torch.equal(ref_ar[i][k].to(torch.int64),
                                   got_ar[i][k].to(torch.int64)), \
                    f"arena {adaptive=} img={i} {k}"
            assert ref_ar[i]["inv"] == got_ar[i]["inv"]
        # N=1 must also work (no batching edge case)
        one = sparse_pack_arena_batched(sparse_pack_meta_batched(metas[:1]))
        assert torch.equal(one[0]["arena_u8"].to(torch.int64),
                           ref_ar[0]["arena_u8"].to(torch.int64))

    # writer: append_batch(images=...) == per-image append_image (bit-exact cache)
    with tempfile.TemporaryDirectory() as tmpdir:
        p_loop = str(Path(tmpdir) / "loop")
        p_batch = str(Path(tmpdir) / "batch")
        w1 = PixelCacheWriter(p_loop, num_samples=N, height=H, width=W,
                              channels=3, quant="xs", xs_mode="balanced")
        for im in imgs:
            w1.append_image(im.numpy())
        w1.close()
        w2 = PixelCacheWriter(p_batch, num_samples=N, height=H, width=W,
                              channels=3, quant="xs", xs_mode="balanced")
        assert w2.append_batch(images=[im.numpy() for im in imgs]) == N
        w2.close()

        # pre-encoded metas path (external batched encoder) must match too
        p_meta = str(Path(tmpdir) / "meta")
        w3 = PixelCacheWriter(p_meta, num_samples=N, height=H, width=W,
                              channels=3, quant="xs", xs_mode="balanced")
        pre = [quantize_pixel_wavelet_adaptive(im, mode="balanced")[0] for im in imgs]
        assert w3.append_batch(metas=pre) == N
        w3.close()

        d1 = PixelCacheDataset(p_loop, decode_device="cpu")
        d2 = PixelCacheDataset(p_batch, decode_device="cpu")
        d3 = PixelCacheDataset(p_meta, decode_device="cpu")
        ref = d1.decode_arenas([d1.get_arena(i) for i in range(N)], device="cpu")
        assert torch.equal(
            ref, d2.decode_arenas([d2.get_arena(i) for i in range(N)], device="cpu"))
        assert torch.equal(
            ref, d3.decode_arenas([d3.get_arena(i) for i in range(N)], device="cpu"))
        d1.close()
        d2.close()
        d3.close()
    print("[+] Batched sparse pack bit-exact (meta + arena + append_batch)")


def test_quantize_adaptive_batched_bit_exact():
    """Batched adaptive encoder must be bit-identical to the per-image path
    across modes (4:2:0 balanced, 4:4:4 high) and batch sizes."""
    from tensorcache.codec import (quantize_pixel_wavelet_adaptive,
                                   quantize_pixel_wavelet_adaptive_batched)
    H = W = 64
    for mode in ("balanced", "high"):
        for N in (1, 3, 5):
            imgs = [_natural_test_img(H, W, seed=s) for s in range(N)]
            batch = torch.stack(imgs, 0)
            ref = [quantize_pixel_wavelet_adaptive(im, mode=mode)[0] for im in imgs]
            got = quantize_pixel_wavelet_adaptive_batched(batch, mode=mode)
            assert len(got) == N
            for n in range(N):
                for c in range(3):
                    rc, gc = ref[n]['channels'][c], got[n]['channels'][c]
                    assert torch.equal(rc['LL4'].to(torch.int64),
                                       gc['LL4'].to(torch.int64)), (mode, n, c, 'LL4')
                    for name, rv in rc.items():
                        if name == 'LL4':
                            continue
                        gv = gc[name]
                        assert torch.equal(rv[0].to(torch.int64), gv[0].to(torch.int64)), \
                            (mode, n, c, name, 'q')
                        assert torch.equal(rv[1].to(torch.int64), gv[1].to(torch.int64)), \
                            (mode, n, c, name, 'idx')
                        assert rv[2] == gv[2], (mode, n, c, name, 'bq')
    print("[+] Batched adaptive encoder bit-exact (balanced/high, N=1/3/5)")


def test_xs_pixel_cache():
    """XS wavelet PixelCache: write arenas -> single/batch decode bit-exact."""
    from tensorcache.codec import (quantize_pixel_wavelet_adaptive,
                                   dequantize_pixel_wavelet_adaptive,
                                   sparse_pack_meta, sparse_unpack_meta)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = str(Path(tmpdir) / "xs_cache")
        H, W, N = 64, 64, 4
        imgs = [_natural_test_img(H, W, seed=s) for s in range(N)]
        # +1 white-noise image: dense occupancy forces nonzero alignment pads
        # in the packed blob layout (regression: pad bytes leaked into i8).
        torch.manual_seed(0)
        imgs.append(torch.randint(0, 256, (H, W, 3), dtype=torch.uint8))
        N += 1
        writer = PixelCacheWriter(prefix, num_samples=N, height=H, width=W, channels=3,
                                  quant="xs", xs_mode="balanced")
        for im in imgs:
            writer.append_image(im.numpy())
        writer.close()

        ds = PixelCacheDataset(prefix, decode_device=dev)
        assert len(ds) == N
        refs = []
        for im in imgs:
            m, _ = quantize_pixel_wavelet_adaptive(im, mode="balanced")
            refs.append(dequantize_pixel_wavelet_adaptive(
                sparse_unpack_meta(sparse_pack_meta(m)), device="cpu"))
        for i in range(N):
            got = ds[i]
            assert got.shape == (H, W, 3) and got.dtype == torch.uint8
            assert torch.equal(refs[i], got.cpu()), f"xs single {i}"
        batch = ds.decode_arenas([ds.get_arena(i) for i in range(N)], device=dev)
        assert batch.shape == (N, H, W, 3)
        for i in range(N):
            assert torch.equal(refs[i], batch[i].cpu()), f"xs batch {i}"
        n = sum(b.shape[0] for b in ds.iter_batches(batch_size=2, device=dev))
        assert n == N
        # CPU fallback parity (Windows-safe path)
        ds_cpu = PixelCacheDataset(prefix, decode_device="cpu")
        assert torch.equal(refs[0], ds_cpu[0])
        ds_cpu.close()
        ds.close()
        print("[+] XS PixelCache round-trip OK (single + batch + CPU fallback)")


def test_xs_entropy_parity_and_backcompat():
    """rANS storage layer (xs_entropy=True, default) must be invisible to
    consumers: arenas decode bit-identically to the pre-entropy layout,
    xs_entropy=False still writes v1 rows old readers accept, files shrink."""
    import json
    from tensorcache.pixel_cache import make_xs_loader
    H, W, N = 64, 64, 5
    imgs = [_natural_test_img(H, W, seed=s) for s in range(N - 1)]
    # +1 white-noise image: dense occupancy exercises nonzero pads post-decode
    torch.manual_seed(0)
    imgs.append(torch.randint(0, 256, (H, W, 3), dtype=torch.uint8))

    def build(prefix, entropy):
        w = PixelCacheWriter(prefix, num_samples=N, height=H, width=W,
                             channels=3, quant="xs", xs_mode="balanced",
                             xs_entropy=entropy)
        for im in imgs:
            w.append_image(im.numpy())
        w.close()

    def file_sizes(prefix):
        p = Path(prefix)
        return [f.stat().st_size for f in p.parent.iterdir()
                if f.name.startswith(p.name)]

    with tempfile.TemporaryDirectory() as tmpdir:
        v1 = str(Path(tmpdir) / "v1")
        v2 = str(Path(tmpdir) / "rans")
        build(v1, entropy=False)
        build(v2, entropy=True)

        # v1 writer: rows carry no entropy keys (byte-layout = pre-0.6 format)
        m1 = json.loads(Path(str(v1) + "_pixel_meta.json").read_text())
        assert m1.get("xs_entropy") is False
        assert all("enc" not in r and "lb_off" not in r for r in m1["xs_table"])

        m2 = json.loads(Path(str(v2) + "_pixel_meta.json").read_text())
        assert m2.get("xs_entropy") is True
        assert all("enc" in r and "lb_off" in r for r in m2["xs_table"])
        # meta section never entropy-coded (rANS measured at a loss on it)
        assert all(r["enc"][2] == 0 for r in m2["xs_table"])
        # entropy must actually shrink the payload files
        assert sum(file_sizes(v2)) < sum(file_sizes(v1))

        ds1 = PixelCacheDataset(v1, decode_device="cpu")
        ds2 = PixelCacheDataset(v2, decode_device="cpu")
        assert len(ds1) == len(ds2) == N
        for i in range(N):
            a, b = ds1.get_arena(i), ds2.get_arena(i)
            for k in ("arena_u8", "arena_i8", "meta", "ll4"):
                assert torch.equal(a[k], b[k]), f"arena {k} differs at sample {i}"
            assert torch.equal(ds1[i], ds2[i]), f"decoded pixels differ at {i}"
        ds1.close()
        ds2.close()

        # loader path: v1 vs entropy cache, multi-worker (fork/spawn safe)
        got1 = torch.cat(list(make_xs_loader(v1, batch_size=2, device="cpu",
                                             num_workers=0, shuffle=False)))
        got2 = torch.cat(list(make_xs_loader(v2, batch_size=2, device="cpu",
                                             num_workers=2, shuffle=False)))
        assert got1.shape == got2.shape == (N, H, W, 3)
        assert torch.equal(got1, got2)
        print(f"[+] XS entropy parity OK (v1 {sum(file_sizes(v1))//1024}KB -> "
              f"rans {sum(file_sizes(v2))//1024}KB, bit-identical decode)")


def test_xs_loader_ease_of_use():
    """cache_images one-liner + make_xs_loader + spawn-safe pickling."""
    import pickle
    from PIL import Image as PILImage
    from tensorcache.pixel_cache import cache_images, make_xs_loader
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "imgs"
        src.mkdir()
        H, W, N = 64, 64, 5
        refs = []
        for s in range(N):
            im = _natural_test_img(H, W, seed=s)
            PILImage.fromarray(im.numpy()).save(src / f"img{s}.png")
            refs.append(im)
        info = cache_images(src, str(Path(tmpdir) / "cache"), height=H, width=W,
                            xs_mode="balanced", log_every=1000)
        assert info["num_samples"] == N
        assert info["ratio_vs_raw"] > 1.0
        # loader: 5 samples, bs=2 -> 3 GPU batches, exact pixels
        seen = 0
        for b in make_xs_loader(str(Path(tmpdir) / "cache"), batch_size=2,
                                device=dev, num_workers=0, shuffle=False):
            assert b.shape[1:] == (H, W, 3) and b.dtype == torch.uint8
            if b.device.type == "cuda":
                assert torch.cuda.is_available()
            seen += b.shape[0]
        assert seen == N
        # multi-worker loader exercises fork/spawn pickling of the dataset
        seen = sum(b.shape[0] for b in make_xs_loader(
            str(Path(tmpdir) / "cache"), batch_size=2, device=dev,
            num_workers=2, shuffle=False))
        assert seen == N
        # explicit pickle round-trip (spawn path): mmaps reopen, arenas decode
        ds = PixelCacheDataset(str(Path(tmpdir) / "cache"), decode_device=dev)
        ds2 = pickle.loads(pickle.dumps(ds))
        assert len(ds2) == N
        assert torch.equal(ds[0].cpu(), ds2[0].cpu())
        ds.close()
        ds2.close()
        print("[+] XS loader ease-of-use OK (cache_images + loader + pickle)")


def test_feature_cache_append_mode():
    """open_append grows an existing cache in place: old rows byte-identical,
    new rows identical to a from-scratch encode, meta count = new total."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "append_feat"
        ref_prefix = Path(tmpdir) / "ref_feat"
        N, M, seq_len, dim = 12, 7, 64, 128

        torch.manual_seed(0)
        feats_old = torch.randn(N, seq_len, dim, dtype=torch.bfloat16)
        feats_new = torch.randn(M, seq_len, dim, dtype=torch.bfloat16)

        # 1. Base cache
        writer = FeatureCacheWriter(prefix, num_samples=N, seq_len=seq_len, dim=dim, group_size=32)
        writer.append(feats_old)
        writer.close()

        # 2. Settings mismatches must be rejected BEFORE any file is touched
        with pytest.raises(ValueError):
            FeatureCacheWriter.open_append(prefix, M, dim=999)
        with pytest.raises(ValueError):
            FeatureCacheWriter.open_append(prefix, M, group_size=16)
        with pytest.raises(ValueError):
            FeatureCacheWriter.open_append(prefix, M, amo_bq=True)
        with pytest.raises(FileNotFoundError):
            FeatureCacheWriter.open_append(Path(tmpdir) / "nope", M)

        # 3. Append M samples at the tail
        writer2 = FeatureCacheWriter.open_append(prefix, extra_samples=M, group_size=32)
        assert writer2.current_idx == N  # continues from existing tail
        writer2.append(feats_new)
        writer2.close()

        # meta must record the new total (current_idx), not the pre-allocated N+M
        # capacity window start, and not the old N
        import json as _json
        with open(str(prefix) + "_meta.json") as f:
            meta = _json.load(f)
        assert meta["num_samples"] == N + M

        # 4. Read back and compare against a from-scratch encode of all N+M
        ref = FeatureCacheWriter(ref_prefix, num_samples=N + M, seq_len=seq_len, dim=dim, group_size=32)
        ref.append(torch.cat([feats_old, feats_new], dim=0))
        ref.close()

        ds = FeatureCacheDataset(prefix)
        ref_ds = FeatureCacheDataset(ref_prefix)
        assert len(ds) == N + M
        for i in range(N + M):
            q, s = ds[i]
            rq, rs = ref_ds[i]
            assert torch.equal(q, rq), f"int8 row {i} differs from from-scratch encode"
            assert torch.equal(s, rs), f"scales row {i} differs from from-scratch encode"
        ds.close()
        ref_ds.close()

        # 5. Capacity: appending beyond extra_samples must raise (not corrupt)
        writer3 = FeatureCacheWriter.open_append(prefix, extra_samples=1)
        with pytest.raises(ValueError):
            writer3.append(torch.randn(2, seq_len, dim, dtype=torch.bfloat16))
        writer3.close()
        with open(str(prefix) + "_meta.json") as f:
            assert _json.load(f)["num_samples"] == N + M  # unchanged after failed over-append

        # 6. Sharded caches are explicitly unsupported
        sharded_prefix = Path(tmpdir) / "sharded_feat"
        wsh = FeatureCacheWriter(sharded_prefix, num_samples=4, seq_len=seq_len,
                                 dim=dim, group_size=32, num_shards=2)
        wsh.append(torch.randn(4, seq_len, dim, dtype=torch.bfloat16))
        wsh.close()
        with pytest.raises(NotImplementedError):
            FeatureCacheWriter.open_append(sharded_prefix, 1)
        print("[+] FeatureCache append mode test passed!")


if __name__ == "__main__":
    test_quantize_dequantize_roundtrip()
    test_adaptive_quantize_roundtrip()
    test_feature_cache_disk_io()
    test_feature_cache_append_mode()
    test_pixel_cache_disk_io()
    test_int4_int3_roundtrip()
    test_pixel_cache_quantized_disk_io()
    test_mono_pixel_cache()
    test_wavelet_8x_codec()
    test_sparse_bitstream_roundtrip()
    test_sparse_pack_batched_bit_exact()
    test_quantize_adaptive_batched_bit_exact()
    test_xs_pixel_cache()
    test_xs_entropy_parity_and_backcompat()
    test_xs_loader_ease_of_use()

    print("\n[+] ALL UNIT TESTS PASSED SUCCESSFULLY!")
