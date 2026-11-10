"""Decoded-error quality search, API opt-in, and cache compatibility."""
from __future__ import annotations

import json

import pytest
import torch

import tensorcache as tc
from tensorcache import codec, fused_ops


@pytest.mark.parametrize("group_size", [16, 24, 32, 48, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quality_matches_exhaustive_decoded_code_search(group_size, dtype):
    """Independent oracle: every wire scale and all 256 decoded codes."""
    torch.manual_seed(41)
    x = torch.randn(group_size + 3, dtype=dtype)
    result = tc.compress(x, mode="quality", group_size=group_size)
    rec = tc.decompress(result[0], result[1], result[3], result[2], group_size=group_size)
    padded = torch.nn.functional.pad(x.float(), (0, (-x.numel()) % group_size))
    oracle_error = 0.0
    for block in padded.view(-1, group_size):
        lower, upper = block.min().clamp(max=0), block.max().clamp(min=0)
        scale0 = (upper / 255 - lower / 255).clamp(min=1e-8 / 255)
        lo = (scale0 * .85).bfloat16().view(torch.int16).item()
        hi = (scale0 * 1.10).bfloat16().view(torch.int16).item()
        assert hi - lo < 64
        scales = torch.arange(lo, hi + 1, dtype=torch.int16).view(torch.bfloat16).float()
        zps = (-lower / scales).round().clamp(0, 255)
        levels = ((torch.arange(256)[None, :] - zps[:, None]) * scales[:, None]).bfloat16().float()
        error = (block[None, :, None] - levels[:, None, :]).square().amin(2).sum(1)
        oracle_error += error.min().item()
    actual_error = (x.float() - rec.float()).square().sum().item()
    assert actual_error == pytest.approx(oracle_error, rel=1e-5, abs=1e-9)
    assert result[0].dtype == result[2].dtype == torch.uint8
    assert result[1].dtype == torch.bfloat16
    assert result[3] == x.shape


@pytest.mark.parametrize("group_size", [16, 32])
def test_quality_empty_nonfinite_and_constant_blocks(group_size):
    for x in (torch.empty(0), torch.tensor([float("nan"), float("inf"), -float("inf")]),
              torch.full((group_size,), 3.0), torch.full((group_size,), -3.0),
              torch.zeros(group_size)):
        q, scales, zps, shape = tc.compress(x, mode="quality", group_size=group_size)
        rec = tc.decompress(q, scales, shape, zps, group_size=group_size)
        assert torch.isfinite(scales).all() and torch.isfinite(rec).all()
        expected = torch.where(torch.isfinite(x), x, 0).bfloat16()
        assert torch.equal(rec, expected)


def test_quality_grid_bound_covers_bf16_mantissas():
    scales = torch.arange(128, 256).float() / 128
    for exponent in (-90, -10, 0, 30, 90):
        scale0 = scales * 2.0 ** exponent
        lo = (scale0 * .85).bfloat16().view(torch.int16).int()
        hi = (scale0 * 1.10).bfloat16().view(torch.int16).int()
        assert ((hi - lo) < 64).all()


def test_quality_cache_append_preserves_settings_and_rows(tmp_path):
    torch.manual_seed(42)
    x = torch.randn(3, 2, 32, dtype=torch.bfloat16)
    prefix = tmp_path / "quality"
    writer = tc.FeatureCacheWriter(prefix, 2, 2, 32, amo_bq=True,
                                   amo_mode="quality", group_size=16)
    try:
        writer.append(x[:2])
    finally:
        writer.close()
    old_bytes = (tmp_path / "quality_int8.bin").read_bytes()
    meta = json.loads((tmp_path / "quality_meta.json").read_text())
    assert meta["amo_mode"] == "quality" and meta["group_size"] == 16
    assert meta["amo_candidates"] == 64
    for settings in ({"amo_mode": "balanced"}, {"group_size": 32}):
        with pytest.raises(ValueError, match="mismatch"):
            tc.FeatureCacheWriter.open_append(prefix, 1, **settings)
        assert (tmp_path / "quality_int8.bin").read_bytes() == old_bytes
    writer = tc.FeatureCacheWriter.open_append(prefix, 1, amo_mode="quality", group_size=16)
    try:
        writer.append(x[2])
    finally:
        writer.close()
    assert (tmp_path / "quality_int8.bin").read_bytes().startswith(old_bytes)
    ds = tc.FeatureCacheDataset(prefix)
    try:
        assert len(ds) == 3
        for i in range(3):
            q, s, z = ds[i]
            expected = tc.compress(x[i], mode="quality", group_size=16)
            for actual, wanted in zip((q.flatten(), s, z), expected[:3]):
                assert torch.equal(actual, wanted)
    finally:
        ds.close()


def test_quality_sharded_cache_roundtrip(tmp_path):
    x = torch.randn(4, 2, 32, dtype=torch.bfloat16)
    prefix = tmp_path / "shards"
    writer = tc.FeatureCacheWriter(prefix, 4, 2, 32, amo_bq=True,
                                   amo_mode="quality", group_size=16, num_shards=2)
    try:
        writer.append(x)
    finally:
        writer.close()
    ds = tc.FeatureCacheDataset(prefix)
    try:
        assert ds.group_size == 16 and len(ds) == 4
        for i in range(4):
            q, s, z = ds[i]
            expected = tc.compress(x[i], mode="quality", group_size=16)
            assert torch.equal(q.flatten(), expected[0])
            assert torch.equal(s, expected[1]) and torch.equal(z, expected[2])
    finally:
        ds.close()


def test_quality_helpers_and_cli_require_opt_in(capsys):
    from tensorcache.cli import main

    x = torch.randn(64, dtype=torch.bfloat16)
    assert "amo_quality" not in tc.benchmark_tensor(x, verbose=False)
    assert "amo_quality" in tc.benchmark_tensor(x, verbose=False, include_quality=True)
    assert tc.auto_select_mode(x, target_rmse=0, group_size=16) == "max"
    assert tc.auto_select_mode(x, target_rmse=0, group_size=16, include_quality=True) == "quality"
    default = tc.compress(x)
    balanced = tc.compress(x, mode="balanced")
    assert all(torch.equal(a, b) for a, b in zip(default[:3], balanced[:3]))
    main(["compress-demo", "--shape", "64", "--device", "cpu", "--mode", "quality", "--group-size", "16"])
    assert "mode=quality G=16" in capsys.readouterr().out


@pytest.mark.skipif(not (torch.cuda.is_available() and fused_ops.HAS_TRITON), reason="CUDA/ROCm + Triton required")
@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quality_triton_matches_portable_decoded_error(group_size, dtype, monkeypatch):
    torch.manual_seed(47)
    x = torch.randn(33, 65, dtype=dtype).t()
    # Include non-finite, one-sided, and constant groups in the same launch.
    flat = x.contiguous().flatten()
    flat[:group_size] = 3
    flat[group_size:2 * group_size] = -3
    flat[2 * group_size] = float("nan")
    flat[3 * group_size] = float("inf")
    flat[-1] = -float("inf")
    x = flat.view(x.shape).t()
    ref = tc.compress(x, mode="quality", group_size=group_size)
    expected = tc.decompress(ref[0], ref[1], ref[3], ref[2], group_size=group_size).float()

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("quality GPU path silently fell back")
    monkeypatch.setattr(codec, "_quantize_amo_quality_torch", forbidden_fallback)
    gpu_x = x.cuda()
    result = tc.compress(gpu_x, mode="quality", group_size=group_size)
    actual = tc.decompress(result[0], result[1], result[3], result[2], group_size=group_size).cpu().float()
    finite = torch.isfinite(x)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[~finite]) == 0
    ref_error = (x[finite].float() - expected[finite]).square().sum()
    actual_error = (x[finite].float() - actual[finite]).square().sum()
    assert actual_error <= ref_error * 1.01 + 1e-7


@pytest.mark.skipif(not (torch.cuda.is_available() and fused_ops.HAS_TRITON), reason="CUDA/ROCm + Triton required")
@pytest.mark.parametrize("num_shards", [1, 2])
def test_quality_gpu_cache_batches_and_ring_streamer(tmp_path, num_shards):
    """GPU cache creation and both training readers honor persisted G16."""
    torch.manual_seed(53)
    x = torch.randn(5, 2, 32, dtype=torch.bfloat16, device="cuda")
    encoded = tc.compress(x, mode="quality", group_size=16)
    expected = tc.decompress(encoded[0], encoded[1], encoded[3], encoded[2], group_size=16)
    prefix = tmp_path / "gpu_quality"
    writer = tc.FeatureCacheWriter(prefix, 5, 2, 32, amo_bq=True,
                                   amo_mode="quality", group_size=16, num_shards=num_shards)
    try:
        writer.append(x)
    finally:
        writer.close()
    ds = tc.FeatureCacheDataset(prefix)
    try:
        batches = [batch.clone() for batch in ds.iter_batches(batch_size=2, shuffle=False, device="cuda")]
        assert torch.equal(torch.cat(batches), expected)
    finally:
        ds.close()
    streamer = tc.ZeroCopyTensorStreamer(prefix, batch_size=2, device="cuda", shuffle=False)
    try:
        batches = [batch.clone() for batch in streamer]
        assert torch.equal(torch.cat(batches), expected)
    finally:
        streamer.close()
