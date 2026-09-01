import torch
import pytest
from tensorcache.codec import quantize_int8_g32, dequantize_int8_g32
from tensorcache.int8_compress import compress_int8_blocks, decompress_int8_blocks, bench_ratio

def test_roundtrip_boring():
    for x in [torch.full((1,446,768), 0.5, dtype=torch.bfloat16),
              torch.full((1,446,768), 0.5, dtype=torch.bfloat16)+torch.randn(1,446,768, dtype=torch.bfloat16)*0.01,
              torch.zeros(1,446,768, dtype=torch.bfloat16)]:
        q,s,_ = quantize_int8_g32(x,32)
        data, offs, bws,_ = compress_int8_blocks(q,32)
        rec = decompress_int8_blocks(data, offs, bws, 32, q.numel())
        assert torch.equal(q, rec)

def test_roundtrip_diverse():
    x=torch.randn(1,446,768, dtype=torch.bfloat16)
    q,s,_ = quantize_int8_g32(x,32)
    data, offs, bws,_ = compress_int8_blocks(q,32)
    rec = decompress_int8_blocks(data, offs, bws, 32, q.numel())
    assert torch.equal(q, rec)

def test_ratio_boring():
    x=torch.full((1,446,768), 0.5, dtype=torch.bfloat16)
    q,s,_ = quantize_int8_g32(x,32)
    r=bench_ratio(q)
    assert r['ratio_real'] > 4.0  # 5.33x

def test_ratio_boring_small_noise():
    x=torch.full((1,446,768), 0.5, dtype=torch.bfloat16)+torch.randn(1,446,768, dtype=torch.bfloat16)*0.01
    q,s,_ = quantize_int8_g32(x,32)
    r=bench_ratio(q)
    assert r['ratio_real'] > 1.3  # 1.45x

def test_ratio_diverse_fallback():
    x=torch.randn(1,446,768, dtype=torch.bfloat16)
    q,s,_ = quantize_int8_g32(x,32)
    r=bench_ratio(q)
    assert r['ratio_real'] == 1.0
    assert r['fallback'] == True

def test_no_error_added():
    x=torch.randn(4,32, dtype=torch.bfloat16)
    q,s,sh = quantize_int8_g32(x,32)
    rec_plain = dequantize_int8_g32(q,s,sh,32)
    data, offs, bws,_ = compress_int8_blocks(q,32)
    rec_q = decompress_int8_blocks(data, offs, bws, 32, q.numel())
    rec = dequantize_int8_g32(rec_q, s, sh, 32)
    assert torch.equal(rec_plain, rec)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_fused():
    from tensorcache.int8_compress import dequantize_fused_gpu
    x=torch.randn(1,446,768, dtype=torch.bfloat16)
    q,s,sh = quantize_int8_g32(x,32)
    data, offs, bws,_ = compress_int8_blocks(q.cpu(),32)
    s_gpu=s.cuda()
    out=torch.empty(sh, dtype=torch.bfloat16, device='cuda')
    # Should work and be close to plain (fallback to raw for diverse)
    plain = dequantize_int8_g32(q.cuda(), s_gpu, sh, 32)
    fused = dequantize_fused_gpu(data, offs, bws, s_gpu, sh, 32, out)
    assert torch.allclose(plain.float(), fused.float(), atol=1e-3)
