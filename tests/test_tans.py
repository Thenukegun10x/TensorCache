"""Unit tests for the block-tANS codec (research track, M0 CPU reference).

Covers: roundtrips (incl. block-size and precision sweeps), normalize parity
with rans at R=12, header layout, corrupt-input rejection, golden
byte-stability, and pack/unpack fallback policy. No dependency on rans
behavior beyond the shared normalize discipline and policy constants.
"""

import struct

import pytest

from tensorcache import tans
from tensorcache.tans import (
    B_DEFAULT,
    FMT_VERSION,
    R_DEFAULT,
    TansError,
    decode,
    decode_block,
    encode,
    encode_block,
    normalize,
    pack_section_tans,
    unpack_section_tans,
    _Codec,
)


def _roundtrip(data: bytes, **kw):
    enc = encode(data, **kw)
    assert decode(enc) == data
    return enc


def test_roundtrip_skewed():
    data = bytes([7] * 1000 + [8] * 300 + [250] * 50) + bytes(range(256))
    enc = _roundtrip(data)
    assert len(enc) < len(data), f"no compression: {len(enc)}"


def test_roundtrip_empty_and_single_symbol():
    assert decode(encode(b"")) == b""
    _roundtrip(bytes([42] * 64))
    big = _roundtrip(bytes([9] * 100_000))
    assert len(big) < 100_000


def test_roundtrip_uniform():
    _roundtrip(bytes(i % 256 for i in range(1024)))


def test_block_residues():
    data = bytes((i * 37 + 11) % 251 for i in range(5000))
    for b in (1, 2, 3, 7, 16, 511, 512, 513, 1024, 4096):
        _roundtrip(data, block=b)


def test_precision_sweep():
    data = bytes((i * 53 + 7) % 251 for i in range(3000))
    for r in (8, 10, 11, 12):
        _roundtrip(data, R=r)


def test_random_sweep_and_corrupt_never_crashes():
    x = 0x123456789ABCDEF1

    def rnd():
        nonlocal x
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        x &= 0xFFFFFFFFFFFFFFFF
        return x

    for _ in range(10):
        n = 1 + rnd() % 5000
        skew = rnd() % 4
        out = bytearray()
        for _ in range(n):
            if skew == 0:
                out.append(rnd() % 256)
            elif skew == 1:
                out.append(rnd() % 4)
            elif skew == 2:
                out.append(200 if rnd() % 8 == 0 else 7)
            else:
                out.append((rnd() >> 37) % 256)
        data = bytes(out)
        enc = _roundtrip(data)
        bad = bytearray(enc)
        for _ in range(3):
            i = rnd() % len(bad)
            bad[i] ^= 1 << (rnd() % 8)
        try:
            decode(bytes(bad))  # any result ok — crash/hang fails the test
        except TansError:
            pass


def test_normalize_matches_rans_at_r12():
    from tensorcache.rans import normalize as rans_normalize
    counts = [0] * 256
    counts[7] = 1000
    counts[8] = 300
    counts[250] = 50
    for i in range(256):
        counts[i] += 1
    assert normalize(counts, 12) == rans_normalize(counts)


def test_header_layout():
    enc = encode(bytes([42] * 64))
    n, fmt, R, K, B, E = struct.unpack_from("<IBBBIB", enc, 0)
    assert (n, fmt, R, K, B, E) == (64, FMT_VERSION, R_DEFAULT, 1, B_DEFAULT, 1)
    (n_used,) = struct.unpack_from("<H", enc, 12)
    assert n_used == 1
    sym, freq = struct.unpack_from("<BH", enc, 14)
    assert (sym, freq) == (42, 1 << R_DEFAULT)
    # per-block rows are u16 now: nblocks, then nblocks u16 bit_lens + u16 finals
    pos = 12 + 2 + 3 * 1
    (nb,) = struct.unpack_from("<I", enc, pos)
    assert nb == 1
    assert pos + 4 + 4 * nb <= len(enc)


def test_format_version_rejected():
    """A blob from another container version must fail loudly, not misparse."""
    enc = bytearray(encode(bytes([7] * 600)))
    assert enc[4] == FMT_VERSION
    enc[4] = FMT_VERSION + 1
    with pytest.raises(TansError, match="unsupported tANS section format"):
        decode(bytes(enc))


def test_block_too_large_rejected():
    # B*R must fit the u16 header
    with pytest.raises(ValueError, match="block\\*R"):
        encode(bytes([7] * 100), block=8192, R=12)


def test_truncated_rejected():
    with pytest.raises(TansError):
        decode(b"\x00\x00")
    with pytest.raises(TansError):
        decode(struct.pack("<I", 5))
    enc = encode(bytes([1, 2, 3]))
    with pytest.raises(TansError):
        decode(enc[:-1])
    enc = encode(bytes([7] * 500))
    bad = bytearray(enc)
    bad[12] = 0  # zero n_used -> bad table size
    with pytest.raises(TansError):
        decode(bytes(bad))


def test_corrupt_table_and_state_rejected():
    enc = bytearray(encode(bytes([7] * 500)))
    # absurd symbol count
    bad = bytearray(enc)
    bad[0:4] = struct.pack("<I", 0xFFFFFFFF)
    with pytest.raises(TansError):
        decode(bytes(bad))
    # bad table size
    bad = bytearray(enc)
    bad[12:14] = struct.pack("<H", 0)
    with pytest.raises(TansError):
        decode(bytes(bad))
    # frequencies don't sum: flip a freq byte (table pair starts at 14)
    bad = bytearray(enc)
    bad[15] ^= 0xFF
    try:
        decode(bytes(bad))
    except TansError:
        pass
    # bad final state: overwrite the block's u16 final (payload is last, so
    # finals sit before it) -> use the exact offset from the header walk
    pos = 12 + 2 + 3 * 1
    (nb,) = struct.unpack_from("<I", enc, pos)
    finals_off = pos + 4 + 2 * nb  # after nblocks + bit_lens
    bad = bytearray(enc)
    bad[finals_off:finals_off + 2] = struct.pack("<H", 0xFFFF)
    with pytest.raises(TansError):
        decode(bytes(bad))


def test_pack_fallback_policy():
    m, stored = pack_section_tans(b"a" * 10)
    assert m == 0 and stored == b"a" * 10  # tiny -> raw
    import os
    rnd = os.urandom(2048)  # incompressible -> raw
    m, stored = pack_section_tans(rnd)
    assert m == 0 and stored == rnd
    data = bytes([7] * 1000 + [8] * 300)
    m, stored = pack_section_tans(data)
    assert m == 2 and unpack_section_tans(m, stored) == data
    with pytest.raises(TansError):
        unpack_section_tans(7, b"junk")


def test_determinism():
    data = bytes((i * 37 + 11) % 251 for i in range(3000))
    assert encode(data) == encode(data)


@pytest.mark.skipif(not tans.HAS_NUMBA, reason="numba not installed")
def test_numba_and_python_paths_byte_identical():
    """njit kernels and the pure-Python reference agree bit-for-bit."""
    vectors = [
        bytes([7] * 500),
        bytes(range(256)) * 3,
        bytes((i * 37 + 11) % 251 for i in range(777)),
        bytes([1] * 4096 + [255] * 13 + [0] * 999),
    ]
    for data in vectors:
        enc_fast = encode(data)
        tans.HAS_NUMBA = False
        try:
            enc_slow = encode(data)
            dec_slow = decode(enc_fast)
        finally:
            tans.HAS_NUMBA = True
        assert enc_fast == enc_slow, "encode paths diverge"
        assert dec_slow == data
        assert decode(enc_slow) == data


def test_block_codec_direct():
    freq = normalize([5 if i % 3 else 100 for i in range(256)], 10)
    codec = _Codec(freq, 10)
    syms = bytes((i * 37 + 11) % 251 for i in range(100))
    pay, blen, fin = encode_block(syms, codec)
    assert decode_block(pay, blen, fin, len(syms), codec) == syms
    with pytest.raises(TansError):
        decode_block(pay, blen, 0, len(syms), codec)  # bad final state
    with pytest.raises(TansError):
        decode_block(pay[:-1] if pay else pay, blen, fin, len(syms), codec)


def test_cross_version_stability():
    """Golden blobs pin encode() byte-for-byte (see gen_tans_goldens.py)."""
    from pathlib import Path

    golden = Path(__file__).parent / "data" / "tans_goldens.bin"
    if not golden.exists():
        pytest.skip("golden fixture missing — regen with: "
                    "python tests/gen_tans_goldens.py")
    blob = golden.read_bytes()
    (nvec,) = struct.unpack_from("<I", blob, 0)
    pos = 4
    for idx in range(nvec):
        (dlen,) = struct.unpack_from("<I", blob, pos)
        data = blob[pos + 4:pos + 4 + dlen]
        pos += 4 + dlen
        (blen,) = struct.unpack_from("<I", blob, pos)
        enc_ref = blob[pos + 4:pos + 4 + blen]
        pos += 4 + blen
        assert decode(enc_ref) == data, f"vector {idx}: cannot decode golden"
        assert encode(data) == enc_ref, (
            f"vector {idx}: encode is not byte-identical to golden"
        )
    assert pos == len(blob), "trailing bytes in golden fixture"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[+] {name}")
    print("\n[+] ALL block-tANS TESTS PASSED")
