"""
Unit tests for the rANS codec (port of Texel/src/rans.rs test suite).

Every Rust test is mirrored here, plus path-equivalence: the numba kernel
and the pure-Python reference must produce byte-identical blobs and decode
each other's output.
"""

import struct

import pytest

from tensorcache.rans import (
    L,
    TOT,
    RansError,
    HAS_NUMBA,
    decode,
    encode,
    normalize,
)


def _roundtrip(data: bytes):
    enc = encode(data)
    assert decode(enc) == data
    return enc


def test_roundtrip_skewed():
    data = bytes([7] * 1000 + [8] * 300 + [250] * 50) + bytes(range(256))
    enc = _roundtrip(data)
    assert len(enc) < len(data), f"no compression: {len(enc)}"


def test_roundtrip_empty_and_single_symbol():
    # Rust parity: encode(&[]) emits the full38-byte header
    # (n + n_used + 4 lens + 4 states), decode returns empty immediately.
    enc = encode(b"")
    assert len(enc) == 38
    assert decode(enc) == b""
    # Single symbol: zero stream bytes (state never grows past L).
    _roundtrip(bytes([42] * 64))
    big = _roundtrip(bytes([9] * 100_000))
    assert len(big) < 100_000


def test_roundtrip_uniform():
    _roundtrip(bytes(i % 256 for i in range(1024)))


def test_interleave_residues():
    # n < 4 leaves streams empty; every residue mod 4 must roundtrip.
    for n in range(32):
        data = bytes((i * 37 + 11) % 251 for i in range(n))
        _roundtrip(data)


def test_normalize_drift_both_directions():
    # Clamp-up overshoot: 200 rare symbols forced to >= 1 push the sum over
    # TOT, exercising the decrement path.
    counts = [0] * 256
    for s in range(200):
        counts[s] = 1
    counts[255] = 100_000
    f = normalize(counts)
    assert sum(f) == TOT
    assert all(x >= 1 for x in f[:200])

    # Round-down undershoot: integer division sheds the remainder.
    counts = [0] * 256
    counts[0] = counts[1] = counts[2] = 1
    counts[3] = 1000
    f = normalize(counts)
    assert sum(f) == TOT

    # Roundtrip through both tables.
    for i, c in enumerate(counts):
        if c > 0:
            assert decode(encode(bytes([i] * c))) == bytes([i] * c)


def test_random_sweep_and_corrupt_never_crashes():
    x = 0x123456789ABCDEF1

    def rnd():
        nonlocal x
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        x &= 0xFFFFFFFFFFFFFFFF
        return x

    for _ in range(30):
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
        except RansError:
            pass


def test_truncated_rejected():
    with pytest.raises(RansError):
        decode(b"\x00\x00")
    # n != 0 but nothing else.
    with pytest.raises(RansError):
        decode(struct.pack("<I", 5))
    enc = encode(bytes([1, 2, 3]))
    with pytest.raises(RansError):
        decode(enc[:-1])
    # Corrupt a table byte so decode walks off the stream.
    enc = encode(bytes([7] * 500))
    bad = bytearray(enc)
    bad[8] = 0
    with pytest.raises(RansError):
        decode(bytes(bad))


def test_corrupt_table_rejected():
    enc = encode(bytes([7] * 500))
    # Duplicate the single table entry (sym 7, freq 4096).
    bad = bytearray(enc)
    entry = bytes(bad[6:9])
    bad[6:6] = entry
    bad[4:6] = struct.pack("<H", 2)
    with pytest.raises(RansError):
        decode(bytes(bad))
    # Frequencies not summing to TOT.
    bad = bytearray(enc)
    bad[7] ^= 0xFF
    bad[8] ^= 0xFF
    with pytest.raises(RansError):
        decode(bytes(bad))


def test_huge_count_rejected_not_oom():
    data = bytes([7] * 64 + [8] * 64)
    enc = bytearray(encode(data))
    enc[0:4] = struct.pack("<I", 0xFFFFFFFF)
    with pytest.raises(RansError):
        decode(bytes(enc))


def test_empty_single_symbol_header_layout():
    # Pin the exact layout from rans.rs: n, n_used, table, 4 lens, streams, states.
    enc = encode(bytes([42] * 64))
    (n,) = struct.unpack_from("<I", enc, 0)
    (n_used,) = struct.unpack_from("<H", enc, 4)
    assert n == 64 and n_used == 1
    sym, freq = struct.unpack_from("<BH", enc, 6)
    assert (sym, freq) == (42, TOT)
    # L is the state rANS starts every stream at.
    assert struct.unpack_from("<I", enc, len(enc) - 4)[0] == L


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_and_python_paths_byte_identical():
    """Both encode paths must emit identical bytes; both decode paths too."""
    import tensorcache.rans as rans

    vectors = [
        b"",
        bytes([7] * 500),
        bytes(range(256)) * 3,
        bytes((i * 37 + 11) % 251 for i in range(777)),
        bytes([1] * 4096 + [255] * 13 + [0] * 999),
    ]
    real_decode = rans.decode
    for data in vectors:
        enc_fast = encode(data)
        # Re-run encode with numba disabled (pure-Python reference path).
        rans.HAS_NUMBA = False
        try:
            enc_slow = encode(data)
            dec_slow = real_decode(enc_fast)
        finally:
            rans.HAS_NUMBA = True
        assert enc_fast == enc_slow, f"encode paths diverge on {len(data)} bytes"
        assert dec_slow == data
        assert decode(enc_slow) == data


def _golden_vectors():
    """Keep in lockstep with `golden_vectors()` in Texel/examples/rans_golden.rs."""
    vecs = []
    v = bytes([7] * 1000 + [8] * 300 + [250] * 50) + bytes(range(256))
    vecs.append(v)
    vecs.append(bytes(i % 256 for i in range(1024)))
    for n in range(33):
        vecs.append(bytes((i * 37 + 11) % 251 for i in range(n)))
    x = 0x123456789ABCDEF1
    out = bytearray()
    for i in range(10000):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        x &= 0xFFFFFFFFFFFFFFFF
        if i % 4 == 0:
            out.append(x % 256)
        elif i % 4 == 1:
            out.append(x % 4)
        elif i % 4 == 2:
            out.append(200 if x % 8 == 0 else 7)
        else:
            out.append((x >> 37) % 256)
    vecs.append(bytes(out))
    vecs.append(bytes((i // 16) % 256 for i in range(4096)))
    return vecs


def test_cross_implementation_with_rust_reference():
    """Golden blobs from Texel's Rust CPU reference (`cargo run --example
    rans_golden`): Python must decode them to the original data AND
    re-encode byte-identically (both encoders are deterministic).
    """
    from pathlib import Path

    golden = Path(__file__).parent / "data" / "rans_goldens.bin"
    if not golden.exists():
        pytest.skip("golden fixture missing — regen with: cd Texel && "
                    "cargo run --example rans_golden -- "
                    "<TensorCache>/tests/data/rans_goldens.bin")
    blob = golden.read_bytes()
    (nvec,) = struct.unpack_from("<I", blob, 0)
    pos = 4
    vecs = _golden_vectors()
    assert nvec == len(vecs), f"fixture has {nvec} vectors, spec has {len(vecs)}"
    for idx in range(nvec):
        (dlen,) = struct.unpack_from("<I", blob, pos)
        data = blob[pos + 4 : pos + 4 + dlen]
        pos += 4 + dlen
        (blen,) = struct.unpack_from("<I", blob, pos)
        enc_rust = blob[pos + 4 : pos + 4 + blen]
        pos += 4 + blen
        assert data == vecs[idx], f"vector {idx}: spec construction diverged"
        assert decode(enc_rust) == data, f"vector {idx}: python cannot decode rust blob"
        assert encode(data) == enc_rust, (
            f"vector {idx}: python encode is not byte-identical to rust"
        )
    assert pos == len(blob), "trailing bytes in golden fixture"


def test_no_numba_warns_loudly_once(monkeypatch):
    """Missing numba must warn big and once, at encode and decode entry."""
    import warnings as _w

    import tensorcache.rans as rans
    import tensorcache.tans as tans

    monkeypatch.setattr(rans, "HAS_NUMBA", False)
    monkeypatch.setattr(rans, "_NO_NUMBA_WARNED", False)
    with pytest.warns(RuntimeWarning, match="PURE-PYTHON"):
        rans.warn_if_no_numba("unit test")
    # once per process: a second call is silent even under -W error
    with _w.catch_warnings():
        _w.simplefilter("error")
        rans.warn_if_no_numba("again")

    data = bytes([7] * 200 + [8] * 50)
    monkeypatch.setattr(rans, "_NO_NUMBA_WARNED", True)
    blob = rans.encode(data)  # pure-Python path, warning suppressed
    monkeypatch.setattr(rans, "_NO_NUMBA_WARNED", False)
    with pytest.warns(RuntimeWarning, match="50x slower"):
        assert rans.decode(blob) == data

    monkeypatch.setattr(rans, "_NO_NUMBA_WARNED", False)
    with pytest.warns(RuntimeWarning, match="50x slower"):
        tans.encode(data, block=128)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[+] {name}")
    print("\n[+] ALL rANS TESTS PASSED")
