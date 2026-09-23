"""Unit tests for tANS dictionary learning + K>1 encode (research track)."""

import numpy as np
import pytest

from tensorcache import tans
from tensorcache.tans import TansError, decode, encode
from tensorcache.tans_dict import (
    collect_block_hists,
    learn_dictionary,
    select_tables,
)


def _toy_tables():
    peaked7 = [1] * 256
    peaked7[7] = 4000 - 255
    peaked200 = [1] * 256
    peaked200[200] = 4000 - 255
    # normalize to exact sums (already sum to 4000; top up argmax to 4096)
    peaked7[7] += 96
    peaked200[200] += 96
    return [peaked7, peaked200]


def test_learn_dictionary_shape_and_determinism():
    rng = np.random.default_rng(0)
    hists = []
    for _ in range(60):
        h = np.zeros(256, dtype=np.int64)
        syms = rng.integers(0, 256, size=2048)
        h[:] = np.bincount(syms, minlength=256)
        hists.append(h)
    H = np.stack(hists)
    t1 = learn_dictionary(H, K=4, seed=1)
    t2 = learn_dictionary(H, K=4, seed=1)
    assert t1 == t2
    assert len(t1) == 4
    for f in t1:
        assert len(f) == 256 and sum(f) == 4096 and min(f) >= 1


def test_select_picks_peaked_table():
    tables = _toy_tables()
    H = np.zeros((2, 256), dtype=np.int64)
    H[0, 7] = 2048
    H[1, 200] = 2048
    assert select_tables(H, tables, 12) == [0, 1]


def test_k_tables_roundtrip():
    tables = _toy_tables()
    data = bytes([7] * 1500 + [200] * 1500 + [7, 200] * 500)
    enc = encode(data, tables=tables)
    assert decode(enc) == data
    # determinism with selection involved
    assert encode(data, tables=tables) == enc


def test_k_tables_unseen_symbol_still_codes():
    rng = np.random.default_rng(3)
    hists = [np.bincount(rng.integers(0, 200, size=2048), minlength=256)
             for _ in range(40)]
    tables = learn_dictionary(np.stack(hists), K=2, seed=0)
    data = bytes(rng.integers(0, 256, size=3000).tolist())  # includes >= 200
    assert decode(encode(data, tables=tables)) == data


def test_referenced_tables_roundtrip():
    """E=0 blobs carry selectors only; decode takes tables explicitly."""
    tables = _toy_tables()
    data = bytes([7] * 1500 + [200] * 1500 + [7, 200] * 500)
    enc = encode(data, tables=tables, embed_tables=False)
    assert decode(enc, tables=tables) == data
    # determinism + smaller than embedded (no table bytes in blob)
    assert encode(data, tables=tables, embed_tables=False) == enc
    assert len(enc) < len(encode(data, tables=tables, embed_tables=True))
    with pytest.raises(TansError):
        decode(enc)  # tables referenced but not supplied
    # single referenced table also roundtrips (per-section table choice)
    enc1 = encode(data, tables=[tables[0]], embed_tables=False)
    assert decode(enc1, tables=[tables[0]]) == data


def test_bad_tables_rejected():
    with pytest.raises(ValueError):
        encode(b"abc", tables=[])
    with pytest.raises(ValueError):
        encode(b"abc", tables=[[1] * 256])  # wrong sum


def test_corrupt_selector_rejected():
    tables = _toy_tables()
    enc = bytearray(encode(bytes([7] * 3000), tables=tables))
    # selector array sits right before the payloads: find it via header walk
    import struct
    n, fmt, R, K, B, E = struct.unpack_from("<IBBBIB", enc, 0)
    assert (R, K, E) == (12, 2, 1)
    pos = 12
    for _ in range(K):
        (nu,) = struct.unpack_from("<H", enc, pos)
        pos += 2 + 3 * nu
    (nb,) = struct.unpack_from("<I", enc, pos)
    pos += 4 + 2 * nb + 2 * nb  # nblocks + u16 bit_lens + u16 finals
    assert nb >= 1
    enc[pos] = 9  # invalid selector
    with pytest.raises(TansError):
        decode(bytes(enc))


def test_collect_block_hists():
    H = collect_block_hists([bytes([5] * 3000)], 2048)
    assert H.shape == (2, 256)
    assert H[0, 5] == 2048 and H[1, 5] == 952


def test_dictionary_meta_roundtrip_and_validation():
    """Cache-embeddable dictionary: compact base64 roundtrip + validation."""
    import json

    from tensorcache.tans_dict import (
        DICT_VERSION, TansDictError, dictionary_from_meta, dictionary_to_meta,
    )
    tables = {"u8": _toy_tables(), "i8": _toy_tables()}
    meta = dictionary_to_meta(tables, R=12, K=2, B=256)
    assert meta["codec"] == "tans" and meta["version"] == DICT_VERSION
    assert set(meta["kinds"]) == {"u8", "i8"}
    assert isinstance(meta["kinds"]["u8"], str)  # base64 blob, not pairs
    # survives a JSON round-trip (it lives in _pixel_meta.json)
    meta = json.loads(json.dumps(meta))
    tabs2, R, K, B = dictionary_from_meta(meta, expected_kinds=("u8", "i8"))
    assert (R, K, B) == (12, 2, 256)
    assert tabs2 == tables
    # sparse tables (realistic) stay small
    t = [0] * 256
    for s in range(0, 200):
        t[s] = 1
    t[7] += 4096 - sum(t)
    small = dictionary_to_meta({"u8": [t, t]}, R=12, K=2, B=256)
    assert len(json.dumps(small)) < 3 * 1024

    # validation failures must raise, never silently decode garbage
    with pytest.raises(TansDictError):
        dictionary_from_meta({"codec": "rans", "version": 1})
    with pytest.raises(TansDictError):
        dictionary_from_meta({**meta, "version": 99})
    with pytest.raises(TansDictError):
        dictionary_from_meta(meta, expected_kinds=("u8", "i8", "ll4_lo"))
    bad = json.loads(json.dumps(meta))
    bad["kinds"]["u8"] = "!!!not-base64!!!"
    with pytest.raises(TansDictError):
        dictionary_from_meta(bad)
    bad2 = json.loads(json.dumps(meta))
    bad2["K"] = 4  # declared K disagrees with the blob
    with pytest.raises(TansDictError):
        dictionary_from_meta(bad2)
    bad3 = json.loads(json.dumps(meta))
    bad3["kinds"]["u8"] = bad3["kinds"]["u8"][:8]  # truncated blob
    with pytest.raises(TansDictError):
        dictionary_from_meta(bad3)
    with pytest.raises(TansDictError):
        dictionary_to_meta({"u8": [[1] * 256]}, R=12, K=1, B=256)  # bad sum


def test_planar_tans_roundtrip():
    """ll4 byte-planar tANS mirrors rANS's wrapper and roundtrips."""
    from tensorcache.tans import pack_byte_planar_tans, unpack_byte_planar_tans
    import numpy as np
    rng = np.random.default_rng(0)
    dense = (rng.integers(-40, 41, size=4096).astype(np.int16).tobytes())
    lo, hi = dense[0::2], dense[1::2]
    tabs_lo = learn_dictionary(
        collect_block_hists([lo], 512), K=2, seed=0)
    tabs_hi = learn_dictionary(
        collect_block_hists([hi], 512), K=2, seed=0)
    m, blob = pack_byte_planar_tans(dense, tabs_lo, tabs_hi, block=512)
    assert m == 1
    assert unpack_byte_planar_tans(blob, tabs_lo, tabs_hi) == dense
    # incompressible -> raw fallback keeps the v1 dense layout
    noise = bytes(rng.integers(0, 256, size=4096, dtype=np.uint8).tolist())
    m2, blob2 = pack_byte_planar_tans(noise, tabs_lo, tabs_hi, block=512)
    assert (m2 == 0 and blob2 == noise) or unpack_byte_planar_tans(
        blob2, tabs_lo, tabs_hi) == noise


def test_gpu_decode_capability_gate():
    from tensorcache.tans_dict import (
        require_tans_gpu_decode, tans_gpu_decode_available,
    )
    import torch
    avail = tans_gpu_decode_available("cpu")
    assert avail is False, "CPU must never claim GPU tANS decode"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            require_tans_gpu_decode("cuda:0")
        with pytest.raises(RuntimeError):
            require_tans_gpu_decode("cpu")
    else:
        # on a GPU host the gate passes for cuda and errors for cpu
        if tans_gpu_decode_available("cuda:0"):
            require_tans_gpu_decode("cuda:0")
        with pytest.raises(RuntimeError):
            require_tans_gpu_decode("cpu")


def test_split_section_components():
    from tensorcache.tans import split_section
    tables = _toy_tables()
    data = bytes([7] * 1500 + [200] * 1500 + [7, 200] * 500)
    for kw in ({}, {"tables": tables}, {"tables": tables, "embed_tables": False}):
        enc = encode(data, **kw)
        sp = split_section(enc, tables=tables if kw.get("embed_tables", True) is False else None)
        assert sp["n"] == len(data)
        assert sp["nblocks"] == (len(data) + 2048 - 1) // 2048
        assert sum(sp["bit_lens"][b] for b in range(sp["nblocks"])) > 0
        # payload regroups to the stored byte total; re-decode via components
        assert len(sp["payload"]) == sum((bl + 7) // 8 for bl in sp["bit_lens"])
        assert decode(enc, tables=tables if kw.get("embed_tables", True) is False else None) == data


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[+] {name}")
    print("\n[+] ALL tANS-DICT TESTS PASSED")