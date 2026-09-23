"""
Block-tANS entropy codec — research track (CPU reference, M0).

Separate from the production 4-way interleaved rANS in `rans.py`: nothing
in the existing cache paths imports this module. On-disk layout is the v3
section format ("tans-v1") with method id 2 (0 = raw, 1 = rANS).

tANS recap (Duda): states x in [NS, 2*NS), NS = 2**R. To encode symbol s
with frequency f from state x, pick k in {k_s, k_s-1} (k_s = R - floor
log2 f) such that j = (x >> k) - f lies in [0, f); emit the k low bits of
x (LSB-first into the block bitstream); the new state is NS + occ_s[j],
where occ_s[j] is the slot of the j-th spread occurrence of s. Decode
starts from the stored final state and inverts each step, consuming bits
from the end of the stream (LIFO, like rANS). Every block is independent:
fresh state NS, own bit-range, no cross-block dependency.

Layout v3 (FMT_VERSION 2):
    [u32 n]                        symbol count
    [u8 fmt][u8 R][u8 K][u32 B][u8 E]
    K x ([u16 n_used][n_used x (u8 sym, u16 freq)])   only if E == 1
    [u32 nblocks]
    nblocks x u16 bit_len, then nblocks x u16 final_state
    [nblocks x u8 selector]        only if K > 1
    [payloads]                     block bitstreams in stable selector-sort
                                   order (grouped by table; identity for K=1)

Both per-block fields fit u16 for R <= 15 and B*R <= 65535, which halves the
per-block header (8 B -> 4 B) — the dominant overhead at the small block
sizes the GPU decoder wants.

Referenced mode (E=0) is the cache design: dictionaries live once per cache
in meta (sidecar), sections carry only selectors; decode takes the tables
as an argument. Embedded mode (E=1, default) keeps blobs self-describing.
K=1 always embeds (the table is the data's own).

The layout reserves K tables + a per-block selector (dictionary tANS);
M0 encodes K=1, decode already handles K>=1 so the format never changes.
"""

from __future__ import annotations

import struct
from typing import List

import numpy as np

from .rans import MAX_SYMBOLS, MIN_GAIN_PCT, MIN_SECTION_BYTES, warn_if_no_numba
from functools import lru_cache

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is an optional accelerator
    HAS_NUMBA = False

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap

    njit = _njit

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

METHOD_TANS = 2
# Section container format. 2 = u16 per-block rows (bit_len, final_state).
# 1 was the u32-row prototype (unreleased); keeping a version byte makes a
# mixed-version blob fail loudly instead of decoding garbage.
FMT_VERSION = 2
R_MAX = 15  # final_state < 2*NS = 2**(R+1) must fit u16
R_DEFAULT = 12
B_DEFAULT = 2048  # single-table overhead (~8 B/block) needs B>=2K for <1%;
                  # dictionary payload wins may buy back smaller B later


class TansError(ValueError):
    """Corrupt or truncated block-tANS section."""


def normalize(counts, R: int) -> List[int]:
    """Quantize raw counts to frequencies summing to exactly 1 << R.

    Same drift-repair discipline as `rans.normalize` (present symbols get
    >= 1; decrement path takes from the argmax; increment path lands the
    whole deficit on one argmax symbol), parameterized by precision R.
    At R=12 output is identical to `rans.normalize` (pinned by test).
    """
    TOT = 1 << R
    total = sum(counts)
    freq = [0] * 256
    if total == 0:
        return freq
    s = 0
    for i in range(256):
        c = counts[i]
        if c > 0:
            q = (c * TOT) // total
            if q < 1:
                q = 1
            if q > TOT:
                q = TOT
            freq[i] = q
            s += q
    while s > TOT:
        best = 0
        for i in range(1, 256):
            if freq[i] > freq[best]:
                best = i
        if freq[best] <= 1:
            break  # unreachable; guards against a hang
        freq[best] -= 1
        s -= 1
    if s < TOT:
        best = 0
        for i in range(1, 256):
            if freq[i] > freq[best]:
                best = i
        freq[best] += TOT - s
    return freq


def spread(freq: List[int], R: int) -> List[int]:
    """Duda fast spread: slot -> symbol, symbol s occupying freq[s] slots.

    Fully deterministic (fixed step, ascending symbol order) — byte-identical
    output for identical input on any platform. Step is odd, hence coprime
    with NS = 2**R, so every slot is visited exactly once.
    """
    NS = 1 << R
    tab = [0] * NS
    pos = 0
    step = (NS >> 1) + (NS >> 3) + 3
    for s in range(256):
        for _ in range(freq[s]):
            tab[pos] = s
            pos = (pos + step) % NS
    return tab


@lru_cache(maxsize=128)
def _cached_codec(freq_tuple: tuple, R: int) -> _Codec:
    """Share table builds across sections using the same dictionary.

    Building a codec walks the full 2**R spread in Python (~ms); dictionary
    tables repeat every section, so caching turns per-section rebuilds into
    pointer lookups. Single-table (per-section) codecs miss and build once,
    same cost as before. maxsize must cover the whole working set (4 section
    kinds x K tables); too small a cache thrashes and rebuilds every call.
    One entry is ~100 KB, so 128 entries is ~13 MB worst case.
    """
    return _Codec(list(freq_tuple), R)


class _Codec:
    """One static tANS table: encode structures + table-driven decode."""

    __slots__ = ("R", "NS", "freq", "enc_k", "enc_slots",
                 "dec_sym", "dec_nb", "dec_base",
                 "f_np", "k_np", "sflat_np", "soff_np",
                 "dsym_np", "dnb_np", "dbase_np")

    def __init__(self, freq: List[int], R: int):
        NS = 1 << R
        self.R = R
        self.NS = NS
        self.freq = freq
        tab = spread(freq, R)
        # occurrence index per slot + per-symbol slot lists (ascending)
        rank = [0] * NS
        slots: List[List[int]] = [[] for _ in range(256)]
        for t, s in enumerate(tab):
            rank[t] = len(slots[s])
            slots[s].append(t)
        self.enc_slots = slots
        enc_k = [0] * 256
        for s in range(256):
            f = freq[s]
            if f:
                enc_k[s] = R - (f.bit_length() - 1)
        self.enc_k = enc_k
        # decode table: slot -> (symbol, bits to read, base state)
        dec_sym = bytearray(NS)
        dec_nb = bytearray(NS)
        dec_base = [0] * NS
        for t in range(NS):
            s = tab[t]
            xf = freq[s] + rank[t]  # in [f_s, 2*f_s)
            k = R - (xf.bit_length() - 1)
            dec_sym[t] = s
            dec_nb[t] = k
            dec_base[t] = xf << k
        self.dec_sym = dec_sym
        self.dec_nb = dec_nb
        self.dec_base = dec_base
        # numpy views for the njit/Triton fast paths (same data, typed)
        self.f_np = np.asarray(freq, dtype=np.int64)
        self.k_np = np.asarray(enc_k, dtype=np.int64)
        flat = []
        off = np.zeros(257, dtype=np.int64)
        for s in range(256):
            off[s + 1] = off[s] + len(slots[s])
            flat.extend(slots[s])
        self.soff_np = off
        self.sflat_np = np.asarray(flat, dtype=np.int64)
        self.dsym_np = np.array(dec_sym, dtype=np.uint8, copy=True)
        self.dnb_np = np.array(dec_nb, dtype=np.uint8, copy=True)
        self.dbase_np = np.asarray(dec_base, dtype=np.int64)


if HAS_NUMBA:

    @njit(cache=True)
    def _encode_block_njit(syms, freq, enc_k, sflat, soff, NS, R):
        """One block forward from state NS into a fresh byte buffer."""
        n = syms.shape[0]
        out = np.zeros((n * (R + 1) + 7) // 8 + 1, dtype=np.uint8)
        x = NS
        pos = 0
        for idx in range(n):
            s = int(syms[idx])
            f = freq[s]
            ks = enc_k[s]
            k = ks if (x >> ks) >= f else ks - 1
            v = x & ((1 << k) - 1) if k > 0 else 0
            kk = k
            vv = v
            while kk > 0:
                sh = pos & 7
                take = 8 - sh
                if take > kk:
                    take = kk
                out[pos >> 3] = out[pos >> 3] | np.uint8(((vv & ((1 << take) - 1)) << sh) & 0xFF)
                vv >>= take
                pos += take
                kk -= take
            x >>= k
            x = NS + sflat[soff[s] + (x - f)]
        nbytes = (pos + 7) >> 3
        return out[:nbytes], pos, x

    @njit(cache=True)
    def _decode_block_njit(payload, bit_len, final, n, dsym, dnb, dbase, NS):
        """Invert one block from its final state. Returns (out, err)."""
        out = np.empty(n, dtype=np.uint8)
        m = payload.shape[0]
        x = final
        p = bit_len
        for i in range(n - 1, -1, -1):
            slot = x - NS
            if slot < 0 or slot >= NS:
                return out, 1
            s = dsym[slot]
            k = int(dnb[slot])
            p -= k
            if p < 0:
                return out, 2
            v = 0
            if k:
                # windowed read: the k bits [p, p+k) span at most 3 bytes
                # (k <= R <= 16); one pass instead of one load per bit
                hi = (p + k - 1) >> 3
                if hi >= m:
                    return out, 2
                w = 0
                sh = 0
                b = p >> 3
                while b <= hi:
                    w |= int(payload[b]) << sh
                    sh += 8
                    b += 1
                v = (w >> (p & 7)) & ((1 << k) - 1)
            x = dbase[slot] + v
            out[i] = s
        return out, 0

else:
    _encode_block_njit = None
    _decode_block_njit = None


def _encode_block_py(syms: bytes, codec: _Codec) -> "tuple[bytes, int, int]":
    """Pure-Python reference: one block forward from state NS."""
    NS = codec.NS
    freq, enc_k, slots = codec.freq, codec.enc_k, codec.enc_slots
    x = NS
    buf = 0
    pos = 0
    for sym in syms:
        f = freq[sym]
        ks = enc_k[sym]
        if (x >> ks) >= f:
            k = ks
        else:
            k = ks - 1
        if k:
            buf |= (x & ((1 << k) - 1)) << pos
            pos += k
            x >>= k
        j = x - f
        assert 0 <= j < f, "tANS encode invariant violated"
        x = NS + slots[sym][j]
    nbytes = (pos + 7) // 8
    return (buf.to_bytes(nbytes, "little") if nbytes else b"", pos, x)


def encode_block(syms: bytes, codec: _Codec) -> "tuple[bytes, int, int]":
    """Encode one block (numba fast path, else the reference above)."""
    if HAS_NUMBA:
        arr = np.frombuffer(syms, dtype=np.uint8)
        buf, blen, fin = _encode_block_njit(arr, codec.f_np, codec.k_np,
                                            codec.sflat_np, codec.soff_np,
                                            codec.NS, codec.R)
        return bytes(buf), int(blen), int(fin)
    return _encode_block_py(syms, codec)


def _decode_block_py(payload: bytes, bit_len: int, final: int,
                     n: int, codec: _Codec) -> bytes:
    """Pure-Python reference: invert one block from its final state."""
    NS = codec.NS
    if not (NS <= final < 2 * NS):
        raise TansError(f"bad final state {final}")
    if not (0 <= bit_len <= len(payload) * 8):
        raise TansError("bad bit length")
    if 8 * len(payload) - bit_len >= 8:
        # more than one byte of slack: encoder never emits that
        raise TansError("bit length inconsistent with payload")
    buf = int.from_bytes(payload, "little") if payload else 0
    dec_sym, dec_nb, dec_base = codec.dec_sym, codec.dec_nb, codec.dec_base
    out = bytearray(n)
    x = final
    p = bit_len
    for i in range(n - 1, -1, -1):
        slot = x - NS
        if not (0 <= slot < NS):
            raise TansError(f"bad decode state {x}")
        s = dec_sym[slot]
        k = dec_nb[slot]
        p -= k
        if p < 0:
            raise TansError("truncated tANS block")
        v = ((buf >> p) & ((1 << k) - 1)) if k else 0
        x = dec_base[slot] + v
        out[i] = s
    return bytes(out)


def decode_block(payload: bytes, bit_len: int, final: int,
                 n: int, codec: _Codec) -> bytes:
    """Decode one block (numba fast path, else the reference above)."""
    if HAS_NUMBA:
        arr = np.frombuffer(payload, dtype=np.uint8) if payload else np.zeros(
            0, dtype=np.uint8)
        if not (codec.NS <= final < 2 * codec.NS):
            raise TansError(f"bad final state {final}")
        if not (0 <= bit_len <= len(payload) * 8):
            raise TansError("bad bit length")
        if 8 * len(payload) - bit_len >= 8:
            raise TansError("bit length inconsistent with payload")
        out, err = _decode_block_njit(arr, bit_len, final, n, codec.dsym_np,
                                      codec.dnb_np, codec.dbase_np, codec.NS)
        if err:
            raise TansError("corrupt tANS block")
        return bytes(out)
    return _decode_block_py(payload, bit_len, final, n, codec)


def _block_histograms(data: bytes, block: int, nb: int) -> "np.ndarray":
    """[nb x 256] per-block symbol counts (numpy, for table selection)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    H = np.zeros((nb, 256), dtype=np.int64)
    for b in range(nb):
        blk = arr[b * block:(b + 1) * block]
        H[b] = np.bincount(blk, minlength=256)
    return H


def encode(data: bytes, R: int = R_DEFAULT, block: int = B_DEFAULT,
           tables: "list[list[int]] | None" = None,
           embed_tables: bool = True) -> bytes:
    """Encode a byte string into a self-describing block-tANS section blob.

    Deterministic: same input bytes give same output bytes on any platform.
    `tables`: optional dictionary of K pre-normalized frequency tables
    (each summing to 2**R); every block is coded with its minimum
    cross-entropy table and the per-block selector is stored (K > 1).
    Without `tables` a single table is learned from the data (K=1, always
    embedded). With `embed_tables=False` the tables are NOT stored — the
    blob references them and `decode` needs the same `tables` argument
    (the cache design: dictionaries live once per cache in meta).
    """
    if not (1 <= R <= R_MAX):
        raise ValueError(f"R must be in [1, {R_MAX}], got {R}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    if block * R > 0xFFFF:
        raise ValueError(
            f"block*R must be <= 65535 for the u16 header, got {block}*{R}")
    warn_if_no_numba("block-tANS encode")
    n = len(data)
    if tables is None:
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8),
                             minlength=256).tolist()
        freqs = [normalize(counts, R)]
    else:
        if not tables:
            raise ValueError("tables must be non-empty")
        freqs = [list(f) for f in tables]
        for f in freqs:
            if len(f) != 256 or sum(f) != (1 << R) or any(v < 0 for v in f):
                raise ValueError("each table must be 256 freqs summing to 2**R")
    K = len(freqs)
    E = 1 if embed_tables else 0
    if K > 1:
        # shared dictionary tables: reuse builds across sections
        codecs = [_cached_codec(tuple(f), R) for f in freqs]
    else:
        codecs = [_Codec(freqs[0], R)]
    used = [[s for s in range(256) if f[s]] for f in freqs]

    nb = (n + block - 1) // block if n else 0
    # dictionary selection: one numpy pass over block histograms
    sels = None
    if K > 1:
        from .tans_dict import select_tables
        H = _block_histograms(data, block, nb)
        sels = select_tables(H, freqs, R)

    rows = []
    for b in range(nb):
        codec = codecs[sels[b]] if sels is not None else codecs[0]
        pay, blen, fin = encode_block(data[b * block:(b + 1) * block], codec)
        rows.append((pay, blen, fin))

    out = bytearray()
    # Section header: [u32 n][u8 fmt][u8 R][u8 K][u32 B][u8 E] = 12 bytes.
    # fmt pins the layout (2 = u16 per-block rows) so a future change is
    # detectable instead of silently misparsed.
    out += struct.pack("<IBBBIB", n, FMT_VERSION, R, K, block, E)
    if E:
        for t, u in enumerate(used):
            out += struct.pack("<H", len(u))
            for s in u:
                out.append(s)
                out += struct.pack("<H", freqs[t][s])
    if nb:
        max_bl = max(r[1] for r in rows)
        max_fin = max(r[2] for r in rows)
        if max_bl > 0xFFFF or max_fin > 0xFFFF:
            raise ValueError(
                f"block too large for the u16 header (bit_len={max_bl}, "
                f"state={max_fin}); use a smaller block (B*R <= 65535)")
    out += struct.pack("<I", nb)
    for _, blen, _ in rows:
        out += struct.pack("<H", blen)
    for _, _, fin in rows:
        out += struct.pack("<H", fin)
    if sels is not None:
        out += bytes(sels)
    if sels is not None:
        # grouped payloads: stable selector sort (deterministic). Same-table
        # blocks become contiguous, which is what makes GPU decode efficient
        # (one table resident per lane-group) and helps CPU locality too.
        # K=1 takes the identity path, so existing blobs are unaffected.
        order = np.argsort(np.asarray(sels), kind="stable")
        for i in order:
            out += rows[int(i)][0]
    else:
        for pay, _, _ in rows:
            out += pay
    return bytes(out)


def decode(blob: bytes, tables: "list[list[int]] | None" = None) -> bytes:
    """Decode a block-tANS section blob. Raises TansError on corrupt input.

    Blobs with referenced tables (E=0) need the same `tables` passed here
    (they live once per cache in meta, not per section).
    """
    warn_if_no_numba("block-tANS decode")
    m = len(blob)
    if m < 4:
        raise TansError("truncated tANS section")
    (n,) = struct.unpack_from("<I", blob, 0)
    if n == 0:
        return b""
    if n > MAX_SYMBOLS:
        raise TansError(f"absurd symbol count {n}")
    if m < 12:
        raise TansError("truncated tANS section")
    n, fmt, R, K, B, E = struct.unpack_from("<IBBBIB", blob, 0)
    if fmt != FMT_VERSION:
        raise TansError(
            f"unsupported tANS section format {fmt} (this build reads "
            f"{FMT_VERSION}); rebuild the cache with a matching version")
    if not (1 <= R <= R_MAX):
        raise TansError(f"bad precision {R}")
    if K == 0:
        raise TansError("bad table count 0")
    if B == 0:
        raise TansError("bad block size 0")
    if E not in (0, 1):
        raise TansError(f"bad embed flag {E}")
    pos = 12
    freqs: list = []
    if E:
        for _ in range(K):
            if m < pos + 2:
                raise TansError("truncated tANS section")
            (n_used,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            if n_used == 0 or n_used > 256:
                raise TansError(f"bad table size {n_used}")
            if m < pos + 3 * n_used:
                raise TansError("truncated tANS section")
            freq = [0] * 256
            seen = [False] * 256
            for _ in range(n_used):
                s = blob[pos]
                f = blob[pos + 1] | (blob[pos + 2] << 8)
                pos += 3
                if seen[s]:
                    raise TansError(f"duplicate symbol {s}")
                if f == 0 or f > (1 << R):
                    raise TansError(f"bad frequency {f}")
                seen[s] = True
                freq[s] = f
            if sum(freq) != (1 << R):
                raise TansError("frequencies do not sum to 2**R")
            freqs.append(freq)
    else:
        if tables is None:
            raise TansError("blob references tables but none were supplied")
        if len(tables) != K:
            raise TansError(f"need {K} tables, got {len(tables)}")
        for f in tables:
            if len(f) != 256 or sum(f) != (1 << R):
                raise TansError("supplied table does not sum to 2**R")
        freqs = [list(f) for f in tables]
    if K > 1:
        codecs = [_cached_codec(tuple(f), R) for f in freqs]
    else:
        codecs = [_Codec(freqs[0], R)]
    if m < pos + 4:
        raise TansError("truncated tANS section")
    (nb,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    if nb != ((n + B - 1) // B):
        raise TansError(f"bad block count {nb}")
    if m < pos + 4 * nb:
        raise TansError("truncated tANS section")
    bit_lens = list(struct.unpack_from("<%dH" % nb, blob, pos)) if nb else []
    pos += 2 * nb
    if m < pos + 2 * nb:
        raise TansError("truncated tANS section")
    finals = list(struct.unpack_from("<%dH" % nb, blob, pos)) if nb else []
    pos += 2 * nb
    byte_lens = [(bl + 7) // 8 for bl in bit_lens]
    sels = None
    if K > 1:
        if m < pos + nb:
            raise TansError("truncated tANS section")
        sels = list(blob[pos:pos + nb])
        pos += nb
        for sel in sels:
            if sel >= K:
                raise TansError(f"bad table selector {sel}")
    if m < pos + sum(byte_lens):
        raise TansError("truncated tANS section")
    # payload order is the stable selector sort (grouped); reconstruct each
    # block's byte offset from it. K=1 takes the identity path.
    if sels is not None:
        order = np.argsort(np.asarray(sels), kind="stable")
    else:
        order = np.arange(nb)
    offs = [0] * nb
    off = 0
    for g in range(nb):
        b = int(order[g])
        offs[b] = off
        off += byte_lens[b]
    out = bytearray()
    for b in range(nb):
        o = offs[b]
        pay = blob[pos + o:pos + o + byte_lens[b]]
        codec = codecs[sels[b]] if sels is not None else codecs[0]
        bn = min(B, n - b * B)
        out += decode_block(pay, bit_lens[b], finals[b], bn, codec)
    return bytes(out)


def split_section(blob: bytes, tables: "list[list[int]] | None" = None) -> dict:
    """Parse a section blob into components without decoding payloads.

    Returns {n,R,K,B,E,freqs,nblocks,bit_lens,finals,sels,payload,groups}
    where payload is the concatenated grouped bitstreams and groups maps
    selector value -> (byte_start, byte_len) into payload. The GPU loader
    path consumes this directly (no byte copies, just views + offsets).
    """
    m = len(blob)
    if m < 4:
        raise TansError("truncated tANS section")
    (n,) = struct.unpack_from("<I", blob, 0)
    if n > MAX_SYMBOLS:
        raise TansError(f"absurd symbol count {n}")
    if n == 0:
        return {"n": 0, "R": R_DEFAULT, "K": 1, "B": B_DEFAULT, "E": 1,
                "freqs": [], "nblocks": 0, "bit_lens": [], "finals": [],
                "sels": None, "payload": b"", "groups": {}}
    if m < 12:
        raise TansError("truncated tANS section")
    n, fmt, R, K, B, E = struct.unpack_from("<IBBBIB", blob, 0)
    if fmt != FMT_VERSION:
        raise TansError(f"unsupported tANS section format {fmt}")
    if not (1 <= R <= R_MAX) or K == 0 or B == 0 or E not in (0, 1):
        raise TansError("bad tANS header")
    pos = 12
    freqs = []
    if E:
        for _ in range(K):
            if m < pos + 2:
                raise TansError("truncated tANS section")
            (n_used,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            if n_used == 0 or n_used > 256 or m < pos + 3 * n_used:
                raise TansError("truncated tANS section")
            freq = [0] * 256
            for _ in range(n_used):
                s = blob[pos]
                f = blob[pos + 1] | (blob[pos + 2] << 8)
                pos += 3
                freq[s] = f
            freqs.append(freq)
    else:
        if tables is None or len(tables) != K:
            raise TansError("referenced tables missing or mismatched")
        freqs = [list(f) for f in tables]
    if m < pos + 4:
        raise TansError("truncated tANS section")
    (nb,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    if nb != ((n + B - 1) // B):
        raise TansError(f"bad block count {nb}")
    if m < pos + 4 * nb:
        raise TansError("truncated tANS section")
    bit_lens = list(struct.unpack_from("<%dH" % nb, blob, pos)) if nb else []
    pos += 2 * nb
    if m < pos + 2 * nb:
        raise TansError("truncated tANS section")
    finals = list(struct.unpack_from("<%dH" % nb, blob, pos)) if nb else []
    pos += 2 * nb
    sels = None
    if K > 1:
        if m < pos + nb:
            raise TansError("truncated tANS section")
        sels = list(blob[pos:pos + nb])
        pos += nb
    byte_lens = [(bl + 7) // 8 for bl in bit_lens]
    if m < pos + sum(byte_lens):
        raise TansError("truncated tANS section")
    payload = blob[pos:pos + sum(byte_lens)]
    groups: dict = {}
    if sels is not None:
        order = np.argsort(np.asarray(sels), kind="stable")
    else:
        order = np.arange(nb)
    off = 0
    for g in range(nb):
        b = int(order[g])
        if sels is not None:
            key = sels[b]
            if key not in groups:
                groups[key] = [off, 0]
            groups[key][1] += byte_lens[b]
        off += byte_lens[b]
    if sels is None and nb:
        groups[0] = [0, off]
    return {"n": n, "R": R, "K": K, "B": B, "E": E, "freqs": freqs,
            "nblocks": nb, "bit_lens": bit_lens, "finals": finals,
            "sels": sels, "payload": bytes(payload),
            "groups": {k: (a, c) for k, (a, c) in groups.items()}}


def pack_section_tans(raw: bytes, R: int = R_DEFAULT,
                      block: int = B_DEFAULT,
                      tables: "list[list[int]] | None" = None,
                      embed_tables: bool = True) -> "tuple[int, bytes]":
    """(method, stored) for one cache section; 2 = block-tANS, 0 = raw.

    Same fallback policy as `rans.pack_section`: tiny sections and gains
    below MIN_GAIN_PCT stay raw.
    """
    if len(raw) <= MIN_SECTION_BYTES:
        return 0, raw
    blob = encode(raw, R=R, block=block, tables=tables,
                  embed_tables=embed_tables)
    if len(blob) * 100 >= (100 - MIN_GAIN_PCT) * len(raw):
        return 0, raw
    return METHOD_TANS, blob


def build_decode_plan(blobs, tables, R: int, block: int, device):
    """One-time GPU decode plan for many section blobs of one kind.

    Parses each blob, groups blocks by selected table, uploads payload + meta,
    and allocates the output. The per-batch path is then `run_decode_plan`
    (kernel launches only). All blobs must share R and block size (a cache
    uses one config throughout).
    """
    if not HAS_TRITON:
        raise RuntimeError("tANS GPU decode requires Triton + CUDA/ROCm")
    import torch

    parts = [split_section(b, tables=tables) for b in blobs]
    for sp in parts:
        if sp["R"] != R:
            raise TansError(f"mixed precision {sp['R']} != {R}")
        if sp["B"] != block:
            raise TansError(f"mixed block size {sp['B']} != {block}")
    NS = 1 << R
    codecs = [_cached_codec(tuple(f), R) for f in tables]
    total = sum(sp["n"] for sp in parts)
    out = torch.empty(total, dtype=torch.uint8, device=device)
    pay_list, rows_by_g, poff, ooff = [], {}, 0, 0
    for sp in parts:
        nb, sels = sp["nblocks"], sp["sels"]
        bl = np.array([(x + 7) // 8 for x in sp["bit_lens"]])
        goff = np.zeros(nb, dtype=np.int64)
        acc = 0
        it = np.argsort(np.asarray(sels), kind="stable") if nb else ()
        for g in it:
            goff[int(g)] = acc
            acc += bl[int(g)]
        for b in range(nb):
            n_b = min(sp["B"], sp["n"] - b * sp["B"])
            rows_by_g.setdefault(int(sels[b]), []).append(
                [poff + int(goff[b]), sp["bit_lens"][b], sp["finals"][b], n_b, ooff])
            ooff += n_b
        pay_list.append(np.frombuffer(sp["payload"], dtype=np.uint8))
        poff += len(sp["payload"])
    payload = (torch.from_numpy(np.concatenate(pay_list)).to(device)
               if pay_list else torch.zeros(0, dtype=torch.uint8, device=device))
    launches = []
    for g, rows in sorted(rows_by_g.items()):
        meta = torch.tensor(np.array(rows, dtype=np.int64), dtype=torch.int32,
                            device=device)
        if meta.shape[0] % 32:  # pad so the timed path allocates nothing
            pad = 32 - meta.shape[0] % 32
            extra = torch.zeros((pad, 5), dtype=torch.int32, device=device)
            extra[:, 2] = NS
            meta = torch.cat([meta, extra], dim=0)
        c = codecs[g]
        launches.append((meta,
                         torch.from_numpy(c.dsym_np).to(device),
                         torch.from_numpy(c.dnb_np).to(device),
                         torch.from_numpy(c.dbase_np.astype(np.int32)).to(device)))
    return {"payload": payload, "out": out, "launches": launches,
            "n": total, "sizes": [sp["n"] for sp in parts],
            "R": R, "block": block}


def run_decode_plan(plan: dict) -> "torch.Tensor":
    """Steady-state GPU decode: kernel launches only (no parse, no H2D)."""
    out = plan["out"]
    for meta, sym, nb_, base in plan["launches"]:
        _tans_unpack_kernel[(meta.shape[0] // 32,)](
            plan["payload"], meta, sym, nb_, base, out,
            NS=1 << plan["R"], BMAX=plan["block"],
            PAY_N=len(plan["payload"]), num_warps=1)
    return out


def split_planar_section(stored: bytes):
    """Split an ll4 byte-planar wrapper -> ((method, bytes), (method, bytes)).

    Mirrors the rANS wrapper the tANS writer reuses; inner methods are 0 or 2.
    """
    if len(stored) < 5:
        raise TansError("truncated planar section")
    m0 = stored[0]
    n0 = int.from_bytes(stored[1:5], "little")
    if len(stored) < 5 + n0 + 5:
        raise TansError("truncated planar section")
    b0 = stored[5:5 + n0]
    p = 5 + n0
    m1 = stored[p]
    n1 = int.from_bytes(stored[p + 1:p + 5], "little")
    if len(stored) < p + 5 + n1:
        raise TansError("truncated planar section")
    b1 = stored[p + 5:p + 5 + n1]
    return (m0, b0), (m1, b1)


def unpack_section_tans(method: int, blob,
                        tables: "list[list[int]] | None" = None) -> bytes:
    """Inverse of pack_section_tans. Accepts any bytes-like (mmap slices)."""
    if method == 0:
        return bytes(blob)
    if method == METHOD_TANS:
        return decode(bytes(blob), tables=tables)
    raise TansError(f"unknown tANS section method {method}")


def pack_byte_planar_tans(raw: bytes, tables_lo, tables_hi,
                          R: int = R_DEFAULT, block: int = B_DEFAULT):
    """Byte-planar + per-plane tANS for int16 LE streams (ll4).

    Format-parity with `rans.pack_byte_planar`: same outer method id (1) and
    `[method][u32 len]` wrapper per plane, but the inner section method is 2
    (block-tANS) instead of 1, so a reader dispatches on the inner flag.
    Returns (0, raw) when neither plane benefits or the wrapper grows it.
    """
    lo, hi = raw[0::2], raw[1::2]
    m0, b0 = pack_section_tans(lo, R, block, tables_lo, embed_tables=False)
    m1, b1 = pack_section_tans(hi, R, block, tables_hi, embed_tables=False)
    if m0 == 0 and m1 == 0:
        return 0, raw
    wrapped = (struct.pack("<BI", m0, len(b0)) + b0
               + struct.pack("<BI", m1, len(b1)) + b1)
    if len(wrapped) >= len(raw):
        return 0, raw
    return 1, wrapped


def unpack_byte_planar_tans(blob, tables_lo=None, tables_hi=None) -> bytes:
    """Inverse of pack_byte_planar_tans -> dense [lo,hi,...] bytes.

    Handles inner methods 0 (raw) and 2 (tANS); method 1 (rANS) is rejected
    because this is the tANS entry point (the cache reader picks the path
    from the meta's `xs_entropy` marker).
    """
    stored = bytes(blob)
    if len(stored) < 5:
        raise TansError("truncated planar section")
    m0 = stored[0]
    n0 = int.from_bytes(stored[1:5], "little")
    b0 = unpack_section_tans(m0, stored[5:5 + n0], tables_lo)
    p = 5 + n0
    if len(stored) < p + 5:
        raise TansError("truncated planar section")
    m1 = stored[p]
    n1 = int.from_bytes(stored[p + 1:p + 5], "little")
    b1 = unpack_section_tans(m1, stored[p + 5:p + 5 + n1], tables_hi)
    if len(b0) != len(b1):
        raise TansError(f"planar planes differ in length: {len(b0)} vs {len(b1)}")
    out = bytearray(len(b0) + len(b1))
    out[0::2] = b0
    out[1::2] = b1
    return bytes(out)


if HAS_TRITON:

    @triton.jit
    def _tans_unpack_kernel(payload_ptr, meta_ptr,
                            sym_ptr, nb_ptr, base_ptr, out_ptr,
                            NS: tl.constexpr, BMAX: tl.constexpr,
                            PAY_N: tl.constexpr):
        """Decode 32 blocks (one lane each) sharing one resident table.

        meta rows (int32): byte_off, bit_len, final, n, out_base. Blocks of
        one launch share a table (writer groups payloads by selector), so a
        single SRAM copy serves all lanes. Bit reads stream from L1 through
        a ≤3-byte window per symbol (k <= 16); no division, no atomics.
        """
        prog = tl.program_id(0)
        lane = tl.arange(0, 32)
        idx = prog * 32 + lane
        byte_off = tl.load(meta_ptr + idx * 5 + 0)
        bit_len = tl.load(meta_ptr + idx * 5 + 1)
        final = tl.load(meta_ptr + idx * 5 + 2)
        nn = tl.load(meta_ptr + idx * 5 + 3)
        obase = tl.load(meta_ptr + idx * 5 + 4)
        x = final
        p = bit_len
        base_bit = byte_off << 3
        for ii in tl.range(0, BMAX, num_stages=2):
            active = ii < nn
            i = nn - 1 - ii
            slot = x - NS
            slot_c = tl.where(active, slot, 0)
            s = tl.load(sym_ptr + slot_c).to(tl.int32)
            k = tl.load(nb_ptr + slot_c).to(tl.int32)
            bs = tl.load(base_ptr + slot_c)
            k = tl.where(active, k, 0)
            pk = p - k
            g = base_bit + pk
            b0 = g >> 3
            nbytes = ((g + k - 1) >> 3) - b0 + 1
            w = tl.zeros([32], dtype=tl.int32)
            for j in range(3):
                bb = b0 + j
                ok = active & (k > 0) & (j < nbytes) & (bb >= 0) & (bb < PAY_N)
                byte = tl.load(payload_ptr + bb, mask=ok, other=0).to(tl.int32)
                w = w | (byte << (j * 8))
            v = (w >> (g & 7)) & ((1 << k) - 1)
            p = pk
            x = tl.where(active, bs + v, x)
            tl.store(out_ptr + obase + i, s.to(tl.uint8), mask=active)

    @triton.jit
    def _tans_lengths_kernel(syms_ptr, sbase_ptr, n_ptr,
                             freq_ptr, enck_ptr, sflat_ptr, soff_ptr,
                             lens_ptr, NS: tl.constexpr, BMAX: tl.constexpr):
        """Pass 1 of GPU encode: bit length per block (state machine only)."""
        prog = tl.program_id(0)
        lane = tl.arange(0, 32)
        idx = prog * 32 + lane
        sb = tl.load(sbase_ptr + idx)
        nn = tl.load(n_ptr + idx)
        x = tl.full([32], NS, tl.int32)
        total = tl.zeros([32], dtype=tl.int32)
        for ii in tl.range(0, BMAX, num_stages=2):
            active = ii < nn
            s = tl.load(syms_ptr + sb + ii, mask=active, other=0).to(tl.int32)
            f = tl.load(freq_ptr + s)
            ks = tl.load(enck_ptr + s)
            use_full = (x >> ks) >= f
            k = tl.where(use_full, ks, ks - 1)
            k = tl.where(active, k, 0)
            total = total + k
            x = tl.where(active, x >> k, x)
            j = x - f
            j_c = tl.where(active, j, 0)
            slot = tl.load(sflat_ptr + tl.load(soff_ptr + s) + j_c)
            x = tl.where(active, NS + slot, x)
        tl.store(lens_ptr + idx, total)

    @triton.jit
    def _tans_pack_kernel(syms_ptr, sbase_ptr, n_ptr, obyte_ptr,
                          freq_ptr, enck_ptr, sflat_ptr, soff_ptr,
                          out_ptr, NS: tl.constexpr, BMAX: tl.constexpr):
        """Pass 2 of GPU encode: repack bits into exclusive byte ranges."""
        prog = tl.program_id(0)
        lane = tl.arange(0, 32)
        idx = prog * 32 + lane
        sb = tl.load(sbase_ptr + idx)
        nn = tl.load(n_ptr + idx)
        ob = tl.load(obyte_ptr + idx)
        x = tl.full([32], NS, tl.int32)
        cb = tl.zeros([32], dtype=tl.int32)
        cb_bits = tl.zeros([32], dtype=tl.int32)
        for ii in tl.range(0, BMAX, num_stages=2):
            active = ii < nn
            s = tl.load(syms_ptr + sb + ii, mask=active, other=0).to(tl.int32)
            f = tl.load(freq_ptr + s)
            ks = tl.load(enck_ptr + s)
            use_full = (x >> ks) >= f
            k = tl.where(use_full, ks, ks - 1)
            k = tl.where(active, k, 0)
            v = x & ((1 << k) - 1)
            x = tl.where(active, x >> k, x)
            j = x - f
            j_c = tl.where(active, j, 0)
            slot = tl.load(sflat_ptr + tl.load(soff_ptr + s) + j_c)
            x = tl.where(active, NS + slot, x)
            rem = k
            vv = v
            for _ in range(3):
                space = 8 - cb_bits
                take = tl.minimum(rem, space)
                cb = cb | ((vv & ((1 << take) - 1)) << cb_bits)
                vv = vv >> take
                cb_bits = cb_bits + take
                rem = rem - take
                full = cb_bits == 8
                tl.store(out_ptr + ob, cb.to(tl.uint8), mask=active & full)
                ob = ob + tl.where(full, 1, 0)
                cb = tl.where(full, 0, cb)
                cb_bits = tl.where(full, 0, cb_bits)
        tl.store(out_ptr + ob, cb.to(tl.uint8),
                 mask=(nn > 0) & (cb_bits > 0))


def tans_unpack_group_gpu(payload, meta, sym, nb_, base, out,
                          NS: int, BMAX: int):
    """Decode one selector-group on GPU. All tensors CUDA; meta [M,5] int32.

    meta rows: byte_off, bit_len, final, n, out_base. M need not be a
    multiple of 32 (padded internally with inactive rows). Requires Triton.
    """
    if not HAS_TRITON:
        raise RuntimeError("tans GPU decode requires Triton + CUDA/ROCm")
    import torch

    if not isinstance(meta, torch.Tensor):
        meta = torch.from_numpy(np.asarray(meta)).to(torch.int32)
    meta = meta.to(torch.int32)
    pad = (-meta.shape[0]) % 32
    if pad:
        extra = torch.zeros((pad, 5), dtype=torch.int32, device=meta.device)
        extra[:, 2] = NS  # valid final state; loads clamped, stores masked
        meta = torch.cat([meta, extra], dim=0)
    grid = (meta.shape[0] // 32,)
    _tans_unpack_kernel[grid](payload, meta, sym, nb_, base, out,
                              NS=NS, BMAX=BMAX, PAY_N=len(payload),
                              num_warps=1)


def tans_lengths_group_gpu(syms, sbase, nn, freq, enck, sflat, soff,
                           lens, NS: int, BMAX: int):
    """GPU encode pass 1: bit length per block (one selector-group).

    syms [N] uint8, sbase/nn [M] int32, freq/enck [256] int32,
    sflat [NS] int32, soff [257] int32, lens [M] int32 out. All CUDA.
    """
    if not HAS_TRITON:
        raise RuntimeError("tans GPU encode requires Triton + CUDA/ROCm")
    import torch

    M = int(sbase.shape[0])
    pad = (-M) % 32
    if pad:
        dev = sbase.device
        sbase = torch.cat([sbase, torch.zeros(pad, dtype=torch.int32,
                                              device=dev)])
        nn = torch.cat([nn, torch.zeros(pad, dtype=torch.int32, device=dev)])
        buf = torch.zeros(M + pad, dtype=torch.int32, device=dev)
    else:
        buf = lens
    _tans_lengths_kernel[(sbase.shape[0] // 32,)](
        syms, sbase, nn, freq, enck, sflat, soff, buf,
        NS=NS, BMAX=BMAX, num_warps=1)
    if pad:
        lens.copy_(buf[:M])
    return lens


def tans_pack_group_gpu(syms, sbase, nn, obyte, freq, enck, sflat, soff,
                        out, NS: int, BMAX: int):
    """GPU encode pass 2: pack bits into exclusive byte ranges (one group).

    obyte [M] int32 byte offsets (prefix over ceil(bit_len/8)); ranges are
    disjoint by construction, so no atomics. All tensors CUDA.
    """
    if not HAS_TRITON:
        raise RuntimeError("tans GPU encode requires Triton + CUDA/ROCm")
    import torch

    M = int(sbase.shape[0])
    pad = (-M) % 32
    if pad:
        dev = sbase.device
        z = torch.zeros(pad, dtype=torch.int32, device=dev)
        sbase = torch.cat([sbase, z])
        nn = torch.cat([nn, z.clone()])
        obyte = torch.cat([obyte, z.clone()])
    _tans_pack_kernel[(sbase.shape[0] // 32,)](
        syms, sbase, nn, obyte, freq, enck, sflat, soff, out,
        NS=NS, BMAX=BMAX, num_warps=1)
