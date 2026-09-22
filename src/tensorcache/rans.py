"""
4-way interleaved rANS entropy codec over the u8 alphabet (12-bit precision).

Bit-exact Python port of Texel's CPU reference (`Texel/src/rans.rs`). The
normalization tie-breaks, arithmetic bounds and byte layout match the Rust
encoder exactly, so blobs are interchangeable between the two implementations
(see tests/test_rans.py, which pins cross-implementation golden vectors).

Section format (all integers little-endian):

    [u32 n]                        symbol count (decoder output length)
    [u16 n_used]                   number of (sym, freq) pairs that follow
    [n_used x (u8 sym, u16 freq)]  nonzero quantized frequencies, sum 4096
    [u32 len x 4]                  byte length of each interleaved stream
    [stream0..3]                   rANS output bytes (encoder emits back-to-front)
    [u32 final_state x 4]          encoder end states = decoder start states

Symbols split round-robin (`symbol[i]` -> stream `i % 4`); each stream is an
independent codec over the shared static table, so decode is 4 branch-
predictable scalar loops with no cross-stream dependencies.

Hot loops run under numba when available (`HAS_NUMBA`); the pure-Python
paths are the correctness reference and the fallback (also used to JIT-warm
comparisons in tests).
"""

from __future__ import annotations

import struct
from typing import List

import numpy as np

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

# Probability precision: quantized frequencies sum to 1 << SCALE_BITS.
SCALE_BITS = 12
TOT = 1 << SCALE_BITS
# Lower bound of the renormalization interval (32-bit state, 8-bit I/O).
L = 1 << 23
NSTREAMS = 4
# Decode allocation guard: absurd stored counts must Err, never OOM.
MAX_SYMBOLS = 1 << 28


class RansError(ValueError):
    """Corrupt or truncated rANS section (mirrors Rust `RansError`)."""


def normalize(counts) -> List[int]:
    """Quantize raw counts to frequencies summing to exactly TOT.

    Present symbols get >= 1; rounding drift is repaired deterministically
    (lowest index wins ties) — bit-identical to `rans.rs::normalize`.
    """
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
    # Repair drift. Decrement terminates (sum > TOT implies some freq >= 2);
    # increment always terminates (ties -> lowest index).
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
        # Increment loop always targets the same symbol: argmax keeps winning
        # after +1 (ties resolve to the lowest index), so the whole deficit
        # lands on one symbol. Apply it in one shot instead of drift*256 scans.
        best = 0
        for i in range(1, 256):
            if freq[i] > freq[best]:
                best = i
        freq[best] += TOT - s
    return freq


# ---------------------------------------------------------------------------
# numba kernels (hot loops). Signature mirrors the Rust put/get exactly.
# ---------------------------------------------------------------------------

if HAS_NUMBA:

    @njit(cache=True)
    def _decode_njit(n, sym_of, freq, start, streams_cat, soff, lens, states):
        """Decode n symbols round-robin from 4 streams.

        Streams live back-to-back in streams_cat (soff = per-stream start).
        Each lane keeps state/cursor in scalars (x0..r3) so the four renorm
        chains are independent and LLVM overlaps them — the array-indexed
        form serializes on states[j]/read[j] memory dependencies. Symbol i
        still goes to lane i & 3, so output is bit-identical.

        Returns (out, err): 0 = ok, 1 = truncated. Table validation already
        happened in the Python wrapper. Arithmetic is int64 but all values
        stay inside the Rust u32 bounds (proven in rans.rs put/get).
        """
        out = np.empty(n, dtype=np.uint8)
        x0 = states[0]; x1 = states[1]; x2 = states[2]; x3 = states[3]
        s0 = soff[0]; s1 = soff[1]; s2 = soff[2]; s3 = soff[3]
        l0 = lens[0]; l1 = lens[1]; l2 = lens[2]; l3 = lens[3]
        r0 = 0; r1 = 0; r2 = 0; r3 = 0
        for i in range(n):
            if i & 3 == 0:
                x = x0
                s = x & (TOT - 1)
                sym = sym_of[s]
                f = freq[sym]
                # f > 0: the table sums to TOT, so every slot is covered.
                x = f * (x >> SCALE_BITS) + (s - start[sym])
                while x < L:
                    if r0 >= l0:
                        return out, 1  # truncated
                    x = (x << 8) | streams_cat[s0 + r0]
                    r0 += 1
                out[i] = sym
                x0 = x
            elif i & 3 == 1:
                x = x1
                s = x & (TOT - 1)
                sym = sym_of[s]
                f = freq[sym]
                x = f * (x >> SCALE_BITS) + (s - start[sym])
                while x < L:
                    if r1 >= l1:
                        return out, 1
                    x = (x << 8) | streams_cat[s1 + r1]
                    r1 += 1
                out[i] = sym
                x1 = x
            elif i & 3 == 2:
                x = x2
                s = x & (TOT - 1)
                sym = sym_of[s]
                f = freq[sym]
                x = f * (x >> SCALE_BITS) + (s - start[sym])
                while x < L:
                    if r2 >= l2:
                        return out, 1
                    x = (x << 8) | streams_cat[s2 + r2]
                    r2 += 1
                out[i] = sym
                x2 = x
            else:
                x = x3
                s = x & (TOT - 1)
                sym = sym_of[s]
                f = freq[sym]
                x = f * (x >> SCALE_BITS) + (s - start[sym])
                while x < L:
                    if r3 >= l3:
                        return out, 1
                    x = (x << 8) | streams_cat[s3 + r3]
                    r3 += 1
                out[i] = sym
                x3 = x
        states[0] = x0; states[1] = x1; states[2] = x2; states[3] = x3
        return out, 0

    @njit(cache=True)
    def _encode_njit(data, freq, start):
        """One descending pass over data; per-lane scalar state/cursor.

        Same ILP rationale as _decode_njit: x0..x3/p0..p3 are independent
        chains LLVM can overlap; symbol i always updates lane i & 3, so the
        bytes are identical to the array-indexed formulation.

        Returns (bufs [4, cap], lens [4], finals [4]). bufs hold bytes in
        emission order — caller reverses each stream (rANS is LIFO).
        """
        n = data.shape[0]
        cap = n + 16  # amortized <= 1 B/symbol + per-stream slack
        bufs = np.zeros((4, cap), dtype=np.uint8)
        x0 = L; x1 = L; x2 = L; x3 = L
        p0 = 0; p1 = 0; p2 = 0; p3 = 0
        x_max_base = (L >> SCALE_BITS) << 8
        for i in range(n - 1, -1, -1):
            sym = data[i]
            f = freq[sym]
            x_max = x_max_base * f
            if i & 3 == 0:
                x = x0
                while x >= x_max:
                    bufs[0, p0] = np.uint8(x & 0xFF)
                    p0 += 1
                    x >>= 8
                x0 = ((x // f) << SCALE_BITS) + (x % f) + start[sym]
            elif i & 3 == 1:
                x = x1
                while x >= x_max:
                    bufs[1, p1] = np.uint8(x & 0xFF)
                    p1 += 1
                    x >>= 8
                x1 = ((x // f) << SCALE_BITS) + (x % f) + start[sym]
            elif i & 3 == 2:
                x = x2
                while x >= x_max:
                    bufs[2, p2] = np.uint8(x & 0xFF)
                    p2 += 1
                    x >>= 8
                x2 = ((x // f) << SCALE_BITS) + (x % f) + start[sym]
            else:
                x = x3
                while x >= x_max:
                    bufs[3, p3] = np.uint8(x & 0xFF)
                    p3 += 1
                    x >>= 8
                x3 = ((x // f) << SCALE_BITS) + (x % f) + start[sym]
        lens = np.empty(4, dtype=np.int64)
        lens[0] = p0; lens[1] = p1; lens[2] = p2; lens[3] = p3
        xs = np.empty(4, dtype=np.int64)
        xs[0] = x0; xs[1] = x1; xs[2] = x2; xs[3] = x3
        return bufs, lens, xs

else:
    _decode_njit = None
    _encode_njit = None


def _freq_start(counts):
    """counts -> (q, freq, start, used) shared by both encode paths."""
    q = normalize(counts)
    freq = [0] * 256
    start = [0] * 256
    used = []
    cum = 0
    for s in range(256):
        start[s] = cum
        freq[s] = q[s]
        cum += q[s]
        if q[s] > 0:
            used.append(s)
    return q, freq, start, used


def encode(data: bytes) -> bytes:
    """Encode a byte string into a self-describing rANS section blob.

    Deterministic and byte-identical to Texel's Rust `rans::encode`.
    """
    # Histogram via numpy (C loop, GIL-free): the pure-Python byte loop was
    # ~74% of encode() even on the numba path. Same counts -> same bitstream.
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8),
                         minlength=256).tolist()
    q, freq, start, used = _freq_start(counts)
    n = len(data)

    if HAS_NUMBA:
        arr = np.frombuffer(data, dtype=np.uint8)
        bufs, lens, finals = _encode_njit(
            arr,
            np.asarray(freq, dtype=np.int64),
            np.asarray(start, dtype=np.int64),
        )
        streams = [bufs[j, : lens[j]][::-1].tobytes() for j in range(NSTREAMS)]
        final_states = [int(finals[j]) for j in range(NSTREAMS)]
    else:
        # Pure-Python reference: one descending pass, per-stream buffers.
        xs = [L] * NSTREAMS
        bufs_py: List[List[int]] = [[], [], [], []]
        for i in range(n - 1, -1, -1):
            sym = data[i]
            j = i % NSTREAMS
            f = freq[sym]
            x = xs[j]
            x_max = ((L >> SCALE_BITS) << 8) * f
            buf = bufs_py[j]
            while x >= x_max:
                buf.append(x & 0xFF)
                x >>= 8
            xs[j] = ((x // f) << SCALE_BITS) + (x % f) + start[sym]
        streams = [bytes(reversed(b)) for b in bufs_py]
        final_states = xs

    out = bytearray()
    out += struct.pack("<I", n)
    out += struct.pack("<H", len(used))
    for s in used:
        out.append(s)
        out += struct.pack("<H", q[s])
    for st in streams:
        out += struct.pack("<I", len(st))
    for st in streams:
        out += st
    for x in final_states:
        out += struct.pack("<I", x)
    return bytes(out)


def decode(blob: bytes) -> bytes:
    """Decode a rANS section blob produced by `encode` (or Rust `rans::encode`).

    Raises RansError on truncated or corrupt input — mirrors the Rust error
    cases exactly (never hangs, never panics).
    """
    m = len(blob)
    if m < 4:
        raise RansError("truncated rans section")
    (n,) = struct.unpack_from("<I", blob, 0)
    if n == 0:
        return b""
    if n > MAX_SYMBOLS:
        raise RansError(f"absurd symbol count {n}")
    if m < 6:
        raise RansError("truncated rans section")
    (n_used,) = struct.unpack_from("<H", blob, 4)
    if n_used == 0 or n_used > 256:
        raise RansError(f"bad table size {n_used}")
    pos = 6
    if m < pos + 3 * n_used:
        raise RansError("truncated rans section")
    freq = [0] * 256
    seen = [False] * 256
    for _ in range(n_used):
        s = blob[pos]
        f = blob[pos + 1] | (blob[pos + 2] << 8)  # u16 LE, no struct overhead
        pos += 3
        if seen[s]:
            raise RansError(f"duplicate symbol {s}")
        if f == 0 or f > TOT:
            raise RansError(f"bad frequency {f}")
        seen[s] = True
        freq[s] = f
    start = [0] * 256
    cum = 0
    for s in range(256):
        start[s] = cum
        cum += freq[s]
    if cum != TOT:
        raise RansError(f"frequencies sum to {cum}")
    if m < pos + 4 * NSTREAMS:
        raise RansError("truncated rans section")
    lens = [0] * NSTREAMS
    for j in range(NSTREAMS):
        lens[j] = (blob[pos] | (blob[pos + 1] << 8)
                   | (blob[pos + 2] << 16) | (blob[pos + 3] << 24))
        pos += 4
    total_streams = sum(lens)
    if m < pos + total_streams + 4 * NSTREAMS:
        raise RansError("truncated rans section")
    streams_cat = blob[pos : pos + total_streams]
    pos += total_streams
    states = [0] * NSTREAMS
    for j in range(NSTREAMS):
        states[j] = (blob[pos] | (blob[pos + 1] << 8)
                     | (blob[pos + 2] << 16) | (blob[pos + 3] << 24))
        pos += 4

    # Decode table: slot -> symbol (full coverage: frequencies sum to TOT).
    # np.repeat(symbol ids, frequencies) builds the same mapping vectorized;
    # the naive double loop costs ~110us/call in Python — half the decode.
    sym_of = np.repeat(np.arange(256, dtype=np.uint8),
                       np.asarray(freq, dtype=np.intp))

    if HAS_NUMBA:
        out, err = _decode_njit(
            n,
            sym_of,
            np.asarray(freq, dtype=np.int64),
            np.asarray(start, dtype=np.int64),
            np.frombuffer(streams_cat, dtype=np.uint8),
            np.asarray(_stream_offsets(lens), dtype=np.int64),
            np.asarray(lens, dtype=np.int64),
            np.asarray(states, dtype=np.int64),
        )
        if err:
            raise RansError("truncated rans section")
        return out.tobytes()

    # Pure-Python reference decode (correctness path; mirrors Rust get()).
    soff = _stream_offsets(lens)
    out_b = bytearray(n)
    read = [0] * NSTREAMS
    for i in range(n):
        j = i % NSTREAMS
        x = states[j]
        s = x & (TOT - 1)
        sym = sym_of[s]
        f = freq[sym]
        x = f * (x >> SCALE_BITS) + (s - start[sym])
        while x < L:
            if read[j] >= lens[j]:
                raise RansError("truncated rans section")
            x = (x << 8) | streams_cat[soff[j] + read[j]]
            read[j] += 1
        out_b[i] = sym
        states[j] = x
    return bytes(out_b)


def _stream_offsets(lens):
    soff = [0] * NSTREAMS
    acc = 0
    for j in range(NSTREAMS):
        soff[j] = acc
        acc += lens[j]
    return soff


# ---------------------------------------------------------------------------
# Section packaging (storage layer used by PixelCacheWriter / -Dataset)
# ---------------------------------------------------------------------------

# Below this size the transmitted tables + stream/state headers (~44 B) outweigh
# the entropy win — store raw (Texel's envelope rule).
MIN_SECTION_BYTES = 48
# ...and a bare win isn't enough either: decoding costs ~0.1-0.2 ms/img, so a
# section must pay for that CPU time. Requiring >=8% keeps u8's measured -4%
# out (74us of decode for ~0.5% of total bytes) while i8 (-49%) and the ll4
# planes (-20..-44%) sail through.
MIN_GAIN_PCT = 8


def pack_section(raw: bytes) -> "tuple[int, bytes]":
    """(method, stored) for one cache section; 0 = raw, 1 = rANS.

    Falls back to raw for tiny sections, for payloads rANS doesn't shrink
    at all, and for gains below MIN_GAIN_PCT (decode cost > byte savings).
    """
    if len(raw) <= MIN_SECTION_BYTES:
        return 0, raw
    blob = encode(raw)
    if len(blob) * 100 >= (100 - MIN_GAIN_PCT) * len(raw):
        return 0, raw
    return 1, blob


def unpack_section(method: int, blob) -> bytes:
    """Inverse of pack_section. Accepts any bytes-like (mmap slices)."""
    if method == 0:
        return bytes(blob)
    if method == 1:
        return decode(bytes(blob))
    raise RansError(f"unknown section method {method}")


def pack_byte_planar(raw: bytes) -> "tuple[int, bytes]":
    """Byte-planar split + per-plane rANS for 16-bit LE element streams.

    Splits [lo,hi,lo,hi,...] into a low plane and a high plane and codes
    each separately (Texel v11 style): the high plane is peaked near zero,
    the low plane near uniform, and coding them apart beats one mixed
    stream by a wide margin. Each plane gets its own [method][u32 len]
    wrapper. Returns (0, raw) — the dense v1 layout — when neither plane
    benefits or the wrapper would grow the payload.
    """
    lo, hi = raw[0::2], raw[1::2]
    m0, b0 = pack_section(lo)
    m1, b1 = pack_section(hi)
    if m0 == 0 and m1 == 0:
        return 0, raw
    wrapped = (struct.pack("<BI", m0, len(b0)) + b0
               + struct.pack("<BI", m1, len(b1)) + b1)
    if len(wrapped) >= len(raw):
        return 0, raw
    return 1, wrapped


def unpack_byte_planar(method: int, blob) -> bytes:
    """Inverse of pack_byte_planar -> dense [lo,hi,lo,hi,...] bytes."""
    if method == 0:
        return bytes(blob)
    if method != 1:
        raise RansError(f"unknown planar section method {method}")
    stored = bytes(blob)
    m0 = stored[0]
    n0 = int.from_bytes(stored[1:5], "little")
    b0 = unpack_section(m0, stored[5:5 + n0])
    p = 5 + n0
    m1 = stored[p]
    n1 = int.from_bytes(stored[p + 1:p + 5], "little")
    b1 = unpack_section(m1, stored[p + 5:p + 5 + n1])
    if len(b0) != len(b1):
        raise RansError(f"planar planes differ in length: {len(b0)} vs {len(b1)}")
    out = bytearray(len(b0) + len(b1))
    out[0::2] = b0
    out[1::2] = b1
    return bytes(out)
