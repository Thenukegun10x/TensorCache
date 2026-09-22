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
    while s < TOT:
        best = 0
        for i in range(1, 256):
            if freq[i] > freq[best]:
                best = i
        freq[best] += 1
        s += 1
    return freq


# ---------------------------------------------------------------------------
# numba kernels (hot loops). Signature mirrors the Rust put/get exactly.
# ---------------------------------------------------------------------------

if HAS_NUMBA:

    @njit(cache=True)
    def _decode_njit(n, sym_of, freq, start, streams_cat, soff, lens, states):
        """Decode n symbols round-robin from 4 streams.

        streams_cat holds all 4 stream byte ranges back-to-back (soff = per-
        stream start offset). Returns (out, err): 0 = ok, 1 = truncated.
        Table validation already happened in the Python wrapper. Arithmetic
        is int64 but all values stay inside the Rust u32 bounds (proven in
        rans.rs put/get), so results are bit-identical.
        """
        out = np.empty(n, dtype=np.uint8)
        read = np.zeros(4, dtype=np.int64)
        for i in range(n):
            j = i & 3
            x = states[j]
            s = x & (TOT - 1)
            sym = sym_of[s]
            f = freq[sym]
            # f > 0: the table sums to TOT, so every slot is covered.
            x = f * (x >> SCALE_BITS) + (s - start[sym])
            while x < L:
                if read[j] >= lens[j]:
                    return out, 1  # truncated
                x = (x << 8) | streams_cat[soff[j] + read[j]]
                read[j] += 1
            out[i] = sym
            states[j] = x
        return out, 0

    @njit(cache=True)
    def _encode_njit(data, freq, start):
        """One descending pass over data; per-stream emission buffers.

        Returns (bufs [4, cap], lens [4], finals [4]). bufs hold bytes in
        emission order — caller reverses each stream (rANS is LIFO).
        """
        n = data.shape[0]
        cap = n + 16  # amortized <= 1 B/symbol + per-stream slack
        bufs = np.zeros((4, cap), dtype=np.uint8)
        lens = np.zeros(4, dtype=np.int64)
        xs = np.full(4, L, dtype=np.int64)
        x_max_base = (L >> SCALE_BITS) << 8
        for i in range(n - 1, -1, -1):
            j = i & 3
            sym = data[i]
            f = freq[sym]
            x = xs[j]
            x_max = x_max_base * f
            p = lens[j]
            while x >= x_max:
                bufs[j, p] = np.uint8(x & 0xFF)
                p += 1
                x >>= 8
            lens[j] = p
            xs[j] = ((x // f) << SCALE_BITS) + (x % f) + start[sym]
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
    counts = [0] * 256
    for b in data:
        counts[b] += 1
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
        (f,) = struct.unpack_from("<H", blob, pos + 1)
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
        (lens[j],) = struct.unpack_from("<I", blob, pos)
        pos += 4
    total_streams = sum(lens)
    if m < pos + total_streams + 4 * NSTREAMS:
        raise RansError("truncated rans section")
    streams_cat = blob[pos : pos + total_streams]
    pos += total_streams
    states = [0] * NSTREAMS
    for j in range(NSTREAMS):
        (states[j],) = struct.unpack_from("<I", blob, pos)
        pos += 4

    # Decode table: slot -> symbol (full coverage: frequencies sum to TOT).
    sym_of = bytearray(TOT)
    for s in range(256):
        for k in range(start[s], start[s] + freq[s]):
            sym_of[k] = s

    if HAS_NUMBA:
        out, err = _decode_njit(
            n,
            np.frombuffer(sym_of, dtype=np.uint8),
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
