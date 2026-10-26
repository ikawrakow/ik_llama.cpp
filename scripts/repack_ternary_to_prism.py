#!/usr/bin/env python3
"""
Lossless repack of upstream llama.cpp ternary GGUFs to the fork's Prism ternary
formats:  TQ1_0 (34) -> PTQ1_0 (143)  and  TQ2_0 (35) -> PQ2_0 (142), QK 256 -> 128.

Both formats carry the same data (one fp16 scale + trits {-1,0,+1}); only the
packing differs, so we copy the 256-block scale into the two 128-blocks. No
re-quantisation: the decoded values are bit-for-bit identical.

Usage: python3 repack_ternary_to_prism.py input.gguf output.gguf
"""

import struct
import sys
import shutil
import os

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

# --- GGUF constants ---------------------------------------------------------
GGUF_MAGIC = b"GGUF"
GGML_TYPE_TQ1_0 = 34   # mainline ternary, QK = 256, 54 bytes / block
GGML_TYPE_TQ2_0 = 35   # mainline ternary, QK = 256, 66 bytes / block
GGML_TYPE_PQ2_0 = 142  # ik_llama.cpp Prism ternary, QK = 128, 34 bytes / block
GGML_TYPE_PTQ1_0 = 143 # ik_llama.cpp Prism ternary, QK = 128, 28 bytes / block

QK_TQ1_0 = 256
QK_TQ2_0 = 256
QK_PQ2_0 = 128
QK_PTQ1_0 = 128
BLOCK_TQ1_0 = 54  # uint8_t qs[48]; uint8_t qh[4]; ggml_half d;
BLOCK_TQ2_0 = 66  # uint8_t qs[64]; ggml_half d;
BLOCK_PQ2_0 = 34  # ggml_half d; uint8_t qs[32];
BLOCK_PTQ1_0 = 28 # uint8_t qs[24]; uint8_t qh[2]; ggml_half d;

FTYPE_MOSTLY_PQ2_0  = 148
FTYPE_MOSTLY_PTQ1_0 = 149

# input type -> (output type, qk_in, qk_out, block_in, block_out, ftype)
TERNARY = {
    GGML_TYPE_TQ1_0: (GGML_TYPE_PTQ1_0, QK_TQ1_0, QK_PTQ1_0, BLOCK_TQ1_0, BLOCK_PTQ1_0, FTYPE_MOSTLY_PTQ1_0),
    GGML_TYPE_TQ2_0: (GGML_TYPE_PQ2_0,  QK_TQ2_0, QK_PQ2_0, BLOCK_TQ2_0, BLOCK_PQ2_0, FTYPE_MOSTLY_PQ2_0),
}

# gguf value type ids
(U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64) = range(13)

_SCALAR = {U8: 'B', I8: 'b', U16: 'H', I16: 'h', U32: 'I', I32: 'i',
           F32: 'f', BOOL: 'B', U64: 'Q', I64: 'q', F64: 'd'}
_SIZEOF = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4,
           BOOL: 1, U64: 8, I64: 8, F64: 8}

POW3 = (1, 3, 9, 27, 81, 243)


def _fp16_to_fp32(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


# --- TQ1_0 (256) <-> trits --------------------------------------------------
def tq1_0_decode_block(block):
    """block: 54 bytes -> (trits: list[int] len 256 in {0,1,2}, d_half: int)."""
    qs = block[0:48]
    qh = block[48:52]
    d = block[52] | (block[53] << 8)
    out = [0] * 256
    p = 0
    # qs: stages 32 then 16 (matches quantize_row_tq1_0_ref)
    j = 0
    while j < (48 - 48 % 32):          # j = 0, c = 32
        c = 32
        for n in range(5):
            for m in range(c):
                q = (qs[j + m] * POW3[n]) & 0xFF
                out[p] = ((q * 3) & 0xFFFF) >> 8
                p += 1
        j += c
    while j < 48:                       # j = 32, c = 16
        c = 16
        for n in range(5):
            for m in range(c):
                q = (qs[j + m] * POW3[n]) & 0xFF
                out[p] = ((q * 3) & 0xFFFF) >> 8
                p += 1
        j += c
    # qh: 4 bytes, 4 trits each
    for n in range(4):
        for jj in range(4):
            q = (qh[jj] * POW3[n]) & 0xFF
            out[p] = ((q * 3) & 0xFFFF) >> 8
            p += 1
    return out, d


def ptq1_0_encode_block(trits, d_half):
    """trits: sequence of 128 values in {0,1,2}; d_half: fp16 bits -> 28 bytes."""
    qs = bytearray(24)
    qh = bytearray(2)
    p = 0
    j = 0
    for c in (16, 8):                   # ptq1_0_stages = {16, 8}
        while j + c <= 24:
            for m in range(c):
                q = 0
                for n in range(5):
                    q = q * 3 + trits[p + m + n * c]
                q = (q * 256 + 242) // 243
                qs[j + m] = q & 0xFF
            p += 5 * c
            j += c
    for h in range(2):
        q = 0
        for m in range(4):
            q = q * 3 + trits[p + h + m * 2]
        q *= 3
        q = (q * 256 + 242) // 243
        qh[h] = q & 0xFF
    p += 8
    assert p == 128, p
    return bytes(qs) + bytes(qh) + struct.pack('<H', d_half)


def repack_tensor_tq1_0_to_ptq1_0(raw, n_elements):
    """raw: bytes of a TQ1_0 tensor (all blocks). Returns bytes of PTQ1_0."""
    assert n_elements % QK_TQ1_0 == 0, n_elements
    nb = n_elements // QK_TQ1_0
    assert len(raw) == nb * BLOCK_TQ1_0, (len(raw), nb * BLOCK_TQ1_0)
    if HAVE_NP:
        return _repack_np(raw, nb)
    out = bytearray(nb * 2 * BLOCK_PTQ1_0)
    for i in range(nb):
        trits, d = tq1_0_decode_block(raw[i * BLOCK_TQ1_0:(i + 1) * BLOCK_TQ1_0])
        o = i * 2 * BLOCK_PTQ1_0
        out[o:o + BLOCK_PTQ1_0] = ptq1_0_encode_block(trits[:QK_PTQ1_0], d)
        out[o + BLOCK_PTQ1_0:o + 2 * BLOCK_PTQ1_0] = ptq1_0_encode_block(trits[QK_PTQ1_0:], d)
    return bytes(out)


# --- numpy vectorised implementation (identical result, ~1e6x faster) ------
_TBL = None


def _trit_table():
    global _TBL
    if _TBL is None:
        b = np.arange(256, dtype=np.int64)
        t = np.zeros((256, 5), np.uint8)
        for n in range(5):
            t[:, n] = (((b * POW3[n]) & 0xFF) * 3) >> 8
        _TBL = t
    return _TBL


def _encode_np(tr, d, out):
    """tr: (nb,128) uint8 trits 0..2; d: (nb,2) fp16 bytes; out: (nb,28) uint8."""
    tr = tr.astype(np.int64)

    def pack(idx):
        packed = np.zeros(tr.shape[0], np.int64)
        for n in idx:
            packed = packed * 3 + tr[:, n]
        return ((packed * 256 + 242) // 243).astype(np.uint8)

    for m in range(16):
        out[:, m] = pack([m + n * 16 for n in range(5)])
    for m in range(8):
        out[:, 16 + m] = pack([80 + m + n * 8 for n in range(5)])
    for h in range(2):
        packed = np.zeros(tr.shape[0], np.int64)
        for m in range(4):
            packed = packed * 3 + tr[:, 120 + h + m * 2]
        packed *= 3
        out[:, 24 + h] = ((packed * 256 + 242) // 243).astype(np.uint8)
    out[:, 26:28] = d


def _tq1_0_decode_np(blk):
    """blk: (nb,54) uint8 -> (nb,256) uint8 trits."""
    qs = blk[:, :48]
    qh = blk[:, 48:52]
    tbl = _trit_table()
    e = np.empty((blk.shape[0], 256), np.uint8)
    for m in range(32):
        t = tbl[qs[:, m]]
        for n in range(5):
            e[:, m + n * 32] = t[:, n]
    for m in range(16):
        t = tbl[qs[:, 32 + m]]
        for n in range(5):
            e[:, 160 + m + n * 16] = t[:, n]
    for j in range(4):
        t = tbl[qh[:, j]]
        for n in range(4):
            e[:, 240 + j + n * 4] = t[:, n]
    return e


def _repack_np(raw, nb):
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, BLOCK_TQ1_0)
    e = _tq1_0_decode_np(blk)
    d = blk[:, 52:54]
    out = np.empty((nb, 2 * BLOCK_PTQ1_0), np.uint8)
    _encode_np(e[:, :128], d, out[:, :BLOCK_PTQ1_0])
    _encode_np(e[:, 128:], d, out[:, BLOCK_PTQ1_0:])
    return out.tobytes()


# --- TQ2_0 (256) <-> trits (2 bits/trit) ------------------------------------
def tq2_0_decode_block(block):
    """block: 66 bytes (qs[64], d) -> (trits len 256 in {0,1,2}, d_half)."""
    qs = block[0:64]
    d = block[64] | (block[65] << 8)
    out = [0] * 256
    p = 0
    for j in range(0, 64, 32):        # 2 groups of 32 bytes = 128 elements each
        for n in range(4):
            for m in range(32):
                out[p] = (qs[j + m] >> (2 * n)) & 3
                p += 1
    return out, d


def pq2_0_encode_block(trits, d_half):
    """trits: 128 values in {0,1,2} -> 34 bytes (d, qs[32])."""
    qs = bytearray(32)
    for j in range(128):
        qs[j // 4] |= (trits[j] & 3) << ((j % 4) * 2)
    return struct.pack('<H', d_half) + bytes(qs)


def repack_tensor_tq2_0_to_pq2_0(raw, n_elements):
    assert n_elements % QK_TQ2_0 == 0, n_elements
    nb = n_elements // QK_TQ2_0
    assert len(raw) == nb * BLOCK_TQ2_0, (len(raw), nb * BLOCK_TQ2_0)
    if HAVE_NP:
        return _repack_tq2_np(raw, nb)
    out = bytearray(nb * 2 * BLOCK_PQ2_0)
    for i in range(nb):
        trits, d = tq2_0_decode_block(raw[i * BLOCK_TQ2_0:(i + 1) * BLOCK_TQ2_0])
        o = i * 2 * BLOCK_PQ2_0
        out[o:o + BLOCK_PQ2_0] = pq2_0_encode_block(trits[:QK_PQ2_0], d)
        out[o + BLOCK_PQ2_0:o + 2 * BLOCK_PQ2_0] = pq2_0_encode_block(trits[QK_PQ2_0:], d)
    return bytes(out)


def _tq2_0_decode_np(blk):
    qs = blk[:, :64]
    e = np.empty((blk.shape[0], 256), np.uint8)
    for j0 in (0, 32):
        for n in range(4):
            for m in range(32):
                e[:, (j0 // 32) * 128 + n * 32 + m] = (qs[:, j0 + m] >> (2 * n)) & 3
    return e


def _pq2_0_encode_np(tr, d, out):
    out[:, 0:2] = d
    qs = np.zeros((tr.shape[0], 32), np.uint8)
    for j in range(128):
        qs[:, j // 4] |= (tr[:, j] & 3) << ((j % 4) * 2)
    out[:, 2:34] = qs


def _repack_tq2_np(raw, nb):
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, BLOCK_TQ2_0)
    e = _tq2_0_decode_np(blk)
    d = blk[:, 64:66]
    out = np.empty((nb, 2 * BLOCK_PQ2_0), np.uint8)
    _pq2_0_encode_np(e[:, :128], d, out[:, :BLOCK_PQ2_0])
    _pq2_0_encode_np(e[:, 128:], d, out[:, BLOCK_PQ2_0:])
    return out.tobytes()


# --- GGUF reader/writer -----------------------------------------------------
class GGUF:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.version = None
        self.kv = []           # list of (key, type, value)
        self.tensors = []      # list of dict(name,n_dims,dims,type,offset)
        self.alignment = 32
        self.data_start = 0
        self._parse()

    def _rs(self):
        n = struct.unpack('<Q', self.f.read(8))[0]
        return self.f.read(n)

    def _rv(self, t):
        if t == STR:
            return self._rs()
        if t == ARR:
            et = struct.unpack('<I', self.f.read(4))[0]
            n = struct.unpack('<Q', self.f.read(8))[0]
            return (et, [self._rv(et) for _ in range(n)])
        if t in _SCALAR:
            return struct.unpack('<' + _SCALAR[t], self.f.read(_SIZEOF[t]))[0]
        raise ValueError('bad gguf value type %d' % t)

    def _parse(self):
        if self.f.read(4) != GGUF_MAGIC:
            raise ValueError('not a GGUF file')
        self.version = struct.unpack('<I', self.f.read(4))[0]
        n_tensors = struct.unpack('<Q', self.f.read(8))[0]
        n_kv = struct.unpack('<Q', self.f.read(8))[0]
        for _ in range(n_kv):
            k = self._rs().decode('utf-8')
            t = struct.unpack('<I', self.f.read(4))[0]
            v = self._rv(t)
            self.kv.append((k, t, v))
            if k == 'general.alignment':
                self.alignment = v
        for _ in range(n_tensors):
            name = self._rs().decode('utf-8')
            n_dims = struct.unpack('<I', self.f.read(4))[0]
            dims = [struct.unpack('<Q', self.f.read(8))[0] for _ in range(n_dims)]
            typ = struct.unpack('<I', self.f.read(4))[0]
            off = struct.unpack('<Q', self.f.read(8))[0]
            self.tensors.append(dict(name=name, dims=dims, type=typ, offset=off))
        hdr_end = self.f.tell()
        self.data_start = (hdr_end + self.alignment - 1) // self.alignment * self.alignment


def _wv(f, t, v):
    if t == STR:
        b = v if isinstance(v, bytes) else v.encode('utf-8')
        f.write(struct.pack('<Q', len(b))); f.write(b); return
    if t == ARR:
        et, items = v
        f.write(struct.pack('<I', et)); f.write(struct.pack('<Q', len(items)))
        for it in items:
            _wv(f, et, it)
        return
    f.write(struct.pack('<' + _SCALAR[t], v))


def _ws(f, s):
    b = s if isinstance(s, bytes) else s.encode('utf-8')
    f.write(struct.pack('<Q', len(b))); f.write(b)


def n_elements(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--selftest':
        return selftest()
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    inp, outp = sys.argv[1], sys.argv[2]

    g = GGUF(inp)
    print(f"input: {inp}  version={g.version}  tensors={len(g.tensors)}  alignment={g.alignment}")

    # rebuild tensor list, converting TQ1_0 -> PTQ1_0 and/or TQ2_0 -> PQ2_0
    new_tensors = []
    total_in = total_out = 0
    ftype_out = None
    for t in g.tensors:
        dims = t['dims']
        ne = n_elements(dims)
        if t['type'] in TERNARY:
            out_type, qk_in, qk_out, blk_in, blk_out, ftype = TERNARY[t['type']]
            assert ne % qk_in == 0, (t['name'], dims)
            new_tensors.append(dict(name=t['name'], dims=dims, type=out_type,
                                    src_off=t['offset'], src_type=t['type'],
                                    size=ne // qk_out * blk_out))
            total_in  += ne // qk_in  * blk_in
            total_out += ne // qk_out * blk_out
            if ftype_out is None:
                ftype_out = ftype
        else:
            new_tensors.append(dict(name=t['name'], dims=dims, type=t['type'],
                                    src_off=t['offset'], src_type=t['type'], size=None))

    # compute original sizes from offset ordering
    order = sorted(range(len(g.tensors)), key=lambda i: g.tensors[i]['offset'])
    for k, i in enumerate(order):
        nxt = g.tensors[order[k + 1]]['offset'] if k + 1 < len(order) else None
        g.tensors[i]['orig_size'] = (nxt - g.tensors[i]['offset']) if nxt is not None else None
    # last tensor size: derive from file size
    file_size = os.path.getsize(inp)
    last = order[-1]
    g.tensors[last]['orig_size'] = file_size - g.data_start - g.tensors[last]['offset']

    # assign new offsets / sizes
    off = 0
    for nt in new_tensors:
        src = next(t for t in g.tensors if t['name'] == nt['name'])
        nt['orig_size'] = src['orig_size']
        nt['new_size'] = nt['size'] if nt['size'] is not None else src['orig_size']
        nt['new_off'] = off
        off += nt['new_size']
        if off % g.alignment:
            off += g.alignment - (off % g.alignment)

    # write header + metadata
    with open(outp, 'wb') as fo:
        fo.write(GGUF_MAGIC)
        fo.write(struct.pack('<I', g.version))
        fo.write(struct.pack('<Q', len(new_tensors)))
        fo.write(struct.pack('<Q', len(g.kv)))
        for k, t, v in g.kv:
            _ws(fo, k)
            if k == 'general.file_type' and t == U32 and ftype_out is not None:
                v = ftype_out
            fo.write(struct.pack('<I', t))
            _wv(fo, t, v)
        for nt in new_tensors:
            _ws(fo, nt['name'])
            fo.write(struct.pack('<I', len(nt['dims'])))
            for d in nt['dims']:
                fo.write(struct.pack('<Q', d))
            fo.write(struct.pack('<I', nt['type']))
            fo.write(struct.pack('<Q', nt['new_off']))
        hdr_end = fo.tell()
        pad = (g.alignment - (hdr_end % g.alignment)) % g.alignment
        fo.write(b'\x00' * pad)
        data_start = hdr_end + pad
        assert data_start == g.alignment * ((hdr_end + g.alignment - 1) // g.alignment)

        # write tensor data
        for nt in new_tensors:
            while fo.tell() < data_start + nt['new_off']:
                fo.write(b'\x00')
            g.f.seek(g.data_start + nt['src_off'])
            raw = g.f.read(nt['orig_size'])
            if nt['src_type'] == GGML_TYPE_TQ1_0:
                out = repack_tensor_tq1_0_to_ptq1_0(raw, n_elements(nt['dims']))
            elif nt['src_type'] == GGML_TYPE_TQ2_0:
                out = repack_tensor_tq2_0_to_pq2_0(raw, n_elements(nt['dims']))
            else:
                out = raw
            assert len(out) == nt['new_size'], (nt['name'], len(out), nt['new_size'])
            fo.write(out)
    print(f"output: {outp}  wrote {off} bytes of tensor data "
          f"({total_in} -> {total_out} for ternary parts)")
    return 0


def selftest():
    """Synthetic round-trip: random trits -> TQ1_0 -> decode -> PTQ1_0 -> decode."""
    import random
    random.seed(1234)
    for _ in range(200):
        d = random.choice([0.0, 0.5, 1.0, 0.125, 7.0])
        dh = struct.unpack('<H', struct.pack('<e', d))[0]
        # random 256 trits (avoid all-zero scale issues)
        trits = [random.randrange(3) for _ in range(256)]
        # encode TQ1_0 block (mimic quantize_row_tq1_0_ref with xi = trits)
        qs = bytearray(48); qh = bytearray(4); p = 0; j = 0
        while j < 32:
            for m in range(32):
                q = 0
                for n in range(5):
                    q = q * 3 + trits[p + m + n * 32]
                q = (q * 256 + 242) // 243
                qs[j + m] = q & 0xFF
            p += 160; j += 32
        while j < 48:
            for m in range(16):
                q = 0
                for n in range(5):
                    q = q * 3 + trits[p + m + n * 16]
                q = (q * 256 + 242) // 243
                qs[j + m] = q & 0xFF
            p += 80; j += 16
        for jj in range(4):
            q = 0
            for m in range(4):
                q = q * 3 + trits[p + jj + m * 4]
            q *= 3
            q = (q * 256 + 242) // 243
            qh[jj] = q & 0xFF
        p += 16
        assert p == 256
        tq = bytes(qs) + bytes(qh) + struct.pack('<H', dh)
        dec, d2 = tq1_0_decode_block(tq)
        assert dec == trits, ("TQ1_0 decode mismatch", dec[:10], trits[:10])
        assert d2 == dh
        # repack to PTQ1_0 and decode back
        out = repack_tensor_tq1_0_to_ptq1_0(tq, 256)
        assert len(out) == 2 * BLOCK_PTQ1_0
        # decode PTQ1_0 block A and B
        def dec_ptq(blk):
            qs = blk[0:24]; qh = blk[24:26]; dd = blk[26] | (blk[27] << 8)
            res = [0] * 128; pp = 0; jj = 0
            for c in (16, 8):
                while jj + c <= 24:
                    for n in range(5):
                        for m in range(c):
                            q = (qs[jj + m] * POW3[n]) & 0xFF
                            res[pp] = ((q * 3) & 0xFFFF) >> 8
                            pp += 1
                    jj += c
            for n in range(4):
                for h in range(2):
                    q = (qh[h] * POW3[n]) & 0xFF
                    res[pp] = ((q * 3) & 0xFFFF) >> 8
                    pp += 1
            return res, dd
        a, da = dec_ptq(out[0:BLOCK_PTQ1_0])
        b, db = dec_ptq(out[BLOCK_PTQ1_0:2 * BLOCK_PTQ1_0])
        assert a == trits[:128], ("PTQ1_0 A mismatch", a[:10], trits[:10])
        assert b == trits[128:], ("PTQ1_0 B mismatch", b[:10], trits[128:130])
        assert da == dh and db == dh
        # float equivalence
        fa = [_fp16_to_fp32(da) * (x - 1) for x in a]
        fb = [_fp16_to_fp32(db) * (x - 1) for x in b]
        ft = [_fp16_to_fp32(dh) * (x - 1) for x in trits]
        assert fa + fb == ft
    # TQ2_0 -> PQ2_0 round trip
    for _ in range(200):
        dh = struct.unpack('<H', struct.pack('<e', random.choice([0.0, 0.5, 1.0, 0.125, 7.0])))[0]
        trits = [random.randrange(3) for _ in range(256)]
        qs = bytearray(64); p = 0
        for j in range(0, 64, 32):
            for n in range(4):
                for m in range(32):
                    qs[j + m] |= (trits[p] & 3) << (2 * n); p += 1
        tq2 = bytes(qs) + struct.pack('<H', dh)
        dec, d2 = tq2_0_decode_block(tq2)
        assert dec == trits, ("TQ2 decode mismatch", dec[:8], trits[:8])
        assert d2 == dh
        out = repack_tensor_tq2_0_to_pq2_0(tq2, 256)
        assert len(out) == 2 * BLOCK_PQ2_0
        def dec_pq2(blk):
            dd = blk[0] | (blk[1] << 8); res = [0] * 128
            for j in range(128):
                res[j] = (blk[2 + j // 4] >> ((j % 4) * 2)) & 3
            return res, dd
        a, da = dec_pq2(out[:BLOCK_PQ2_0])
        b, db = dec_pq2(out[BLOCK_PQ2_0:])
        assert a == trits[:128], ("PQ2 A mismatch", a[:8], trits[:8])
        assert b == trits[128:], ("PQ2 B mismatch", b[:8], trits[128:136])
        assert da == dh and db == dh
        fa = [_fp16_to_fp32(da) * (x - 1) for x in a]
        fb = [_fp16_to_fp32(db) * (x - 1) for x in b]
        ft = [_fp16_to_fp32(dh) * (x - 1) for x in trits]
        assert fa + fb == ft
    print("selftest OK: 200 random blocks, TQ1_0 -> PTQ1_0 and TQ2_0 -> PQ2_0 lossless (trits + scale + floats)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
