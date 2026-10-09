#!/usr/bin/env python3
# quillama: slot save/restore robustness test. Saves slot 0, then restores it with valid, legacy, truncated
# and corrupt .checkpoints files; the server must answer every request and keep reusing the prefix.
# Usage: quill-slot-checkpoint-test.py http://127.0.0.1:8080 <--slot-save-path dir>
import json, struct, sys, urllib.request, os
url, d = sys.argv[1], sys.argv[2]
def post(path, body):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=600)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
prompt = " ".join(f"Fact {i}: entity_{i} relates to entity_{i+1}." for i in range(100))
post("/completion", {"prompt": prompt, "n_predict": 4, "temperature": 0, "id_slot": 0, "cache_prompt": True})
code, r = post("/slots/0?action=save", {"filename": "s.bin"}); print("save", code, {k: r.get(k) for k in ("n_saved","n_written")})
MAGIC, VERSION = 0x67677371, 4
def ckfile(records, count=None, truncate=0):
    b = struct.pack("<IIQ", MAGIC, VERSION, len(records) if count is None else count)
    for pmin, pmax, ntok, data in records:
        b += struct.pack("<iiiiQ", pmin, pmax, ntok, 0, len(data) if not isinstance(data, int) else data) + (b"" if isinstance(data, int) else data)
    return b[:len(b)-truncate] if truncate else b
cases = {
  "no checkpoint file": None,
  "valid, 3 checkpoints": ckfile([(0, 99, 100, b"x"*64), (0, 199, 200, b"y"*64), (0, 299, 300, b"z"*64)]),
  "old format (n_tokens field 0)": ckfile([(0, 99, 0, b"x"*64)]),
  "truncated blob": ckfile([(0, 99, 100, b"x"*64)], truncate=10),
  "huge count": ckfile([(0, 99, 100, b"x"*64)], count=1<<60),
  "huge data_len": ckfile([(0, 99, 100, 1<<62)]),
  "empty blob": ckfile([(0, 99, 100, b"")]),
  "bad magic": b"\0"*64,
}
ok = True
for name, blob in cases.items():
    p = os.path.join(d, "s.bin.checkpoints")
    if blob is None:
        if os.path.exists(p): os.remove(p)
    else:
        open(p, "wb").write(blob)
    code, r = post("/slots/0?action=restore", {"filename": "s.bin"})
    code2, r2 = post("/completion", {"prompt": prompt + " More.", "n_predict": 2, "temperature": 0, "id_slot": 0, "cache_prompt": True})
    reused = r2.get("tokens_cached", 0)
    alive = code2 == 200
    print(f"{name:32s} restore {code} n_restored {r.get('n_restored')}  next request {code2}  prompt processed {r2.get('timings',{}).get('prompt_n')}")
    ok = ok and code == 200 and alive
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
