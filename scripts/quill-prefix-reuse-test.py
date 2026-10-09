#!/usr/bin/env python3
# quillama: regression test for prompt-prefix reuse. A prompt that shares ~98% of its prefix with an
# earlier one must only process the new suffix, also after an unrelated request evicted it from the
# slot (restored from --cache-ram). Usage: quill-prefix-reuse-test.py http://127.0.0.1:8080
import json, sys, urllib.request
url = sys.argv[1]
base = " ".join(f"Fact {i}: entity_{i} relates to entity_{i+1} in project Alpha." for i in range(120))
def ask(prompt):
    body = {"messages":[{"role":"user","content":prompt}], "max_tokens":8, "temperature":0}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(url+"/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=900).read())
    t = r["timings"]; return t["prompt_n"], r["usage"]["prompt_tokens"]
a = ask(base + " Question one?")
b = ask("Something completely different about cooking pasta. " * 20)
c = ask(base + " Question two, slightly different?")
d = ask(base + " Question two, slightly different? And more.")
for name, (n, tot) in zip(["A first", "B unrelated", "A' after B (from RAM cache)", "A'' extends A'"], [a, b, c, d]):
    print(f"{name:32s} processed {n:5d} of {tot:5d} prompt tokens")
ok = c[0] < 0.1*c[1] and d[0] < 0.1*d[1]
print("PASS" if ok else "FAIL: near-identical prefix was re-processed")
sys.exit(0 if ok else 1)
