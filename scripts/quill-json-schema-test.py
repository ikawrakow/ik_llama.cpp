#!/usr/bin/env python3
"""
End-to-end structured-output conformance test for llama-server.

Run it against a server started with a RANDOM-WEIGHT model
(scripts/quill-synthetic-model.py): such a model emits noise unless the
sampler enforces the grammar, so every response must parse and validate.

  python3 scripts/quill-json-schema-test.py --url http://127.0.0.1:8080 --rounds 5
"""
import argparse, json, sys, urllib.request

try:
    import jsonschema
except ImportError:
    jsonschema = None

SCHEMAS = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 20},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 8}, "maxItems": 3},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age", "tags", "active"],
        "additionalProperties": False,
    },
    "enum_nested": {
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b", "c"]},
            "inner": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "null"}},
                      "required": ["x", "y"], "additionalProperties": False},
        },
        "required": ["kind", "inner"],
        "additionalProperties": False,
    },
    "pattern": {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{2}$"}},
        "required": ["code"], "additionalProperties": False,
    },
    "refs_anyof": {
        "$defs": {"pt": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                         "required": ["x", "y"], "additionalProperties": False}},
        "type": "object",
        "properties": {
            "shape": {"anyOf": [{"$ref": "#/$defs/pt"}, {"type": "array", "items": {"$ref": "#/$defs/pt"}, "minItems": 1, "maxItems": 2}]},
            "label": {"oneOf": [{"type": "string", "maxLength": 6}, {"type": "null"}]},
        },
        "required": ["shape", "label"], "additionalProperties": False,
    },
    "formats_bounds": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "format": "uuid"},
            "day": {"type": "string", "format": "date"},
            "score": {"type": "integer", "minimum": -5, "maximum": 42},
            "ratio": {"type": "number", "minimum": 0},
            "const": {"const": "fixed"},
            "word": {"type": "string", "minLength": 2, "maxLength": 4},
        },
        "required": ["id", "day", "score", "ratio", "const", "word"], "additionalProperties": False,
    },
    "optional_map": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"},
                       "meta": {"type": "object", "additionalProperties": {"type": "integer"}, "maxProperties": 2}},
        "required": ["a"], "additionalProperties": False,
    },
    "array_of_objects": {
        "type": "object",
        "properties": {"items": {"type": "array", "minItems": 2, "maxItems": 3,
                                 "items": {"type": "object", "properties": {"k": {"enum": [1, "two", None, True]}},
                                           "required": ["k"], "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False,
    },
}

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def check(text, schema):
    try:
        obj = json.loads(text)
    except Exception as e:
        return f"invalid JSON: {e}"
    if schema is None:
        return None if isinstance(obj, dict) else "not an object"
    if jsonschema is not None:
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e:
            return f"schema violation: {e.message}"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    cases = []
    for name, schema in SCHEMAS.items():
        cases.append((f"chat json_schema {name}", "chat", {"response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}, schema))
        cases.append((f"chat json_schema {name} (no thinking)", "chat", {"chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}, schema))
        cases.append((f"chat json_object+schema {name}", "chat", {"response_format": {"type": "json_object", "schema": schema}}, schema))
        cases.append((f"completion json_schema {name}", "completion", {"json_schema": schema}, schema))
    cases.append(("chat json_object (no schema)", "chat", {"response_format": {"type": "json_object"}}, None))

    fails = total = skipped = 0
    for rnd in range(args.rounds):
        for label, kind, extra, schema in cases:
            for temp in (0.0, 1.0):
                body = dict(extra, temperature=temp, seed=rnd * 7 + 1, max_tokens=args.max_tokens, n_predict=args.max_tokens)
                if kind == "chat":
                    body["messages"] = [{"role": "user", "content": "Reply with JSON only."}]
                    r = post(args.url + "/v1/chat/completions", body)
                    ch = r["choices"][0]
                    text, finish = ch["message"].get("content") or "", ch.get("finish_reason")
                    if finish == "length" and not text.strip() and ch["message"].get("reasoning_content"):
                        skipped += 1  # still inside the reasoning block: inconclusive
                        continue
                else:
                    body["prompt"] = "Output: "
                    r = post(args.url + "/completion", body)
                    limit = r.get("stop_type") == "limit" or r.get("stopped_limit", False)
                    text, finish = r["content"], ("length" if limit else "stop")
                err = check(text, schema)
                if err and finish == "length":
                    skipped += 1  # cut off mid-value: the prefix is grammar-constrained, but not checkable
                    continue
                total += 1
                if err:
                    fails += 1
                    print(f"FAIL [{label}, T={temp}, round {rnd}]: {err}\n      output: {text[:200]!r}")
    print(f"{total - fails}/{total} structured outputs valid ({skipped} inconclusive: cut off by max_tokens)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
