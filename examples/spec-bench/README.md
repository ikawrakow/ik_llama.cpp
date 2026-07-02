# ik_llama.cpp/example/spec-bench

`llama-spec-bench` is a direct C++ speculative benchmark for prompt-driven tasks.
It reuses the normal `llama-common` startup and speculative lifecycle instead of
benchmarking through `llama-server`.

## Scope

- built-in canonical tasks: `code`, `extract`, `story`
- optional JSONL dataset override
- baseline and speculative runs use the same binary and normal model/sampler args
- one JSONL row per task attempt plus one summary row at the end

## Benchmark-specific flags

- `--dataset <path>`: JSONL dataset override
- `--task <name[,name...]>`: select built-in tasks
- `--repeat <n>`: repeat each task `n` times
- `--retry <n>`: retry transient task failures up to `n` times
- `--output <path>`: write JSONL rows to a file instead of `stdout`

## Dataset format

Each JSONL line must contain a `prompt` string. Optional fields:

```json
{"id":"task-1","name":"math","category":"reasoning","prompt":"Solve 12*17.","max_tokens":64}
```

## Example

```bash
./build/bin/llama-spec-bench \
  -m model.gguf \
  --seed 123 \
  --temp 0 \
  --task code,extract,story \
  --repeat 1 \
  --output results.jsonl
```
