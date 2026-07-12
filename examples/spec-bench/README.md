# ik_llama.cpp/examples/spec-bench

`llama-spec-bench` is a direct C++ speculative benchmark for prompt-driven tasks.
It reuses the normal `llama-common` startup and speculative lifecycle instead of
benchmarking through `llama-server`.

## Scope

- built-in canonical tasks: `code`, `extract`, `story`
- optional strict JSONL prompt-file override
- baseline and speculative runs use the same binary and normal model/sampler args
- one JSONL row per task attempt plus one summary row at the end
- per-stage drafted and accepted counts by speculative position

## Benchmark-specific flags

- `--prompts <path>`: replace the built-in tasks with a strict JSONL prompt file
- `--task <name[,name...]>`: select built-in tasks
- `--repeat <n>`: repeat each task `n` times
- `--retry <n>`: retry transient task failures up to `n` times
- `--output-format jsonl`: select the common JSONL output convention; output is written to `stdout`
- `--predict <n>` / `-n <n>`: command-level generation budget for every task without a row override

## Dataset format

Each JSONL line must be an object containing a non-empty `prompt` string. Optional fields are
`id`, `name`, `category`, and positive integer `max_tokens`:

```json
{"id":"task-1","name":"math","category":"reasoning","prompt":"Solve 12*17.","max_tokens":64}
```

IDs default to the one-based input line number and must be unique. Unknown fields,
duplicate IDs, empty prompts, malformed JSON, and invalid `max_tokens` values are rejected.
The file replaces the built-in task set for that invocation.

The built-in extraction task uses the versioned
`fixtures/youtube-extract.txt` snapshot. Its source URL, retrieval date, and Wikipedia
revision ID are recorded in `fixtures/youtube-extract.meta.json`; no network request is
made during a benchmark.

Speculative output includes `drafted_by_position`, `accepted_by_position`,
`acceptance_rate_by_position`, and `conditional_acceptance_rate` arrays for every stage.
Array element zero is speculative position one. The first conditional rate is `null`
because it has no preceding position.

## Example

```bash
./build/bin/llama-spec-bench \
  -m model.gguf \
  --seed 123 \
  --temp 0 \
  --predict 256 \
  --output-format jsonl \
  --task code,extract,story > results.jsonl
```
