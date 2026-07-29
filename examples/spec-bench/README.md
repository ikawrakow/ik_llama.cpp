# ik_llama.cpp/examples/spec-bench

`llama-spec-bench` is a direct C++ speculative benchmark for prompt-driven tasks.
It reuses the normal `llama-common` startup and speculative lifecycle instead of
benchmarking through `llama-server`.

## Scope

- built-in canonical tasks: `code`, `extract`, `story`
- all three canonical built-in workloads by default, or one plain custom prompt via `-p` / `-f`
- optional strict JSONL prompt-file override for structured multi-prompt workloads
- baseline and speculative runs use the same binary and normal model/sampler args
- Markdown report by default; compact JSONL is available with `--output-format jsonl`
- per-stage drafted and accepted counts by speculative position

## Benchmark-specific flags

- `--prompts <path>`: replace the built-in tasks with a strict JSONL prompt file
- `-p, --prompt <text>`: run one inline custom prompt
- `-f, --file <path>`: run one plain-text custom prompt file; the file is one prompt, not one task per line
- `--task <name[,name...]>`: select built-in tasks
- `--repeat <n>`: repeat each task `n` times
- `--retry <n>`: retry transient task failures up to `n` times
- `--output-format jsonl`: select the common JSONL output convention; output is written to `stdout`
- `--output-details`: print prompts and responses first, followed by normal and detailed Markdown metrics; JSONL includes complete details
- `--predict <n>` / `-n <n>`: command-level generation budget for every task without a row override

## Input modes

Choose exactly one mode: built-ins (optionally narrowed with `--task`), one `-p` prompt, one `-f` file, or one `--prompts` JSONL dataset.
Examples:

```bash
./build/bin/llama-spec-bench -m model.gguf -n 4 -p "Write a merge sort in C++."
./build/bin/llama-spec-bench -m model.gguf -n 4 -f examples/spec-bench/prompts/code.txt
```

## Dataset format

Each JSONL line must be an object containing a non-empty `prompt` string. Optional fields are
`id`, `name`, `category`, and positive integer `max_tokens`:

```json
{"id":"task-1","name":"math","category":"reasoning","prompt":"Solve 12*17.","max_tokens":64}
```

IDs default to the one-based input line number and must be unique. Unknown fields,
duplicate IDs, empty prompts, malformed JSON, and invalid `max_tokens` values are rejected.
The file replaces the built-in task set for that invocation.

The canonical prompts are embedded into the executable at configure time from `prompts/code.txt`,
`prompts/extract.txt`, and `prompts/story.txt`; no source-tree or network access is needed at runtime.

Compact JSONL includes raw `drafted_by_position` and `accepted_by_position` arrays for
every stage. Detailed JSONL additionally includes the derived
`acceptance_rate_by_position` and `conditional_acceptance_rate` arrays. Array element
zero is speculative position one; the first conditional rate is `null` because it has
no preceding position.

Acceptance length is defined consistently as `1 + accepted_tokens / num_drafts` in detailed JSON, compact JSON, Markdown, and repeat summaries.

Repeated attempts are executed in one process. Stateful drafting stages, including
adaptive n-gram stages and lookup caches, may therefore carry learned state from an
earlier task or repeat; use `--repeat 1` and separate invocations when independent
samples are required. Pin the chat-template mode (`--jinja` or `--no-jinja`) when
comparing runs because it changes the effective prompt. Speculative verification can
also diverge from a baseline after a near-tie because batched evaluation changes
floating-point reduction order, so this tool is a performance and acceptance benchmark,
not a bit-identical output checker.

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
