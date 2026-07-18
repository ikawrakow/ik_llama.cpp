# RTR auto: план реализации pre-PR follow-up

Связанная спецификация: [RTR_AUTO_PR_FOLLOWUP_SPEC.md](RTR_AUTO_PR_FOLLOWUP_SPEC.md).

## Этап 1 — безопасные helpers и cgroup

1. В `llama-cgroup-resolver.h` заменить выбор most-specific mount на сбор всех
   host-relative mappings с dedup `(mountpoint,path)`; оставить namespace
   fallback только при отсутствии host-relative mapping.
2. Вынести cgroup headroom traversal в testable seam, позволяющий fixture reader
   симулировать limits/ошибки чтения. Пересекать все applicable hierarchy;
   unreadable unique hierarchy => unknown.
3. Добавить fixture full root + bind mount, stricter parent, unreadable mapping,
   v1/v2 mixed membership и duplicate dedup. Для v2 отдельно зафиксировать:
   missing `memory.max` + `memory.current` допустим только ровно на mountpoint;
   partial pair и missing pair ниже mountpoint дают unknown.

**Gate:** `test-cgroup-resolver` доказывает отсутствие ancestor loss и fail-closed
результат.

## Этап 2 — conservative peak budget

1. В `llama-model-loader.h/.cpp` создать shared constants для worker count и
   CUDA staging size, используемые `load_all_data()`.
2. Вынести overflow-safe peak arithmetic в deterministic helper: model bytes,
   workspace, `workers * max_read_buffer`, optional CUDA staging.
3. Derive helper inputs from the same finalized placement/buffer-type decision
   as loader; unknown backend applicability => AUTO_UNKNOWN. Largest probe
   tensor is admitted only after asserting it bounds every `read_buf.resize()`
   path, including Windows graph/split path.
4. Добавить unit tests для CPU, active CUDA, requested-but-inactive CUDA,
   Windows non-CUDA split read path, 8-worker term и overflow.

**Gate:** policy cannot keep when helper cannot bound loader transient memory.

## Этап 3 — mmap/prefetch/defer lifecycle

1. Добавить trailing `prefetch_experts` compatibility field в
   `llama_model_params`, default params и propagation всех model-load callers
   (common, bench/server/examples and direct aggregate initializers); обновить
   public ABI/matching-header documentation.
2. On Linux, `-rtr auto + prefetch` => AUTO_UNKNOWN before memory probe. On
   unsupported platforms, prefetch does not alter RTR policy and is explicitly
   disabled with a warning during context creation.
3. После construction loader записать effective mmap; обновить после tensor
   load. `vocab_only` получает correct constructor-effective value.
4. `defer_experts` index использует `ml.use_mmap`; forced incompatibility
   логируется и не индексируется.
5. В context creation gate prefetch на `!model->mappings.empty()`; при false —
   единственный warning, без init/register. Define idempotent ownership for
   global init/registration across repeated contexts and cleanup. Forced path
   не выдаёт false success.
6. Добавить loader/context test seam или tiny GGUF tests: forced mmap, auto
   prefetch UNKNOWN, forced defer/prefetch, direct API caller, repeated context
   creation и vocab-only.

**Gate:** no mmap-only feature silently no-ops.

## Этап 4 — versioned benchmark schema и consumer

1. Перевести SQL writer c `test_v2` на `test_v3`; сохранить historical
   `use_mmap`, добавить explicit requested/effective/buffer columns и
   require all v3 RTR/mmap columns before reading.
2. Расширить reader allowlist на `test`, `test_v2`, `test_v3`; fixed internal
   `source_schema` включить в every UNION projection and JOIN. Historical
   schemas compare only within the same schema; `test↔v2` и `v2↔v3` never join.
3. For v3 use `use_mmap_effective` as canonical comparison key; requested mmap
   remains report-only. Replace NULL wildcard with SQLite `IS`; centralize an
   early no-common diagnostic before any `rows[0]` or rendering in default and
   explicit-show flows; nullable bool renders `Unknown`.
4. Persist and SQLite-inspect v3 requested/effective/buffer getters, including
   forced `requested=1,effective=0,mmap_backed=0` fixture.
5. Update README/examples and Python fixtures: legacy-only, v2-only, v3-only,
   test/v2/v3 mixed rejection, malformed v3, multi-RTR variants, every nullable
   bool display, and no-common default/show.

**Gate:** no existing v2 schema is modified; no semantic cross-version speedup.

## Этап 5 — review and verification

1. Run formatter/style checks and `git diff --check`.
2. Build MSVC Release targets: `test-rtr-params`, `test-cgroup-resolver`,
   new policy/bench tests, `llama-bench`.
3. Run relevant CTest and Python consumer tests, then `llama-bench --help`
   smoke.
4. Review final `upstream/main...HEAD` diff specifically for public ABI fields,
   cgroup fail-closed paths, SQL migrations and direct API context behaviour.
5. Document Linux cgroup test as CI-required if no Linux environment is present.

## Commit structure

1. `fix: make RTR cgroup and peak policy conservative`
2. `fix: guard RTR mmap-only feature interactions`
3. `fix: version RTR benchmark mmap metadata`
4. `test: cover RTR policy and benchmark schema regressions`
5. `docs: record RTR pre-PR remediation` (spec + plan)

Commits must remain independently buildable where practical; public API and its
tests land in the same commit.
