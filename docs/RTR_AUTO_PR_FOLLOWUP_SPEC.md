# RTR auto: спецификация закрытия pre-PR review

**Дата:** 2026-07-19
**Ветка:** `feature/rtr-auto-pr-prep`
**База:** `upstream/main`

## Цель и границы

Довести `--run-time-repack auto` до safety-first контракта: авто-режим не делает
`AUTO_KEEP`, когда память/placement/mmap consequences неизвестны или оценены
неполно; benchmark и SQL consumer не выдают ложный speedup.

Forced `-rtr 1` остаётся explicit opt-in и по-прежнему отключает mmap в loader.
Эта серия не меняет его в silent auto-disable, но делает несовместимые mmap-only
features явными и не позволяет им притворяться работающими.

## Findings, которые закрывает серия

1. **P1 cgroup ancestor loss:** наиболее специфичный bind mount вытесняет full
   hierarchy mount; скрытый stricter parent limit не участвует в headroom.
2. **P1 loader peak undercount:** policy не учитывает persistent `read_bufs`
   всех 8 workers и CUDA staging buffers.
3. **P1 benchmark mmap mislabel:** `use_mmap` хранит CLI request, а не
   loader-effective состояние.
4. **P1 SQL wildcard:** legacy `NULL` RTR fields сопоставляются с любым known
   `test_v2` setting и смешивают результаты.
5. **P2 prefetch/defer incompatibilities:** AUTO_KEEP может убрать mmap для
   `--prefetch-experts`; forced path проверяет requested вместо loader mmap.
6. **P2 consumer crashes:** no common configuration и nullable boolean ломают
   штатный compare script.
7. **P3 effective-state edge:** `vocab_only` не записывает loader mmap state.
8. **Coverage gap:** current tests не исполняют policy/loader interactions.

## Инварианты

- `AUTO_KEEP` допустим, только когда placement, memory peak и последствия
  отключения mmap полностью смоделированы.
- Любая неопределённость => `AUTO_UNKNOWN`, repack выключается, requested mmap
  сохраняется.
- Все host-relative cgroup mappings, которые могут ограничивать process,
  участвуют в пересечении. Непрочитываемая единственная applicable hierarchy
  делает результат unknown; duplicate mapping той же полной hierarchy может
  быть deduplicated после доказательства одинакового `(mountpoint,path)`.
- Unknown legacy RTR/mmap-effective metadata не равна known setting.
- `llama_model_loader_mmap_enabled()` означает loader decision, а
  `llama_model_has_mmap_buffers()` — отдельный факт наличия mmap-backed buffer.
  Getters meaningful только для успешно возвращённой модели.

## A. Cgroup hierarchy

`llama_resolve_cgroup_mounts()` сохраняет все host-relative matching mounts:
full hierarchy и bind mounts. Namespace-relative fallback возможен только при
отсутствии любого host-relative match. Каждая уникальная resolved
`(mountpoint,path)` hierarchy читается до mountpoint; итоговый headroom —
minimum по всем successfully resolved applicable hierarchy.

Для cgroup v2 отсутствие одновременно `memory.max` и `memory.current` ровно на
mountpoint означает unlimited/неэкспонированный корневой уровень и пропускается.
Та же отсутствующая пара ниже mountpoint, несовпадающая пара файлов или ошибка
доступа остаются unknown (fail closed).

Нельзя выбирать только «самый специфичный» root. Неразборчивый membership/mount
или unreadable unique applicable hierarchy => unknown, а не host fallback. Tests
должны доказать: (1) full + bind mapping оба сохранены, (2) stricter ancestor
полной hierarchy выигрывает, (3) unreadable applicable hierarchy даёт unknown,
(4) missing pair пропускается только на mountpoint.

## B. Conservative loader peak

Общий helper с checked multiplication/addition считает:

```text
CPU-resident model bytes
+ maximum repack workspace
+ n_load_workers * maximum non-mmap read buffer
+ n_load_workers * CUDA staging buffer size, если фактический loader path
  способен использовать CUDA async upload
```

`maximum non-mmap read buffer` допускается консервативно принять равным largest
tensor среди probe tensors. Это покрывает persistent worker capacities и Windows
parallel split path. CUDA staging учитывается по runtime-applicability
(`!use_mmap`, `!check_tensors`, active CUDA upload backend); если определить путь
по metadata/placement нельзя, policy возвращает UNKNOWN. Shared constants worker
count/staging size имеют единый источник с `load_all_data()`.

Overflow любого промежуточного расчёта => UNKNOWN. Tests покрывают CPU-only,
active CUDA, unknown backend applicability, `8 * max_read_buffer` и overflow.

## C. mmap metadata и versioned SQL schema

Нельзя менять значение existing `test_v2.use_mmap`: historical rows используют
его как requested setting. SQL writer переключается на immutable **`test_v3`**:

- `use_mmap` — retained requested compatibility field;
- `use_mmap_requested` — явный requested setting;
- `use_mmap_effective` — `llama_model_loader_mmap_enabled()`;
- `mmap_backed_buffers` — `llama_model_has_mmap_buffers()`.

`test_v3` avoids `CREATE TABLE IF NOT EXISTS test_v2` schema collision. Consumer
reads `test`, `test_v2`, `test_v3`. Comparison of v3 uses
`use_mmap_effective`; historical tables with no effective field may compare only
with the same historical schema. **`test_v2` and `test_v3` are never compared**:
v2 has no effective-mmap semantics and must never be interpreted as
requested==effective.

Loader records `use_mmap_loader_enabled` immediately after construction (for
successful `vocab_only`) and again after tensor loading (for later fallback).

## D. SQL consumer contract

SQLite-only equality for optional RTR/effective mmap properties is:

```sql
tb.column IS tc.column
```

Thus `NULL` matches only `NULL`. No wildcard semantics. If no full comparable
configuration remains, every output mode (default and explicit `--show`) exits
non-zero and prints `no comparable configurations` to stderr. Nullable boolean
display is `Unknown`, never `int(None)`.

Tests cover legacy-only, v2-only, v3-only, legacy/v3 rejection, multiple v3 RTR
variants without cross-product, `--show repack` on NULL, and no-common failure
in default and explicit-show modes.

## E. Prefetch/defer interactions and API callers

Add `prefetch_experts` to model-load params and propagate it from common CLI so,
on Linux where expert prefetch is implemented, `-rtr auto --prefetch-experts`
returns AUTO_UNKNOWN before memory probe and preserves mmap. On unsupported
platforms the request must not affect RTR auto and context creation must emit an
explicit warning before disabling prefetch.

Because prefetch is also a public **context** API request, context creation is
the final authority. Its exact eligibility predicate is
`!model->mappings.empty()`: these are the address ranges registered with the
prefetch backend. When false, context emits one deterministic warning and does
not initialize/register prefetch. `mmap_backed_buffers` is benchmark metadata,
not this eligibility predicate. Direct API callers therefore cannot get a silent
no-op even if model was loaded without the CLI compatibility flag.

For forced RTR, prefetch/defer are explicitly disabled at their effective use
point, not merely logged at model load. Deferred expert index checks
`ml.use_mmap`, not requested `params.use_mmap`, and logs its incompatibility.
All new model-param fields are trailing, default-initialized and documented as
C-ABI matching-header requirements; defaults and aggregate initializers update.

## F. Tests and acceptance

1. Cgroup resolver plus injectable/fixture headroom seam prove A.
2. Deterministic peak helper tests prove B.
3. Tiny GGUF or test seam proves forced requested=true/effective=false plus
   actual `n_repacked>0`; auto UNKNOWN prefetch keeps mmap; vocab-only records
   constructor-effective mmap; forced defer/prefetch takes explicit disabled
   path.
4. SQL/output integration proves test_v3 stores requested/effective/buffer
   state and compare keys use effective state; historical v2 remains readable
   but v2↔v3 is rejected. README/examples and schema-detection tests update for
   `test_v3`.
5. MSVC Release build, relevant CTest/Python tests and `llama-bench` smoke pass.
   Linux cgroup build/test is required in CI or explicitly reported as an
   external pending check; local Windows success is not evidence for Linux.

## Не-цели

- Точный Windows Job Object accounting (current AUTO_UNKNOWN fallback remains).
- SQL ALTER/migration или dual-write existing tables.
- Изменение forced RTR в automatic disable.
