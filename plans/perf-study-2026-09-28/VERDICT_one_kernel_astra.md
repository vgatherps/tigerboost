status: raise

The requested cleanup is implemented. All specified command gates and assembly comparisons pass.
The hot benchmark remains slower than the captured baseline: 84.07 ns before, 91.22 ns on the
first after run, then 89.97 / 90.16 / 90.35 ns in three serial repeats. The repeat median is
90.16 ns, 7.24% above the baseline. This remains unresolved under the instruction that passing
tests with a benchmark regression is not sufficient for signoff. No kernel tuning was attempted.

All three emitted assembly files are identical after removing exactly one trailing NUL from the
before files. The benchmark harness now measures one kernel instead of eleven variants, as
requested. Reference and driver timings also increased. These facts do not establish the cause
of the timing difference; this report does not attribute it to either the kernel or host noise.

The starting checkout was a88324b. Before capture completed before any source edits. No agents,
commits, stash, checkout, cargo fmt, rustfmt, or saturating arithmetic were used.

| Item | Result |
|---|---|
| 1 | `plan(&model)` uses shared-feature order, private `CHUNK_TREES = 8` and `SUMS = 2`. Sum clamping is preserved. `SortOptions`, `TreeOrdering`, and model-order planning are deleted. `plan_with` is crate-private under `cfg(test)`; every caller is a crate unit test. |
| 2 | Both integration test files use `plan(&model)`. The unit differential test compiles 96 seeded forests across all 3 × 3 combinations of chunks 1/3/8 and sums 1/2/4, with 20 rows per kernel and disjoint leaf bit fields. An explicit compiled test covers chunk 1, chunk larger than the forest, and sums larger than the forest. |
| 3 | `EmitOptions` already contained only `avx2`. `compile_with` now rejects unsupported host AVX2. The test exercises rejection on this ARM64 host. `compile_with` and `emit_assembly_with` remain. `emit_assembly` remains a default-options wrapper, used by `inspect_code` and documented in README. |
| 4 | `scalar_bench` measures reference, kernel, driver floor, and the AVX2 twin only on an x86-64 host supporting AVX2. `cold_bench` rejects both removed flags and drops their report fields and CSV column. `inspect_code` emits exactly three assembly files. `x86_check` emits one selected ISA and shares one native comparison loop between tests and output generation. All examples use `support::Workload::parse`; named special/depth fixtures moved into support without changing their data. The historical workloads are deleted. Cold/distinct workloads share the existing SplitMix step and retain their original seeds. |
| 5 | `emit_assembly_with` strips at most one terminal NUL. The x86 workaround is deleted. A JIT unit test checks no NUL and a final newline on both targets; the acceptance comparison also checks the AVX2 output. |
| 6 | `PredicateId` directly indexes threshold slots; the randomized planner test checks consecutiveness and complete coverage. The identity map and redundant `evaluated_trees` vector are deleted. `Constant` replaces `Admit`/`Retire`; `Chunk.tests` is deleted. Local `add_leaf` and `feature_vector` helpers share the requested tails without changing the emitted instruction sequence. The named tautological tests and duplicate JIT planner-shape assertions are deleted. AVX2 constant-order assertions remain; ARM64 compares each leaf block's multiset against original leaves plus padding zeros. |
| 7 | README uses `plan(&model)`, explains the fixed order/chunks/sums, documents current commands/workloads and host AVX2 rejection, and records the requested 85/228 ns historical measurements with a RESULTS.md link. PLAN describes the fixed kernel and rejected reuse experiments. The README doctest passes. |

Builds used `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21`. The doctest used
`RUSTDOCFLAGS=-C link-arg=/opt/homebrew/lib/libzstd.dylib`.

| Gate | Result | Evidence |
|---|---|---|
| Before `inspect_code -- /tmp/tb-cl-before 100x100` | PASS, before edits | `/tmp/tb-cl-before/` |
| `cargo build --offline --release --examples` | PASS | `/tmp/tb-cl-build-final.log` |
| `cargo test --offline --all-targets` | PASS, 60 tests | `/tmp/tb-cl-debug-tests.log` |
| `cargo test --offline --release --all-targets` | PASS, 60 tests | `/tmp/tb-cl-release-tests.log` |
| `cargo test --offline --doc` | PASS, 1 doctest | `/tmp/tb-cl-doctest.log` |
| `cargo clippy --offline --all-targets -- -D warnings` | PASS | `/tmp/tb-cl-clippy-final.log` |
| After `inspect_code -- /tmp/tb-cl-after 100x100` | PASS, exactly three emissions | `/tmp/tb-cl-inspect-after-final.log` |
| Byte comparison | PASS, all three files | `/tmp/tb-cl-byte-comparison-final.json` |
| `tools/x86_check.sh special` | `ok 256 rows`, `PASS kernel` | `/tmp/tb-cl-x86-special.log` |
| `tools/x86_check.sh special --avx2` | `ok 256 rows`, `PASS kernel-avx2` | `/tmp/tb-cl-x86-special-avx2.log` |
| `tools/x86_check.sh depths --avx2` | `ok 256 rows`, `PASS kernel-avx2` | `/tmp/tb-cl-x86-depths-avx2.log` |
| `tools/x86_check.sh depths2` | `ok 256 rows`, `PASS kernel` | `/tmp/tb-cl-x86-depths2.log` |
| `cargo run --offline --release --example scalar_bench -- 100x100` | Exit 0; slower timings raised above | `/tmp/tb-cl-bench-after.log`, `/tmp/tb-cl-bench-after-{2,3,4}.log` |
| `git diff --check` | PASS | Final diff |

Debug/release test counts are identical:

| Test binary | Passed per profile |
|---|---:|
| Library unit tests | 25 |
| differential | 1 |
| model_reference | 11 |
| scalar_jit | 8 |
| cold_bench | 6 |
| inspect_code | 0 |
| scalar_bench | 3 |
| x86_check | 6 |
| Total | 60 |

The library tests also passed after the planner changes and again after the emitter helpers.
The final scalar example tests passed after restoring planning inside the compile-time timer.
Rosetta link commands emitted `-no_pie` deprecation and pointer-alignment warnings; all linked
executables ran successfully and each compared 256 rows bit for bit.

Each before file below is in `/tmp/tb-cl-before`; each after file is in `/tmp/tb-cl-after`.
Comparison requires the before file to end in NUL, removes exactly that byte, and compares every
remaining byte. Each after file has no NUL and ends with a newline.

| Before file | After file | Bytes before → after | Result |
|---|---|---:|---|
| `aarch64-apple-darwin-tree-major-8-2sums-shared.s` | `aarch64-apple-darwin-kernel.s` | 227581 → 227580 | PASS |
| `x86_64-unknown-linux-gnu-tree-major-8-2sums-shared.s` | `x86_64-unknown-linux-gnu-kernel.s` | 239597 → 239596 | PASS |
| `x86_64-unknown-linux-gnu-tree-major-8-2sums-shared-avx2.s` | `x86_64-unknown-linux-gnu-kernel-avx2.s` | 225439 → 225438 | PASS |

| Run | Kernel hot median ns | Reference median ns | Driver floor median ns |
|---|---:|---:|---:|
| Before, retained shared-order variant | 84.07 | 283.17 | 5.09 |
| After, first run | 91.22 | 316.88 | 5.84 |
| After, repeat 1 | 89.97 | 292.51 | 5.86 |
| After, repeat 2 | 90.16 | 290.72 | 5.89 |
| After, repeat 3 | 90.35 | 297.92 | 5.92 |

Every run used 100x100 and five rounds of 50,000 dependent calls. Both versions visited 35
rows in the dependent trace. These are indicative timings including the driver, with no floor
subtraction. Before log: `/tmp/tb-cl-bench-before.log`. README's 85/228 ns values remain the
requested recorded results from RESULTS.md, not measurements from this cleanup run.

Line counts include every allowed file, including the three unchanged files. The new report
and generated build/log artifacts are excluded from this source total.

| File | Before | After | Change |
|---|---:|---:|---:|
| `PLAN.md` | 68 | 71 | +3 |
| `README.md` | 156 | 167 | +11 |
| `examples/cold_bench.rs` | 698 | 623 | -75 |
| `examples/inspect_code.rs` | 73 | 63 | -10 |
| `examples/scalar_bench.rs` | 256 | 163 | -93 |
| `examples/support/bench.rs` | 41 | 41 | +0 |
| `examples/support/distinct.rs` | 53 | 46 | -7 |
| `examples/support/mod.rs` | 80 | 222 | +142 |
| `examples/x86_check.rs` | 639 | 371 | -268 |
| `src/jit.rs` | 1878 | 1711 | -167 |
| `src/lib.rs` | 5 | 5 | +0 |
| `src/sort_policy.rs` | 241 | 213 | -28 |
| `src/sort_policy/continuation.rs` | 43 | 29 | -14 |
| `src/sort_policy/tests.rs` | 619 | 550 | -69 |
| `tests/differential.rs` | 94 | 79 | -15 |
| `tests/scalar_jit.rs` | 403 | 330 | -73 |
| `tools/x86_check.sh` | 63 | 63 | +0 |
| Total | 5410 | 4747 | -663 |

Self-review found no required deletion left undone and no unrelated source change. Auxiliary
removals are tied to the deleted configuration: model-order tests and feature-reuse comparison
helper, example variant generators/assertions, cold benchmark option/report fields and its
`chunk_trees` CSV column, and outdated four-tree static measurements in README. The old workload
shape test was deleted because all workloads it exercised were deleted. Surviving prediction,
leaf-bit, ownership, NaN, output-size, threshold, and instruction checks remain.

The review alternatives beyond the task remain: `PredicateId` is retained as requested instead
of replacing it with a range; `TreePath`, the planner replay oracle, and the detailed AVX2
instruction tests are retained. No SSE2 algorithm, leaf layout, ABI, code model, continuation
attributes, or vzeroupper behavior was changed. `src/lib.rs` already includes README for the
doctest. `tools/x86_check.sh` already consumes the generated manifest, now containing only the
selected kernel, so neither file needed an edit. `examples/support/bench.rs` is also unchanged.

All tracked edits are within the allowed files; VERDICT.md is the requested new report. The only
unresolved signoff issue is the worse indicative hot timing. Resolving its cause would require a
controlled measurement comparison; the fixed kernel has not been changed to chase these numbers.
