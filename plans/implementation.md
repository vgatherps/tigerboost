# Implementation notes

## Baseline

`cargo test --offline` passes: zero tests, six warnings in the unfinished `sort_policy.rs` sketch. No working predictor or latency baseline exists yet.

Host: `aarch64-apple-darwin`; Rust `1.93.0-nightly (c90bcb957 2025-11-06)`.

## Review order

One implementation agent at a time. Each chunk must pass parent review before another starts. Review the approach, unsafe boundaries, dedicated tests, and evidence; passing tests alone is insufficient. No placeholder implementation may become a dependency.

1. LLVM integration following breadboard; Cranelift experiments are now historical.
2. Typed model and straightforward reference evaluator; focused tests with constructed trees.
3. Scalar JIT baseline; then measured independent accumulation.
4. Rolling schedule and interpreter; then scheduled scalar codegen and measurements.
5. Portable SIMD and differential tests; then both-target lowering and latency comparison.
6. Complete model-to-prediction validation, reproducible benchmarks, and API handoff.

## Cranelift probe review

Accepted after parent and independent read-only review. `cargo test --offline --all-targets`: three dedicated tests pass. Scoped rustfmt and Clippy checks pass; existing sketch warnings remain. Review required explicit function-pointer cast types, now fixed.

Verified checked input bounds/alignment, native ABI, immutable reads, and executable-memory cleanup against pinned dependency code. Direct Cranelift crates are 0.128.2; supporting internals include 0.128.4 and regalloc2 0.13.5, pinned by Cargo.lock.

Small forward-reuse case: eight input loads become four, 56 code bytes on ARM64. At 64/128 features, genuine stack spills occur. Ordinary loads also merge; do not attribute all merging/spills to `readonly`.

## Backend switch

User switched to LLVM. Cranelift work stopped; no latency-correction edits or commands were left in flight. Seven Cranelift tests pass. Parent and independent review verified the spill evidence but did not accept independent-call timing as a single-row latency gate.

LLVM integration follows breadboard's Inkwell/MCJIT setup. Breadboard remains read-only. Subsequent timing must include dependent calls and separately reported driver cost.

## LLVM integration review

Accepted by parent review of the complete example and pinned bindings. Verified native C ABI, checked slice length, inbounds offsets/alignment, read-only memory effects, reassociation-only flags, and `JitFunction`'s engine ownership/context lifetime. The secondary reviewer was stopped after following a wrong checkout path; acceptance does not rely on that unfinished review.

Parent reran five dedicated tests in debug and release, scoped Clippy and rustfmt, and the release example. All pass. The full all-target suite has twelve passing tests, including seven historical Cranelift tests; six pre-existing sketch warnings remain.

`llvm_smoke` executes through LLVM 21.1.8. Separate generic ARM64 target emission is `ldp`, `fadd`, `ldr`, `fadd`, `ret`; this is not a dump of MCJIT memory or a performance result. No x86 execution was tested.

## Next chunk

Typed model and reference accepted after parent code/test review; eleven tests pass in debug and release. Checked original level-bit ordering, validation before indexing/shifting, NaN modes, scale/bias, and empty/constant trees. Existing scheduler sketch remains unused.

Scalar tree-order LLVM kernel accepted after parent code/test review. `src/jit.rs` has 200 implementation lines, retains `JitFunction` under a caller-scoped `Context`, copies leaves into private constant globals, and checks rows before invocation. Parent reran five functional tests plus one IR test in debug/release, scoped Clippy, and rustfmt. All pass; six old scheduler warnings remain.

Benchmark accepted after parent review and rerun: `cargo run --offline --release --example scalar_bench`. Synthetic 64 depth-6 trees, 16 features, 256 rows; the dependent trace visits 47 rows and does not settle on one row. All row outputs and the trace agree before timing. Five alternating rounds of 50,000 calls, checked APIs, warm code/data.

Parent run on Apple M4 Pro, Rust 1.93.0-nightly, LLVM 21.1.8, generic target, early-cse/backend Less:

| Variant | Median ns/call | Min–max |
|---|---:|---:|
| Reference | 179.33 | 178.59–183.93 |
| Scalar JIT | 90.97 | 90.46–92.32 |
| Driver floor | 5.19 | 5.07–5.71 |

Compile: 76.484 ms. Driver cost is included, not subtracted. These are one synthetic workload's results, not deployment or scheduling speedups. No scheduling/SIMD changes in this slice; no model-kernel assembly/spill claims yet. Next is explicit rolling reuse in another reviewed slice.

## Scope correction

User rejected the Python/CatBoost setup and native fixture pipeline as overengineering. Dropped that requirement and deferred JSON import. The naive Rust evaluator plus focused constructed-tree tests is the reference for JIT work. No further native dependency setup; no importer work before the evaluator/JIT.

## Rolling planner review

Replaced the unfinished sorting sketch with a 230-line crate-private planner. Parent reviewed the action representation, exact-bit predicate interning, bounded active set, deterministic choices, constant-tree handling, immediate retirement/refill, and retained-feature reuse. Six focused interpreter tests pass in debug/release; edition-2024 formatting and diff checks pass. Prediction code is unchanged, so this slice makes no latency claim. Planner dead-code warnings disappear when LLVM lowering consumes it in the next slice.

## Rolling LLVM review

Both orders now use one action lowerer; retirement clears its partial index and updates a round-robin accumulator. Parent and independent review found no correctness defect or need for another abstraction. Fourteen library/scalar tests pass in release (independent reviewer reran debug); benchmark trace validation also passes. No warnings. Default remains tree-order/one accumulator until broader measurements and emitted-code review.

Same 64-tree workload, parent medians in ns/call (driver included):

| Order | 1 accumulator | 2 accumulators | 4 accumulators |
|---|---:|---:|---:|
| Tree order | 92.07 | 94.69 | 95.05 |
| Rolling, cap 1 | 87.17 | 90.33 | 89.11 |
| Rolling, cap 2 | 83.53 | 82.66 | 82.08 |
| Rolling, cap 4 | 86.39 | 79.68 | 79.12 |
| Rolling, cap 8 | 85.53 | 82.92 | 83.91 |

Extra accumulators alone regress. Rolling cap 4 with two accumulators improves about 13% over tree order on this workload; four accumulators have similar timing and need more logical state. Reference 169.28 ns, floor 5.22 ns. Compilation ranged 36–73 ms, with first-call LLVM setup included only in the first case. Logical action lifetimes are tested; physical register pressure is not yet established.

## Emitted-code inspection

Added a narrow `emit_assembly` API sharing target setup/passes with JIT compilation, and a small `inspect_code` example. Parent checked the diff, reran fifteen library/scalar tests, emitted both targets, and assembled their output with LLVM 21 `llvm-mc`. Workload extraction preserves the exact original model/rows. These are separate emissions, not MCJIT byte disassemblies.

The inspection exposed substantial spilling despite bounded planned indices. ARM64 text shrinks from 12,232 to 11,192 bytes (tree-order1 → rolling4/sums2), but rolling still allocates 1,536 local stack bytes plus 160 bytes of callee saves. x86 text grows 8,650 → 9,044 bytes; local stack allocation grows 344 → 496 bytes. Body stores/reloads include generated indices and FP temporaries, not only ABI saves. Resolve emission/backend ordering before adding SIMD; do not claim physical boundedness from the planner cap.

Independent review found an inspection mismatch: MCJIT's EngineBuilder defaults to `CodeModel::JITDefault`, while explicit emission used `Default`. Corrected the shared target setup to JITDefault. See installed LLVM 21 `ExecutionEngine.h` and [AArch64 target selection](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/AArch64/AArch64TargetMachine.cpp). Corrected emissions still spill heavily: ARM64 rolling has 1,472 local bytes plus 160 callee-save bytes; x86 rolling has 552 local bytes. Separate target emission remains distinct from MCJIT byte disassembly.

Tried backend `Default` with the previous code model: emitted ARM64 assembly was byte-identical and timings unchanged. Dropped that experiment; backend stays `Less`. A shared private constant now keeps both creation sites consistent. After code-model correction, same-workload tree-order1/rolling4-sums2 medians were 93.12/79.98 ns; sixteen focused release tests pass.

Also tried `None` with JITDefault. ARM64 rolling emission has no stack accesses and begins accumulating leaves around line 157 rather than near the end; x86 stack-access instructions decrease 205 → 170. Warm compilation falls to about 3 ms. However, rolling4/sums2 latency regresses 79.98 → 87.15 ns on the same workload. Dropped the change rather than trade away the primary latency goal. Retain `Less`; record the physical-liveness limitation explicitly. SIMD exploration follows on the faster accepted baseline, not on the regressing experiment.
