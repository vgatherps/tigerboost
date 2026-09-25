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
