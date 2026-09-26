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

## Two-lane SIMD review

Parent and independent review accepted same-feature pairs of distinct predicates, with all consumers updated before retirement/refill. Odd tails stay scalar. Both vector lanes are defined; ordered comparisons retain NaN behavior. One shared consumer loop handles both modes. Focused lane-order, fanout/refill, odd-tail and IR tests pass; parent reran 18 library/JIT/benchmark tests. Agent also passed release, Clippy and formatting gates.

Same-workload parent medians: tree order 91.59 ns, scalar rolling2/sums2 81.89 ns, SIMD rolling2/sums2 73.92 ns. SIMD rolling4/sums2 regresses 79.31 → 80.35 ns; two earlier runs also regressed at cap4. Defaults remain unchanged pending broader shapes. Driver floor 5.19 ns included; SIMD cap2 compilation 31.843 ms.

Both targets emit packed comparisons (ARM64 `fcmgt.2d`, x86-64 `cmpltpd`/`movmskpd`). Six outputs assembled successfully. For cap4/sums2, ARM64 text decreases 11,176 → 9,880 bytes and local stack 1,472 → 736 bytes (plus 160 callee saves). x86-64 text grows 11,883 → 12,286 bytes and local stack 552 → 888 bytes. SIMD reduces comparison count, but extraction and scheduling costs remain target/model dependent. No x86 execution result.

## Broader synthetic workloads

Added fixed mixed, reuse-heavy, disjoint, and wide workloads shared by timing and assembly examples. Mixed retains its original RNG sequence/data. Reuse-heavy has 32 possible predicates; disjoint uses a different feature for every split; wide has 128 trees and 128 features. All have depth six and 256 input rows. Parent and independent review accepted the small example-only change; three dedicated benchmark tests pass.

The old dependent driver stalled on reuse-heavy row 70. Changed every workload to an odd, prediction-dependent row step, preventing a fixed point modulo 256. Row storage is now variable-width. Remeasured all variants; do not compare these timings directly with the older harness. The traces visit 9/2/13/30 rows respectively, so these are hot synthetic measurements, not broad input-distribution or deployment claims.

Apple M4 Pro, same generic LLVM setup; medians in ns/call, including the roughly 5 ns driver:

| Workload | Reference | Tree order / 1 sum | Scalar rolling4 / 2 sums | SIMD rolling2 / 2 sums |
|---|---:|---:|---:|---:|
| Mixed | 179.07 | 92.25 | 80.43 | 74.38 |
| Reuse-heavy | 176.51 | 52.04 | 51.99 | 75.85 |
| Disjoint | 179.36 | 105.96 | 107.00 | 107.12 |
| Wide | 346.32 | 226.25 | 184.95 | 202.34 |

Scalar rolling4/sums2 compilation: 35.4/13.7/55.0/104.4 ms, after LLVM initialization. Disjoint repeat: tree order 104.76 ns versus scalar rolling4/sums2 107.06 ns; ranges overlap (102.70–107.08 versus 99.61–107.27). Keep the small slower median visible; there is no reuse to exploit in that model.

SIMD is not a blanket win. Reuse-heavy tree order emits 32 scalar comparisons; SIMD cap2 emits 70 packed plus 20 scalar comparisons on both targets. Pair formation prevents some scalar-level common-expression sharing and adds extraction/spill costs. ARM64 text grows 3,484 → 5,532 bytes and total frame 448 → 640 bytes; x86 text grows 5,032 → 8,021 bytes and local stack 672 → 920 bytes (plus 48 saved-register bytes). Mixed SIMD cap2 text is 9,804/11,742 bytes on ARM64/x86-64. No per-model classifier or further backend tuning is justified here.

Choose scalar rolling4/sums2 as the fixed portable default in the next chunk: useful mixed/wide gains, reuse-heavy parity, a small disjoint tradeoff. Keep tree order and SIMD explicit options. This is a measured compromise on one host, not a universal optimum.

## Default and seeded differential check

Default now uses scalar rolling4/sums2. Named tree-order benchmark/inspection controls remain explicit. Mixed rerun: old tree-order default 94.14 ns, selected default 79.98 ns; all row/trace checks pass.

Added one 97-line seeded test: 96 forests, up to eight depth-0–6 trees, 20 rows, and three configurations per forest. Disjoint integer leaf-index fields occupy at most 48 bits, so scores stay exact and uniquely expose every selected leaf. Feature counts, live caps, and accumulator counts cycle independently; no debug API or fuzz dependency. Parent and secondary review accepted the oracle and small representation. Nineteen focused tests pass; parent reran the seeded test in debug/release after removing a generator correlation.

## Obsolete probe cleanup

Removed the Cranelift probe/sweep and LLVM smoke example with their unused dev-dependencies; all three files remain available at revision `ae90c35`. Production code and saved measurement artifacts are unchanged. Current validation uses the model, planner, JIT, differential, and benchmark tests; historical reproduction notes now name the required revision.

All 33 current tests pass in debug/release with `--all-targets`; Clippy passes with `-D warnings`. Cargo refreshed the lockfile offline without upgrading retained packages. The dependency tree has no Cranelift; llvm-sys remains only through Inkwell.

## Final handoff

The README documents the in-memory API, model/row ownership, numeric semantics, selected scalar default, explicit SIMD option, and synthetic measurements with their regressions and short-trace limitations. Its single Rust example is included as crate documentation and executes as a doctest. PLAN.md now marks the agreed in-memory scope complete without claiming deployment performance or x86-64 execution coverage. Prediction code is unchanged.

All 33 tests pass in debug/release with `--all-targets`; the README doctest, Clippy with `-D warnings`, and formatting checks pass offline. On this Homebrew ARM64 setup, the doctest requires `RUSTDOCFLAGS="-C link-arg=/opt/homebrew/lib/libzstd.dylib"`: rustdoc does not inherit the normal Cargo linker path and otherwise selects the existing Intel zstd in `/usr/local/lib`. The documented command links the correct native library explicitly; no source workaround or dependency change was needed.

## Resource-aware planning

Cold model compilation (planning + LLVM) remains higher: scalar 500 trees / 100 features rose from 485.28 to 639.88 ms; 500 / 1000 rose from 606.54 to 819.00 ms. Caching each waiting-sort key avoids repeated score calculation without changing plans; it does not eliminate the compilation regression.

Final same-harness comparison: old cap 4 versus independent GP/FP budgets of 12/12, both with two sums. Five alternating rounds of 50,000 dependent calls; median ns/call includes the driver, without subtracting its floor. All models have depth six, reused features, and globally distinct thresholds.

| Trees / features | Scalar before → after | SIMD before → after |
|---|---:|---:|
| 100 / 100 | 142.40 → 144.34 | 142.79 → 138.36 |
| 500 / 100 | 691.60 → 690.79 | 652.81 → 640.11 |
| 500 / 1000 | 805.41 → 764.56 | 863.48 → 799.70 |

Earlier 500 / 100 scalar medians were 708.53 and 726.37 ns; that regression did not recur in the final 690.79 ns run. No cause or noise explanation is established, and these results are not a universal win. The timed traces visit only 35/13/3 rows respectively; all 256 rows were separately correctness-checked for every kernel.

Raw artifacts: `/private/tmp/tigerboost-sort-after.wfA3TV/`. The final pair is `old-final.txt` / `new-cached-final.txt`; `new1.txt` / `new2.txt` retain the earlier regressions. Default remains scalar, GP/FP 12/12, two sums; the LLVM backend is unchanged. Logical resource limits do not bound LLVM's physical allocation.

Validation passed: 23 standalone planner tests, all 48 tests in both debug and release, and the README doctest. Review covered primitive peaks, mixed-pair destination unions, actual short-tail progress, reload IDs, deterministic metrics, and complete plan/JIT separation.
