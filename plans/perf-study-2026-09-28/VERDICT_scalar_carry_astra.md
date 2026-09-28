status: pass

Carry6 is implemented and remains off by default. It adds instructions in every measured variant. ARM64 hot medians regress at 100 trees and with shared-feature ordering at both sizes. Model order at 300 trees improves by 0.32%; its timing ranges overlap the control.

Work started at master `6cae781bcc69c23d295aac2eb12e87b1f3648e78`. LLVM is 21.1.8; rustc is 1.93.0-nightly (c90bcb957, 2025-11-06); the execution host is ARM64. No commits, formatter runs, sub-agents, or edits outside the seven allowed source files and report/probe artifacts.

The source changes are:

| File | Change |
| --- | --- |
| `src/jit.rs` | Adds `EmitOptions.carry_features`, default false, including `from_host()`. With carry enabled and fewer than eight sums, every continuation gets exactly `8 - sum_count` trailing scalar doubles. Wrapper and unused slots pass `undef`. Incoming values replace lane loads; newly loaded values remain available for the next boundary. Selection counts next-chunk lane fills, including actual padding reuse, then breaks ties by feature index. |
| `tests/scalar_jit.rs` | Checks master assembly byte lengths and FNV-1a fingerprints for both triples, both AVX2 settings, and both tree orders. Also checks carried NaN-normalized values and repeated predictions through the same row allocation. Fingerprints are pinned to master with LLVM 21.1.8. |
| `tests/differential.rs` | Runs carry off/on and both orderings for all 96 seeded forests, three chunk sizes, special values, and exact leaf-index encodings. |
| `examples/scalar_bench.rs` | Adds the two requested carry6 labels; verifies every variant before timing. |
| `examples/cold_bench.rs` | Adds `--carry`, rejects duplicates, records the option in the report, and tests the complete benchmark path with carry off/on. |
| `examples/inspect_code.rs` | Adds `tree-major-8-2sums-carry6` and `tree-major-8-2sums-carry6-shared`, with both targets and x86 AVX2 twins. Existing filenames and contents remain unchanged. |
| `examples/x86_check.rs` | Adds both carry variants to native verification, assembly generation, and the execution manifest. |

Tests inside `src/jit.rs` inspect production IR before and after `early-cse`. They check scalar signatures and replaced loads across ten chunks with 1, 2, 7, 8, and 9 sums; `undef` slots; byte-identical carry on/off output when sums are at least eight; frequency ranking, feature-index ties, target-specific padding, and a feature carried through two boundaries. Sums remain scalar, loads remain ordinary, `early-cse` remains enabled, and transfers remain `musttail`.

All requested gates passed with `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'`:

| Gate | Result and evidence |
| --- | --- |
| BEFORE `inspect_code`, 100x100 | Captured before edits in `/tmp/tb-c6-before-62989`; 18 files. |
| `cargo build --offline --release --examples` | Pass; [build log](probes/build.log). |
| `cargo test --offline --all-targets` | Pass, 67 tests; [debug log](probes/test-debug.log). |
| `cargo test --offline --release --all-targets` | Pass, 67 tests; [release log](probes/test-release.log). |
| `cargo test --offline --doc` | Pass, 1 test; [doc log](probes/test-doc.log). |
| `cargo clippy --offline --all-targets -- -D warnings` | Pass; [Clippy log](probes/clippy.log). |
| AFTER `inspect_code`, 100x100 and 300x100 | Pass; [100-tree log](probes/inspect-100.log), [300-tree log](probes/inspect-300.log). |
| `cmp` every pre-existing output | All 18 files byte-identical, including assembly and plans; [comparison log](probes/byte-identity.log). |
| `tools/x86_check.sh 100x100` | All 6 variants pass, 256 rows each; [log](probes/x86-100.log). |
| `tools/x86_check.sh special --avx2` | All 6 variants pass, 256 rows each; [log](probes/x86-special-avx2.log). |
| `scalar_bench`, 100x100 and 300x100 | Pass; [100-tree log](probes/hot-100.log), [300-tree log](probes/hot-300.log). |
| `cold_bench`, 100x100, 3000 samples, default and `--carry` | Pass; [default log](probes/cold-off.log), [carry log](probes/cold-carry.log). |
| `git diff --check` | Pass. |

The x86 linker prints `-no_pie` deprecation and unaligned-pointer warnings for control and carry variants; both execution gates return success. During development, the new load-count assertion initially missed the optimized feature-zero address, and Clippy rejected arithmetic in a test expectation. Both tests were corrected; the implementation was unchanged by those corrections.

Assembly counts follow. All kernels use eight trees per chunk and two sums. ARM64 means `aarch64-apple-darwin`; SSE2 and AVX2 mean `x86_64-unknown-linux-gnu` with AVX2 off/on. Instructions include `predict` and every continuation, excluding alignment padding and directives. The wrapper contributes 26 instructions on ARM64 and 15 on x86 in every variant.

Row loads count assembly instructions that read the row: an ARM64 pair load counts once, and an x86 instruction with a folded row read counts once. IR loads count scalar row loads after `early-cse`. Carry mean/max count non-`undef` feature arguments over the 12 or 37 continuation boundaries. Added register moves are the change versus the same-order control in register-to-register `mov`/`fmov` instructions, including lane copies and excluding x86 movemask operations. Stack operations count memory-access instructions inside continuations, including folded reloads and pair operations once; wrapper saves/restores are excluded.

[The IR probe](probes/emit_ir.rs) includes the production emitter and calls `prepare_module`; all 24 probe assemblies match both the linked production crate and `inspect_code` output. [The counting script](probes/count_assembly.py), [machine-readable counts](probes/counts.json), [assembly and IR](probes/ir/), and [comparison log](probes/probe-identity.log) preserve the evidence.

ARM64 counts:

| Trees | Kernel | Instructions | Row loads | IR loads | Carry mean / max | Added register moves | Stack ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | continuation | 2143 | 429 | 468 | 0.00 / 0 | +0 | 0 |
| 100 | carry6 | 2195 | 374 | 396 | 6.00 / 6 | +137 | 0 |
| 100 | shared order | 2086 | 332 | 358 | 0.00 / 0 | +0 | 0 |
| 100 | carry6 + shared order | 2112 | 279 | 297 | 5.08 / 6 | +101 | 2 |
| 300 | continuation | 6521 | 1315 | 1419 | 0.00 / 0 | +0 | 0 |
| 300 | carry6 | 6621 | 1109 | 1197 | 6.00 / 6 | +402 | 0 |
| 300 | shared order | 6273 | 886 | 974 | 0.00 / 0 | +0 | 0 |
| 300 | carry6 + shared order | 6388 | 710 | 770 | 5.51 / 6 | +372 | 7 |

SSE2 counts:

| Trees | Kernel | Instructions | Row loads | IR loads | Carry mean / max | Added register moves | Stack ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | continuation | 2300 | 470 | 468 | 0.00 / 0 | +0 | 6 |
| 100 | carry6 | 2422 | 397 | 396 | 6.00 / 6 | +123 | 49 |
| 100 | shared order | 2310 | 360 | 358 | 0.00 / 0 | +0 | 13 |
| 100 | carry6 + shared order | 2452 | 301 | 297 | 5.08 / 6 | +102 | 98 |
| 300 | continuation | 6867 | 1414 | 1419 | 0.00 / 0 | +0 | 9 |
| 300 | carry6 | 7166 | 1193 | 1197 | 6.00 / 6 | +334 | 103 |
| 300 | shared order | 6940 | 977 | 974 | 0.00 / 0 | +0 | 76 |
| 300 | carry6 + shared order | 7444 | 774 | 770 | 5.51 / 6 | +364 | 371 |

AVX2 counts:

| Trees | Kernel | Instructions | Row loads | IR loads | Carry mean / max | Added register moves | Stack ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | continuation | 1451 | 470 | 468 | 0.00 / 0 | +0 | 0 |
| 100 | carry6 | 1489 | 397 | 396 | 6.00 / 6 | +55 | 25 |
| 100 | shared order | 1405 | 360 | 358 | 0.00 / 0 | +0 | 0 |
| 100 | carry6 + shared order | 1436 | 301 | 297 | 5.08 / 6 | +35 | 30 |
| 300 | continuation | 4304 | 1414 | 1419 | 0.00 / 0 | +0 | 0 |
| 300 | carry6 | 4383 | 1193 | 1197 | 6.00 / 6 | +147 | 33 |
| 300 | shared order | 4166 | 977 | 974 | 0.00 / 0 | +0 | 0 |
| 300 | carry6 + shared order | 4329 | 774 | 770 | 5.51 / 6 | +137 | 146 |

ARM64 hot medians are ns/prediction, five rounds of 50,000 dependent calls, with the driver included. The workload uses 100 features and 256 rows. The dependent chains visit 35 distinct rows at 100 trees and 12 at 300 trees.

| Trees | Continuation | Carry6 | Shared order | Carry6 + shared order |
| --- | ---: | ---: | ---: | ---: |
| 100 | 88.39 | 88.55 | 85.29 | 87.12 |
| 300 | 239.79 | 239.03 | 228.17 | 236.77 |

Compared with the same order without carry, model-order carry6 is 0.18% slower at 100 trees and 0.32% faster at 300. Shared-order carry6 is 2.15% slower at 100 and 3.77% slower at 300. The 300-tree model-order ranges are 236.21–242.62 ns without carry and 237.66–241.67 ns with carry; the small median improvement does not establish a repeatable speedup.

Cold percentiles below are ns for 100x100, 3000 samples per condition. Default and carry ran separately. Timer resolution was about 41 ns; timer-only rows are included as measured.

Cold benchmark, shuffled rows:

| Condition | Default p50 | Default p90 | Default p99 | Carry p50 | Carry p90 | Carry p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hot | 83 | 84 | 125 | 83 | 125 | 125 |
| data 256KiB | 167 | 209 | 250 | 167 | 209 | 250 |
| data 4MiB | 167 | 209 | 375 | 167 | 208 | 292 |
| data 16MiB | 458 | 584 | 1917 | 292 | 458 | 1334 |
| data 64MiB | 875 | 2000 | 3333 | 875 | 2208 | 3334 |
| other model | 208 | 209 | 250 | 208 | 250 | 292 |
| timer only | 0 | 42 | 42 | 0 | 42 | 42 |
| timer only after data 64MiB | 0 | 42 | 42 | 0 | 42 | 42 |

Cold benchmark, dependent rows:

| Condition | Default p50 | Default p90 | Default p99 | Carry p50 | Carry p90 | Carry p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hot | 83 | 84 | 125 | 83 | 84 | 125 |
| data 256KiB | 167 | 209 | 250 | 167 | 209 | 209 |
| data 4MiB | 167 | 208 | 250 | 167 | 208 | 209 |
| data 16MiB | 417 | 583 | 1750 | 292 | 417 | 1834 |
| data 64MiB | 833 | 1958 | 3500 | 875 | 2042 | 3833 |
| other model | 208 | 250 | 292 | 208 | 250 | 292 |
| timer only | 0 | 42 | 42 | 0 | 42 | 42 |
| timer only after data 64MiB | 0 | 42 | 42 | 0 | 42 | 42 |

On ARM64, model-order carry6 removes 55 / 206 row-load instructions and adds 52 / 100 instructions at 100 / 300 trees; shared-order carry6 removes 53 / 176 and adds 26 / 115. Carry6 does not beat its same-order ARM64 hot control at 100 trees or with shared ordering at either size. It narrowly beats the model-order control at 300 trees by the measured median, with overlapping timing ranges.
