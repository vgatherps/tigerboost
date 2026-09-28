status: pass

The experiment is implemented and all required gates pass. Source-order scheduling reduces spills but does not remove them on ARM64, x86 SSE2, or x86 AVX2. The continuation kernel still has zero body stack-memory instructions on all three targets and remains faster on native ARM64.

Nine tracked files changed: `src/jit.rs`, `Cargo.toml`, `Cargo.lock`, `tests/differential.rs`, `tests/scalar_jit.rs`, `examples/scalar_bench.rs`, `examples/cold_bench.rs`, `examples/inspect_code.rs`, and `examples/x86_check.rs`. `tools/x86_check.sh` is unchanged because it already reads every variant from the generated manifest. `VERDICT.md` is the requested untracked report. No commits, merges, stashes, checkouts, or formatting commands were made.

The supplied anchors were checked against HEAD `6cae781bcc69c23d295aac2eb12e87b1f3648e78`. The direct `llvm-sys = "=211.0.1"` dependency matches the version already resolved in `Cargo.lock`.

1. `EmitOptions` now has `single_function`, `backend_schedule`, and `backend_level`. Their defaults preserve continuation emission, LLVM's default scheduler, and `OptimizationLevel::Less`. `EmitOptions::from_host` changes only `avx2`.
2. `BackendSchedule::Source` invokes `LLVMParseCommandLineOptions` through one `std::sync::Once` before target-machine creation. The arguments are exactly `-pre-RA-sched=source`, `-enable-misched=false`, and `-enable-post-misched=false`. Its documentation states that the setting is process-wide and affects later compilations.
3. `BackendLevel::None` selects `OptimizationLevel::None` for both the target machine and MCJIT. `early-cse` still runs before backend emission.
4. Single-function mode emits only C-ABI `predict(row)`. All plan chunks feed the same SSA sums in one basic block. There are no calls, `musttail` transfers, fences, continuations, or volatile loads. The row retains `align`, `dereferenceable`, `readonly`, and `noalias`; `predict` retains `memory(read)`. Threshold and leaf GEPs use absolute slots from the global constants pointer.
5. The continuation branch remains the default. All 18 pre-existing `inspect_code 100x100` files, comprising nine assemblies and nine plan dumps, are byte-identical to the before-edit snapshot.
6. The harnesses include continuation, `single`, `single source`, `single none`, and `single source shared`. Continuation and other default-scheduler variants are compiled before the first Source variant. `cold_bench` accepts `--single`, `--single source`, and `--single none`; it compiles the default continuation used for the other-model disturbance before setting Source.
7. Tests cover disabled-option assembly identity on both triples, the one-function IR shape and attributes, ordinary non-volatile loads, source-order row offsets, and differential execution while cycling both scheduler values and both backend levels. The source-order test runs in a child process so LLVM's process-wide setting cannot contaminate unrelated tests. Its Default probe recorded `[40, 0, 32, 8, 24, 16]`; Source emitted the same IR order.

Static counts include `predict` and every continuation body, excluding directives and data. Row cells are doubles read / memory instructions in final assembly. The CSE column is the number of the original feature loads removed in prepared IR. Threshold and leaf columns count memory instructions; one ARM64 `ldp q` counts as one threshold instruction and transfers two threshold vectors. Body stack excludes ABI saves/restores and stack-pointer arithmetic. Frame bytes include saved registers and explicit allocation, but not implicit x86 return addresses. Prologue size is instructions / encoded bytes.

## ARM64 static counts

| Trees | Variant | Instructions | Row doubles/ins | CSE merged | Threshold loads | `ldp q` | Leaf loads | Body stack | Frame B | Prologue ins/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | continuation | 2143 | 468/429 | 132 | 163 | 137 | 100 | 0 | 160 | 10/40 |
| 100 | single | 3078 | 99/75 | 501 | 300 | 0 | 100 | 421 | 1744 | 6/24 |
| 100 | single source | 3027 | 99/97 | 501 | 300 | 0 | 100 | 282 | 1584 | 6/24 |
| 100 | single none | 3331 | 99/99 | 501 | 300 | 0 | 100 | 664 | 3616 | 6/24 |
| 100 | single source shared | 2979 | 99/99 | 501 | 300 | 0 | 100 | 221 | 1392 | 6/24 |
| 300 | continuation | 6521 | 1419/1315 | 381 | 463 | 437 | 300 | 0 | 160 | 10/40 |
| 300 | single | 9071 | 100/75 | 1700 | 900 | 0 | 300 | 1091 | 2128 | 6/24 |
| 300 | single source | 8946 | 100/100 | 1700 | 900 | 0 | 300 | 874 | 2128 | 6/24 |
| 300 | single none | 10262 | 100/100 | 1700 | 900 | 0 | 300 | 2482 | 11200 | 7/28 |
| 300 | single source shared | 8783 | 100/100 | 1700 | 900 | 0 | 300 | 627 | 1952 | 6/24 |

The 100-tree constants block has 45 leaf bases above 32 KiB; its maximum leaf offset is 55,992 bytes. The 300-tree block has 264 leaf bases above 32 KiB; its maximum leaf offset is 167,992 bytes. Threshold offsets remain below 32 KiB at both sizes. No alternate bases were added. Every single-function mode loses all ARM64 threshold pairing: `ldp q` falls from 137 to zero at 100 trees and from 437 to zero at 300, so threshold memory instructions rise from 163 to 300 and from 463 to 900.

## x86 SSE2 static counts

| Trees | Variant | Instructions | Row doubles/ins | CSE merged | Threshold loads | Leaf loads | Body stack | Frame B | Prologue ins/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | continuation | 2300 | 470/470 | 132 | 300 | 100 | 0 | 48 | 7/11 |
| 100 | single | 3098 | 105/102 | 501 | 300 | 100 | 888 | 3176 | 7/17 |
| 100 | single source | 2867 | 105/102 | 501 | 300 | 100 | 414 | 1432 | 1/7 |
| 100 | single none | 3458 | 105/102 | 501 | 300 | 100 | 902 | 4776 | 1/7 |
| 100 | single source shared | 2871 | 105/102 | 501 | 300 | 100 | 335 | 1336 | 1/7 |
| 300 | continuation | 6867 | 1414/1414 | 381 | 900 | 300 | 0 | 48 | 8/12 |
| 300 | single | 9087 | 118/109 | 1700 | 900 | 300 | 2666 | 5288 | 7/17 |
| 300 | single source | 8322 | 118/109 | 1700 | 900 | 300 | 1214 | 1928 | 1/7 |
| 300 | single none | 10417 | 118/109 | 1700 | 900 | 300 | 3058 | 13912 | 1/7 |
| 300 | single source shared | 8451 | 118/109 | 1700 | 900 | 300 | 934 | 1752 | 1/7 |

## x86 AVX2 static counts

The folded column counts threshold comparisons whose threshold remains a memory operand. One zero threshold uses a register in the 100-tree Less emissions. FastISel under `single none` folds none of the threshold operands.

| Trees | Variant | Instructions | Row doubles/ins | CSE merged | Threshold loads | Folded compares | Leaf loads | Body stack | Frame B | Prologue ins/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | continuation | 1451 | 470/470 | 132 | 200 | 200 | 100 | 0 | 40 | 5/9 |
| 100 | single | 2148 | 107/102 | 501 | 200 | 199 | 100 | 710 | 3624 | 7/17 |
| 100 | single source | 1802 | 107/102 | 501 | 200 | 199 | 100 | 391 | 2568 | 1/7 |
| 100 | single none | 2811 | 107/102 | 501 | 200 | 0 | 100 | 709 | 4888 | 1/7 |
| 100 | single source shared | 1780 | 107/102 | 501 | 200 | 199 | 100 | 311 | 2408 | 1/7 |
| 300 | continuation | 4304 | 1414/1414 | 381 | 600 | 600 | 300 | 0 | 40 | 5/9 |
| 300 | single | 6476 | 130/109 | 1700 | 600 | 600 | 300 | 2437 | 6632 | 7/17 |
| 300 | single source | 5120 | 122/106 | 1700 | 600 | 600 | 300 | 1122 | 3480 | 1/7 |
| 300 | single none | 8326 | 122/106 | 1700 | 600 | 0 | 300 | 2260 | 8664 | 1/7 |
| 300 | single source shared | 5111 | 122/106 | 1700 | 600 | 600 | 300 | 833 | 3384 | 1/7 |

`early-cse` sees the full single-function kernel. At 100 trees, 600 emitted feature loads cover 99 distinct features, so it removes 501. At 300 trees, 1,800 loads cover all 100 features, so it removes 1,700. Final machine row reads can exceed those prepared-IR counts because the backend rematerializes or widens some x86 loads.

## Hot native ARM64 timings

Values are median ns/prediction from five rounds of 50,000 dependent calls, including the driver. Default and None were compiled before Source set the process-wide options.

| Variant | 100 trees | 300 trees |
|---|---:|---:|
| continuation | 90.22 | 242.33 |
| single | 110.03 | 293.42 |
| single source | 103.52 | 283.35 |
| single none | 164.28 | 565.00 |
| single source shared | 95.89 | 257.12 |

No native x86 timing was available on the ARM64 host. The x86 gates assemble, link, and execute every row under Rosetta for correctness; the static x86 tables use the requested Linux triple.

## Cold native ARM64 timings

Values are p50 / p90 / p99 ns from 3,000 retained samples for `100x100`. The timer resolution was about 41 ns. The default and Source runs used separate processes.

| Sequence | Condition | continuation | single source |
|---|---|---:|---:|
| shuffled | hot | 83 / 84 / 125 | 84 / 125 / 125 |
| shuffled | 64 MiB | 875 / 1084 / 2625 | 1000 / 1458 / 4083 |
| shuffled | other model | 208 / 250 / 291 | 250 / 292 / 334 |
| dependent | hot | 83 / 84 / 125 | 84 / 125 / 125 |
| dependent | 64 MiB | 875 / 1125 / 2708 | 1084 / 3000 / 4500 |
| dependent | other model | 208 / 250 / 292 | 292 / 334 / 417 |

## Validation

| Gate | Result |
|---|---|
| Before-edit release `inspect_code 100x100` | Pass; 18-file snapshot at `/tmp/tb-sched-before.VxqHiG/output` |
| `cargo build --offline --release --examples` | Pass |
| `cargo test --offline --all-targets` | Pass; 66 tests |
| `cargo test --offline --release --all-targets` | Pass; 66 tests |
| `cargo test --offline --doc` | Pass; 1 doctest |
| `cargo clippy --offline --all-targets -- -D warnings` | Pass |
| After-edit release `inspect_code 100x100` | Pass; all requested variants emitted |
| Byte comparison of every pre-existing output | Pass; 18/18 identical |
| Release `inspect_code 300x100` | Pass; all requested variants emitted |
| `tools/x86_check.sh 100x100` | Pass; 8 variants × 256 rows |
| `tools/x86_check.sh special --avx2` | Pass; 8 variants × 256 rows |
| Release `scalar_bench 100x100` and `300x100` | Pass; all predictions verified and variants measured |
| Release `cold_bench 100x100 --samples 3000` | Pass |
| Release cold run with `--single source` | Pass |
| `git diff --check` | Pass |

Validation ran on native ARM64, macOS 15.6, LLVM 21.1.8, and rustc 1.93.0-nightly. The required `LLVM_SYS_211_PREFIX` and `RUSTDOCFLAGS` were set. The x86 linker printed its existing `-no_pie` deprecation and pointer-alignment warnings; every correctness check passed.

After Source is set, a later `BackendSchedule::Default` request is not byte-identical to the baseline on ARM64, generic x86, or x86 AVX2. This is the documented consequence of LLVM's process-wide options. Pre-existing and Default variants are therefore emitted first, and benchmark modes that need isolation run in separate processes.

Source-order scheduling does not remove the unchunked spills on any target. It reduces them, most strongly on x86 AVX2, but leaves 282/874 ARM64, 414/1214 x86 SSE2, and 391/1122 x86 AVX2 body stack-memory instructions at 100/300 trees. It also leaves the single-function kernels with more instructions than continuation: 3027 versus 2143 on ARM64, 2867 versus 2300 on SSE2, and 1802 versus 1451 on AVX2 at 100 trees. On native ARM64, Source takes 103.52 versus 90.22 ns at 100 trees and 283.35 versus 242.33 ns at 300 trees. The continuation design remains better on spills, instruction count, and measured hot and cold time.
