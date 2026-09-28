status: complete

Round 4 is implemented on `aafd45b`; all required gates pass. It does not enforce a physical register budget: ARM64 spills at every tested K, and ordinary row loads change during machine-code generation. The emitter still produces exactly the Belady miss count in IR. The default continuation outputs remain byte-identical to `5a915cd`. Nothing was committed or merged.

Tracked files changed: `src/jit.rs`, `tests/differential.rs`, `examples/scalar_bench.rs`, `examples/cold_bench.rs`, `examples/inspect_code.rs`, `examples/x86_check.rs`. `src/sort_policy.rs`, `tests/scalar_jit.rs`, and `tools/x86_check.sh` are unchanged. The existing scheduled dropped-model test remains active; the shell checker reads the renamed variants from its manifest. Measurement sources, logs, counts, and x86 binaries are under `probes/`.

The amendment supersedes the volatile-load experiment. Per-item implementation notes follow.

| Item | Implementation and validation |
| --- | --- |
| 1 | `EmitOptions` contains `avx2`, `scheduled`, and `resident_features`; defaults remain false, false, and 12. `volatile_leaves` and all emitter volatility marking are removed. `early-cse` still runs only for continuation mode. |
| 2 | The C wrapper `predict(row)` calls internal C `noinline kernel(row, constants)`. Both functions have `memory(read)`; row and constants parameters have `readonly`, `noalias`, alignment and bounds. The kernel receives per-chunk threshold and leaf GEPs based on its constants argument. Assembly still materializes some large addresses and loses most threshold pairing after early chunks; counts below. |
| 3 | Exactly `chunks − 1` calls to `asm sideeffect "", ""()` appear between chunks. No operands, clobbers, or emitted instructions. A 13-tree production-path test uses feature 0 in trees 0 and 12: K1 loads it on both sides of the boundary; K64 loads it once. Assembly checks pass for ARM64, SSE2, and AVX2. |
| 4 | Scheduled leaf blocks follow `EvaluateTree`/`Retire` visit order, while `leaf_base` remains indexed by `TreeId`. A reordered mixed-depth model verifies constant contents, chunk-relative GEPs, and predictions, including its constant tree and odd-depth trees. Continuation layout is unchanged. |
| 5 | Round-1 `feature_fills`, `residency_schedule`, and eviction tie-breaking are unchanged. Cross-boundary hits reuse the same SSA value. The K2 sequence A B C A B C misses on fills 0, 1, 2, and 4; the no-next-use tie test evicts the lower feature index. |
| 6 | Benchmarks use K8/K12/K16/K24 barrier and K16 shared barrier. `cold_bench --scheduled [K]` selects this implementation. `inspect_code --sweep` adds K1–K24 emissions on all three targets. `x86_check` names are `scheduled-k16-barrier` and `scheduled-k16-barrier-shared`. Fenced and volatile-leaf variants are absent. |
| 7 | Tests check wrapper/kernel attributes, empty barriers, IR miss counts and inserted-value provenance, both orders and NaN modes, scheduled-off identity on both triples with AVX2 on/off, reference predictions at K1/K4/K16/K64, and model lifetime. The workload IR probe adds 288 checks: two sizes × two orders × 24 budgets × three targets. |

Host was ARM64 macOS 15.6; Rust 1.93.0-nightly (c90bcb957), LLVM 21.1.8, inkwell 0.9.0. Gates ran offline with `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS=-C link-arg=/opt/homebrew/lib/libzstd.dylib`. No formatter, stash, checkout, or commit was used. The baseline came from a `git archive 5a915cd` scratch source tree; its 72 source files were verified against that commit before cleanup. The scratch source tree was removed; baseline output remains in `/tmp/tb-residency4-base-out`.

| Gate | Result |
| --- | --- |
| `git stash list`; `git show 5a915cd:src/jit.rs` | Empty; success |
| Scratch baseline `inspect_code 100x100` | 18 files emitted |
| `cargo build --offline --release --examples` | Pass |
| `cargo test --offline --all-targets` | 70 passed |
| `cargo test --offline --release --all-targets` | 70 passed |
| `cargo test --offline --doc` | 1 passed |
| `cargo clippy --offline --all-targets -- -D warnings` | Pass; initial style diagnostics in the edited emitter were corrected |
| Final `cargo run ... inspect_code ... 100x100` + `cmp` | 18/18 baseline files identical; all final scheduled files also match the measured sweep |
| `inspect_code 100x100 --sweep`; `300x100 --sweep` | 168 assemblies; both sizes, three targets, K1–K24 plus shared K16 and continuation controls |
| `tools/x86_check.sh 100x100` | 6/6 variants, each 256 rows |
| `tools/x86_check.sh special --avx2` | 6/6 variants, each 256 rows |
| `scalar_bench 100x100`; `300x100` | Pass; all variants verified before timing |
| `cold_bench ... --samples 3000` | Default, K8, K16, K16 shared at both sizes; every condition and both row sequences complete |
| Additional x86 AOT timing probe | SSE2 and AVX2 under Rosetta; six variants at both sizes. Every row and a 50,000-call dependent trace checked; 2,000-tree disturbance model checked too. |

Gate logs are `probes/tb-residency4-*.log`; full measurements are `probes/hot-arm-*.log`, `probes/cold-arm-*.log`, and `probes/timing-x86-*.log`. The x86 linker issued deprecated `-no_pie` and unaligned absolute-pointer warnings; assembly, linking, and execution succeeded.

Counts come from the emitted assembly, following row/constants pointers and spilled address registers. Every file has its expected number of threshold vectors and leaf loads, with no unclassified memory references. LLVM object disassembly independently agrees with the K16 instruction totals and prologue byte counts on all three targets. Full counts and scripts: `probes/assembly-counts.csv`, `probes/count_assembly.py`, `probes/ir-counts.csv`.

In the tables, instructions include the wrapper plus all kernel/continuation instructions, excluding directives, padding, and empty asm comments. Row counts are IR loads / assembly scalar values read / assembly memory instructions: an ARM64 `ldp d` reads two values. Threshold counts are memory instructions, including folded compares; `ldp q` reads two threshold vectors. Leaf counts include folded additions. Inserts count register lane fills, including x86 register `vinsertf128`; memory lane fills are listed separately. Stack is body instructions / all explicit kernel stack-memory instructions; body excludes callee-save prologue and epilogue. Frames are maximum kernel allocation plus x86 red-zone bytes used. Prologues are bytes summed across kernel/continuation functions. Wrapper costs are separate below. Empty barriers add zero machine instructions.

100 trees, depth 6, 100 features; eight trees per chunk and two sums. All scheduled rows use barriers.

| Target | Variant | Instructions | Row IR / values / insns | Inserts reg / mem | Threshold | Leaf | Stack body / all | Frame + red zone B | Prologue B | `ldp q` threshold / all | Immediate materializations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM64 | Continuation | 2143 | — / 468 / 429 | 125 / 172 | 163 | 100 | 0 / 0 | 0 + 0 | 0 | 137 / 137 | 47 |
| ARM64 | K8 barrier | 2261 | 423 / 406 / 376 | 160 / 137 | 273 | 100 | 119 / 139 | 496 + 0 | 44 | 27 / 28 | 13 |
| ARM64 | K12 barrier | 2286 | 375 / 370 / 342 | 181 / 116 | 275 | 100 | 146 / 166 | 512 + 0 | 44 | 25 / 25 | 13 |
| ARM64 | K16 barrier | 2358 | 338 / 337 / 310 | 194 / 99 | 276 | 100 | 201 / 221 | 592 + 0 | 44 | 24 / 24 | 13 |
| ARM64 | K24 barrier | 2434 | 281 / 281 / 261 | 225 / 68 | 277 | 100 | 305 / 321 | 752 + 0 | 36 | 23 / 31 | 13 |
| ARM64 | K16 shared barrier | 2276 | 268 / 268 / 250 | 235 / 58 | 275 | 100 | 163 / 177 | 560 + 0 | 32 | 25 / 28 | 13 |
| SSE2 | Continuation | 2300 | — / 470 / 470 | 120 / 174 | 300 | 100 | 0 / 4 | 8 + 0 | 2 | 0 / 0 | 25 |
| SSE2 | K8 barrier | 2292 | 423 / 407 / 407 | 151 / 139 | 300 | 100 | 27 / 29 | 8 + 96 | 1 | 0 / 0 | 1 |
| SSE2 | K12 barrier | 2353 | 375 / 370 / 370 | 163 / 117 | 300 | 100 | 85 / 97 | 72 + 128 | 14 | 0 / 0 | 1 |
| SSE2 | K16 barrier | 2418 | 338 / 337 / 337 | 168 / 100 | 300 | 100 | 203 / 215 | 232 + 128 | 17 | 0 / 0 | 1 |
| SSE2 | K24 barrier | 2465 | 281 / 282 / 282 | 183 / 68 | 300 | 100 | 299 / 311 | 360 + 128 | 17 | 0 / 0 | 1 |
| SSE2 | K16 shared barrier | 2381 | 268 / 271 / 271 | 218 / 58 | 300 | 100 | 151 / 163 | 200 + 128 | 17 | 0 / 0 | 1 |
| AVX2 | Continuation | 1451 | — / 470 / 470 | 200 / 162 | 200 | 100 | 0 / 0 | 0 + 0 | 0 | 0 / 0 | 12 |
| AVX2 | K8 barrier | 1388 | 423 / 407 / 407 | 217 / 130 | 200 | 100 | 0 / 12 | 48 + 0 | 10 | 0 / 0 | 0 |
| AVX2 | K12 barrier | 1394 | 375 / 370 / 370 | 234 / 108 | 200 | 100 | 18 / 30 | 72 + 128 | 14 | 0 / 0 | 0 |
| AVX2 | K16 barrier | 1517 | 338 / 337 / 337 | 238 / 92 | 200 | 100 | 153 / 165 | 296 + 120 | 17 | 0 / 0 | 0 |
| AVX2 | K24 barrier | 1611 | 281 / 282 / 282 | 245 / 64 | 200 | 100 | 294 / 306 | 664 + 120 | 17 | 0 / 0 | 0 |
| AVX2 | K16 shared barrier | 1421 | 268 / 271 / 271 | 280 / 55 | 200 | 100 | 99 / 111 | 216 + 120 | 17 | 0 / 0 | 0 |

300 trees, depth 6, 100 features; eight trees per chunk and two sums. All scheduled rows use barriers.

| Target | Variant | Instructions | Row IR / values / insns | Inserts reg / mem | Threshold | Leaf | Stack body / all | Frame + red zone B | Prologue B | `ldp q` threshold / all | Immediate materializations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM64 | Continuation | 6521 | — / 1419 / 1315 | 359 / 532 | 463 | 300 | 0 / 0 | 0 + 0 | 0 | 437 / 437 | 92 |
| ARM64 | K8 barrier | 6951 | 1242 / 1214 / 1128 | 492 / 398 | 872 | 300 | 685 / 705 | 1168 + 0 | 44 | 28 / 28 | 13 |
| ARM64 | K12 barrier | 7101 | 1087 / 1076 / 1003 | 566 / 324 | 874 | 300 | 789 / 809 | 1408 + 0 | 44 | 26 / 26 | 13 |
| ARM64 | K16 barrier | 7299 | 964 / 959 / 894 | 614 / 277 | 874 | 300 | 972 / 992 | 1456 + 0 | 44 | 26 / 34 | 13 |
| ARM64 | K24 barrier | 7476 | 780 / 779 / 727 | 705 / 187 | 875 | 300 | 1187 / 1207 | 1408 + 0 | 44 | 25 / 33 | 13 |
| ARM64 | K16 shared barrier | 6952 | 681 / 680 / 635 | 774 / 108 | 877 | 300 | 654 / 674 | 832 + 0 | 44 | 23 / 26 | 13 |
| SSE2 | Continuation | 6867 | — / 1414 / 1414 | 364 / 527 | 900 | 300 | 0 / 6 | 8 + 0 | 3 | 0 / 0 | 75 |
| SSE2 | K8 barrier | 6773 | 1242 / 1211 / 1211 | 497 / 392 | 900 | 300 | 11 / 17 | 24 + 32 | 5 | 0 / 0 | 1 |
| SSE2 | K12 barrier | 7016 | 1087 / 1075 / 1075 | 521 / 320 | 900 | 300 | 324 / 336 | 120 + 128 | 14 | 0 / 0 | 1 |
| SSE2 | K16 barrier | 7232 | 964 / 958 / 958 | 560 / 274 | 900 | 300 | 631 / 643 | 216 + 128 | 17 | 0 / 0 | 1 |
| SSE2 | K24 barrier | 7339 | 780 / 778 / 778 | 578 / 185 | 900 | 300 | 898 / 910 | 360 + 128 | 17 | 0 / 0 | 1 |
| SSE2 | K16 shared barrier | 7153 | 681 / 683 / 683 | 731 / 109 | 900 | 300 | 490 / 502 | 200 + 128 | 17 | 0 / 0 | 1 |
| AVX2 | Continuation | 4304 | — / 1414 / 1414 | 602 / 513 | 600 | 300 | 0 / 0 | 0 + 0 | 0 | 0 / 0 | 37 |
| AVX2 | K8 barrier | 4129 | 1242 / 1211 / 1211 | 696 / 374 | 600 | 300 | 0 / 10 | 40 + 0 | 9 | 0 / 0 | 0 |
| AVX2 | K12 barrier | 4110 | 1087 / 1075 / 1075 | 738 / 300 | 600 | 300 | 32 / 44 | 48 + 120 | 10 | 0 / 0 | 0 |
| AVX2 | K16 barrier | 4573 | 964 / 958 / 958 | 741 / 251 | 600 | 300 | 548 / 560 | 344 + 128 | 17 | 0 / 0 | 0 |
| AVX2 | K24 barrier | 4790 | 780 / 778 / 778 | 723 / 168 | 600 | 300 | 882 / 894 | 760 + 128 | 17 | 0 / 0 | 0 |
| AVX2 | K16 shared barrier | 4196 | 681 / 683 / 683 | 853 / 101 | 600 | 300 | 300 / 312 | 296 + 120 | 17 | 0 / 0 | 0 |

Immediate materializations count ARM64 integer `mov`-immediate/`movk`/`movz`/`movn`, and x86 `movabsq`, inside kernels. Address `add` instructions with encodable immediates remain part of total instructions. For example, 100-tree ARM64 K16 still materializes 4800, 8896, 12992, and subsequent chunk leaf addresses. It has 24 threshold `ldp q` instructions versus 137 for continuation; at 300 trees, 26 versus 437. Later threshold loads use offsets such as 1104 directly from x1; the chunk GEPs have folded back into offsets from the shared constants argument. Relative GEPs in IR did not preserve small addressing offsets throughout the emitted kernel.

| Target | Wrapper | Instructions | Stack instructions | Frame B | Prologue B |
| --- | --- | --- | --- | --- | --- |
| ARM64 | Continuation | 26 | 20 | 160 | 40 |
| ARM64 | Scheduled | 6 | 2 | 16 | 4 |
| SSE2 | Continuation | 15 | 10 | 40 | 9 |
| SSE2 | Scheduled | 6 | 2 | 8 | 1 |
| AVX2 | Continuation | 15 | 10 | 40 | 9 |
| AVX2 | Scheduled | 6 | 2 | 8 | 1 |

Wrapper costs are the same at both sizes and all scheduled budgets. Continuation kernels themselves have no body stack traffic in these workloads.

The 100-tree ARM64 K16 kernel emits its first `fadd` only after the second barrier. Before the first barrier, eight leaf values are stored to the stack. The boundary orders memory instructions, but sum arithmetic can be scheduled later, so leaf values and address registers also consume registers beyond the K feature slots.

Belady misses below equal prepared-IR row loads on all three targets. Machine-code counts need not equal them: for 100-tree K8, 423 IR loads become 406 ARM64 scalar reads and 407 x86 reads. At K16 shared, 268 IR loads become 271 x86 reads. Removing volatility permits backend merging and duplication despite skipping `early-cse`. No workaround was added.

| K | 100 model order | 100 shared order | 300 model order | 300 shared order |
| --- | --- | --- | --- | --- |
| 1 | 595 | 592 | 1775 | 1768 |
| 2 | 550 | 510 | 1637 | 1511 |
| 3 | 517 | 455 | 1540 | 1335 |
| 4 | 493 | 423 | 1462 | 1212 |
| 5 | 472 | 396 | 1397 | 1118 |
| 6 | 454 | 376 | 1340 | 1044 |
| 7 | 438 | 361 | 1289 | 987 |
| 8 | 423 | 347 | 1242 | 936 |
| 9 | 409 | 334 | 1198 | 894 |
| 10 | 397 | 322 | 1158 | 857 |
| 11 | 386 | 310 | 1121 | 822 |
| 12 | 375 | 299 | 1087 | 789 |
| 13 | 364 | 288 | 1053 | 758 |
| 14 | 354 | 280 | 1021 | 730 |
| 15 | 346 | 274 | 992 | 704 |
| 16 | 338 | 268 | 964 | 681 |
| 17 | 330 | 263 | 938 | 662 |
| 18 | 323 | 258 | 913 | 645 |
| 19 | 316 | 253 | 889 | 628 |
| 20 | 309 | 248 | 866 | 611 |
| 21 | 302 | 243 | 843 | 594 |
| 22 | 295 | 238 | 821 | 578 |
| 23 | 288 | 233 | 800 | 564 |
| 24 | 281 | 228 | 780 | 550 |

ARM64 spill sweep in model order. Stack traffic excludes callee saves/restores; `ldp q` counts threshold pairs only.

| K | 100 stack body | 100 frame B | 100 `ldp q` | 300 stack body | 300 frame B | 300 `ldp q` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 140 | 624 | 26 | 696 | 1104 | 29 |
| 2 | 148 | 640 | 27 | 709 | 1216 | 30 |
| 3 | 113 | 592 | 26 | 685 | 1088 | 29 |
| 4 | 145 | 592 | 26 | 663 | 1056 | 29 |
| 5 | 131 | 528 | 26 | 713 | 1216 | 29 |
| 6 | 134 | 560 | 26 | 643 | 1040 | 29 |
| 7 | 157 | 592 | 28 | 661 | 1040 | 29 |
| 8 | 119 | 496 | 27 | 685 | 1168 | 28 |
| 9 | 141 | 496 | 27 | 665 | 1008 | 27 |
| 10 | 136 | 464 | 26 | 678 | 1040 | 27 |
| 11 | 152 | 576 | 24 | 734 | 1120 | 27 |
| 12 | 146 | 512 | 25 | 789 | 1408 | 26 |
| 13 | 159 | 544 | 25 | 800 | 1280 | 26 |
| 14 | 164 | 512 | 26 | 871 | 1360 | 24 |
| 15 | 189 | 656 | 24 | 908 | 1376 | 25 |
| 16 | 201 | 592 | 24 | 972 | 1456 | 26 |
| 17 | 226 | 544 | 25 | 980 | 1472 | 25 |
| 18 | 246 | 656 | 25 | 1033 | 1440 | 26 |
| 19 | 232 | 592 | 23 | 1046 | 1472 | 27 |
| 20 | 245 | 640 | 25 | 1128 | 1472 | 26 |
| 21 | 280 | 832 | 23 | 1163 | 1504 | 26 |
| 22 | 249 | 672 | 24 | 991 | 1232 | 22 |
| 23 | 261 | 624 | 25 | 1208 | 1440 | 23 |
| 24 | 305 | 752 | 23 | 1187 | 1408 | 25 |

First observed body spills and budgets that still reduce assembly row reads without body spills follow. These are workload-dependent observations; SSE2 at 300 trees spills at K5 but returns to zero body spills at K6.

| Target | Trees | First K with body spills | K with fewer row reads and no body spills |
| --- | --- | --- | --- |
| ARM64 | 100 | 1 | None |
| ARM64 | 300 | 1 | None |
| SSE2 | 100 | 2 | None |
| SSE2 | 300 | 5 | 2, 3, 4, 6 |
| AVX2 | 100 | 9 | 2, 3, 4, 5, 6, 7, 8 |
| AVX2 | 300 | 10 | 2, 3, 4, 5, 6, 7, 8, 9 |

Hot medians are ns/prediction, including the dependent-row driver; five rounds of 50,000 calls, after warmup. ARM64 uses the requested Rust JIT harness. x86 uses separately emitted Mach-O code under Rosetta and `probes/x86_timing.c`; these are translated timings, not native x86 hardware results. Mach-O and Linux emissions have identical instruction totals and row-read counts for all timed variants. Compare variants within a target/harness, not timings across targets. Timings are indicative; cold tails also reflect host activity during this experiment.

| Target | Variant | 100 median | 100 min–max | 300 median | 300 min–max |
| --- | --- | --- | --- | --- | --- |
| ARM64 | Continuation | 90.58 | 89.58–92.58 | 239.23 | 238.38–241.97 |
| ARM64 | K8 barrier | 92.58 | 91.91–94.47 | 256.12 | 255.09–257.72 |
| ARM64 | K12 barrier | 93.78 | 92.36–94.84 | 253.44 | 251.68–256.13 |
| ARM64 | K16 barrier | 91.53 | 90.98–93.09 | 254.94 | 253.24–257.80 |
| ARM64 | K24 barrier | 93.99 | 93.46–95.18 | 255.43 | 254.30–257.44 |
| ARM64 | K16 shared barrier | 91.72 | 89.20–94.02 | 238.16 | 238.05–242.15 |
| SSE2 | Continuation | 111.86 | 111.31–113.53 | 312.67 | 311.94–313.36 |
| SSE2 | K8 barrier | 113.20 | 112.64–116.52 | 313.84 | 312.55–329.46 |
| SSE2 | K12 barrier | 112.75 | 112.01–114.38 | 319.22 | 318.41–319.92 |
| SSE2 | K16 barrier | 112.87 | 111.58–113.72 | 313.01 | 310.80–314.07 |
| SSE2 | K24 barrier | 113.40 | 112.58–115.54 | 317.05 | 313.38–319.39 |
| SSE2 | K16 shared barrier | 111.53 | 111.25–112.30 | 312.22 | 310.97–313.11 |
| AVX2 | Continuation | 420.87 | 416.21–543.05 | 1333.81 | 1333.20–1339.33 |
| AVX2 | K8 barrier | 396.04 | 394.69–511.14 | 1275.76 | 1274.15–1283.80 |
| AVX2 | K12 barrier | 396.63 | 396.36–402.40 | 1232.74 | 1230.16–1239.66 |
| AVX2 | K16 barrier | 377.66 | 372.60–378.16 | 1166.06 | 1162.42–1173.39 |
| AVX2 | K24 barrier | 384.23 | 383.63–384.69 | 1204.26 | 1199.14–1236.10 |
| AVX2 | K16 shared barrier | 380.44 | 379.17–382.53 | 1189.01 | 1181.65–1190.80 |

Cold entries are p50 / p90 / p99 in ns; 3,000 retained single-call samples after a discarded pass. Rows are copied into one buffer after disturbance. The 64 MiB condition visits shuffled cache lines; the other-model condition executes a separate 2,000-tree continuation model. The ARM harness runs all its original conditions; the extra x86 probe runs hot / 64 MiB / other-model / timer conditions. Timer cost is not subtracted; timer-only p50 is generally zero and p99 about 42 ns because of clock quantization.

Shuffled row sequence.

| Target | Trees | Variant | Hot p50/p90/p99 | 64 MiB p50/p90/p99 | Other model p50/p90/p99 |
| --- | --- | --- | --- | --- | --- |
| ARM64 | 100 | Continuation | 83 / 125 / 125 | 792 / 1000 / 2666 | 250 / 292 / 334 |
| ARM64 | 100 | K8 barrier | 83 / 84 / 125 | 708 / 917 / 2542 | 208 / 250 / 292 |
| ARM64 | 100 | K16 barrier | 83 / 84 / 125 | 750 / 917 / 2375 | 209 / 250 / 292 |
| ARM64 | 100 | K16 shared barrier | 83 / 84 / 84 | 750 / 1167 / 3000 | 208 / 250 / 250 |
| ARM64 | 300 | Continuation | 250 / 250 / 291 | 1917 / 2250 / 5500 | 500 / 625 / 708 |
| ARM64 | 300 | K8 barrier | 250 / 292 / 334 | 2250 / 2666 / 7083 | 583 / 625 / 750 |
| ARM64 | 300 | K16 barrier | 250 / 292 / 375 | 2459 / 6833 / 8625 | 584 / 750 / 834 |
| ARM64 | 300 | K16 shared barrier | 209 / 250 / 291 | 2292 / 6333 / 9167 | 583 / 625 / 709 |
| SSE2 | 100 | Continuation | 125 / 125 / 125 | 667 / 833 / 3875 | 209 / 459 / 667 |
| SSE2 | 100 | K8 barrier | 125 / 125 / 125 | 833 / 2917 / 4250 | 292 / 292 / 375 |
| SSE2 | 100 | K16 barrier | 125 / 125 / 375 | 833 / 2750 / 3791 | 333 / 334 / 375 |
| SSE2 | 100 | K16 shared barrier | 125 / 125 / 125 | 708 / 2167 / 3417 | 291 / 292 / 458 |
| SSE2 | 300 | Continuation | 292 / 334 / 417 | 2125 / 4917 / 11583 | 625 / 792 / 875 |
| SSE2 | 300 | K8 barrier | 292 / 334 / 334 | 2292 / 2667 / 9125 | 1000 / 1000 / 1292 |
| SSE2 | 300 | K16 barrier | 292 / 334 / 334 | 2541 / 2875 / 9833 | 1000 / 1042 / 1209 |
| SSE2 | 300 | K16 shared barrier | 292 / 333 / 334 | 2208 / 2625 / 9167 | 1000 / 1041 / 1292 |
| AVX2 | 100 | Continuation | 417 / 459 / 500 | 1708 / 7250 / 9833 | 625 / 667 / 792 |
| AVX2 | 100 | K8 barrier | 416 / 458 / 541 | 1542 / 1875 / 7458 | 875 / 875 / 1084 |
| AVX2 | 100 | K16 barrier | 375 / 417 / 459 | 1375 / 1708 / 7625 | 875 / 917 / 1125 |
| AVX2 | 100 | K16 shared barrier | 375 / 417 / 458 | 1417 / 1792 / 7583 | 834 / 875 / 1000 |
| AVX2 | 300 | Continuation | 1375 / 1500 / 1708 | 7041 / 22500 / 33667 | 1750 / 1834 / 2292 |
| AVX2 | 300 | K8 barrier | 1292 / 1375 / 1500 | 5500 / 21333 / 30041 | 2667 / 2709 / 2958 |
| AVX2 | 300 | K16 barrier | 1167 / 1209 / 1375 | 4917 / 10250 / 23625 | 2792 / 2834 / 3250 |
| AVX2 | 300 | K16 shared barrier | 1208 / 1250 / 1542 | 4916 / 21250 / 25291 | 2708 / 2792 / 3416 |

Dependent row sequence.

| Target | Trees | Variant | Hot p50/p90/p99 | 64 MiB p50/p90/p99 | Other model p50/p90/p99 |
| --- | --- | --- | --- | --- | --- |
| ARM64 | 100 | Continuation | 83 / 84 / 125 | 792 / 1000 / 2917 | 250 / 292 / 334 |
| ARM64 | 100 | K8 barrier | 84 / 125 / 208 | 750 / 1917 / 3167 | 209 / 250 / 292 |
| ARM64 | 100 | K16 barrier | 83 / 125 / 208 | 792 / 1959 / 3125 | 209 / 250 / 292 |
| ARM64 | 100 | K16 shared barrier | 83 / 84 / 84 | 750 / 958 / 2458 | 208 / 250 / 292 |
| ARM64 | 300 | Continuation | 250 / 250 / 250 | 2458 / 2791 / 6958 | 500 / 625 / 833 |
| ARM64 | 300 | K8 barrier | 250 / 292 / 292 | 2542 / 7000 / 8584 | 584 / 708 / 2292 |
| ARM64 | 300 | K16 barrier | 250 / 292 / 292 | 2292 / 6750 / 10708 | 583 / 667 / 750 |
| ARM64 | 300 | K16 shared barrier | 209 / 250 / 250 | 2125 / 4667 / 6625 | 583 / 625 / 750 |
| SSE2 | 100 | Continuation | 125 / 125 / 125 | 708 / 3458 / 5625 | 250 / 291 / 333 |
| SSE2 | 100 | K8 barrier | 125 / 125 / 125 | 792 / 2625 / 3708 | 292 / 292 / 334 |
| SSE2 | 100 | K16 barrier | 125 / 292 / 333 | 875 / 2958 / 3791 | 333 / 625 / 792 |
| SSE2 | 100 | K16 shared barrier | 125 / 125 / 125 | 708 / 2541 / 3541 | 292 / 333 / 375 |
| SSE2 | 300 | Continuation | 292 / 333 / 334 | 2125 / 4833 / 11459 | 666 / 709 / 875 |
| SSE2 | 300 | K8 barrier | 292 / 334 / 417 | 2334 / 3000 / 9958 | 1000 / 1042 / 1292 |
| SSE2 | 300 | K16 barrier | 292 / 333 / 334 | 2584 / 7500 / 10583 | 1000 / 1042 / 1167 |
| SSE2 | 300 | K16 shared barrier | 292 / 334 / 416 | 2625 / 3459 / 11792 | 1000 / 1042 / 1125 |
| AVX2 | 100 | Continuation | 417 / 459 / 500 | 1666 / 2958 / 8833 | 625 / 667 / 791 |
| AVX2 | 100 | K8 barrier | 375 / 417 / 459 | 2042 / 7708 / 13916 | 875 / 917 / 1250 |
| AVX2 | 100 | K16 barrier | 375 / 417 / 417 | 1417 / 7042 / 9250 | 875 / 917 / 1125 |
| AVX2 | 100 | K16 shared barrier | 375 / 417 / 417 | 1416 / 2417 / 7667 | 875 / 875 / 958 |
| AVX2 | 300 | Continuation | 1333 / 1375 / 1459 | 7250 / 25500 / 37209 | 1750 / 1875 / 2292 |
| AVX2 | 300 | K8 barrier | 1250 / 1334 / 1458 | 4875 / 6666 / 20875 | 2708 / 2750 / 3291 |
| AVX2 | 300 | K16 barrier | 1166 / 1208 / 1209 | 5125 / 7250 / 23500 | 2833 / 2875 / 2958 |
| AVX2 | 300 | K16 shared barrier | 1167 / 1209 / 1250 | 4875 / 9166 / 24834 | 2667 / 2709 / 2875 |

The requested performance comparison uses the default model-order continuation. The following variants have both fewer assembly row reads and a lower hot median in this run; small differences within the min–max ranges do not establish a repeatable speedup.

| Target | Trees | Fewer row reads and lower hot median |
| --- | --- | --- |
| ARM64 | 100 | None |
| ARM64 | 300 | K16 shared barrier (0.45% lower median) |
| SSE2 | 100 | K16 shared barrier (0.30% lower median) |
| SSE2 | 300 | K16 shared barrier (0.14% lower median) |
| AVX2 | 100 | K8 barrier (5.90% lower median); K12 barrier (5.76% lower median); K16 barrier (10.27% lower median); K24 barrier (8.71% lower median); K16 shared barrier (9.61% lower median) |
| AVX2 | 300 | K8 barrier (4.35% lower median); K12 barrier (7.58% lower median); K16 barrier (12.58% lower median); K24 barrier (9.71% lower median); K16 shared barrier (10.86% lower median) |

Spilling first appears at K1 on ARM64 for both sizes, K2/K5 on SSE2 for 100/300 trees, and K9/K10 on AVX2. No ARM64 budget tested reduces row reads without spilling. AVX2 K8 does so at both sizes; SSE2 has a few spill-free lower budgets at 300 trees. In the measured hot runs, ARM64 K16 shared beats default continuation on both row reads and median time only at 300 trees; SSE2 K16 shared does so at both sizes by less than 0.3%; AVX2 K8/K12/K16/K24 and K16 shared do so at both sizes under Rosetta. ARM64 K16 shared at 300 trees is only 238.16 versus 239.23 ns; the existing shared-order continuation is faster at 229.70 ns. These tiny ARM64/SSE2 differences do not establish repeatable speedups. The barriers preserve the tested cross-chunk ordering and reuse, but do not make K a bound on machine-register pressure.
