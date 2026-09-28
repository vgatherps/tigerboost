status: pass

The experiment is implemented and all gates passed. Native ARM64 hot latency regressed for every scheduled variant: continuation took 88.91 ns at 100 trees and 242.14 ns at 300; scheduled variants took 101.41–108.86 ns and 284.97–313.13 ns. ARM64 spills even at K1. SSE2 and AVX2 have budgets that reduce row loads without body spills. No barriers or compensating changes were added.

Seven tracked files changed: `src/jit.rs`, `tests/differential.rs`, `tests/scalar_jit.rs`, `examples/scalar_bench.rs`, `examples/cold_bench.rs`, `examples/inspect_code.rs`, and `examples/x86_check.rs`. `src/sort_policy.rs` and `tools/x86_check.sh` are unchanged; the shell script already consumes the example's variant manifest. `VERDICT.md` and `probes/` contain the report and measurement artifacts. No commits, merges, stashes, or formatting commands were made.

The supplied anchors were checked against `9f2fa6c`. `EmitOptions` was at line 27, `prepare_module` at 137 with the pass guard at 162, and `load_feature` at 806 rather than 742. Both `EvaluateTree` arms and action order were read before editing.

1. `EmitOptions` now has `avx2`, `scheduled`, `resident_features`, and `volatile_leaves`. Defaults are false, false, 12, false. Unscheduled emission ignores the last two fields; all 18 baseline files match `5a915cd` byte for byte.
2. Scheduled emission defines only C-ABI `predict(row)` and skips `early-cse`. No inline assembly or continuation definitions remain in that mode. Row alignment and dereferenceability remain; `readonly`, `noalias`, and `memory(read)` are absent.
3. Scheduled `load_feature` emits volatile aligned double loads, followed by the existing NaN selection. Threshold loads stay ordinary. All three leaf-loading sites become volatile only when both `scheduled` and `volatile_leaves` are true.
4. Before module creation, `feature_fills` walks real feature lanes in action order and records tree-visit and lane positions. Existing synthetic padding copies stay unchanged; they request no feature load. A reverse pass computes next uses. `residency_schedule` scans at most K resident entries per fill, evicts farthest-next-use, and breaks ties by lower feature index. Emission overwrites the evicted SSA slot on misses and reuses it on hits. `scheduled_feature_loads`, used by `inspect_code`, reports the production schedule's misses. The simulation uses O(fills × K) work after next-use preprocessing.
5. LLVM 21.1.8 retained some load-and-insert folding despite volatile IR. At 100 trees, K16 has 100 ARM64 `ld1.d` row inserts, 100 SSE2 `movhpd` row inserts, and 91 AVX2 `vmovhpd` row inserts. No workaround was added. Counts below include this actual behavior.
6. All requested benchmark and inspection variants replace fenced variants. `inspect_code` additionally emits K20 and K16 volatile leaves to complete the requested tables. Every variant was emitted at both 100 and 300 trees. Both x86 execution gates include K16 and its shared-feature twin.
7. Tests cover exact off-mode assembly identity on both triples with AVX2 on/off, the four-miss Belady sequence and eviction tie, volatility/alignment and exact SSA reuse after `prepare_module`, all three NaN modes, both orders, K1/K4/K16/K64 differential execution, volatile leaves on/off, model destruction, and assembly load order. The hand sequence misses on A, B, C, and the second B: `[true, true, true, false, true, false]`. Independent assembly probes also checked every scheduled row address, in order, for all 42 requested scheduled kernels and 144 budget-sweep kernels.

| Gate | Result |
|---|---|
| `git stash list`; `git show 5a915cd:src/jit.rs` | Empty stash; baseline present |
| Scratch baseline worktree; release `inspect_code 100x100` | 18 files emitted; temporary worktree removed |
| `cargo build --offline --release --examples` | Pass |
| `cargo test --offline --all-targets` | 69 tests passed |
| `cargo test --offline --release --all-targets` | 69 tests passed |
| `cargo test --offline --doc` | 1 doctest passed |
| `cargo clippy --offline --all-targets -- -D warnings` | Pass |
| `inspect_code 100x100`; `cmp` of every baseline file | 18/18 identical: 9 assemblies and 9 plan dumps |
| `inspect_code 300x100` | All variants emitted |
| `tools/x86_check.sh 100x100` | 6 variants × 256 rows passed under Rosetta |
| `tools/x86_check.sh special --avx2` | 6 variants × 256 rows passed under Rosetta |
| `scalar_bench 100x100`; `scalar_bench 300x100` | Pass; all variants verified |
| `cold_bench 100x100 --samples 3000`, default and `--scheduled 16` | Pass |
| Independent assembly and K1–K24 probes | Load counts and address order match Belady |
| Additional x86 timing probes | All rows and 50,000 dependent calls verified for every variant |
| `git diff --check` | Pass |

Gates used `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'`. Host was Apple M4 Pro, macOS 15.6; LLVM 21.1.8, inkwell 0.9.0, rustc 1.93.0-nightly (c90bcb957). Development checks caught and resolved a test's missing `get_alignment().unwrap()` and its initial failure to recognize ARM broadcast loads and row-address aliases. Final gates passed. x86 linking printed deprecation and pointer-alignment warnings; execution and score checks passed.

Belady misses are target-independent. K16 volatile leaves has the same misses as K16. The shared-feature ordering uses the existing eight-tree grouping.

| Trees | Order | K8 | K12 | K16 | K20 | K24 |
|---:|---|---:|---:|---:|---:|---:|
| 100 | Model | 423 | 375 | 338 | 309 | 281 |
| 100 | SharedFeatures | 347 | 299 | 268 | 248 | 228 |
| 300 | Model | 1242 | 1087 | 964 | 866 | 780 |
| 300 | SharedFeatures | 936 | 789 | 681 | 611 | 550 |

Static counts include `predict` and all continuation bodies, excluding directives, data, and assembler alignment padding. Row counts give doubles read / memory instructions; an ARM paired load can read two doubles. Threshold and leaf columns count memory instructions, including folded operands; paired threshold loads count once. Each kernel consumes exactly 3 threshold vectors/tree on ARM64/SSE2 or 2 on AVX2 and one leaf/tree.

Register inserts count ARM `mov.d` into a vector lane and `ins`; x86 register-only `unpcklpd`, `movlhps`, `vinsertf128`, `vblendpd`, and `vperm2f128`, including their VEX forms. The second number counts register broadcasts (`dup.2d`, `movddup`, `vbroadcastsd`). Whole-register copies and mask extraction shuffles are excluded. Folded row inserts count `ld1.d` / `movhpd` / `vmovhpd`.

Body stack memory excludes ABI register saves/restores. Frame bytes give explicit stack allocation, including saved registers, plus red-zone depth separately; implicit x86 return addresses are excluded. Continuation frame allocation is wrapper plus largest continuation frame. Prologue instruction/byte counts sum the stack setup and saves across all functions.

Counts for ARM64.

| Trees | Variant | Instructions | Row doubles/ins | Reg inserts/broadcasts | Folded row inserts | Threshold loads | Leaf loads | Body stack memory | Frame + red zone, B | Prologue ins/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | continuation | 2143 | 468/429 | 125/0 | 172 | 163 | 100 | 0 | 160 + 0 | 10/40 |
| 100 | K8 | 2320 | 423/423 | 148/0 | 149 | 268 | 100 | 140 | 816 + 0 | 11/44 |
| 100 | K12 | 2305 | 375/375 | 178/0 | 119 | 268 | 100 | 129 | 816 + 0 | 11/44 |
| 100 | K16 | 2299 | 338/338 | 193/1 | 100 | 268 | 100 | 136 | 768 + 0 | 7/28 |
| 100 | K20 | 2300 | 309/309 | 212/1 | 81 | 268 | 100 | 141 | 752 + 0 | 6/24 |
| 100 | K24 | 2325 | 281/281 | 225/1 | 68 | 268 | 100 | 163 | 864 + 0 | 6/24 |
| 100 | K16 shared | 2267 | 268/268 | 235/0 | 58 | 268 | 100 | 131 | 736 + 0 | 6/24 |
| 100 | K16 volatile leaves | 2299 | 338/338 | 193/1 | 100 | 268 | 100 | 136 | 768 + 0 | 7/28 |
| 300 | continuation | 6521 | 1419/1315 | 359/0 | 532 | 463 | 300 | 0 | 160 + 0 | 10/40 |
| 300 | K8 | 7258 | 1242/1242 | 468/1 | 422 | 868 | 300 | 707 | 2512 + 0 | 11/44 |
| 300 | K12 | 7222 | 1087/1087 | 558/1 | 332 | 868 | 300 | 660 | 2480 + 0 | 11/44 |
| 300 | K16 | 7206 | 964/964 | 610/1 | 281 | 868 | 300 | 639 | 2496 + 0 | 11/44 |
| 300 | K20 | 7172 | 866/866 | 664/1 | 228 | 868 | 300 | 615 | 2496 + 0 | 11/44 |
| 300 | K24 | 7156 | 780/780 | 704/2 | 188 | 868 | 300 | 607 | 2576 + 0 | 11/44 |
| 300 | K16 shared | 7029 | 681/681 | 773/1 | 109 | 868 | 300 | 516 | 2336 + 0 | 11/44 |
| 300 | K16 volatile leaves | 7206 | 964/964 | 610/1 | 281 | 868 | 300 | 639 | 2496 + 0 | 11/44 |

Counts for x86 SSE2.

| Trees | Variant | Instructions | Row doubles/ins | Reg inserts/broadcasts | Folded row inserts | Threshold loads | Leaf loads | Body stack memory | Frame + red zone, B | Prologue ins/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | continuation | 2300 | 470/470 | 120/0 | 174 | 300 | 100 | 0 | 48 + 0 | 7/11 |
| 100 | K8 | 2662 | 423/423 | 150/0 | 149 | 300 | 100 | 0 | 0 + 0 | 0/0 |
| 100 | K12 | 2721 | 375/375 | 174/0 | 119 | 300 | 100 | 47 | 0 + 104 | 0/0 |
| 100 | K16 | 2756 | 338/338 | 174/0 | 100 | 300 | 100 | 119 | 104 + 128 | 1/4 |
| 100 | K20 | 2793 | 309/309 | 180/0 | 81 | 300 | 100 | 178 | 168 + 128 | 1/7 |
| 100 | K24 | 2807 | 281/281 | 183/0 | 68 | 300 | 100 | 221 | 232 + 128 | 1/7 |
| 100 | K16 shared | 2755 | 268/268 | 221/0 | 58 | 300 | 100 | 104 | 104 + 128 | 1/4 |
| 100 | K16 volatile leaves | 2756 | 338/338 | 174/0 | 100 | 300 | 100 | 119 | 104 + 128 | 1/4 |
| 300 | continuation | 6867 | 1414/1414 | 364/0 | 527 | 900 | 300 | 0 | 48 + 0 | 8/12 |
| 300 | K8 | 7980 | 1242/1242 | 476/0 | 422 | 900 | 300 | 0 | 0 + 0 | 0/0 |
| 300 | K12 | 8189 | 1087/1087 | 533/0 | 332 | 900 | 300 | 175 | 8 + 128 | 1/1 |
| 300 | K16 | 8319 | 964/964 | 541/0 | 281 | 900 | 300 | 394 | 104 + 128 | 1/4 |
| 300 | K20 | 8395 | 866/866 | 561/0 | 228 | 900 | 300 | 542 | 184 + 128 | 1/7 |
| 300 | K24 | 8457 | 780/780 | 563/0 | 188 | 900 | 300 | 666 | 280 + 128 | 1/7 |
| 300 | K16 shared | 8328 | 681/681 | 735/0 | 109 | 900 | 300 | 335 | 104 + 128 | 1/4 |
| 300 | K16 volatile leaves | 8319 | 964/964 | 541/0 | 281 | 900 | 300 | 394 | 104 + 128 | 1/4 |

Counts for x86 AVX2.

| Trees | Variant | Instructions | Row doubles/ins | Reg inserts/broadcasts | Folded row inserts | Threshold loads | Leaf loads | Body stack memory | Frame + red zone, B | Prologue ins/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | continuation | 1451 | 470/470 | 224/0 | 161 | 200 | 100 | 0 | 40 + 0 | 5/9 |
| 100 | K8 | 1681 | 423/423 | 253/0 | 138 | 200 | 100 | 0 | 0 + 0 | 0/0 |
| 100 | K12 | 1663 | 375/375 | 282/0 | 109 | 200 | 100 | 0 | 0 + 0 | 0/0 |
| 100 | K16 | 1734 | 338/338 | 283/0 | 91 | 200 | 100 | 89 | 200 + 128 | 1/7 |
| 100 | K20 | 1752 | 309/309 | 289/0 | 74 | 200 | 100 | 134 | 392 + 128 | 1/7 |
| 100 | K24 | 1784 | 281/281 | 286/0 | 64 | 200 | 100 | 190 | 600 + 128 | 1/7 |
| 100 | K16 shared | 1682 | 268/268 | 322/0 | 55 | 200 | 100 | 69 | 200 + 128 | 1/7 |
| 100 | K16 volatile leaves | 1734 | 338/338 | 283/0 | 91 | 200 | 100 | 89 | 200 + 128 | 1/7 |
| 300 | continuation | 4304 | 1414/1414 | 658/0 | 512 | 600 | 300 | 0 | 40 + 0 | 5/9 |
| 300 | K8 | 5029 | 1242/1242 | 782/0 | 401 | 600 | 300 | 0 | 0 + 0 | 0/0 |
| 300 | K12 | 4967 | 1087/1087 | 871/0 | 311 | 600 | 300 | 2 | 0 + 24 | 0/0 |
| 300 | K16 | 5190 | 964/964 | 886/0 | 257 | 600 | 300 | 292 | 296 + 128 | 1/7 |
| 300 | K20 | 5264 | 866/866 | 901/0 | 210 | 600 | 300 | 442 | 472 + 128 | 1/7 |
| 300 | K24 | 5320 | 780/780 | 908/0 | 170 | 600 | 300 | 566 | 664 + 128 | 1/7 |
| 300 | K16 shared | 4994 | 681/681 | 1049/0 | 102 | 600 | 300 | 217 | 264 + 128 | 1/7 |
| 300 | K16 volatile leaves | 5190 | 964/964 | 886/0 | 257 | 600 | 300 | 292 | 296 + 128 | 1/7 |

Hot medians below are ns/prediction, five rounds of 50,000 dependent calls after warmup, including driver cost. ARM64 uses the requested Rust example. x86 uses separately emitted Darwin assembly and the C probe under Rosetta 2; static x86 counts above use the requested Linux triple. Rosetta timings do not establish native Intel/AMD performance. SSE2 and AVX2 also have different translation costs. These are indicative, unpinned single-run measurements.

| Variant | ARM64 100 | ARM64 300 | SSE2 Rosetta 100 | SSE2 Rosetta 300 | AVX2 Rosetta 100 | AVX2 Rosetta 300 |
|---|---:|---:|---:|---:|---:|---:|
| continuation | 88.91 | 242.14 | 111.21 | 318.25 | 467.83 | 1331.42 |
| K8 | 108.86 | 313.13 | 111.62 | 321.39 | 430.52 | 1199.77 |
| K12 | 105.44 | 296.05 | 111.85 | 321.33 | 402.16 | 1203.97 |
| K16 | 102.10 | 287.50 | 112.52 | 324.00 | 402.30 | 1172.98 |
| K20 | 101.41 | 288.30 | 112.05 | 324.69 | 411.25 | 1185.64 |
| K24 | 101.51 | 287.60 | 111.14 | 326.59 | 418.02 | 1183.54 |
| K16 shared | 102.72 | 284.97 | 110.84 | 321.34 | 406.79 | 1206.40 |
| K16 volatile leaves | 101.81 | 287.51 | 111.06 | 320.87 | 406.71 | 1165.03 |

Cold values are p50 / p90 / p99 in ns, 3,000 retained samples per condition and sequence, at 100 trees. The native harness reports approximately 41 ns timer resolution. Both drivers copy the selected row after cache disturbance and time only prediction. “64 MiB” updates one word per cache line in shuffled order. “Other model” calls a separate 2,000-tree, 100-feature continuation model first.

Native runs use the full cold harness with shuffled condition order. The additional Rosetta probe runs just the three requested conditions in fixed order, discarding a first pass before retaining a second; it uses the same row data, dependent step, shuffled-row seed, and pollution-line seed. Cross-driver cold numbers are not directly comparable. The full native logs also contain timer-only controls and the other cache sizes.

| Target | Sequence | Kernel | Hot p50/p90/p99 | 64 MiB p50/p90/p99 | Other model p50/p90/p99 |
|---|---|---|---:|---:|---:|
| ARM64 | shuffled | continuation | 83/84/125 | 833/1291/2875 | 208/250/291 |
| ARM64 | shuffled | k16 | 84/125/125 | 750/917/2250 | 209/250/292 |
| ARM64 | dependent | continuation | 83/84/125 | 833/1000/2250 | 208/250/292 |
| ARM64 | dependent | k16 | 84/125/125 | 667/916/2083 | 250/250/292 |
| SSE2 Rosetta | shuffled | continuation | 125/125/125 | 750/959/3667 | 208/250/292 |
| SSE2 Rosetta | shuffled | k16 | 125/125/125 | 750/1041/3917 | 375/417/459 |
| SSE2 Rosetta | dependent | continuation | 125/125/125 | 792/3166/5500 | 209/250/333 |
| SSE2 Rosetta | dependent | k16 | 125/167/292 | 750/3375/4417 | 375/417/459 |
| AVX2 Rosetta | shuffled | continuation | 417/500/583 | 1666/5500/8333 | 584/666/792 |
| AVX2 Rosetta | shuffled | k16 | 375/375/459 | 1750/7875/10500 | 875/917/1167 |
| AVX2 Rosetta | dependent | continuation | 417/458/500 | 4667/8500/17875 | 583/625/791 |
| AVX2 Rosetta | dependent | k16 | 375/417/500 | 1709/7875/12041 | 875/958/1167 |

Spill onset was measured at every integer K from 1 through 24, with model order and ordinary leaves. The frame column alone is insufficient on x86: spills can use the red zone without moving `rsp`.

| Target | Trees | First K with body spills | Last K without body spills | Row doubles at that last K | Continuation row doubles |
|---|---:|---:|---:|---:|---:|
| ARM64 | 100 | 1 | none in K1–K24 | — | 468 |
| ARM64 | 300 | 1 | none in K1–K24 | — | 1419 |
| SSE2 | 100 | 11 | 10 | 397 | 470 |
| SSE2 | 300 | 11 | 10 | 1158 | 1414 |
| AVX2 | 100 | 13 | 12 | 375 | 470 |
| AVX2 | 300 | 12 | 11 | 1121 | 1414 |

SSE2 and AVX2 beat continuation row-load counts without spilling. Among the requested variants, SSE2 K8 does so at both sizes; AVX2 K8 does so at both sizes and K12 does so at 100 trees. ARM64 has no such budget in the measured K1–K24 range. K16 shared reduces misses further but spills on every target. K16 volatile leaves changes neither instruction counts nor body stack counts at either model size.

The feature budget limits cached feature SSA values, not threshold vectors, assembled vectors, leaf values, or sums. For example, ARM64 K16 loads the first tree's leaf with `ldr d1, [x9, #5088]`, then stores it at `[sp, #664]`; the feature schedule itself still matches all 338 emitted row loads. No attempt was made to alter accumulation, add barriers, or hide this result.

Raw native timings are in `probes/hot100.log`, `probes/hot300.log`, and `probes/cold-*.log`; x86 timings are in `probes/rosetta/*/timings.log`. Gate logs are in `probes/gates/`. `probes/inspect100/` and `probes/inspect300/` retain assembly and plan dumps; `probes/baseline100/` retains the 18 baseline files. `probes/count_assembly.py` produces `probes/counts.json` and checks every scheduled address against an independent Belady simulation. `probes/count_sweep.py` and `probes/sweep-counts.json` record the exact spill sweep; its raw assembly remains in `/tmp/tb-res-sweep`. The small Rust emission probes and C Rosetta driver are retained under `probes/`.
