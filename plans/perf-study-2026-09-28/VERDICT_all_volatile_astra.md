status: complete

Scheduled mode reduces row loads but loses native ARM64 hot time at both sizes. Every ARM64 K spills. All correctness gates pass, and all 18 baseline files match 5a915cd byte for byte.

Started at aafd45b, the requested round-1 checkpoint. The 9f2fa6c line anchors describe the preceding fenced experiment; current code already had Belady residency and its tests. No commits, merges, stashes, formatting runs, or sort-policy edits.

Changed files: src/jit.rs, tests/differential.rs, examples/scalar_bench.rs, examples/cold_bench.rs, examples/inspect_code.rs, examples/x86_check.rs. Existing scheduled dropped-model coverage in tests/scalar_jit.rs and manifest-driven tools/x86_check.sh remain unchanged. probes/ contains measurement sources, assembly, raw samples, and gate logs; VERDICT.md is this report.

1. EmitOptions now contains avx2, scheduled, resident_features. Defaults remain false, false, K12. volatile_leaves is removed; scheduled means all emitter loads are volatile.
2. Scheduled mode still emits one C-convention predict function, no fences or inline assembly, and skips early-cse. Row align 8 and dereferenceable remain; readonly, noalias, and memory(read) are absent. The emitter places each group's missed feature loads before its threshold load and compare; each computed leaf address is followed by the leaf load and add. Pure arithmetic can still move during backend scheduling.
3. Every emitted row, threshold-vector, and leaf load is volatile and aligned to 8 in prepared IR, including constant-tree leaves. Scheduled leaf blocks follow plan.actions() visit order, with leaf_base indexed by TreeId. Threshold and leaf addresses use absolute constants slots. Continuation layout and emitted outputs are unchanged.
4. Round-1 deterministic O(fills × K) Belady scheduling is unchanged, including lower-feature-index eviction ties. Recorded misses drive row loads and SSA reuse. All measured scheduled assembly row counts equal simulated misses.
5. LLVM 21.1.8 retains some folding despite volatility: ARM64 uses ld1.d lane loads; x86 uses movhpd/vmovhpd; AVX2 threshold compares and leaf adds retain memory operands. SSE2 thresholds use separate loads. ARM64 scheduled thresholds are individual ldr q instructions. These are observed backend results, with no workaround. Backend spills, the constants GOT read, and SSE2 pack-mask constant-pool reads are additional machine loads, not emitter IR loads.
6. Harness names now end in all-volatile for K8/K12/K16/K24 and K16 shared. cold_bench --scheduled [K] selects all-volatile mode. x86_check includes K16 and K16 shared. The superseded volatile-leaves variant is removed. K20 is measured by the sweep probe, not added back to the amended harness set.
7. all_loads_are_volatile checks prepared IR for three targets, both orders, three NaN modes, and K1/K2/K4/K16/K64; counts equal misses + threshold groups + trees. It also checks every feature lane against its scheduled SSA load/reuse. loads_appear_in_emitted_order checks exact row/threshold/leaf memory-access order and offsets on ARM64/SSE2/AVX2, plus threshold→compare and leaf→add adjacency in IR. scheduled_leaf_blocks_follow_visit_order checks real constant arrays for reordered, odd-depth, and constant trees. Belady, scheduled-off byte identity, row-load order, differential K1/K4/K16/K64 with both orders, and dropped-model tests pass.

Gate commands used LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 and RUSTDOCFLAGS=-C link-arg=/opt/homebrew/lib/libzstd.dylib. Native timings are ARM64 on Darwin 24.6.0; target code uses generic ARM64/SSE2 and x86-64-v3 AVX2. x86 correctness runs use Rosetta. No x86 hot/cold timing claim is made.

| Gate | Result |
| --- | --- |
| git stash list; git show 5a915cd:src/jit.rs | Empty stash; baseline present |
| Scratch 5a915cd inspect_code 100x100 | Pass; git archive snapshot built offline; scratch source removed; outputs retained in probes/baseline-100 |
| cargo build --offline --release --examples | Pass |
| cargo test --offline --all-targets | Pass |
| cargo test --offline --release --all-targets | Pass |
| cargo test --offline --doc | Pass |
| cargo clippy --offline --all-targets -- -D warnings | Pass |
| inspect_code 100x100 and 300x100 | Pass, both triples and AVX2 twins |
| cmp baseline files | 18/18 identical, including plan dumps |
| tools/x86_check.sh 100x100 | 6/6 variants pass, 256 rows each |
| tools/x86_check.sh special --avx2 | 6/6 variants pass, 256 rows each |
| scalar_bench 100x100 and 300x100 | Pass, all variants verified and timed |
| cold_bench at both sizes: continuation, K8, K16, K16 shared | Pass, 3,000 samples per condition and sequence in all eight runs |
| Assembly count validation | 42 table rows + 144 sweep rows match model/schedule counts |

The new assembly-test parser initially misread ARM lane brackets and retained an overwritten x86 address register; both test-parser errors were fixed before the final gates. Clippy requested is_multiple_of(2) in the edited leaf-layout loop; that equivalent expression is now used. x86 linking reports unaligned embedded pointers; every checker still exits successfully. Logs are in probes/gates.log, probes/x86-100.log, and probes/x86-special-avx2.log.

Counts come from separately emitted target assembly, not disassembly of MCJIT executable bytes. Instruction counts cover predict and every continuation, each once, excluding assembler directives, data, and alignment padding. Row cells are memory instructions / scalar feature values; a continuation ARM64 ldp reads two values. Threshold cells are memory instructions / vector groups; every AVX2 threshold load is folded into its compare. Leaf cells are total loads / loads folded into adds. Scheduled row counts equal misses; threshold groups are 3 per tree on ARM64/SSE2 and 2 on AVX2. Lane inserts count register-to-register mov.d/ins/dup on ARM64 and unpack/movlhps/128-bit inserts/blends/broadcasts on x86; full-register copies are excluded. Stack cells are body / all explicit stack-memory instructions, excluding ABI saves/restores from body. Frame cells are peak explicitly allocated bytes + red-zone bytes, excluding hardware return addresses. Prologue cells are total ABI save/stack-allocation instructions / encoded bytes across emitted functions.

100 trees × 100 features, ARM64.

| Variant | Instructions | Row I/values | Lane inserts | Threshold I/groups | Leaves/folded | Stack body/all | Frame + red zone B | Prologue I/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation | 2143 | 429/468 | 125 | 163/300 | 100/0 | 0/20 | 160+0 | 10/40 |
| k8 | 2354 | 423/423 | 148 | 300/300 | 100/0 | 142/162 | 816+0 | 11/44 |
| k12 | 2342 | 375/375 | 178 | 300/300 | 100/0 | 134/154 | 816+0 | 11/44 |
| k16 | 2335 | 338/338 | 194 | 300/300 | 100/0 | 142/152 | 752+0 | 6/24 |
| k20 | 2331 | 309/309 | 213 | 300/300 | 100/0 | 140/150 | 752+0 | 6/24 |
| k24 | 2349 | 281/281 | 226 | 300/300 | 100/0 | 159/169 | 848+0 | 6/24 |
| k16-shared | 2297 | 268/268 | 235 | 300/300 | 100/0 | 128/138 | 736+0 | 6/24 |

100 trees × 100 features, x86 SSE2.

| Variant | Instructions | Row I/values | Lane inserts | Threshold I/groups | Leaves/folded | Stack body/all | Frame + red zone B | Prologue I/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation | 2300 | 470/470 | 120 | 300/300 | 100/99 | 0/14 | 48+0 | 7/11 |
| k8 | 2662 | 423/423 | 150 | 300/300 | 100/98 | 0/0 | 0+0 | 0/0 |
| k12 | 2716 | 375/375 | 176 | 300/300 | 100/98 | 38/38 | 0+104 | 0/0 |
| k16 | 2758 | 338/338 | 178 | 300/300 | 100/98 | 108/108 | 72+128 | 1/4 |
| k20 | 2796 | 309/309 | 182 | 300/300 | 100/98 | 172/172 | 168+128 | 1/7 |
| k24 | 2822 | 281/281 | 186 | 300/300 | 100/98 | 219/219 | 264+128 | 1/7 |
| k16-shared | 2779 | 268/268 | 219 | 300/300 | 100/98 | 114/114 | 104+128 | 1/4 |

100 trees × 100 features, x86 AVX2.

| Variant | Instructions | Row I/values | Lane inserts | Threshold I/groups | Leaves/folded | Stack body/all | Frame + red zone B | Prologue I/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation | 1451 | 470/470 | 224 | 200/200 | 100/100 | 0/10 | 40+0 | 5/9 |
| k8 | 1681 | 423/423 | 253 | 200/200 | 100/100 | 0/0 | 0+0 | 0/0 |
| k12 | 1663 | 375/375 | 282 | 200/200 | 100/100 | 0/0 | 0+0 | 0/0 |
| k16 | 1734 | 338/338 | 283 | 200/200 | 100/100 | 89/89 | 200+128 | 1/7 |
| k20 | 1752 | 309/309 | 291 | 200/200 | 100/100 | 131/131 | 408+128 | 1/7 |
| k24 | 1781 | 281/281 | 289 | 200/200 | 100/100 | 188/188 | 568+128 | 1/7 |
| k16-shared | 1688 | 268/268 | 324 | 200/200 | 100/100 | 72/72 | 232+128 | 1/7 |

300 trees × 100 features, ARM64.

| Variant | Instructions | Row I/values | Lane inserts | Threshold I/groups | Leaves/folded | Stack body/all | Frame + red zone B | Prologue I/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation | 6521 | 1315/1419 | 359 | 463/900 | 300/0 | 0/20 | 160+0 | 10/40 |
| k8 | 7297 | 1242/1242 | 469 | 900/900 | 300/0 | 707/727 | 2496+0 | 11/44 |
| k12 | 7260 | 1087/1087 | 559 | 900/900 | 300/0 | 657/677 | 2480+0 | 11/44 |
| k16 | 7243 | 964/964 | 611 | 900/900 | 300/0 | 645/665 | 2528+0 | 11/44 |
| k20 | 7200 | 866/866 | 665 | 900/900 | 300/0 | 614/634 | 2528+0 | 11/44 |
| k24 | 7186 | 780/780 | 706 | 900/900 | 300/0 | 604/624 | 2592+0 | 11/44 |
| k16-shared | 7059 | 681/681 | 774 | 900/900 | 300/0 | 514/534 | 2320+0 | 11/44 |

300 trees × 100 features, x86 SSE2.

| Variant | Instructions | Row I/values | Lane inserts | Threshold I/groups | Leaves/folded | Stack body/all | Frame + red zone B | Prologue I/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation | 6867 | 1414/1414 | 364 | 900/900 | 300/299 | 0/16 | 48+0 | 8/12 |
| k8 | 7982 | 1242/1242 | 476 | 900/900 | 300/298 | 0/0 | 0+0 | 0/0 |
| k12 | 8218 | 1087/1087 | 534 | 900/900 | 300/298 | 179/179 | 24+128 | 1/4 |
| k16 | 8312 | 964/964 | 546 | 900/900 | 300/298 | 382/382 | 120+128 | 1/4 |
| k20 | 8408 | 866/866 | 565 | 900/900 | 300/298 | 540/540 | 200+128 | 1/7 |
| k24 | 8469 | 780/780 | 566 | 900/900 | 300/298 | 674/674 | 264+128 | 1/7 |
| k16-shared | 8335 | 681/681 | 733 | 900/900 | 300/298 | 326/326 | 104+128 | 1/4 |

300 trees × 100 features, x86 AVX2.

| Variant | Instructions | Row I/values | Lane inserts | Threshold I/groups | Leaves/folded | Stack body/all | Frame + red zone B | Prologue I/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation | 4304 | 1414/1414 | 658 | 600/600 | 300/300 | 0/10 | 40+0 | 5/9 |
| k8 | 5029 | 1242/1242 | 782 | 600/600 | 300/300 | 0/0 | 0+0 | 0/0 |
| k12 | 4967 | 1087/1087 | 871 | 600/600 | 300/300 | 2/2 | 0+24 | 0/0 |
| k16 | 5187 | 964/964 | 887 | 600/600 | 300/300 | 289/289 | 280+128 | 1/7 |
| k20 | 5272 | 866/866 | 908 | 600/600 | 300/300 | 427/427 | 504+128 | 1/7 |
| k24 | 5332 | 780/780 | 902 | 600/600 | 300/300 | 577/577 | 648+128 | 1/7 |
| k16-shared | 4994 | 681/681 | 1051 | 600/600 | 300/300 | 209/209 | 264+128 | 1/7 |

At 100 trees, all 45 scheduled ARM64 leaf offsets above the 32 KiB load-immediate range use mov plus a register-offset ldr. At 300 trees, 264 offsets exceed that range: 64 use mov plus a register-offset ldr, and 200 use an additional shifted-immediate add before ldr. These counts are the same for every scheduled K and order shown. Continuation uses 45 materializations at 100 trees; at 300 trees it uses 65 mov materializations and 200 shifted adds. The emitter itself keeps absolute offsets in both scheduled threshold and leaf GEPs.

Belady misses, independent of target width. Shared means TreeOrdering::SharedFeatures; the 8-tree grouping remains part of that ordering.

| K | 100 model | 100 shared | 300 model | 300 shared |
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

ARM64 spill sweep in model order. Every K has body stack traffic. Saves/restores are excluded; frames include ABI saves.

| K | 100 body stack I | 100 frame B | 300 body stack I | 300 frame B |
| --- | --- | --- | --- | --- |
| 1 | 229 | 1008 | 912 | 2640 |
| 2 | 204 | 960 | 854 | 2624 |
| 3 | 171 | 864 | 826 | 2624 |
| 4 | 167 | 848 | 798 | 2592 |
| 5 | 160 | 848 | 770 | 2576 |
| 6 | 161 | 864 | 749 | 2560 |
| 7 | 146 | 816 | 730 | 2544 |
| 8 | 142 | 816 | 707 | 2496 |
| 9 | 145 | 832 | 697 | 2496 |
| 10 | 143 | 832 | 685 | 2512 |
| 11 | 139 | 832 | 668 | 2480 |
| 12 | 134 | 816 | 657 | 2480 |
| 13 | 139 | 800 | 652 | 2480 |
| 14 | 135 | 784 | 643 | 2464 |
| 15 | 134 | 752 | 638 | 2496 |
| 16 | 142 | 752 | 645 | 2528 |
| 17 | 144 | 752 | 632 | 2496 |
| 18 | 143 | 752 | 622 | 2464 |
| 19 | 142 | 752 | 617 | 2512 |
| 20 | 140 | 752 | 614 | 2528 |
| 21 | 148 | 768 | 603 | 2496 |
| 22 | 150 | 816 | 605 | 2528 |
| 23 | 156 | 816 | 604 | 2560 |
| 24 | 159 | 848 | 604 | 2592 |

First body spill in the model-order K1..K24 sweep:

| Trees | ARM64 | SSE2 | AVX2 |
| --- | --- | --- | --- |
| 100 | 1 | 11 | 13 |
| 300 | 1 | 11 | 12 |

ARM64 spills include row-address temporaries and delayed leaf results, even at K1. In the 100-tree K16 assembly, 279 of 300 vector comparisons precede the first fadd; the emitter placed each leaf add directly after its load in IR. Volatility fixes memory-access order, not arithmetic scheduling. K bounds resident feature values in the emitter; it does not bound every backend temporary. On SSE2, K6..K10 at 100 trees and K5..K10 at 300 trees reduce row loads without spills. AVX2 does so at K6..K12 and K5..K11 respectively. No measured ARM64 K is spill-free.

Native ARM64 hot medians in ns/prediction, including the dependent row-selection driver. Each value is the median of five 50,000-call rounds; variant order reverses between rounds. Timings are indicative.

| Variant | 100 trees ns | Change vs continuation | 300 trees ns | Change vs continuation |
| --- | --- | --- | --- | --- |
| continuation | 90.34 | +0.0% | 247.47 | +0.0% |
| k8 | 109.39 | +21.1% | 318.20 | +28.6% |
| k12 | 104.63 | +15.8% | 303.00 | +22.4% |
| k16 | 103.02 | +14.0% | 292.28 | +18.1% |
| k24 | 100.80 | +11.6% | 291.00 | +17.6% |
| k16-shared | 100.08 | +10.8% | 287.02 | +16.0% |

Driver floors were 5.84 ns at 100 trees and 5.86 ns at 300 trees; neither is subtracted. Full hot output: probes/hot-100.log and probes/hot-300.log.

Cold tables give p50 / p90 / p99 in ns, 3,000 retained samples per condition and row sequence. The harness discards its first pass, touches pollution data outside the timed region, and copies the row after pollution. The other model contains 2,000 trees. Timer resolution is about 41 ns; timer-only and other pollution sizes remain in the raw logs. Every benchmark ran serially. Raw samples are in probes/cold-*.csv.

100 trees × 100 features, native ARM64 cold latency.

| Variant | Condition | Shuffled p50/p90/p99 | Dependent p50/p90/p99 |
| --- | --- | --- | --- |
| continuation | hot | 83 / 125 / 125 | 83 / 84 / 125 |
| continuation | data 64MiB | 1167 / 2792 / 3792 | 1041 / 2708 / 3542 |
| continuation | other model | 125 / 167 / 250 | 166 / 167 / 250 |
| k8 | hot | 84 / 125 / 125 | 125 / 125 / 167 |
| k8 | data 64MiB | 1083 / 2750 / 3708 | 1125 / 2875 / 4417 |
| k8 | other model | 208 / 250 / 292 | 250 / 250 / 333 |
| k16 | hot | 84 / 125 / 125 | 84 / 125 / 208 |
| k16 | data 64MiB | 1167 / 2917 / 4000 | 1000 / 2750 / 3875 |
| k16 | other model | 208 / 250 / 292 | 250 / 250 / 333 |
| k16-shared | hot | 84 / 125 / 125 | 84 / 125 / 125 |
| k16-shared | data 64MiB | 917 / 2417 / 3375 | 1042 / 2708 / 3750 |
| k16-shared | other model | 209 / 250 / 333 | 250 / 250 / 334 |

300 trees × 100 features, native ARM64 cold latency.

| Variant | Condition | Shuffled p50/p90/p99 | Dependent p50/p90/p99 |
| --- | --- | --- | --- |
| continuation | hot | 250 / 250 / 333 | 250 / 250 / 375 |
| continuation | data 64MiB | 2666 / 6875 / 8917 | 2708 / 6875 / 8875 |
| continuation | other model | 500 / 625 / 750 | 500 / 666 / 1000 |
| k8 | hot | 333 / 334 / 417 | 292 / 334 / 334 |
| k8 | data 64MiB | 3417 / 7958 / 13250 | 2250 / 4875 / 8167 |
| k8 | other model | 583 / 750 / 833 | 583 / 750 / 792 |
| k16 | hot | 292 / 292 / 334 | 292 / 292 / 375 |
| k16 | data 64MiB | 2541 / 7292 / 12333 | 2584 / 7500 / 13583 |
| k16 | other model | 542 / 667 / 750 | 583 / 708 / 792 |
| k16-shared | hot | 292 / 292 / 375 | 292 / 292 / 334 |
| k16-shared | data 64MiB | 2458 / 6583 / 8292 | 2375 / 5833 / 7916 |
| k16-shared | other model | 583 / 708 / 834 | 584 / 750 / 833 |

Both scalar_bench gates and all eight 3,000-sample cold runs exited successfully. All correctness gates, measurements, and scope checks are complete; nothing was committed or merged.

In model order, spilling starts at K1 on ARM64 and K11 on SSE2 at both sizes. AVX2 first spills at K13 for 100 trees and K12 for 300 trees. K16 shared also spills on all targets. Some x86 K values reduce row loads without spilling, but no measured scheduled variant beats continuation on both row loads and native ARM64 hot time. x86 hot time was not measured.
