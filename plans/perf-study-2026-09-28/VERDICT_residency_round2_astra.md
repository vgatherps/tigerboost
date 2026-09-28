status: pass

The requested emission and tests are complete. The fences do not solve ARM64 spilling or restore continuation-level threshold pairing. Native ARM64 hot medians regress for every scheduled variant: continuation is 90.25 ns / 238.88 ns at 100 / 300 trees; the best fenced times are 99.04 ns / 281.20 ns. Leaf packing does work: every fenced ARM64 leaf uses a chunk-relative immediate displacement of at most 3,872 bytes, including SharedFeatures. Performance and instruction-folding failures are reported as experiment results; no additional barriers or semantic changes were added.

Seven tracked files differ from starting commit aafd45b: src/jit.rs, tests/differential.rs, tests/scalar_jit.rs, examples/scalar_bench.rs, examples/cold_bench.rs, examples/inspect_code.rs, examples/x86_check.rs. Only src/jit.rs needed further tracked edits for amendment 2. src/sort_policy.rs and tools/x86_check.sh are unchanged; the shell script already reads the example’s variant manifest. VERDICT.md and probes/round3 contain this report and its artifacts. No commits, merges, stashes, or formatting commands were made.

The 9f2fa6c anchors were checked: EmitOptions starts at 27, prepare_module at 137, early-cse at 164, and load_feature at 806 rather than 742. The initial branch already contained round 1; round-2 partial edits were retained and completed. Prior reports remain in probes/round2/round1-VERDICT.md and probes/round3/round2-VERDICT.md.

1. EmitOptions has avx2, scheduled, fence_chunks, resident_features, volatile_leaves. Boolean defaults are false; the residency budget defaults to 12. Scheduled-only options do not affect continuation emission.
2. Scheduled emission defines only C-ABI predict(row), skips early-cse, and retains row align 8 and dereferenceable without readonly/noalias/memory(read). fence_chunks adds exactly chunks − 1 empty side-effecting assembly calls with constraints =r,=r,0,1,~{memory}. Both extracted pointer outputs feed subsequent chunks. Each fenced chunk gets threshold and leaf base GEPs from its current constants pointer.
3. Feature loads remain volatile double loads with alignment 8; the NaN select remains after each load. Thresholds stay ordinary. Leaves become volatile only with scheduled && volatile_leaves. All three leaf sites, including constant trees, use chunk-relative addressing when fenced.
4. feature_fills and residency_schedule remain unchanged from round 1. Real lane fills follow action order; synthetic padding reuses its paired value. Reverse next-use preprocessing plus a scan of at most K residents implements Belady in O(fills × K), with no-next-use first and lower feature index breaking ties. SSA residents survive fences. scheduled_feature_loads reports production misses. Leaf blocks now follow action visit order whenever scheduled is true, including unfenced and constant trees; leaf_base remains indexed by TreeId. Continuations retain model-order packing.
5. LLVM 21 still folds some volatile feature loads into ARM ld1.d and x86 movhpd/vmovhpd. The assembly probes validate the exact miss-address sequence, so these folds do not change the measured row-load count/order. No workaround was added. Threshold pairing and leaf materialization results appear below.
6. scalar_bench and inspect_code expose fenced K8/K12/K16/K24 and K16 SharedFeatures plus unfenced K16. Both sizes and all three target configurations were emitted. cold_bench accepts --scheduled [K] --fence-chunks and --shared-features. x86_check runs the fenced K16 variants.
7. Tests retain Belady’s four-miss A B C A B C case and eviction tie, off-mode assembly identity, aligned volatile load counts and exact SSA reuse, NaN behavior, assembly miss order, and dropped-model execution. Differential tests cycle K1/K4/K16/K64, both orders, volatility, and fences. Fence tests check constraints, pointer chaining, and both chunk bases after prepare_module. The new scheduled_shared_leaf_blocks_follow_visit_order test checks the complete production constants initializer chunk by chunk on ARM64/SSE2/AVX2 with fences on/off, including odd depths and constants, and checks all 16 binary input rows against the reference. All 71 debug and release tests pass.

| Gate | Result |
|---|---|
| git stash list; git show 5a915cd:src/jit.rs | Empty stash; baseline available |
| Fresh 5a915cd scratch source; release inspect_code 100x100 | git archive snapshot built offline; scratch source removed |
| cargo build --offline --release --examples | Pass |
| cargo test --offline --all-targets | 71 passed |
| cargo test --offline --release --all-targets | 71 passed |
| cargo test --offline --doc | 1 passed |
| cargo clippy --offline --all-targets -- -D warnings | Pass |
| inspect_code 100x100; cmp against baseline | 18/18 byte-identical: 9 assemblies and 9 plan dumps |
| inspect_code 300x100 | All variants emitted |
| Unfenced K16 controls against saved round 1 | 6/6 assemblies byte-identical across both sizes and all targets |
| tools/x86_check.sh 100x100 | 6 variants × 256 rows passed under Rosetta |
| tools/x86_check.sh special --avx2 | 6 variants × 256 rows passed under Rosetta |
| scalar_bench 100x100; scalar_bench 300x100 | Pass; all variants verified |
| cold_bench, 3000 samples | Pass at both sizes: continuation, unfenced K16, fenced K8/K16/K16 shared |
| Independent assembly and K1–K24 sweep | 42 table kernels and 144 sweep kernels counted; every scheduled row address matches Belady in order |
| x86 timing probes | Both sizes/SSE2/AVX2; all rows and 50,000 dependent calls verified for every variant |
| git diff --check | Pass |

Gates used LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 and RUSTDOCFLAGS=-C link-arg=/opt/homebrew/lib/libzstd.dylib. macOS 15.6, LLVM 21.1.8, inkwell 0.9.0, rustc 1.93.0-nightly (c90bcb957). Native timings use ARM64; x86 timings use x86_64-apple-darwin code under Rosetta, while static x86 counts use the requested x86_64-unknown-linux-gnu output. These are indicative timings, not native x86 CPU measurements. Clippy initially rejected the touched odd-depth modulo check; is_multiple_of resolved it, then all gates were rerun. The additional C probe initially used a constant-bound VLA rejected by -Werror; an enum bound fixed it. Linker deprecation/alignment warnings did not affect execution checks.

Belady misses are target-independent. The budgets below apply to the unchanged feature schedule with either fence setting.

| Trees | Order | K8 | K12 | K16 | K20 | K24 |
|---|---|---|---|---|---|---|
| 100 | Model | 423 | 375 | 338 | 309 | 281 |
| 100 | SharedFeatures | 347 | 299 | 268 | 248 | 228 |
| 300 | Model | 1242 | 1087 | 964 | 866 | 780 |
| 300 | SharedFeatures | 936 | 789 | 681 | 611 | 550 |

Static counts include predict and every continuation body, excluding data, directives, alignment padding, and empty fences. Row counts give doubles read / memory instructions. Threshold and leaf counts include folded memory operands; an ldp counts once. Each tree reads three threshold vectors on ARM64/SSE2 or two on AVX2, and one leaf. All totals are checked by the probe.

Register inserts include ARM mov.d into a vector lane / ins and x86 register-only unpcklpd, movlhps, vinsertf128, vblendpd, vperm2f128; broadcasts count dup.2d, movddup, vbroadcastsd separately. Whole-register copies and mask extraction are excluded. Folded row inserts count ld1.d / movhpd / vmovhpd. Body stack instructions exclude ABI saves/restores. Frame bytes give explicit allocation (including saves) + red-zone depth; continuation allocation is wrapper plus largest continuation frame. Prologue instructions/bytes sum stack setup and saves across functions; x86 return addresses are excluded.

Static counts for ARM64.

| Trees | Variant | Instructions | Row doubles/ins | Reg inserts/broadcasts | Folded row inserts | Threshold loads | Threshold ldp q | Leaf loads | Body stack | Frame + red zone, B | Prologue ins/B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | continuation | 2143 | 468/429 | 125/0 | 172 | 163 | 137 | 100 | 0 | 160 + 0 | 10/40 |
| 100 | K8 fenced | 2347 | 423/423 | 148/0 | 149 | 274 | 26 | 100 | 126 | 704 + 0 | 6/24 |
| 100 | K12 fenced | 2324 | 375/375 | 178/0 | 119 | 273 | 27 | 100 | 127 | 736 + 0 | 6/24 |
| 100 | K16 fenced | 2312 | 338/338 | 193/1 | 100 | 273 | 27 | 100 | 135 | 752 + 0 | 6/24 |
| 100 | K24 fenced | 2324 | 281/281 | 225/1 | 68 | 273 | 27 | 100 | 164 | 880 + 0 | 6/24 |
| 100 | K16 shared fenced | 2251 | 268/268 | 235/0 | 58 | 271 | 29 | 100 | 120 | 736 + 0 | 6/24 |
| 100 | K16 unfenced | 2299 | 338/338 | 193/1 | 100 | 268 | 32 | 100 | 136 | 768 + 0 | 7/28 |
| 300 | continuation | 6521 | 1419/1315 | 359/0 | 532 | 463 | 437 | 300 | 0 | 160 + 0 | 10/40 |
| 300 | K8 fenced | 7153 | 1242/1242 | 468/1 | 422 | 872 | 28 | 300 | 495 | 2176 + 0 | 6/24 |
| 300 | K12 fenced | 7095 | 1087/1087 | 558/1 | 332 | 872 | 28 | 300 | 505 | 2208 + 0 | 6/24 |
| 300 | K16 fenced | 7058 | 964/964 | 610/1 | 281 | 872 | 28 | 300 | 515 | 2240 + 0 | 6/24 |
| 300 | K24 fenced | 7029 | 780/780 | 704/2 | 188 | 872 | 28 | 300 | 569 | 2448 + 0 | 6/24 |
| 300 | K16 shared fenced | 6883 | 681/681 | 773/1 | 109 | 873 | 27 | 300 | 517 | 2256 + 0 | 6/24 |
| 300 | K16 unfenced | 7206 | 964/964 | 610/1 | 281 | 868 | 32 | 300 | 639 | 2496 + 0 | 11/44 |

Static counts for SSE2.

| Trees | Variant | Instructions | Row doubles/ins | Reg inserts/broadcasts | Folded row inserts | Threshold loads | Threshold ldp q | Leaf loads | Body stack | Frame + red zone, B | Prologue ins/B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | continuation | 2300 | 470/470 | 120/0 | 174 | 300 | 0 | 100 | 0 | 48 + 0 | 7/11 |
| 100 | K8 fenced | 2294 | 423/423 | 150/0 | 149 | 300 | 0 | 100 | 0 | 0 + 0 | 0/0 |
| 100 | K12 fenced | 2381 | 375/375 | 170/0 | 119 | 300 | 0 | 100 | 67 | 0 + 120 | 0/0 |
| 100 | K16 fenced | 2378 | 338/338 | 175/0 | 100 | 300 | 0 | 100 | 115 | 88 + 128 | 1/4 |
| 100 | K24 fenced | 2451 | 281/281 | 179/0 | 68 | 300 | 0 | 100 | 234 | 280 + 128 | 1/7 |
| 100 | K16 shared fenced | 2396 | 268/268 | 215/0 | 58 | 300 | 0 | 100 | 110 | 88 + 128 | 1/4 |
| 100 | K16 unfenced | 2756 | 338/338 | 174/0 | 100 | 300 | 0 | 100 | 119 | 104 + 128 | 1/4 |
| 300 | continuation | 6867 | 1414/1414 | 364/0 | 527 | 900 | 0 | 300 | 0 | 48 + 0 | 8/12 |
| 300 | K8 fenced | 6817 | 1242/1242 | 476/0 | 422 | 900 | 0 | 300 | 0 | 0 + 0 | 0/0 |
| 300 | K12 fenced | 7048 | 1087/1087 | 525/0 | 332 | 900 | 0 | 300 | 200 | 40 + 128 | 1/4 |
| 300 | K16 fenced | 7160 | 964/964 | 540/0 | 281 | 900 | 0 | 300 | 409 | 120 + 128 | 1/4 |
| 300 | K24 fenced | 7334 | 780/780 | 572/0 | 188 | 900 | 0 | 300 | 706 | 296 + 128 | 1/7 |
| 300 | K16 shared fenced | 7140 | 681/681 | 732/0 | 109 | 900 | 0 | 300 | 347 | 104 + 128 | 1/4 |
| 300 | K16 unfenced | 8319 | 964/964 | 541/0 | 281 | 900 | 0 | 300 | 394 | 104 + 128 | 1/4 |

Static counts for AVX2.

| Trees | Variant | Instructions | Row doubles/ins | Reg inserts/broadcasts | Folded row inserts | Threshold loads | Threshold ldp q | Leaf loads | Body stack | Frame + red zone, B | Prologue ins/B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | continuation | 1451 | 470/470 | 224/0 | 161 | 200 | 0 | 100 | 0 | 40 + 0 | 5/9 |
| 100 | K8 fenced | 1405 | 423/423 | 253/0 | 138 | 200 | 0 | 100 | 0 | 0 + 0 | 0/0 |
| 100 | K12 fenced | 1388 | 375/375 | 282/0 | 109 | 200 | 0 | 100 | 0 | 0 + 0 | 0/0 |
| 100 | K16 fenced | 1463 | 338/338 | 283/0 | 91 | 200 | 0 | 100 | 94 | 232 + 128 | 1/7 |
| 100 | K24 fenced | 1511 | 281/281 | 287/0 | 64 | 200 | 0 | 100 | 188 | 584 + 128 | 1/7 |
| 100 | K16 shared fenced | 1413 | 268/268 | 325/0 | 55 | 200 | 0 | 100 | 67 | 200 + 128 | 1/7 |
| 100 | K16 unfenced | 1734 | 338/338 | 283/0 | 91 | 200 | 0 | 100 | 89 | 200 + 128 | 1/7 |
| 300 | continuation | 4304 | 1414/1414 | 658/0 | 512 | 600 | 0 | 300 | 0 | 40 + 0 | 5/9 |
| 300 | K8 fenced | 4153 | 1242/1242 | 782/0 | 401 | 600 | 0 | 300 | 0 | 0 + 0 | 0/0 |
| 300 | K12 fenced | 4093 | 1087/1087 | 870/0 | 311 | 600 | 0 | 300 | 4 | 0 + 24 | 0/0 |
| 300 | K16 fenced | 4300 | 964/964 | 890/0 | 257 | 600 | 0 | 300 | 275 | 296 + 128 | 1/7 |
| 300 | K24 fenced | 4474 | 780/780 | 914/0 | 170 | 600 | 0 | 300 | 576 | 664 + 128 | 1/7 |
| 300 | K16 shared fenced | 4114 | 681/681 | 1051/0 | 102 | 600 | 0 | 300 | 212 | 264 + 128 | 1/7 |
| 300 | K16 unfenced | 5190 | 964/964 | 886/0 | 257 | 600 | 0 | 300 | 292 | 296 + 128 | 1/7 |

The bases exist in IR, but they do not force machine-address bases to remain separate. In 100-tree fenced K16, late threshold loads still use offsets such as [x8, #4608] through [x8, #4784]; LLVM folded the two constant GEP offsets together after each opaque pointer output. Only 27 threshold ldp q remain versus 137 in continuation; at 300 trees, 28 remain versus 437. Thus continuation-level pairing did not return.

Leaf packing and relative addressing do survive: every fenced ARM64 leaf load uses an immediate displacement at most 3,872 bytes, with zero register-offset leaf loads at both sizes and both orders. The unfenced K16 control has 45 / 64 register-offset leaf loads at 100 / 300 trees. Chunk base addresses still need setup: the 100-tree K16 assembly materializes 13 base constants (4,800, 8,896, …, 53,952), one per chunk; it no longer materializes each tree’s large leaf offset. At 300 trees LLVM also uses shifted immediate adds to form larger chunk bases.

The pointer fence does not constrain floating-point additions. In 100-tree ARM64 fenced K16, the first eleven chunks execute zero fadd instructions; the final two execute 48 and 54. Early leaves are loaded and stored to the stack until their late additions. At 300 trees, the first 34 chunks execute zero fadd instructions. The explicit K budget bounds feature residency, but does not bound those other values.

ARM64 fenced sweep, model order. Each row-load count equals the independent Belady miss count. Frame bytes include ABI saves; every capacity has body stack traffic.

| K | 100 loads | 100 stack ins | 100 frame B | 100 ldp q | 300 loads | 300 stack ins | 300 frame B | 300 ldp q |
|---|---|---|---|---|---|---|---|---|
| 1 | 595 | 112 | 624 | 26 | 1775 | 484 | 2112 | 26 |
| 2 | 550 | 111 | 640 | 26 | 1637 | 484 | 2128 | 26 |
| 3 | 517 | 115 | 656 | 26 | 1540 | 488 | 2128 | 26 |
| 4 | 493 | 119 | 672 | 26 | 1462 | 492 | 2128 | 26 |
| 5 | 472 | 122 | 672 | 26 | 1397 | 494 | 2144 | 27 |
| 6 | 454 | 126 | 672 | 26 | 1340 | 492 | 2144 | 27 |
| 7 | 438 | 124 | 688 | 26 | 1289 | 493 | 2160 | 28 |
| 8 | 423 | 126 | 704 | 26 | 1242 | 495 | 2176 | 28 |
| 9 | 409 | 132 | 720 | 27 | 1198 | 493 | 2176 | 28 |
| 10 | 397 | 131 | 736 | 27 | 1158 | 500 | 2192 | 28 |
| 11 | 386 | 131 | 736 | 27 | 1121 | 505 | 2192 | 28 |
| 12 | 375 | 127 | 736 | 27 | 1087 | 505 | 2208 | 28 |
| 13 | 364 | 133 | 736 | 27 | 1053 | 512 | 2224 | 28 |
| 14 | 354 | 133 | 736 | 27 | 1021 | 517 | 2224 | 28 |
| 15 | 346 | 139 | 752 | 27 | 992 | 518 | 2240 | 28 |
| 16 | 338 | 135 | 752 | 27 | 964 | 515 | 2240 | 28 |
| 17 | 330 | 140 | 768 | 27 | 938 | 526 | 2256 | 28 |
| 18 | 323 | 141 | 768 | 27 | 913 | 530 | 2272 | 28 |
| 19 | 316 | 135 | 768 | 27 | 889 | 540 | 2288 | 28 |
| 20 | 309 | 144 | 768 | 27 | 866 | 545 | 2320 | 28 |
| 21 | 302 | 140 | 800 | 27 | 843 | 548 | 2352 | 28 |
| 22 | 295 | 150 | 816 | 27 | 821 | 555 | 2336 | 28 |
| 23 | 288 | 159 | 848 | 27 | 800 | 563 | 2384 | 28 |
| 24 | 281 | 164 | 880 | 27 | 780 | 569 | 2448 | 28 |

First fenced spill, model order, from the full K1–K24 sweep. K8 reduces row loads without spilling on both x86 paths. No ARM64 capacity does.

| Target | 100 first spill | 100 fewer loads without spills | 300 first spill | 300 fewer loads without spills |
|---|---|---|---|---|
| ARM64 | K1 | none | K1 | none |
| SSE2 | K11 | K6–K10 | K10 | K5–K9 |
| AVX2 | K13 | K6–K12 | K12 | K5–K11 |

Hot medians, ns/prediction. Five alternating-order rounds × 50,000 dependent calls, with a 10,000-call warmup. Driver cost is included and not subtracted. The native driver visits 35 / 12 distinct rows at 100 / 300 trees. Raw logs also report min/max.

| Variant | ARM64 100 | ARM64 300 | SSE2 Rosetta 100 | SSE2 Rosetta 300 | AVX2 Rosetta 100 | AVX2 Rosetta 300 |
|---|---|---|---|---|---|---|
| continuation | 90.25 | 238.88 | 109.84 | 303.93 | 410.26 | 1392.57 |
| K8 fenced | 107.08 | 299.89 | 110.41 | 305.46 | 368.71 | 1239.50 |
| K12 fenced | 105.21 | 288.18 | 111.49 | 308.73 | 347.22 | 1181.58 |
| K16 fenced | 103.20 | 283.38 | 111.42 | 310.33 | 361.12 | 1245.74 |
| K24 fenced | 99.04 | 284.37 | 110.22 | 306.52 | 375.06 | 1206.06 |
| K16 shared fenced | 99.65 | 281.20 | 111.44 | 304.61 | 375.16 | 1201.30 |
| K16 unfenced | 101.79 | 284.63 | 110.26 | 309.22 | 361.75 | 1202.02 |

The 300-tree AVX2 Rosetta hot run was noisy: continuation ranged from 1,328.61 to 2,195.75 ns across the five rounds, and fenced K8 ranged from 1,193.54 to 2,175.73 ns. All five fenced variants had lower medians, but these timings do not establish a native x86 speedup.

Cold measurements, ns, shown as p50 / p90 / p99. Every condition keeps 3,000 single-call samples after a discarded first pass, for shuffled and prediction-dependent row sequences. 64 MiB means touching all randomized cache lines before each call; other model runs the 2,000-tree continuation model first. Rows are copied to a fixed buffer before timing. Native cold_bench also measures other buffer sizes and timer-only controls in the raw logs. Timers have roughly 41–42 ns granularity; no timer floor is subtracted.

Cold results for ARM64.

| Trees | Variant | Sequence | Hot | 64 MiB | Other model |
|---|---|---|---|---|---|
| 100 | continuation | shuffled | 83 / 84 / 125 | 959 / 2250 / 3709 | 208 / 250 / 292 |
| 100 | continuation | dependent | 83 / 84 / 208 | 1333 / 2708 / 4208 | 208 / 250 / 292 |
| 100 | K8 fenced | shuffled | 84 / 125 / 125 | 916 / 2000 / 3625 | 209 / 250 / 292 |
| 100 | K8 fenced | dependent | 84 / 125 / 125 | 1042 / 2625 / 4209 | 250 / 291 / 333 |
| 100 | K16 fenced | shuffled | 84 / 125 / 125 | 1083 / 2667 / 3792 | 208 / 250 / 333 |
| 100 | K16 fenced | dependent | 84 / 125 / 125 | 958 / 1291 / 3458 | 250 / 250 / 292 |
| 100 | K16 shared fenced | shuffled | 84 / 125 / 125 | 833 / 1167 / 3291 | 208 / 250 / 292 |
| 100 | K16 shared fenced | dependent | 84 / 125 / 125 | 917 / 2417 / 4750 | 209 / 250 / 292 |
| 100 | K16 unfenced | shuffled | 84 / 125 / 125 | 1000 / 2542 / 3750 | 208 / 250 / 292 |
| 100 | K16 unfenced | dependent | 84 / 125 / 334 | 916 / 2333 / 3875 | 250 / 250 / 292 |
| 300 | continuation | shuffled | 250 / 250 / 333 | 2208 / 5791 / 9500 | 583 / 667 / 750 |
| 300 | continuation | dependent | 250 / 250 / 375 | 2125 / 2458 / 6000 | 583 / 667 / 750 |
| 300 | K8 fenced | shuffled | 292 / 333 / 334 | 2166 / 2458 / 6625 | 583 / 709 / 791 |
| 300 | K8 fenced | dependent | 292 / 333 / 375 | 2083 / 2292 / 6209 | 584 / 709 / 792 |
| 300 | K16 fenced | shuffled | 292 / 292 / 334 | 2125 / 2417 / 6500 | 583 / 709 / 875 |
| 300 | K16 fenced | dependent | 292 / 292 / 375 | 2125 / 2375 / 6458 | 542 / 667 / 750 |
| 300 | K16 shared fenced | shuffled | 291 / 292 / 292 | 1875 / 2125 / 6167 | 542 / 584 / 709 |
| 300 | K16 shared fenced | dependent | 292 / 292 / 334 | 2083 / 6333 / 9792 | 542 / 625 / 750 |
| 300 | K16 unfenced | shuffled | 292 / 292 / 333 | 2125 / 2375 / 6500 | 583 / 625 / 750 |
| 300 | K16 unfenced | dependent | 292 / 292 / 334 | 2125 / 2875 / 7083 | 583 / 584 / 750 |

Cold results for SSE2 under Rosetta.

| Trees | Variant | Sequence | Hot | 64 MiB | Other model |
|---|---|---|---|---|---|
| 100 | continuation | shuffled | 125 / 125 / 125 | 667 / 833 / 3709 | 208 / 250 / 292 |
| 100 | continuation | dependent | 125 / 125 / 125 | 708 / 2791 / 4292 | 209 / 250 / 334 |
| 100 | K8 fenced | shuffled | 125 / 125 / 125 | 625 / 833 / 2959 | 291 / 292 / 375 |
| 100 | K8 fenced | dependent | 125 / 125 / 125 | 625 / 750 / 2833 | 291 / 292 / 334 |
| 100 | K16 fenced | shuffled | 125 / 125 / 125 | 625 / 750 / 2833 | 291 / 292 / 334 |
| 100 | K16 fenced | dependent | 125 / 125 / 125 | 625 / 750 / 2709 | 292 / 292 / 334 |
| 100 | K16 shared fenced | shuffled | 125 / 125 / 125 | 667 / 917 / 3125 | 250 / 292 / 334 |
| 100 | K16 shared fenced | dependent | 125 / 125 / 125 | 666 / 792 / 2792 | 250 / 292 / 334 |
| 300 | continuation | shuffled | 292 / 334 / 375 | 2333 / 6250 / 12041 | 625 / 667 / 833 |
| 300 | continuation | dependent | 292 / 334 / 334 | 2333 / 5000 / 11458 | 625 / 667 / 750 |
| 300 | K8 fenced | shuffled | 292 / 333 / 334 | 1917 / 2125 / 7833 | 959 / 1000 / 1250 |
| 300 | K8 fenced | dependent | 333 / 334 / 417 | 1917 / 2417 / 8542 | 959 / 1000 / 1084 |
| 300 | K16 fenced | shuffled | 333 / 334 / 375 | 1959 / 2333 / 8708 | 959 / 1000 / 1042 |
| 300 | K16 fenced | dependent | 292 / 334 / 334 | 1958 / 2125 / 8125 | 959 / 1000 / 1208 |
| 300 | K16 shared fenced | shuffled | 292 / 334 / 334 | 2041 / 7459 / 10708 | 958 / 1000 / 1292 |
| 300 | K16 shared fenced | dependent | 292 / 334 / 334 | 1958 / 7125 / 21500 | 917 / 959 / 1167 |

Cold results for AVX2 under Rosetta.

| Trees | Variant | Sequence | Hot | 64 MiB | Other model |
|---|---|---|---|---|---|
| 100 | continuation | shuffled | 417 / 459 / 500 | 1750 / 5833 / 9459 | 584 / 708 / 792 |
| 100 | continuation | dependent | 417 / 459 / 500 | 2083 / 8167 / 10542 | 625 / 709 / 792 |
| 100 | K8 fenced | shuffled | 375 / 417 / 500 | 1834 / 7625 / 9709 | 792 / 834 / 1042 |
| 100 | K8 fenced | dependent | 375 / 417 / 458 | 1750 / 6667 / 8792 | 833 / 834 / 1000 |
| 100 | K16 fenced | shuffled | 375 / 417 / 500 | 2250 / 8125 / 10583 | 833 / 875 / 1084 |
| 100 | K16 fenced | dependent | 375 / 375 / 417 | 1958 / 7292 / 9917 | 833 / 875 / 1000 |
| 100 | K16 shared fenced | shuffled | 375 / 417 / 500 | 2250 / 8041 / 30584 | 792 / 834 / 1041 |
| 100 | K16 shared fenced | dependent | 375 / 417 / 459 | 1792 / 7084 / 9375 | 792 / 834 / 1000 |
| 300 | continuation | shuffled | 1334 / 1417 / 1708 | 7542 / 25208 / 33083 | 1750 / 1875 / 2333 |
| 300 | continuation | dependent | 1334 / 1458 / 1791 | 6667 / 24584 / 32667 | 1791 / 1875 / 2291 |
| 300 | K8 fenced | shuffled | 1250 / 1375 / 1750 | 19541 / 25042 / 47834 | 2750 / 2792 / 2875 |
| 300 | K8 fenced | dependent | 1167 / 1250 / 1333 | 6542 / 23041 / 30000 | 2708 / 2791 / 3500 |
| 300 | K16 fenced | shuffled | 1166 / 1209 / 1375 | 6833 / 22792 / 29042 | 2625 / 2708 / 3292 |
| 300 | K16 fenced | dependent | 1166 / 1208 / 1291 | 6625 / 24084 / 58625 | 2667 / 2750 / 3459 |
| 300 | K16 shared fenced | shuffled | 1167 / 1250 / 1541 | 6209 / 21625 / 26584 | 2583 / 2625 / 3292 |
| 300 | K16 shared fenced | dependent | 1167 / 1250 / 1500 | 7916 / 23583 / 29667 | 2625 / 2708 / 3333 |

All logs, assemblies, plan dumps, parser source, and JSON counts are under probes/round3. probes/emitter uses production emit_assembly_with and scheduled_feature_loads; the Python counter independently simulates Belady from plan dumps and validates each emitted miss address. x86 timings use generated production assembly linked to a C driver and verify predictions before measuring.

ARM64 variants beating continuation on both row loads and measured hot median — 100 trees: none; 300 trees: none.
SSE2 under Rosetta variants beating continuation on both row loads and measured hot median — 100 trees: none; 300 trees: none.
AVX2 under Rosetta variants beating continuation on both row loads and measured hot median — 100 trees: K8 fenced, K12 fenced, K16 fenced, K24 fenced, K16 shared fenced, K16 unfenced; 300 trees: K8 fenced, K12 fenced, K16 fenced, K24 fenced, K16 shared fenced, K16 unfenced.

Fenced single-function kernels first spill at K1 on ARM64 at both sizes, K11 / K10 on SSE2 at 100 / 300 trees, and K13 / K12 on AVX2. These are workload-specific backend results, not universal register-budget limits. No fenced ARM64 variant beats continuation on hot time. Small x86 timing differences under Rosetta are indicative only; the row-load reductions and spill counts are deterministic for these emitted kernels.
