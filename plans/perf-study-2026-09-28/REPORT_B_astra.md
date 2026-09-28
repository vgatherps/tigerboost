# Study B: scheduler, feature residency, and continuation size

Keep the scheduler for the next experiment, with GP4/FP9 and two sums on both targets. Keep 16-test chunks for that first comparison. This reduces emitted instructions and code bytes at both measured forest sizes. Increasing the FP budget saves feature loads but adds stack arguments. Increasing the GP budget gives little extra reuse and can make LLVM fail register allocation on x86-64.

These are study results, not an implementation change. All workloads use depth six, 100 declared features, and globally distinct thresholds. The 100-tree workload actually uses 99 features; the 300-tree workload uses all 100. The exact lower bounds are therefore 99 and 100 loads. The brief's 394 default loads and 600 tree-order loads are correct.

## 1. Answers and evidence

### B1. Physical registers and arguments

Eight scalar floating-point arguments cross a `preserve_nonecc` boundary in registers on both targets. That limit matters more than the number of vector registers available inside a continuation. Two sums leave six argument registers for resident features. FP9 already permits those six features plus one comparison temporary.

For AArch64 Darwin, the integer argument order is x20–x28, x0–x7, x10–x14, x9, x15: 24 registers. Scalar FP arguments use d0–d7; vector arguments use the corresponding v0–v7 aliases. x0–x17 and x19–x28 are caller-saved working registers. x18 is reserved on this target; x29 and x30 remain preserved. All v0–v31 are caller-saved under LLVM's implementation. The argument lists and preserved set are in [LLVM 21.1.0 AArch64CallingConvention.td, lines 472–510 and 628](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.0/llvm/lib/Target/AArch64/AArch64CallingConvention.td). [LLVM 21.1.8 AArch64RegisterInfo.cpp, lines 72–74](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/AArch64/AArch64RegisterInfo.cpp) selects that preserved set.

For x86-64 Linux, integer arguments use r12, r13, r14, r15, rdi, rsi, rdx, rcx, r8, r9, r11, rax: 12 registers. Scalar FP/vector arguments use xmm0–xmm7. rbp remains callee-saved; rsp remains the stack pointer. The other 14 ordinary GP registers and xmm0–xmm15 are caller-saved. AVX2 aliases the same vector register numbers. AVX-512 adds vector register numbers 16–31, but does not enlarge this scalar argument list. See [LLVM 21.1.8 X86CallingConv.td, lines 521–540, 940–955, and 1080](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/X86/X86CallingConv.td) and [X86RegisterInfo.cpp, lines 299–301](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/X86/X86RegisterInfo.cpp).

The installed package has CallingConv.h but no target calling-convention source. The version-pinned 21.1.8 AArch64 calling-convention URL failed to fetch, so the source citation uses 21.1.0 and the local probes check the installed 21.1.8 behavior. `probes/abi/aarch64.s:9–21` confirms the argument order; `:32–38` loads the ninth and tenth FP arguments from the stack. `probes/abi/x86_64.s:43–52` does the same for x86. The AVX-512 probe also puts those arguments on the stack. Clobber-only functions compile to `ret` without saves for all working registers listed above. The empty inline asm exists only to test preservation; it is not a proposed kernel barrier.

The [LLVM 21 LangRef paragraph](https://releases.llvm.org/21.1.0/docs/LangRef.html#calling-conventions) says non-general registers retain the C convention. That describes FP argument placement here, but it does not describe AArch64's actual preserved FP set. The source selects only FP/LR, and the local clobber probe saves no d8–d15. Prefer the implementation and the installed-tool probe for this distinction.

| Target | Caller-saved working GP / FP registers | Largest logical GP / FP budget allowed by boundary capacity, scalar and two sums | Largest tested GP at FP9 with zero body stack accesses at both sizes |
| --- | --- | --- | --- |
| AArch64 Darwin | 28 / 32 | 26 / 9 | 16 |
| x86-64 Linux, current generic emission | 14 / 16 | 14 / 9 | 8 |

The middle column is a capacity bound, not a spill guarantee. Scalar planning permits at most GP−3 active selectors: row pointer plus two scratch registers use the other slots (`src/sort_policy.rs:336–360`). AArch64's 24 integer arguments allow the row plus 23 selectors, hence GP26. x86's 12 allow the row plus 11 selectors, hence GP14. LLVM can keep more temporary values live than this arithmetic counts. Indeed, the tested x86 GP14 case fails allocation, and GP12 already uses rbp saves and a spill.

Within one body the raw resource ceilings are GP28/FP32 on AArch64 and GP14/FP16 on baseline x86 if no callee-save traffic is allowed. Those ceilings do not apply to carried state. `src/jit.rs:109–125` builds one signature from maximum slot-vector lengths, then `:196–225` reads live values from those slots. Unused slots receive undef (`:171–175`, `:421–438`); live features in slots beyond the eight FP arguments still need stack storage. GP12/FP9 is therefore conservative in AArch64 GP capacity, but FP9 is already at the useful scalar boundary limit.

### B2. Residency curve

No tested budget approaches one load per used feature without stack traffic. The best load count in the requested grid is 220, over twice the 99-feature lower bound. All grid points have 600 tests and 38 chunks. Maximum residency is exactly FP−3.

| GP | FP | Loads | Reloads | Chunks | Maximum resident features | Maximum active trees |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 9 | 394 | 295 | 38 | 6 | 9 |
| 12 | 12 | 347 | 248 | 38 | 9 | 9 |
| 12 | 16 | 320 | 221 | 38 | 13 | 9 |
| 12 | 24 | 268 | 169 | 38 | 21 | 9 |
| 12 | 32 | 224 | 125 | 38 | 29 | 9 |
| 20 | 9 | 393 | 294 | 38 | 6 | 17 |
| 20 | 12 | 365 | 266 | 38 | 9 | 16 |
| 20 | 16 | 322 | 223 | 38 | 13 | 17 |
| 20 | 24 | 264 | 165 | 38 | 21 | 17 |
| 20 | 32 | 224 | 125 | 38 | 29 | 17 |
| 28 | 9 | 393 | 294 | 38 | 6 | 20 |
| 28 | 12 | 365 | 266 | 38 | 9 | 16 |
| 28 | 16 | 322 | 223 | 38 | 13 | 17 |
| 28 | 24 | 265 | 166 | 38 | 21 | 20 |
| 28 | 32 | 220 | 121 | 38 | 29 | 19 |

These counts come from the production `plan` function. `probes/study.rs` counts `Action::Load`, live load IDs, and incoming slots. Full actions and boundaries are in `probes/curve/`; the summary is `probes/curve.csv:2–16`. GP20/FP12 is worse than GP12/FP12: extra GP capacity does not produce a monotonic improvement.

The following emissions use the unchanged generic CPU, empty features, early-cse, and backend `Less` settings in `src/jit.rs:71–97`. Each successful AArch64 case also agrees with `Trees::predict` on all 256 generated rows. x86 code was assembled, never executed.

| GP/FP | ARM instructions | ARM body stack instructions | ARM text bytes | x86 instructions | x86 body stack instructions | x86 text bytes |
| --- | --- | --- | --- | --- | --- | --- |
| 12/9 | 4453 | 0 | 17812 | 4015 | 28 | 21458 |
| 12/12 | 4496 | 69 | 17984 | 4197 | 234 | 22629 |
| 12/16 | 4536 | 139 | 18144 | 4371 | 448 | 23840 |
| 12/24 | 4573 | 213 | 18292 | 5410 | 1339 | 29748 |
| 12/32 | 4610 | 295 | 18440 | 5621 | 1632 | 31622 |

At FP12, an AArch64 tail stores a feature with `str d8, [sp, #8]` before `b _continuation2` (`probes/curve/100-gp12-fp12-aarch64-apple-darwin.s:278–281`). x86 stores `%xmm0` at `8(%rsp)` before the next tail jump (`...-x86_64-unknown-linux-gnu.s:172–176`). These are transfer costs caused by argument placement, even though many vector registers are available for local calculations.

At FP9, GP20 emits 4,458 AArch64 instructions with eight body stack instructions; GP28 emits 4,484 with fifteen. Every GP20 and GP28 x86 emission in the requested grid exits 1 with `ran out of registers during register allocation`. The GP20/FP9 failure names `continuation20`; its complete diagnostic is `probes/curve/100-gp20-fp9-x86_64-unknown-linux-gnu.log:1`. A separate GP sweep also fails at GP14 and above on x86. These failed configurations supply no assembly or speed evidence. The planner counts remain valid. Each emission ran in a separate process after the first failure stopped a combined run.

Instruction totals include the wrapper. Stack counts count instructions that access stack memory, including push/pop and paired ARM loads/stores, but exclude SP arithmetic and wrapper accesses. x86 `%rbp` used as a leaf-table address or index is not a stack access; the parser checks whether a function establishes a frame pointer. The default x86 body has 26 rbp push/pop instructions and two actual spill accesses at 100 trees, or 52 and 24 at 300. AArch64's default wrapper has 20 stack instructions; x86's has ten. Text bytes include function alignment, and exclude thresholds and leaf data. `probes/asm_metrics.py` uses llvm-mc and llvm-size; `probes/assembly.csv` contains the results. All 74 measured assembly files in that table assembled successfully.

### B3. What residency is worth

The full saving is 501 loads at 100 trees and 1,700 at 300. The brief's rounded 500-load estimate gives almost the same percentage. These are upper bounds under the assumption that one avoided feature load removes one instruction and the row is in L1. They are not latency forecasts. ARM can pair loads; LLVM can merge repeated loads within a chunk; selector formation and state transfer also change when the order changes.

| Trees | ARM default instructions | x86 default instructions | Maximum removed loads | ARM fraction | x86 fraction | Loads already saved versus tree order |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 4,453 | 4,015 | 501 | 11.25% | 12.48% | 206 |
| 300 | 13,338 | 12,033 | 1700 | 12.75% | 14.13% | 608 |

Using exactly 500 instead of 501 gives 11.23% on ARM and 12.45% on x86 at 100 trees. The existing 206-load saving is only 4.63% and 5.13% of those kernels. The remaining theoretical saving from the default is 295 loads at 100 trees and 1,092 at 300, but keeping every used feature in an ordinary scalar register is impossible on either target.

ARM threshold materialization is roughly three instructions per split: about 1,800 instructions at 100 trees and 5,400 at 300. A representative `mov`/`movk`/`fmov` sequence appears in the supplied ARM capture at lines 55–61. Replacing three instructions with one would save roughly 1,200/3,600 before address setup, substantially more than residency's entire upper bound. This is a cost comparison, not verification of piece A's replacement. x86's repeated constant-address `movabsq` plus `ucomisd` is visible at `probes/curve/100-gp12-fp12-x86_64-unknown-linux-gnu.s:157–159`; its code-model alternatives belong to piece A.

Scheduling has a modest measured benefit on ARM. `plans/implementation.md:169–175` records 121.151 versus 124.656 ns at 100/100, 571.641 versus 587.466 at 500/100, and 586.101 versus 613.198 at 500/1000, all with the same continuation backend and two sums. Those median gains are about 2.8–4.4%. They do not establish a universal advantage, and they do not show what happens after cheaper threshold lowering. Calling the gain noise would also go beyond the evidence.

A fair static tree-order comparison needs two sums. The supplied `tree-order-1` capture uses one; this study emits its own two-sum baseline.

| Trees | Plan | ARM instructions | x86 instructions | x86 text bytes |
| --- | --- | --- | --- | --- |
| 100 | Tree order, two sums | 4565 | 3980 | 22073 |
| 100 | Scheduled GP12/FP9 | 4453 | 4015 | 21458 |
| 100 | Scheduled GP4/FP9 | 4413 | 3892 | 20309 |
| 300 | Tree order, two sums | 13658 | 11889 | 65686 |
| 300 | Scheduled GP12/FP9 | 13338 | 12033 | 64082 |
| 300 | Scheduled GP4/FP9 | 13152 | 11474 | 59642 |

On x86, tree order already has fewer instructions than GP12/FP9, but more code bytes. There is no x86 timing evidence. GP4/FP9 has fewer instructions and bytes than either, on both targets. Some scheduling remains worth testing; the evidence does not support expanding multi-tree residency. It also does not justify replacing the scheduler with tree order on the assumption that a cheaper backend makes the difference disappear.

### B4. Heuristic quality and reloads

FP9 permits six features, not nine. At a scalar comparison the arithmetic is `2 + resident + !loaded + 1 <= 9` (`src/sort_policy.rs:502–513`). A new feature therefore requires eviction when six are resident. The maximum measured residency is six.

Default planning loads 99 features for the first time, then reloads features 295 times. All 295 later reloads follow an earlier release while future uses remained. In contrast, the 99 final releases occur after all uses are consumed. The eviction loop removes the resident feature with the fewest total remaining uses; equal counts evict the larger feature index. It does not consider how soon that feature will be used (`src/sort_policy.rs:456–470`, `:505–511`).

The scheduler does share loaded values across trees, but most loads serve only one tree. The mean is 1.502538 distinct consumer trees per loaded value. The distribution counts distinct tree IDs for each `FeatureValueId` from load to release; multiple levels in the same tree count once.

| Trees consuming one loaded value | Load instances | Fraction |
| --- | --- | --- |
| 1 | 281 | 71.32% |
| 2 | 68 | 17.26% |
| 3 | 22 | 5.58% |
| 4 | 12 | 3.05% |
| 5 | 7 | 1.78% |
| 6 | 2 | 0.51% |
| 7 | 2 | 0.51% |

There are 592 distinct tree/value pairs across 600 tests. 113 of 394 loads serve more than one tree; 281 serve only one. Tree order serves one tree per load. `waiting_order` ranks overlap with resident features or active pending uses, then total feature-frequency score and tree ID (`src/sort_policy.rs:313–325`). This distribution shows that the combined admission and selection policy clusters some consumers. It does not isolate the causal contribution of `waiting_order` from `select`.

`operation` scans active trees before waiting trees and fits the requested comparison width into the budgets (`src/sort_policy.rs:329–391`). `select` first favors progress on the shortest active tree when GP pressure is high, then use count, completions, existing residency, and feature ID (`:394–451`). Residency is only the fourth ranking item. That explains why spare capacity does not imply minimum reloads.

For the exact default comparison order, a six-feature cache that always evicts the feature whose next use is farthest away needs 357 loads, versus the actual 394. This is a 37-load gap with the order held fixed, not a tested replacement scheduler. `probes/reuse_analysis.py` reconstructs the 600 feature requests and calculates this offline minimum; `probes/optimal-fixed-order.csv` records the result. At GP12/FP32, the corresponding numbers are 202 versus 224 at capacity 29. Thus, the eviction choice wastes some capacity, but a different eviction rule alone cannot reach 99 loads with six slots on the current order. Changing eviction in the actual planner would also change later overlap rankings.

Increasing GP from 12 to 20 at FP9 saves one load at 100 trees and adds 21 loads at 300. GP4 permits only one active tree; it uses 402 loads at 100 and 1,099 at 300, versus 394 and 1,192 for GP12. More simultaneous trees are not reliably better, even for feature loads. The current heuristic is not optimal, but the measured gap does not warrant a new planner during this study.

### B5. Chunk boundaries

Keep 16 tests with the current GP12 budget. The plan alone suggests trying 32 because it roughly halves carried-state transfers without increasing the maximum incoming state. The emitted code rejects that choice on ARM at GP12: body stack traffic appears and instruction count rises. GP4 changes the result; 32 tests then stays spill-free in both measured sizes and targets.

For GP12/FP9, these are sums of live values crossing boundaries, excluding chunk zero. They are not counts of emitted moves. Every boundary also carries the row pointer and two sum arguments.

| Trees | Test limit | Chunks | Boundaries | Selector crossings | Feature crossings | Maximum live selectors | Maximum live features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 16 | 38 | 37 | 281 | 178 | 9 | 6 |
| 100 | 32 | 19 | 18 | 139 | 86 | 9 | 6 |
| 100 | 64 | 10 | 9 | 71 | 40 | 9 | 6 |
| 100 | Unlimited | 1 | 0 | 0 | 0 | 0 | 0 |
| 300 | 16 | 113 | 112 | 773 | 585 | 9 | 6 |
| 300 | 32 | 57 | 56 | 395 | 290 | 9 | 6 |
| 300 | 64 | 29 | 28 | 195 | 145 | 9 | 6 |
| 300 | Unlimited | 1 | 0 | 0 | 0 | 0 | 0 |

All split versions use a nine-selector, six-feature slot layout, plus row and two sums. Slots are stable and their vectors do not shrink (`src/sort_policy/continuation.rs:20–35`), so vector lengths differ from live counts. For example, boundary 26 has five live selectors in nine slots; boundary 36 has zero live features in six slots. The emitter reserves argument positions from vector lengths, not live counts.

Every default 100-tree boundary follows below. Chunk 0 has no incoming selectors or features. The two sums and row are omitted from this table.

| Incoming chunk | Live selectors | Live features |
| --- | --- | --- |
| 1 | 5 | 6 |
| 2 | 5 | 6 |
| 3 | 7 | 5 |
| 4 | 7 | 5 |
| 5 | 7 | 6 |
| 6 | 8 | 5 |
| 7 | 7 | 6 |
| 8 | 8 | 5 |
| 9 | 8 | 5 |
| 10 | 9 | 5 |
| 11 | 8 | 5 |
| 12 | 7 | 5 |
| 13 | 8 | 5 |
| 14 | 8 | 6 |
| 15 | 7 | 5 |
| 16 | 8 | 5 |
| 17 | 7 | 6 |
| 18 | 8 | 6 |
| 19 | 9 | 5 |
| 20 | 8 | 5 |
| 21 | 8 | 5 |
| 22 | 8 | 5 |
| 23 | 8 | 5 |
| 24 | 8 | 6 |
| 25 | 8 | 5 |
| 26 | 5 | 5 |
| 27 | 8 | 5 |
| 28 | 8 | 5 |
| 29 | 8 | 5 |
| 30 | 9 | 5 |
| 31 | 8 | 6 |
| 32 | 9 | 4 |
| 33 | 9 | 4 |
| 34 | 8 | 3 |
| 35 | 8 | 3 |
| 36 | 8 | 0 |
| 37 | 4 | 0 |

The full table, including action ranges and slot lengths, is `probes/curve/100-gp12-fp9.boundaries.csv:1–39`. The corresponding 32/64/unlimited files are under `probes/chunks-<limit>/`. Dedicated probe tests check unchanged actions, contiguous ranges, exact incoming live sets, and the requested limits for GP4 and GP12 at both forest sizes.

For 100 trees with GP12/FP9:

| Limit | ARM instructions | ARM stack instructions | x86 instructions | x86 stack instructions |
| --- | --- | --- | --- | --- |
| 16 | 4453 | 0 | 4015 | 28 |
| 32 | 4552 | 118 | 4000 | 117 |
| 64 | 4712 | 369 | 3978 | 176 |
| Unlimited | 5026 | 944 | 4120 | 619 |

For split cases the stack column excludes the C wrapper. Unlimited emits one C `predict`; its counts include that function's saves and local traffic. There is no continuation body in that case. ARM allocates another 3,168 bytes after the save area (`probes/chunks-18446744073709551615/100-gp12-fp9-aarch64-apple-darwin.s:16`). This reproduces the cause recorded in `plans/implementation.md:179`; removing the boundaries is not a sound simplification. The four chunk experiments contain 48 successful target emissions, all assembled; their AArch64 kernels pass the 256-row checks.

The current boundary placement is sensible. `continuation.rs:60–63` includes trailing releases and retirements before cutting, so dead features and completed selectors are not passed onward. A retirement does not imply that every other tree has retired. In the default 100-tree plan, the active set becomes empty only after comparison 600; at 300 trees, only after comparison 1,800. Cutting only when no selector is live would therefore remove all internal boundaries. Cutting after any retirement still carries the other selectors.

GP4 is different: one active tree finishes every six comparisons. A boundary after every two trees would give 50 chunks at 100 trees, with no carried selectors but still carried features and sums. That is cleaner state, but more chunks than the current 38. It needs a separate emission comparison; this report does not assume that fewer selector arguments compensate for extra tails.

### B6. Planner cost

The planner is below the brief's roughly 100 ms threshold at both sizes. No planner-speed change is recommended. These measurements are indicative under machine load, not isolated benchmarks.

| Trees | Features | Samples | Minimum ms | Median ms | Maximum ms |
| --- | --- | --- | --- | --- | --- |
| 100 | 100 | 11 | 12.169 | 12.573 | 12.895 |
| 300 | 100 | 11 | 85.603 | 86.908 | 89.375 |

The model and 256 rows are created before timing. One plan call warms the code; eleven timed calls measure only `plan`, including plan allocation and partitioning. Dropping each returned plan occurs after its timer stops. LLVM is not called in this mode. Commands and all samples are in `probes/timing.csv` and the replay section below. The scans in `waiting_order`, `uses`, and `remaining_uses` are visible at `src/sort_policy.rs:307–325`, `:418–431`, and `:456–470`; these timings are not an asymptotic-complexity proof.

## 2. Hypotheses touched

H2 is partly confirmed. The default saves 206 loads at 100 trees, and the historical ARM timing gain is small. Its physical-register premise is wrong for carried FP values: both targets pass only eight scalar FP arguments in registers. More GP capacity also gives little reuse here. The exact best possible load count is 99 for this 100-tree model, so the total saving is 501, not 500. The grid's lowest count is 220, with stack traffic. GP12 is already tight enough on x86 to cause body stack accesses.

H1's observed threshold sequences are consistent with the supplied captures. Their relative cost is much larger than the default's feature-load saving on ARM. This study does not verify alternate code models, threshold storage, or their instruction deltas. Those remain piece A's work.

H3–H7 are outside this piece's experiments. Two sums, scalar comparisons, leaf-bit assignments, and missing-value behavior are unchanged.

## 3. Ranked recommendations

First, keep the scheduler with GP4/FP9, two sums, and 16-test chunks for both AArch64 and x86-64. Change the configured `RegisterBudget` in `SortOptions` or at its caller; no emitter change is needed. GP4 limits the existing scalar algorithm to one active tree, while retained feature values still influence which tree is admitted next. It removes most interleaved selector state. At 100 trees it even loads eight more features than GP12, yet emits fewer total instructions. This is why instruction count, code bytes, and stack traffic should decide the budget.

| Target | Trees | Instruction delta | Delta per tree | Delta per split | Text-byte delta | Body stack instructions |
| --- | --- | --- | --- | --- | --- | --- |
| AArch64 | 100 | -40 | -0.400 | -0.067 | -160 | 0 → 0 |
| x86-64 | 100 | -123 | -1.230 | -0.205 | -1149 | 28 → 0 |
| AArch64 | 300 | -186 | -0.620 | -0.103 | -744 | 0 → 0 |
| x86-64 | 300 | -559 | -1.863 | -0.311 | -4440 | 76 → 0 |

These deltas are direct measurements of separate 100- and 300-tree workloads, not a threefold extrapolation. They project the static change for those models only. Risks are loss of reuse on other feature distributions and different scheduling on another CPU or backend configuration. No latency improvement is established by these counts. AVX-512 production settings need their own emissions; the ABI probe alone is not a performance test. Preserve the current default if serial timing or a representative model rejects GP4.

Second, after that budget comparison, test 32-test chunks with GP4/FP9 on both targets. This changes the partition limit in `src/sort_policy.rs:172`; actions and planner load counts stay unchanged. It halves the number of tails approximately and allows LLVM to combine work over a larger region.

| Target | Trees | Additional instruction delta | Delta per tree | Delta per split | Additional text-byte delta | Body stack instructions |
| --- | --- | --- | --- | --- | --- | --- |
| AArch64 | 100 | -46 | -0.460 | -0.077 | -184 | 0 |
| x86-64 | 100 | -55 | -0.550 | -0.092 | -93 | 0 |
| AArch64 | 300 | -158 | -0.527 | -0.088 | -632 | 0 |
| x86-64 | 300 | -174 | -0.580 | -0.097 | -560 | 0 |

The combined GP4/FP9 and 32-test option removes 86/344 ARM instructions and 178/733 x86 instructions at 100/300 trees versus the current default. It remains spill-free in these cases. Rank it second because longer chunks already regress ARM at GP12 and at 64 tests with GP4. A global limit increase without the budget comparison is unsupported.

Do not increase FP budgets under the current scalar continuation signatures. Do not remove chunking. Do not rewrite the planner for compilation speed on these measurements.

## 4. Replay and proposed serial experiments

The initial required `cargo build --offline --release --examples` passed before any assembly was used. The chunk probe passed that build gate again. Its first full `cargo test --offline --release` passed source tests but failed the doctest link with undefined ZSTD symbols. An attempted rustdoc `-l zstd` flag was rejected as an unrecognized option. Passing the installed dylib as a linker argument made the full gate pass, including the doctest. Logs are `probes/chunk-tests.log`, `probes/chunk-tests-zstd.log`, and `probes/chunk-tests-linked-zstd.log`; statuses are in `probes/chunk-gates.txt`. No source change addresses the environment issue.

The only tracked edit is the temporary helper below. It exposes the same planner with a supplied limit and leaves normal `plan` at 16. This avoids changing tests that explicitly require the production limit. All alternative limits pass the dedicated state/range test and native row comparisons. The helper is reverted before delivery. Apply this patch only while replaying the probes:

```diff
diff --git a/src/sort_policy.rs b/src/sort_policy.rs
index fcc41e0..d893a64 100644
--- a/src/sort_policy.rs
+++ b/src/sort_policy.rs
@@ -141,6 +141,12 @@ impl Plan<'_> {
 }
 
 pub fn plan(model: &Trees, options: SortOptions) -> Result<Plan<'_>, InsufficientBudget> {
+    plan_with_study_limit(model, options, 16)
+}
+
+pub fn plan_with_study_limit(
+    model: &Trees, options: SortOptions, limit: usize,
+) -> Result<Plan<'_>, InsufficientBudget> {
     let sum_count = options.accumulators.get().min(model.trees().len().max(1));
     let mut plan = match options.order {
         Order::TreeOrder => tree_order(model, sum_count),
@@ -169,7 +175,7 @@ pub fn plan(model: &Trees, options: SortOptions) -> Result<Plan<'_>, Insufficien
     } else {
         Action::ReturnScaled(SumId(0))
     });
-    plan.chunks = continuation::partition(&plan, 16);
+    plan.chunks = continuation::partition(&plan, limit);
     Ok(plan)
 }
 
```

From the worktree root:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
export CARGO_TARGET_DIR="$PWD/target"
export RUSTDOCFLAGS='-C link-arg=/opt/homebrew/opt/zstd/lib/libzstd.dylib'
patch -p1 < probes/chunk-probe.patch
cargo build --offline --release --examples
cargo test --offline --release
cargo build --offline --release --manifest-path probes/Cargo.toml --bins
cargo test --offline --release --manifest-path probes/Cargo.toml --bin study-b
./target/release/study-b probes/curve curve > probes/curve.csv
./target/release/study-b probes/baseline baseline > probes/baseline.csv
./target/release/study-b probes timing > probes/timing.csv
python3 probes/emissions.py probes/curve curve
python3 probes/emissions.py probes/baseline baseline
python3 probes/chunks.py
python3 probes/reuse_analysis.py
python3 probes/asm_metrics.py probes/curve probes/baseline captures/100x100 > probes/assembly.csv
python3 probes/asm_metrics.py probes/chunks-* > probes/chunk-assembly.csv
```

The required grid's x86 GP20/GP28 failures are expected to reappear; the status CSV preserves each exit code. AArch64 should show the same eight-FP-argument limit, and GP12/FP12 should add stack transfers despite fewer loads. `probes/emissions.py` records failures rather than treating missing assembly as zero instructions.

For serial latency, use `probes/serial_bench.rs`, generated by `probes/make_serial_bench.py` from the existing benchmark. Only its support path, 300x100 argument, and planner-limit wrapper differ. It retains the existing dependent driver, trace checks, five alternating rounds, 10,000 warmup calls, 50,000 timed calls, and un-subtracted driver floor. These timing commands were not run in this study:

```sh
./target/release/serial-bench 100x100 > probes/serial-100-limit16.txt
./target/release/serial-bench 300x100 > probes/serial-300-limit16.txt
STUDY_B_CHUNK_LIMIT=32 ./target/release/serial-bench 100x100 > probes/serial-100-limit32.txt
STUDY_B_CHUNK_LIMIT=32 ./target/release/serial-bench 300x100 > probes/serial-300-limit32.txt
```

Compare scalar two-sum GP4/FP9, GP12/FP9, and tree order within each run. The expected direction from static evidence is a small GP4 gain, then another gain from 32 tests with GP4. Reject that projection if paired timing ranges or minima regress. Do not compare the 1/4-sum or SIMD variants as if only the budget changed. Repeat the planner-only timing without LLVM after the other agents finish; expected medians remain below roughly 100 ms.

The ABI probes replay with:

```sh
/opt/homebrew/opt/llvm@21/bin/llc -O1 -mtriple=aarch64-apple-darwin -mcpu=generic probes/abi/aarch64.ll -o probes/abi/aarch64.s
/opt/homebrew/opt/llvm@21/bin/llc -O1 -mtriple=x86_64-unknown-linux-gnu -mcpu=generic -mattr=+avx2 probes/abi/x86_64.ll -o probes/abi/x86_64.s
/opt/homebrew/opt/llvm@21/bin/llc -O1 -mtriple=x86_64-unknown-linux-gnu -mcpu=generic -mattr=+avx512f,+avx512vl probes/abi/x86_64-avx512.ll -o probes/abi/x86_64-avx512.s
```

Expect stack loads for the ninth/tenth double arguments and no saves in `clobber_working_registers`. All three commands passed here. After replay, restore only the tracked probe edit:

```sh
git checkout -- src/sort_policy.rs
git status --short
git diff
```

One artifact-placement error occurred during analysis: the first metrics script wrote derived objects/counts beside `captures/100x100`. Those generated files were removed immediately, and the script now writes capture derivatives under `probes/capture-metrics/`. The original 16 capture files remain unchanged. The final `git checkout -- src/sort_policy.rs` hit a sandbox permission error on the external worktree index. Reversing `probes/chunk-probe.patch` with `patch -R -p1` restored the file without changing Git metadata. Final `git diff` is empty, and the restored-tree release example build passes. No README, PLAN, plans/, or tests/ file was edited; no formatter or commit was run.

## 5. Questions that would change the recommendation

Which production CPU and operating system will run these kernels? This study's x86 target is Linux; Windows argument and preservation rules need a separate check. Host specialization or AVX-512 may change local allocation, though the tested scalar FP argument limit remains eight.

Can a representative model supply per-feature split counts, repeated predicates, and border counts? GP4's advantage on globally distinct thresholds does not establish its value when one predicate feeds many trees or feature overlap is much stronger.

How common are depths four and eight, mixed depths, and forests near 300 trees? The recommendation is measured at depth six; both selector construction and safe chunk length can change with depth.

Is a 1–5% serial latency gain worth retaining the current planner? Its measured planning cost is about 13 ms at 100 trees and 87 ms at 300 under load. That product tradeoff can justify a later simplification, but it is not a reason to assume that tree order has equal latency today.
