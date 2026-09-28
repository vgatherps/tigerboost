The harness refused to write the report file ("Subagents should return findings as text, not write report files"), so `/private/tmp/claude-501/-Users-samschetterer-src-tigerboost/2605435d-dccd-4bbe-8fe4-e15501b78509/scratchpad/REPORT_B_opus.md` does not exist. The full report is below. Please save it to that path if you need the file.

In `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-ac54dfbc22d4b0ea1`, `git status --short` shows only `?? probes/`, so there are no tracked-file changes. `git diff --stat` is empty. I deleted the temporary `examples/probe_b.rs`.

**Summary**
1. Keep the scheduler with the budget unchanged at GP12/FP9, on both targets. On AArch64 it saves 112 of 4565 instructions (2.5%). Timed under load it was about 5% faster at 100 trees and about 2% at 500.
2. Raising the budgets makes it worse. Anything live at a chunk boundary is passed as a `musttail` argument, and `preserve_nonecc` has only 8 FP argument registers on both targets and 12 GP on x86-64. FP9 with 2 sums fills the 8 FP registers exactly. Every FP above 9 turns row loads into stack loads and stores (FP12: +69 stack accesses on AArch64, +234 on x86). GP14 and above makes LLVM fail on x86-64 ("ran out of registers").
3. Residency is not worth more work. Even with every feature loaded exactly once (99–100 loads), the saving is 261 instructions (5.9%) on AArch64 and 285 (7.1%) on x86 at 100 trees. That is only reachable by passing features through the stack. The threshold-constant fix on AArch64 is worth about 1200 (27%).
4. The planner's heuristics are fine. It never evicts a feature and reloads it within a few tests. Eviction is within 10–14% of the optimal eviction order. The weak sharing (71% of loaded values feed one test) comes from the 6-feature limit, not from how trees are admitted.
5. On x86-64 alone, the static count favors tree order with chunk limit 64: 3800 vs 4015 instructions at 100 trees, 11301 vs 12033 at 300. This is unverified (x86 never runs here), and it should be re-checked after piece A lands, because the `movabsq` register pressure is part of why the scheduled plan loses on x86.

---

# Study B report: scheduler, feature residency, budgets, chunking

This covers the worktree at da93993. Every tracked-file edit has been reverted.

All numbers come from the `distinct` workloads: depth 6, every threshold unique, 100 declared features. "Instructions" means continuation bodies plus the `predict` wrapper, counted from separately emitted assembly with `probes/count_asm.py`.

The line anchors in the brief match the code: `src/sort_policy.rs:143-174, 225-238, 329-392, 394-454, 502-514, 515-527, 549-555, 556-559`, `continuation.rs:60-63` and `plans/implementation.md:175, 179`.

The probe files are untracked, under `probes/` in the worktree:
- `probe_b.rs`: copy it to `examples/probe_b.rs` to run it. Modes are `curve`, `reuse`, `belady`, `emit` and `time`.
- `count_asm.py`, `emit_all.sh`, `limit_run.sh`, `gp_sweep.sh`, `bench_run.sh`.
- Calling-convention IR probes `cc_args.ll`, `clobber_a64.ll`, `clobber_x86.ll`, `clobber_x86z.ll`, plus their `llc` output.

The emitted assembly is in the scratchpad under `b_asm/`, `b_asm_gp/`, `b_asm_L16/`, `b_asm_L32/`, `b_asm_L64/`, `b_asm_L128/`, `b_asm_Lmax/` and `b_asm_Lmax_inv/`.

## 1. Answers per question

### B1. Physical registers under `preserve_nonecc` (LLVM 21.1.8)

Neither LLVM 21 `.td` sources nor the LangRef are installed. The only `AArch64CallingConvention.td` / `X86CallingConv.td` on disk are old capstone copies from before `preserve_none` existed.

So I answered empirically with the installed `llc` 21.1.8:
- `probes/cc_args.ll` calls a `preserve_nonecc` function with a pointer, 30 `i64` arguments and 24 `double` arguments, to see which registers carry them.
- `probes/clobber_*.ll` defines a `preserve_nonecc` function whose inline asm clobbers every register, next to a C-convention copy, to see what each prologue saves.

| | AArch64 (`-mcpu=generic`) | x86-64 (`+avx2` / `+avx512f`) |
|---|---|---|
| GP argument registers, in order | x20, x21-x28, x0-x7, x10-x14, x9, x15 (24) | r12, r13, r14, r15, rdi, rsi, rdx, rcx, r8, r9, r11, rax (12) |
| FP argument registers | d0-d7 (8); the 9th double goes on the stack | xmm0-xmm7 (8); the 9th double goes on the stack |
| What a body that clobbers everything saves | only x29/x30 | only rbp |
| GP a body can use without saving | about 28 (x0-x17, x19-x28; x18 is reserved on Darwin, x30 holds the return into `predict`) | 14 (all but rsp; rbp costs a `pushq`/`popq`) |
| FP/vector a body can use | 32 (v0-v31; unlike AAPCS, d8-d15 are not preserved) | 16 xmm (32 zmm with AVX-512) |

Because of this, the C wrapper `predict` saves x19-x28 and d8-d15 on AArch64 (its 160-byte frame), and rbx and r12-r15 on x86.

Everything live at a chunk boundary is a `musttail` argument: the row pointer, `index_slots` selectors, the sums and `feature_slots` features (`src/jit.rs:106-126`). So the binding limit is the argument registers, not the register file.

**FP.** Sums plus crossing features must be at most 8 on both targets.
- The planner's rule is `sum_count + resident + !loaded + comparisons <= floating` (`src/sort_policy.rs:502`). With FP9 and 2 sums, at most 6 features are resident after a load.
- So at most 6 features cross a boundary, and 2 + 6 = 8. FP9 sits exactly on the FP argument limit on both targets.
- Every FP budget above 9 puts features into stack arguments at every boundary.

**GP.** The GP arguments are the row pointer plus the active selectors.
- GP12 allows 9 active trees (`1 + active + scratch <= general`, `src/sort_policy.rs:337`), so 10 GP arguments.
- AArch64 has room for up to 24.
- On x86-64, GP13 still compiles, with 88 stack accesses in the bodies.
- GP14 and above fails to compile: `ran out of registers during register allocation in function 'continuation20'` (`probes/gp_sweep.sh`). GP20 and GP28 fail the same way. This is a hard compile error, not a spill, and the default is one step below it.

The largest spill-free budgets I measured at 100x100:
- AArch64: GP up to 16 at FP9. GP17 shows 4 stack accesses.
- x86-64: only GP12 at FP9, and even that has 28 stack accesses in the bodies. 26 of them are `pushq/popq %rbp` pairs to get a 15th GP register in 13 chunks. The other 2 are a real spill, `movq %rax, -8(%rsp)`.
- No FP budget above 9 is spill-free on either target.

### B2. Residency curve

Settings are scalar, 2 sums, chunk limit 16. Column meanings:
- "After in-chunk CSE" is what `early-cse` leaves: the number of distinct features loaded in each chunk, summed over all chunks.
- "Reloads" are loads of a feature index that was loaded before.

**100 trees / 100 features.** 99 features are used, so the minimum is 99 loads.

| order | GP | FP | loads | after in-chunk CSE | reloads | chunks | peak resident | peak active | feature slots | index slots | mean features crossing | mean selectors crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tree | - | - | 600 | 562 | 501 | 38 | 1 | 1 | 1 | 1 | 0.00 | 0.68 |
| sched | 12 | 9 | 394 | 384 | 295 | 38 | 6 | 9 | 6 | 9 | 4.81 | 7.59 |
| sched | 12 | 12 | 347 | 340 | 248 | 38 | 9 | 9 | 9 | 9 | 7.84 | 7.59 |
| sched | 12 | 16 | 320 | 320 | 221 | 38 | 13 | 9 | 13 | 9 | 11.30 | 7.51 |
| sched | 12 | 24 | 268 | 268 | 169 | 38 | 21 | 9 | 21 | 9 | 18.19 | 7.89 |
| sched | 12 | 32 | 224 | 224 | 125 | 38 | 29 | 9 | 29 | 9 | 24.97 | 7.81 |
| sched | 20 | 9 | 393 | 384 | 294 | 38 | 6 | 17 | 6 | 17 | 4.78 | 10.05 |
| sched | 20 | 12 | 365 | 360 | 266 | 38 | 9 | 16 | 9 | 16 | 7.46 | 11.73 |
| sched | 20 | 16 | 322 | 314 | 223 | 38 | 13 | 17 | 13 | 17 | 11.16 | 10.81 |
| sched | 20 | 24 | 264 | 261 | 165 | 38 | 21 | 17 | 21 | 17 | 18.08 | 11.78 |
| sched | 20 | 32 | 224 | 224 | 125 | 38 | 29 | 17 | 29 | 17 | 24.51 | 12.27 |
| sched | 28 | 9 | 393 | 384 | 294 | 38 | 6 | 20 | 6 | 20 | 4.70 | 10.32 |
| sched | 28 | 12 | 365 | 360 | 266 | 38 | 9 | 16 | 9 | 16 | 7.46 | 11.73 |
| sched | 28 | 16 | 322 | 314 | 223 | 38 | 13 | 17 | 13 | 17 | 11.08 | 10.89 |
| sched | 28 | 24 | 265 | 262 | 166 | 38 | 21 | 20 | 21 | 20 | 17.78 | 11.97 |
| sched | 28 | 32 | 220 | 220 | 121 | 38 | 29 | 19 | 29 | 19 | 24.78 | 13.27 |
| sched | 12 | 64 | 123 | 123 | 24 | 38 | 61 | 9 | 61 | 9 | 46.22 | 7.73 |
| sched | 12 | 128 | 99 | 99 | 0 | 38 | 76 | 9 | 76 | 9 | 53.22 | 7.92 |

**300 trees / 100 features.** The minimum is 100 loads.

| order | GP | FP | loads | after in-chunk CSE | chunks | peak resident | feature slots |
|---|---:|---:|---:|---:|---:|---:|---:|
| tree | - | - | 1800 | 1671 | 113 | 1 | 1 |
| sched | 12 | 9 | 1192 | 1137 | 113 | 6 | 6 |
| sched | 12 | 12 | 1109 | 1068 | 113 | 9 | 9 |
| sched | 12 | 16 | 1007 | 967 | 113 | 13 | 13 |
| sched | 12 | 24 | 829 | 806 | 113 | 21 | 21 |
| sched | 12 | 32 | 653 | 646 | 113 | 29 | 29 |
| sched | 20 | 9 | 1213 | 1161 | 113 | 6 | 6 |
| sched | 28 | 9 | 1213 | 1161 | 113 | 6 | 6 |
| sched | 12 | 64 | 237 | 237 | 113 | 61 | 61 |
| sched | 12 | 128 | 100 | 100 | 113 | 93 | 93 |

Loads depend almost entirely on the FP budget:
- Raising GP from 12 to 28 saves 1 load at 100 trees, and adds 21 loads at 300 trees.
- Loads only reach the minimum when FP allows 76 resident features (100 trees) or 93 (300 trees). All of those then cross every chunk boundary.

**Emitted assembly at 100x100, chunk limit 16.**
- Stack accesses are counted in continuation bodies only. For x86, the number in parentheses is how many are `pushq/popq`.
- Row refs are body instructions that address the row register: x20 on AArch64, r12 on x86-64.

| budget | AArch64 insns | AArch64 stack | AArch64 row refs | x86-64 insns | x86-64 stack | x86-64 row refs |
|---|---:|---:|---:|---:|---:|---:|
| tree order, 2 sums | 4565 | 0 | 519 | 3980 | 0 | 562 |
| GP12/FP9 (default) | 4453 | 0 | 360 | 4015 | 28 (26) | 384 |
| GP13/FP9 | 4443 | 0 | 353 | 4091 | 88 (50) | 381 |
| GP12/FP12 | 4496 | 69 | 320 | 4197 | 234 (32) | 340 |
| GP12/FP16 | 4536 | 139 | 302 | 4371 | 448 (28) | 320 |
| GP12/FP32 | 4610 | 295 | 216 | 5621 | 1632 (24) | 224 |
| GP12/FP128 | 4711 | 502 | 98 | 8038 | 4168 (32) | 99 |
| GP20/FP9 | 4458 | 8 | 361 | compile error | - | - |
| GP28/FP9 | 4484 | 15 | 357 | compile error | - | - |

The default budget reproduces the pre-generated captures exactly (4453 and 4015 instructions).

The extra stack accesses above FP9 are features passed on the stack. At FP12 on AArch64, each chunk reads its three extra features with `ldr d, [sp, #n]` and writes them back with `str d, [sp, #n]` before the `musttail` branch. A reload from the row is also one `ldr d, [x20, #n]`. So these budgets swap row loads for stack loads plus stack stores.

Every budget above FP9 emits more instructions than FP9 on both targets. No budget gets near 100 loads without spilling. FP128 reaches 99 loads, but costs 502 stack accesses on AArch64, and 4168 on x86-64, where the kernel doubles to 8038 instructions.

### B3. What residency is worth

This table is the theoretical saving if every feature were loaded once and held for free. It is counted on emitted row loads, which already include `early-cse` merging and, on AArch64, `ldp` pairing.

| target, trees | kernel insns (default) | row refs (default) | minimum | max saving vs default | kernel insns (tree order) | row refs (tree order) | max saving vs tree order |
|---|---:|---:|---:|---:|---:|---:|---:|
| AArch64, 100 | 4453 | 360 | 99 | 261 (5.9%) | 4565 | 519 | 420 (9.2%) |
| AArch64, 300 | 13338 | 1074 | 100 | 974 (7.3%) | 13658 | 1543 | 1443 (10.6%) |
| x86-64, 100 | 4015 | 384 | 99 | 285 (7.1%) | 3980 | 562 | 463 (11.6%) |
| x86-64, 300 | 12033 | 1137 | 100 | 1037 (8.6%) | 11889 | 1671 | 1571 (13.2%) |

This ceiling cannot be reached, because anything beyond 8 FP values crossing a boundary goes through the stack (B1, B2).

For comparison, take the threshold constants that piece A is working on. On AArch64 each split's constant costs `mov` + `movk` + `fmov`: about 1800 of 4453 instructions at 100 trees, and 5400 at 300. Cutting that to one load per split removes about 1200 instructions (27%) at 100 trees and 3600 at 300. The whole theoretical residency ceiling is about a fifth of that on AArch64 at 100 trees, and the part reachable in registers is already used by FP9.

**What the scheduler buys today, static count, chunk limit 16:**

| target, trees | tree order insns | scheduled insns | difference |
|---|---:|---:|---:|
| AArch64, 100 | 4565 | 4453 | -112 (-2.5%) |
| AArch64, 300 | 13658 | 13338 | -320 (-2.3%) |
| x86-64, 100 | 3980 | 4015 | +35 (+0.9%) |
| x86-64, 300 | 11889 | 12033 | +144 (+1.2%) |

At the same chunk limit, the scheduled plan emits more instructions than tree order on x86-64. The reason is GP pressure:
- The scheduled plan carries up to 9 selectors plus the row pointer across every boundary.
- Each comparison in flight needs a `movabsq` address register.
- Together these exceed the 14 usable GP registers, which produces the `pushq %rbp` pairs and spills above.
- Tree order carries on average 0.68 selectors and no features per boundary.

**Indicative AArch64 timing**, taken while other agents were loading the machine. This is `scalar_bench`, median ns per call including the driver, three runs each. Both columns use 2 sums; the scheduled one is GP12/FP9.

| workload, chunk limit | tree order medians | scheduled medians |
|---|---|---|
| 100x100, 16 | 125.81 / 125.20 / 125.01 | 118.78 / 119.47 / 120.00 |
| 500x100, 16 | 584.07 / 589.06 / 585.80 | 572.31 / 580.17 / 567.47 |
| 100x100, 32 | 124.91 / 126.90 / 128.90 | 126.51 / 126.95 / 129.38 |
| 500x100, 32 | 589.24 / 581.35 / 593.18 | 597.98 / 589.24 / 606.02 |

At limit 16 this agrees with `plans/implementation.md:175` (2.8–4.4%). Here it is about 5% at 100 trees and 2% at 500. At limit 32 the scheduler's advantage disappears.

In short:
- On AArch64 with the current backend, scheduling is a real but small win: 2–5% timed, 2.5% by instruction count.
- Residency beyond the current FP9 is worth nothing on either target, because the extra values go through the stack.
- On x86-64, by static count, the scheduler loses to tree order at the same limit. Tree order with a longer chunk does better still (B5).
- Tree order plus a cheap backend would be within 2.5–5% on AArch64, and ahead on x86-64.
- None of this justifies more planner complexity. The threshold-constant work is several times larger on both targets.

### B4. Heuristic quality

By the planner's own arithmetic at FP9, 2 sums, scalar, a new load needs `2 + resident + 1 + 1 <= 9`. So at most 5 features can be resident before the load, and 6 after it. The observed peak is 6 at every GP value.

Reloads are 295 of 394 loads at 100 trees, and 1092 of 1192 at 300. Every reload follows an eviction by the loop at `src/sort_policy.rs:502-514`. The release at 549-555 only drops features with `remaining_uses == 0`, and those are never loaded again.

This table shows the gap between an eviction and the matching reload, in comparison tests:

| trees | reloads | 0-4 tests | 5-16 | 17-64 | 65+ |
|---:|---:|---:|---:|---:|---:|
| 100 | 295 | 6 | 31 | 87 | 171 |
| 300 | 1092 | 26 | 146 | 325 | 595 |

This is not thrashing: most evicted features are not needed again for more than 64 tests.

Next, the planner's eviction choice against the optimum: evict the feature whose next use is furthest away, on the same sequence of compared features (`probe_b belady`). Each cell is loads at 100 trees / 300 trees.

| resident capacity | tree order, optimal eviction | scheduled order, optimal eviction |
|---:|---:|---:|
| 4 | 493 / 1462 | 398 / 1180 |
| 6 | 454 / 1340 | 357 / 1044 |
| 8 | 423 / 1242 | 330 / 944 |
| 16 | 338 / 964 | 259 / 691 |
| 32 | 234 / 643 | 181 / 437 |

- At capacity 6 the planner issues 394 loads (100 trees) and 1192 (300 trees). Optimal eviction on the planner's own order gives 357 and 1044. So eviction by `remaining_uses` is within 10% of optimal at 100 trees and 14% at 300.
- The scheduled order itself is worth about 100 loads at 100 trees: with optimal eviction at capacity 6, tree order needs 454 and the scheduled order 357.

To measure how well `waiting_order` clusters trees, I counted the distinct trees that consume each loaded feature value before it is released:

| trees | loaded values | mean trees per value | 1 tree | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 394 | 1.50 | 281 | 68 | 22 | 12 | 7 | 2 | 2 | 0 |
| 300 | 1192 | 1.49 | 908 | 160 | 57 | 22 | 12 | 16 | 9 | 8 |

- Each loaded value feeds 1.52 tests on average. The model itself offers about 6.1 uses per feature at 100 trees (600 splits over 99 features) and 18 at 300.
- About 71–76% of loaded values feed exactly one test in one tree.
- The clustering is weak. But raising GP to 20–28 (up to 20 active trees) does not reduce loads, so admitting more trees would find no more sharing within FP9. The limit is the 6 resident features, not the admission order.

### B5. Chunking

State crossing each boundary, as a mean over the `chunks - 1` boundaries with the maximum in parentheses:

| limit | trees | chunks | scheduled selectors | scheduled features | tree-order selectors | tree-order features |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 100 | 38 | 7.59 (9) | 4.81 (6) | 0.68 (1) | 0 |
| 32 | 100 | 19 | 7.72 (9) | 4.78 (6) | 0.67 (1) | 0 |
| 64 | 100 | 10 | 7.89 (9) | 4.44 (6) | 0.67 (1) | 0 |
| 128 | 100 | 5 | 8.25 (9) | 5.00 (6) | 0.75 (1) | 0 |
| 16 | 300 | 113 | 6.90 (9) | 5.22 (6) | 0.67 (1) | 0 |
| 32 | 300 | 57 | 7.05 (9) | 5.18 (6) | 0.68 (1) | 0 |
| 64 | 300 | 29 | 6.96 (9) | 5.18 (6) | 0.68 (1) | 0 |

Longer chunks do not shrink what crosses a boundary: the scheduled plan always carries its full working set. Only the number of boundaries changes.

Emitted instructions by chunk limit, with body stack accesses in parentheses. For "none", everything is in one function, `predict`, and the stack count includes its 20 (AArch64) or 12 (x86-64) save/restore instructions.

| limit | AArch64 tree, 100 | AArch64 sched, 100 | x86 tree, 100 | x86 sched, 100 | AArch64 tree, 300 | AArch64 sched, 300 | x86 tree, 300 | x86 sched, 300 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4565 (0) | 4453 (0) | 3980 (0) | 4015 (28) | 13658 (0) | 13338 (0) | 11889 (0) | 12033 (76) |
| 32 | 4483 (8) | 4552 (118) | 3890 (0) | 4000 (117) | 13402 (12) | 13556 (349) | 11596 (0) | 11892 (290) |
| 64 | 4730 (342) | 4712 (369) | 3800 (4) | 3978 (176) | 14223 (1076) | 14136 (1201) | 11301 (0) | 11754 (473) |
| 128 | 4919 (649) | 4762 (511) | 3899 (208) | 4004 (286) | 14721 (1926) | 14213 (1563) | 11667 (670) | 12015 (987) |
| none | 5500 (1387) | 5026 (944) | 3976 (547) | 4120 (619) | 16529 (4271) | 14806 (2652) | 11766 (1639) | 12123 (1814) |

**What limits chunk length on each target:**
- On AArch64, the spills that appear at limits 32 and 64 are all GP registers (`str x`/`ldr x`/`stp x`/`ldp x`). For example, tree order at limit 64 has 167 `str x`.
- The start of `continuation1` at limit 64 shows why: the machine scheduler hoists many `fcmp`/`cset` pairs ahead of the `orr` that merges each result into its selector, and the `cset` results get spilled.
- So on AArch64 the chunk limit is set by GP pressure from comparison bits, not by feature residency.
- On x86-64, tree order stays spill-free up to limit 64.

**Chunk boundaries also block CSE.** In the unsplit kernel, `early-cse` merges every load of a feature into one value, so only 99 or 100 row loads survive, each kept live across the whole kernel. The chunk boundaries stop this from happening across the whole kernel.
- I tried marking feature loads `!invariant.load` (diff in section 4), to see whether the register allocator would reload from the row instead of spilling.
- The unsplit AArch64 output was byte-identical with and without the marking, and the x86-64 instruction and stack-access counts were unchanged. So this is not a cheap fix.

**Where the boundaries go.** `continuation.rs:60-63` extends a chunk over the `Release`/`Retire` actions that follow its last comparison, so released features and retired selectors never cross. That is correct and cheap.

Cutting only at points where no selector is live works for tree order (every 6 tests) but not for the scheduled plan. At GP12/FP9 on 100x100, the number of active trees reaches zero only once, after the 600th test. That rule would produce one chunk: the unsplit kernel, with 944 (AArch64) / 619 (x86-64) stack accesses.

**Limits the plans suggest:**
- Scheduled plan on AArch64: 16. At 32 it already spills, and the timing advantage is gone.
- Scheduled plan on x86-64: no limit is spill-free.
- Tree order: 32 on AArch64, 64 on x86-64.

### B6. Planner cost

`plan` alone at the default budget, measured under load (`probe_b time`), in ms:

| trees / features | tree order median | scheduled GP12/FP9 min | scheduled median |
|---|---:|---:|---:|
| 100 / 100 | 0.014 | 12.1 | 12.6 |
| 300 / 100 | 0.037 | 84.5 | 85.3 |
| 500 / 100 | 0.070 | 220.8 | 221.3 |
| 500 / 1000 | 0.062 | 331.2 | 336.5 |

- Going from 100 to 300 trees costs 6.8x, and from 300 to 500 trees 2.6x: roughly the square of the tree count.
- 300 trees is just under the 100 ms line; somewhere above about 300 trees it crosses.
- `scalar_bench` compile times agree: at 500x100 the scheduled compile takes about 440–450 ms against 161–167 ms for tree order, most of the difference being planning.

## 2. Hypotheses

**H2 (residency is a small lever): confirmed, with corrections.**
- The 206-load gap in the plans (394 vs 600) shrinks in emitted code: 178 row loads on x86-64 and 159 on AArch64. `early-cse` already merges 38 repeated loads in tree order (600 down to 562), and AArch64 pairs neighboring loads with `ldp`.
- The net instruction saving on AArch64 is 112 (2.5%), not about 5%. On x86-64 the scheduled plan is 35 instructions larger than tree order.
- "The logical budgets are far below what `preserve_nonecc` makes available" is true of the register file inside a body: 32 FP and about 28 GP on AArch64. It is wrong about what matters here, because everything live at a boundary is a `musttail` argument. There are only 8 FP argument registers on both targets and 12 GP on x86-64. FP9 with 2 sums fills the FP arguments exactly, and GP12 is one step below an x86-64 register-allocation failure.
- "Unlimited residency caps the saving at 500" is too high. The ceiling on emitted code is 261 instructions (AArch64) or 285 (x86-64) against the default, and 420 / 463 against tree order at 100 trees. It is only reachable by moving features to the stack.

**H1 (threshold constants dominate): not tested here, but it affects this piece.** x86-64 `movabsq` needs one GP register per comparison in flight. That GP pressure is what makes the scheduled plan lose to tree order on x86 (the `pushq %rbp` pairs, the `movq %rax, -8(%rsp)` spills, and the compile failure at GP14). If piece A moves thresholds to RIP-relative or base-register operands, the x86 comparisons in B3 and B5 need redoing.

## 3. Ranked recommendations

**1. Keep the scheduler, with the default budget at GP12/FP9 on both targets. Do not raise FP or GP.**
- Verified:
  - Every larger budget emits more instructions on both targets (B2).
  - FP above 9 turns row loads into stack loads and stores.
  - GP14 and above fails to compile for x86-64.
  - GP above 12 on AArch64 changes the count by at most 15 instructions, and never below the default by more than 10.
- Instruction change: 0 at 100 and 300 trees. Risk: none.
- A related gap: `plan` accepts budgets that are unusable on some targets. GP14 or more makes LLVM abort on x86-64, and FP above 9 compiles but only adds stack traffic.
- The comment on `RegisterBudget` ("Logical working registers, not a guarantee about LLVM's physical allocation") is true inside a body, but it hides that the limit at chunk boundaries is physical.
- A check would belong in `plan` (`src/sort_policy.rs:143-157`) or in `emit_plan`, and would need the target, which the planner does not see today. That changes the interface, so I left it as question 2 in section 5.

**2. Stop adding planner complexity for residency.**
- On 100x100, the residency gain still reachable in registers is zero.
- The theoretical gain is 5.9% (AArch64) and 7.1% (x86-64) at 100 trees, and 7.3% / 8.6% at 300. That is about a fifth of what the threshold-constant change should give on AArch64.
- The one planner improvement I found, optimal eviction, would cut 394 loads to 357 at 100 trees: 37 fewer row loads, under 1% of the kernel. Not worth the code.

**3. On x86-64 only, after piece A's threshold/code-model change lands: evaluate tree order with chunk limit 64 in place of the scheduled plan at limit 16.**
- Static count today at 100 trees: 3800 vs 4015 instructions (-215, -5.4%), with 4 stack accesses.
- At 300 trees: 11301 vs 12033 (-732, -6.1%), with 0 stack accesses.
- This needs the chunk limit chosen per target or per order. Today it is a literal at `src/sort_policy.rs:172`.
- Unverified: x86 code never runs here, and part of the scheduled plan's x86 penalty comes from the `movabsq` GP register that piece A may remove.
- On AArch64, keep the scheduled plan at limit 16. Tree order at limit 32 is 30 instructions larger at 100 trees and 64 at 300. Its timing matched the scheduled plan at 32, and both were slower than the scheduled plan at 16.

**Answer for the deliverable: keep the scheduler, with the budget unchanged** (GP12/FP9, 2 sums, limit 16).
- On AArch64 it is worth 2.5% by instruction count and 2–5% timed.
- On x86-64, the candidate replacement is tree order at limit 64 (recommendation 3).
- The projected change for the first-ranked option is 0 instructions at 100 and 300 trees on both targets. The finding is that there is nothing to gain above the current budget.

## 4. Experiments for the orchestrator

Setup, in a worktree:
- `cp probes/probe_b.rs examples/probe_b.rs`
- `export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21`
- Delete `examples/probe_b.rs` afterwards.
- The scripts use absolute paths into this agent's worktree (`agent-ac54dfbc22d4b0ea1`); adjust `W`/`BIN` if running elsewhere.

**E1. AArch64 timing, tree order vs scheduled by chunk limit.** Run serially on a quiet machine.
1. For each limit L in 16, 32 and 64, set `plan.chunks = continuation::partition(&plan, L);` at `src/sort_policy.rs:172`.
2. Run `TAG=L$L WORKLOAD=100x100 bash probes/bench_run.sh`, then the same with `WORKLOAD=500x100`.
3. Compare the `Tree order, sums 2` rows with the `general: 12, floating: 9 }, sums 2` rows.

Expected: the scheduled plan at 16 is fastest (about 119 ns at 100x100), and tree order at 32 is within 1–2% of tree order at 16. Only 16 and 32 were timed; limit 64 was not. The static counts at 64 (4730 / 4712 instructions with 342 / 369 stack accesses, against 4453 for the scheduled plan at 16) suggest both orders are slower there.

**E2. After piece A lands, redo the x86 static comparison.**
1. Run `OUT=<fresh dir> TREES=100 FEATURES=100 bash probes/emit_all.sh`.
2. Run `TAG=<tag> bash probes/limit_run.sh` at limits 16 and 64.
3. Count with `python3 probes/count_asm.py <dir>/*.s`.

Expected: the scheduled plan's push/pop and spill counts on x86 drop. Whether tree order at 64 still beats the scheduled plan at 16 decides recommendation 3.

**E3. When an x86-64 host exists.** Time tree order at limit 64 against scheduled GP12/FP9 at limit 16, with `scalar_bench 100x100` and `500x100`. Expected from the static counts: tree order about 5–6% faster.

**Throwaway probe diff (all reverted).**
- The chunk-limit edit was applied with 32, 64, 128 and `usize::MAX` in turn.
- The `!invariant.load` edit was applied only for the `b_asm_Lmax_inv` run, together with `usize::MAX`.

```
--- a/src/sort_policy.rs
+++ b/src/sort_policy.rs
@@ -169,7 +169,7 @@ pub fn plan(model: &Trees, options: SortOptions) -> Result<Plan<'_>, Insufficien
     } else {
         Action::ReturnScaled(SumId(0))
     });
-    plan.chunks = continuation::partition(&plan, 16);
+    plan.chunks = continuation::partition(&plan, usize::MAX);
     Ok(plan)
 }
--- a/src/jit.rs
+++ b/src/jit.rs
@@ -239,6 +239,10 @@
                     let loaded = builder.build_load(f64_type, address, "feature")?;
                     loaded.as_instruction_value().unwrap().set_alignment(8)?;
+                    loaded.as_instruction_value().unwrap().set_metadata(
+                        context.metadata_node(&[]),
+                        context.get_kind_id("invariant.load"),
+                    )?;
                     let mut value = loaded.into_float_value();
```

## 5. Questions for the user

1. **Deployment target.** The AArch64 and x86-64 answers differ: on AArch64 the scheduler is a small win, and on x86-64 it is a small loss by static count. Which target sets the default?
2. **Should `RegisterBudget` stay a user-facing option?** It accepts values that make LLVM fail on x86-64 (GP 14 or more) and values that only add stack traffic (FP above 9). The alternatives are to fix it per target from the calling convention, or to pass the target triple into `plan` so it can check. Either one changes the public interface.
3. **Real model statistics.** Real models reuse features and thresholds much more than the `distinct` workload, which has about 6 uses per feature at 100 trees and every threshold unique. Residency and sharing would be worth more on a model where a few features take most of the splits. Is a histogram of splits per feature available for at least one representative model?
4. **Planning cost.** Planning takes about 85 ms at 300 trees and 220 ms at 500, growing roughly with the square of the tree count. Is that acceptable?