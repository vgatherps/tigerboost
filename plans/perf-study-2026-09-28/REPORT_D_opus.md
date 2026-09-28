# Study D report: latency chain, front-end and cache bounds, benchmark fidelity

Worktree `agent-a6b75ac99417fc84c` at `da93993`. All brief anchors checked: `src/jit.rs` 22-56, 356-415, 421-442 and `src/sort_policy.rs` 143-174, 478-484 match. `plans/implementation.md` accumulator sweep is at lines 71-82 (brief said 72-82), current tables at 165-192.

The short version: on the M4 the current 100-tree kernel is bound by rename width (10 per cycle) and runs at about 90% of it. The sum chain is hidden by a wide margin, and it stays hidden for any formulation that keeps one flag-setting compare per split. The reason is a port limit, not instruction count: the M4 issues about 2 flag-producing compares per cycle, and the x86 cores I could cite issue 1 `ucomisd` per cycle. That gives a floor of 300 cycles (M4) or 600 cycles (Zen 4, Golden Cove) per 100 trees no matter how many other instructions are removed. The benchmark measures a hot case. When other work evicts the caches between calls, one call costs 0.7 to 1.4 us at 100 trees and 2 to 3 us at 300 trees, 6 to 10 times the hot number. Leaf-table misses dominate that cost.

Every timing below was taken while other agents loaded the machine (load average 3 to 5 on 14 cores). Microbenchmark results are ratios against a dependent `add` chain in the same run, so frequency drift cancels out. Treat the kernel timings as indicative.

## 1. Answers per question

### Evidence base: M4 port and latency measurements

There is no public M4 instruction table, so I measured what the bounds need. The sources are `probes/ubench/ub.c`, `ub2.c`, `ub3.c` and `ub4.c`. `ub5.c` and `ub6.c` are generated from `ub.c` by the inline python recorded in section 4. Build with `clang -O2`. Cycles are calibrated against a dependent `add x10,x10,#1` chain, which reported 4.17 to 4.48 GHz. Each figure is the best of 9 runs of 2M iterations, and every run was repeated at least twice.

| Sequence (independent unless stated) | cycles per op | Reading |
|---|---:|---|
| `fcmp d,d` | 0.49-0.50 | 2 flag-setting FP compares per cycle |
| `fcmp` + `cset` pair | 0.50-0.51 | the pair is limited by `fcmp` |
| `cmp x,x` alone / `cset` alone | 0.25 / 0.26 | 4 per cycle each |
| `cmp` + `cset` pair (integer) | 0.51-0.52 | same 2 per cycle: they share units |
| `fmov d,x` (GPR to FP) | 0.33-0.34 | 3 per cycle |
| `ldr d` | 0.33-0.34 | 3 per cycle |
| `fmov d,x` alternating with `ldr d` | 0.32-0.33 per instr | 3 per cycle combined: the same pipes |
| emitted split `mov`,`movk`,`fmov`,`fcmp`,`cset` | 0.52-0.54 per split | compare-bound |
| split with threshold from memory `ldr d`,`fcmp`,`cset` | 0.52 per split | compare-bound |
| split as integer `mov`,`movk`,`cmp`,`cset` | 0.51-0.52 per split | compare-bound |
| `fcmgt d` scalar / `fcmgt.2d` | 0.24 / 0.24-0.25 | 4 per cycle, no flags |
| `orr x, x, x, lsl #k` | 0.25; latency 2.0 | |
| `mov`+`movk` wide-immediate pair | 0.22-0.23 per pair | about 9 instr per cycle, `mov #imm` handled at rename |
| 8 `add` + 4 `fadd` + 4 `ldr d` mix | 0.106 per instr | about 9.4 IPC sustained |
| taken `b` to next instruction | 0.50-0.51 | |
| `fadd d0,d0,d1` dependent | 2.09-2.2 | fadd-to-fadd latency |
| `fmul` dependent | 3.07-3.23 | |
| random pointer chase in 32 KB | 3.0 | L1 load-to-use |
| `ldr d` > `fcmp` > `cset` > `add` > `ldr x`, loop-carried, unpredictable | 16.1-16.3 | compare path about 12 plus int load 3 |
| same plus 3 dependent `orr`-shift steps | 21.4 | about 1.8 per `orr` |
| same with an `fadd` between load and `fcmp` | 19.2-19.4 | load-to-fadd-to-fcmp adds about 3 |

Two traps showed up in the first attempts. Loads with constant or short-period values run far faster than their latency, at 0.35 to 1.4 cycles (see `ub.c` `ldr_dep` and `ub2.c` `cmp_idx_ld`). M3 and M4 cores predict load values and addresses. `ub3.c` defeats this with a 1024-node random cycle, and only its numbers are used.

Cited figures. The M4 P-core does 10-way decode and 10-way rename and has a 361-entry coalesced ROB ([jia.je M4 P-core diagram](https://jia.je/cpu/m4_pcore.svg), built from Geekerwan measurements and Apple's CPU optimization guide). On Firestorm (M1), `fcmp` is limited to 1 per cycle on unit 14, fadd latency is 3, and `mov`+`movk` is not fused ([Dougall Johnson, Firestorm SIMD table](https://dougallj.github.io/applecpu/firestorm-simd.html), [overview](https://dougallj.github.io/applecpu/firestorm.html)). The M4 measures twice M1's compare rate and a 2-cycle fadd. Host caches from `sysctl hw.perflevel0`: L1D 128 KiB, L1I 192 KiB, L2 16 MiB shared by the P cluster, 128-byte lines.

x86 figures come from [uops.info](https://uops.info). `UCOMISD xmm, m64` has a reciprocal throughput of 1.00 cycle on both Zen 4 (2 uops FP23+FP45) and Alder Lake-P (p0 plus a load port), and 0.50 on Zen 5. Its register-form latency is 7 on Zen 4 and 3 on ADL-P. `ADDSD` latency is 3 on Zen 4 and 2 on ADL-P and Zen 5, with memory-operand latency of 11 on Zen 4 and 7 on ADL-P. `SHL r32, imm8` runs on p06 on ADL-P and on 2 ALUs on Zen 4. For the front-end: Golden Cove has 6 decoders, 32 B per cycle L1i fetch, a 4K-entry uop cache delivering 8 per cycle, and 6-wide rename ([Chips and Cheese](https://chipsandcheese.com/p/popping-the-hood-on-golden-cove)). Zen 4 has a 6.75K-entry op cache delivering 9 per cycle, 4-wide decoders, 6-wide rename and a 320-entry ROB ([Chips and Cheese](https://chipsandcheese.com/p/amds-zen-4-part-1-frontend-and-execution-engine)). Some figures are assumptions from memory that I did not re-fetch. `SETA`/`SETNBE` is 2 uops on p06 on Intel P-cores (Agner Fog) and 1 op on Zen 4. L1D is 48 KiB with 5-cycle latency on Golden Cove and 32 KiB with 4-cycle latency on Zen 4. DRAM latency is about 100 ns on all three platforms.

### D1. Bound model for the current kernel

Static counts come from `probes/count.sh`. It assembles each file with `llvm-mc` and sizes it with `llvm-size`. The 300-tree emissions come from `probes/d_emit.rs` (same generator as `100x100`, `Workload::Distinct(300, 100)`, default budget GP12/FP9).

| Default kernel (GP12/FP9, 2 sums) | ARM64 100 | ARM64 300 | x86-64 100 | x86-64 300 |
|---|---:|---:|---:|---:|
| instructions | 4453 | 13338 | 4015 | 12033 |
| text bytes | 17,812 | 53,352 | 21,458 | 64,082 |
| flag compares (`fcmp` / `ucomisd`) | 600 | 1800 | 600 | 1800 |
| GPR-to-FP `fmov` / `movabsq` | 600 / - | 1803 / - | - / 737 | - / 2212 |
| load-pipe ops (`ldr`+`ldp`+`stp`+`fmov d,x`) | 1179 | 3497 | - | - |
| stack references (wrapper included) | 20 | 20 | 38 | 86 |
| chunks | 38 | 113 | 38 | 113 |
| x86 constant pool `.rodata.cst8` | - | - | 4,792 B | - |

Throughput on the M4 at 100 trees has three candidate bounds. Rename width gives 4453 / 10 = 445 cycles. The load and GPR-to-FP pipes give 1179 / 3 = 393. The compare port gives 600 / 2 = 300. Measured per call (d_probe, `independent` column, driver about 2 to 3 ns) is 108 to 114 ns, or about 480 to 500 cycles at 4.47 GHz, which is 8.9 to 9.3 IPC. The dependent column is 113 to 126 ns. The kernel therefore runs at about 90% of the rename-width bound, the load pipes are about 80% busy and the compare port about 62%. At 300 trees the bounds are 1334 (rename), 1166 (load pipes) and 900 (compare). The measured 330 to 346 ns independent is about 1475 to 1520 cycles, again about 90% of rename width. The orchestrator's "about 8.7 IPC, close to width" holds.

Latency on the M4 splits into a per-tree part and a sum part. Once the row pointer is known, one tree needs a feature `ldr d`, then `fcmp`, then `cset`, which measures about 12 cycles. One or two `orr` steps after its last compare add 2 to 4 cycles, the leaf `ldr d` about 4 (assumption: FP load a cycle slower than the 3-cycle integer chase), and the leaf `fadd` 2. That totals about 20 to 25 cycles. Across trees the only serial dependency is the sum chain (`src/sort_policy.rs:478-484` assigns round robin). It is trees/2 fadds at 2.1 cycles, plus `AddSums` and the bias add. In the capture the scale multiply by 1.0 was folded away but `+ 0.0` bias remains (`fadd d0,d0,d1; movi d1,#0; fadd d0,d0,d1` at the end of `_continuation37`). That gives about 25 + 50 x 2.1 + 4 = 135 cycles at 100 trees and 25 + 150 x 2.1 + 4 = 345 at 300. A one-tree model measured end to end, dependent, takes 13.8 ns against a 4.3 ns driver floor, so about 42 cycles of call-to-call latency including call, return and conversion. That agrees with the estimate.

LLVM does not rebalance the per-tree `orr` chain. It folds the shifts into `orr xd, xn, xm, lsl #k` and emits a linear chain in comparison order (`_continuation0` lines 90-150 of the capture). Only the `orr` steps after a tree's last compare are on its path, and those are 1 to 2 deep in the scheduled order.

On the x86 cores, per-tree latency is about 18 cycles on Golden Cove: `movsd` 5, `ucomisd` 3, `seta` 1, `shl`/`or` 2, and `addsd` with its memory operand 7. On Zen 4 it is about 25 (4 + 7 + 1 + 2 + 11). The 2-sum chain comes to about 25 + 50 x 2 + 4 = 130 (Golden Cove) and 30 + 50 x 3 + 6 = 185 (Zen 4) at 100 trees, and about 330 and 485 at 300.

On x86, throughput is set by the compare port, rename and front-end together. `ucomisd` alone floors the kernel at 600 cycles (100 trees) and 1800 (300 trees) on both Zen 4 and Golden Cove. On Golden Cove the uop count is about 4015 + 600 (`ucomisd m64` takes 2 retire slots per uops.info) + 600 (`seta` second uop, assumption) = 5215 over 6-wide rename, or 869 cycles. Ports 0 and 6 also carry `ucomisd`, `seta` and 340 `shll`, which is 1070 cycles if `seta` is 2 uops and 770 if it is 1. On Zen 4, 4015 / 6 = 669.

Front-end on the M4: 17.8 KB of straight-line code at 10 instructions (40 B) per cycle equals the rename bound, and 53 KB at 300 trees still fits the 192 KB L1I. On x86, 21.5 KB at 32 B per cycle takes 671 cycles, and 64 KB at 300 trees takes 2003. Golden Cove's uop cache holds 4K entries against about 5.2K uops at 100 trees and 15.6K at 300, so it cannot hold the kernel and legacy decode feeds most of it. Zen 4's op cache (6.75K) could in principle hold the 4015 macro-ops of the 100-tree kernel. I cannot quantify how much the ten-byte `movabsq` immediates reduce its effective capacity, so treat that case as unknown. If the op cache misses, Zen 4 decodes 4 instructions per cycle, which is 1004 cycles at 100 trees and 3008 at 300. The large code model accounts for 737 `movabsq` x 10 B = 7,370 B, or 34% of x86 text.

Data cache on the M4: leaf tables are 51.2 KB at 100 trees and fit the 128 KB L1D. At 300 trees they are 153.6 KB and do not fit, but each call touches only one 128-byte line per tree (38 KB). With the benchmark's 35 hot rows everything stays in L1 or L2, and the measured throughput shows no data stall. On x86, each call also touches the whole constant pool, 75 lines of 64 B at 100 trees. ARM64 has no constant-pool traffic because the thresholds are immediates. ARM64 does have a different overhead: leaf tables are reached through the GOT, with 100 `adrp` + 100 `ldr x` per call, which are 200 instructions and 100 load-pipe slots.

Cold data was measured on the M4 (next table). The cost scales linearly with tree count, which means the misses are not overlapping. My explanation, a model I have not verified: each leaf load sits about 45 instructions after the previous one. A 361-entry coalesced ROB (roughly 600 instructions) therefore spans about 13 trees, so about 13 misses can be outstanding before retirement blocks. At 100 trees that is about 8 serialized DRAM round trips of roughly 100 ns each, or about 800 ns. The warm-code control below measured 731 ns, which fits. On x86 the same model with a 320-entry ROB (Zen 4) or a roughly 512-entry ROB (Golden Cove, assumption) predicts 0.6 to 1.2 us at 100 trees.

| Per-call latency, M4, 100 trees, ns (Instant, 41.7 ns ticks) | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| timer only, no call | 13-17 | 0 | 42 | 42 |
| hot, no pollution | 111-131 | 125 | 125 | 125-209 |
| after reading 256 KiB (L1D evicted) | 124-135 | 125 | 125-167 | 167 |
| after reading 4 MiB (L2-resident) | 141-143 | 125 | 167 | 167-208 |
| after reading 16 MiB | 163-225 | 125-166 | 208-292 | 542-2375 |
| after reading 32 MiB | 667-899 | 541-625 | 750-2625 | 3875-3917 |
| after reading 64 MiB (DRAM-cold) | 1121-1409 | 708-792 | 3083-3583 | 4625-5208 |
| 64 MiB, timer only (control) | 16-17 | 0 | 42 | 42 |
| 64 MiB, then one call on another row (code, GOT, row warm; about 75% of leaf lines cold) | 731 | 625 | 750 | 3500 |
| after one call of a 2000-tree model (L1I, L1D and BTB interference) | 306-325 | 292-333 | 334-375 | 417-458 |

| Same, 300 trees (one run) | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| hot | 354 | 334 | 375 | 459 |
| 256 KiB / 4 MiB | 378 / 388 | 375 / 375 | 375 / 458 | 459 / 625 |
| 16 MiB / 32 MiB / 64 MiB | 1034 / 2378 / 3203 | 875 / 1833 / 2083 | 1250 / 2333 / 9042 | 7041 / 11458 / 12000 |
| 64 MiB then warm call on another row | 1901 | 1875 | 2042 | 2500 |
| after one 2000-tree call | 973 | 958 | 1000 | 1250 |

Which bound is active today? On the M4 it is rename width when hot; the measurement reconciles at about 90% of that bound. When DRAM-cold, it is leaf-table misses paced by the ROB. On x86 there is no measurement. By static count the active bound is the `ucomisd` port (600 per 100 trees) with Golden Cove's rename or port 0/6 pressure (870 to 1070) and legacy-decode fetch (671 to 1004) close behind. My best estimate is 670 to 1070 cycles per 100 trees, against the M4's roughly 490.

### D2. After the instruction cut

Pieces A and C aim for 1500 to 2500 instructions at 100 trees (assume 3x at 300). A scalar recoding keeps one flag-setting compare per split, which pins the kernel to the compare-port floor. On the M4 that is 300 cycles at 100 trees and 900 at 300. The load and GPR-to-FP pipes are nearly as tight: each split needs its threshold moved into an FP register by either `fmov d,x` or `ldr d`, and both use the same 3 pipes. That pipe floor is about 600/3 + (394 feature + 200 leaf/GOT)/3, or about 400 cycles, unless thresholds are loaded two at a time with `ldp` (then about 280). The integer form (`cmp`/`cset` on order-preserving keys) measured the same 0.52 cycles per split, so it does not escape the floor. On Zen 4 and Golden Cove the floor is 600 and 1800 cycles; Zen 5 halves it to 300 and 900. Only compares that write a mask instead of flags (`fcmgt`, `vcmppd`) remove the floor. `fcmgt` measured 4 per cycle, and each `fcmgt.2d` does 2 lanes.

The sum chain does not change when instructions are cut. With 2 sums it stays under the scalar compare floor at both sizes on every target (M4 135/345 against 300/900; Golden Cove 130/330 against 600/1800; Zen 4 185/485 against 600/1800). It becomes relevant only with mask compares. At 300 trees an M4 vector kernel of 4500 to 7500 instructions takes 450 to 750 cycles against a 345-cycle 2-sum chain. Those are within a factor of 1.3 to 2.2, and because leaves arrive spread over the kernel the chain cannot start early. On Zen 4 a 100-tree vector kernel at 6-wide rename (250 to 420 cycles) meets a 185-cycle chain. The chain stops mattering when trees/sums x fadd latency is under about half the kernel time. With 4 sums the chain is M4 85/190, Golden Cove 80/180 and Zen 4 115/265, which clears every projected vector kernel except the smallest 100-tree Zen 4 case.

Measured today, the accumulator count makes no difference, as expected. d_probe dependent medians at 100 trees are 118-120 (1 sum), 123-126 (2), 122-123 (4) and 123-127 (8) ns, and at 300 trees 347/350/350/358. scalar_bench GP12 medians are 119.75/118.82/120.71 ns for 1/2/4 sums.

"More sums cost registers only" is false with this ABI. LLVM 21's `CC_AArch64_Preserve_None` lists only integer registers and then delegates to `CC_AArch64_AAPCS` ([AArch64CallingConvention.td at llvmorg-21.1.8](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/AArch64/AArch64CallingConvention.td)). FP arguments therefore go in v0-v7 and anything beyond goes on the stack. SysV x86-64 has the same limit of 8 (xmm0-7; assumption that x86 preserve_none also delegates FP, which the emitted spills agree with). Every carried sum takes one of those 8 slots from carried features. Measured at 100 trees with FP budget 7+sums: ARM64 stack references go 20 (wrapper only) -> 20 -> 64 -> 163 for 1/2/4/8 sums, and x86 goes 40 -> 38 -> 139 -> 349. The `_continuation0` of the 4-sum ARM64 file ends with `stp d8, d3, [sp]` and `_continuation1` reloads `ldr d6, [sp]`. The old 4-sum regression (`plans/implementation.md:71-82`) was measured under the spilling single-function backend, so it does not carry over. But 4 scalar sums are still not free now.

Recommendation: keep 2 round-robin scalar sums while compares set flags. When a mask-compare formulation lands, switch to 4 lane sums carried as two `<2 x double>` values, which take 2 FP argument slots instead of 4. On ARM64, a pair of trees costs `ldr d` + `ld1 {v.d}[1]` + `fadd.2d`, three instructions against four for two scalar `ldr d` + `fadd`. That saves about 1 instruction per 2 trees, minus a 2-instruction final `faddp` + `fadd`. This is unverified: I did not probe the latency of `ld1` into a lane. On x86 the pair costs `movsd` + `movhpd` + `addpd` against two memory-operand `addsd`, so +1 instruction per 2 trees (+50 at 100 trees, +150 at 300). On x86, prefer 4 scalar xmm sums if the planner can give up two carried feature slots, and the vector pair only if it cannot. A pairwise tree reduction is rejected because it needs all leaf values live at once.

### D3. Continuation and wrapper cost

On ARM64 each boundary is one direct `b` plus almost no argument shuffling: the whole 100-tree kernel has 6 `mov x,x` and 1 `fmov d,d`, because carried values keep stable registers. The measured cost of a boundary is near zero. I swept the comparison limit at `src/sort_policy.rs:172` (throwaway sed, reverted; script `probes/chunk_sweep.sh`).

| limit | chunks | ARM64 instr | ARM64 stack refs | x86 instr | x86 stack refs | M4 dep / indep median, 2 sums, ns |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 75 | 4518 | 20 | 4121 | 10 | 119.4 / 111.7 |
| 16 | 38 | 4453 | 20 | 4015 | 38 | 123.7 / 113.1 |
| 32 | 19 | 4552 | 138 | 4000 | 127 | 128.1 / 117.6 |
| 64 | 10 | 4712 | 389 | 3978 | 186 | 131.0 / 122.5 |

Doubling the boundaries from 38 to 75 did not cost time: the 75-chunk kernel was 1 to 4 ns faster, within noise. With a noise band of about 3 ns, the cost per boundary is under about 0.4 cycles. Larger chunks already spill at 32 comparisons. The wrapper is 23 instructions: 5 `stp`, `mov`, `bl`, 5 `ldp`, `ret`. The callee-save reloads come about 500 cycles after the stores, so no store forwarding is involved. I estimate about 3 to 5 cycles for the wrapper (unmeasured). `CompiledModel::predict` adds a length compare and branch, plus a call through the `JitFunction` pointer, which is loop-invariant and so off the data path. An empty model runs at 1.4 ns per call, or about 6 cycles of throughput. Boundaries, wrapper and entry together come to about 10 cycles: 2% of 500 cycles and about 5% of 200. They are noise at both sizes.

x86 differs in two ways. First, under the large code model every transfer is `movabsq $continuationN, %rax; jmpq *%rax`, and the wrapper uses `callq *%rax`. That is 37 indirect jumps and one indirect call. When the kernel is hot the BTB predicts them, but after other work evicts the BTB each one can cost a front-end redirect. I cannot measure that here; a small code model makes them direct. Second, the "zero continuation-body stack accesses" result in `plans/implementation.md:178` is ARM64-only. In the x86 100x100 default, 13 of 38 continuations `pushq %rbp`/`popq %rbp` and one spills `%rax` to `-8(%rsp)` (capture lines 1247-5434).

### D4. Does the benchmark measure the use case?

The benchmark gets the latency shape right: `next_row` feeds each prediction into the next row index, so calls are chained. With everything hot, chaining costs only 5 to 13 ns per call at 100 trees and 10 to 16 ns at 300. The kernel is mostly throughput work (threshold materialization does not depend on the row), and the core overlaps it across the call boundary.

The benchmark hides the cold case, which is 6 to 10 times slower. Its 50,000 calls revisit 35 rows, so code, GOT, the 100 touched leaf lines, the rows and the BTB all stay warm. The measurements above show what happens when the caller's other work between calls evicts caches. Evicting only L1 costs 5 to 15%, and L2-resident data costs 15 to 30%. Evicting the 16 MiB L2 costs 6 to 10 times at the median, with p99 in the microseconds. One call of another large model in between costs 2.5 to 3 times. The warm-code control shows that most of the DRAM-cold cost comes from leaf-table misses, not from code: 731 ns remains at 100 trees with code and GOT warm, out of 1121 to 1409 ns. The cold cost grows linearly with trees (about 1.9 us warm-code at 300), which fits the ROB-paced miss model in D1. At that point instruction count barely matters for cold calls; memory-level parallelism over leaf loads does.

Proposed harness (described, not added under `examples/`; my throwaway version is `probes/d_probe.rs`): a new `examples/cold_bench.rs` taking `<workload> <scenario>`. For each of about 4,000 calls it:
1. runs the scenario's "other work": nothing, a streaming read of N bytes (N in 256 KiB, 4 MiB, 16 MiB, 64 MiB), or one call of an unrelated large compiled model;
2. copies a fresh row (from a stride-97 walk over all 256 rows, not the 35-row chain) into the input buffer, so the row is hot as it would be from a producer;
3. times only `predict` with `Instant`.

It then reports mean, p50, p90 and p99 plus a timer-only control per scenario. On macOS `Instant` ticks at 41.7 ns, so the percentiles are quantized and only the means are fine-grained. For sharper per-call numbers use kperf cycle counters (needs root) on macOS, or `rdtscp`/`perf_event_open` cycles on Linux x86. Expected results are the M4 tables above: hot about 110 ns, DRAM-cold about 1.1 to 1.4 us at 100 trees. On x86, expect the same shape plus indirect-jump redirects.

`CompiledModel::predict` overhead does not matter even at 100-cycle kernels: about 6 cycles of throughput, off the critical path (D3).

### D5. scalar_bench 100x100, one run under load (load average 5.5)

| Variant | median | min |
|---|---:|---:|
| Tree order, sums 1 / 2 / 4 | 123.02 / 125.55 / 126.28 | 122.04 / 121.17 / 124.19 |
| Scheduled GP12/FP9, sums 1 / 2 / 4 | 119.75 / 118.82 / 120.71 | 114.99 / 114.96 / 117.31 |
| Scheduled GP20/FP9, sums 1 / 2 / 4 | 122.65 / 120.67 / 122.43 | 119.40 / 117.33 / 118.25 |
| SIMD2 GP12/FP9, sums 2 | 124.68 | 121.68 |
| Driver floor | 5.66 | 5.07 |

The sum count is flat within about 2 ns, which matches the D2 estimate that the chain (135 to 240 cycles for 2 or 1 sums) sits well under the roughly 490-cycle throughput.

## 2. Hypotheses touched

H4 is partly refuted. The M4 fadd-to-fadd latency is 2.1 cycles, not 3 (measured). The chain is hidden today (measured: 1 to 8 sums are equal). It stays hidden after any cut that keeps one flag-setting compare per split, because the compare port floors the kernel at 300/900 cycles on the M4 and 600/1800 on Zen 4 and Golden Cove, above the 2-sum chain at both 100 and 300 trees. It surfaces only with mask compares (`fcmgt`/`vcmppd`). "More sums cost registers only" is refuted: FP continuation arguments beyond 8 go to the stack (LLVM td file; stack references 20 -> 64 -> 163 at 4 and 8 sums).

The orchestrator's throughput claim ("about 8.7 IPC, throughput bound on instruction count") is confirmed for the M4 today at about 90% of 10-wide rename. The correction is that instruction count is not the only resource. The M4's load and GPR-to-FP pipes run about 80% busy (393 of about 490 cycles), and the compare port 62%. On x86 the binding resources are the `ucomisd` port and the front-end, not instruction count.

H1 is confirmed for x86 and extended. `movabsq` is 34% of x86 text bytes, and the same code model also makes all 37 continuation transfers and the wrapper call indirect. On ARM64 the extra cost is somewhere else: 100 GOT `adrp`/`ldr x` pairs for leaf tables. For M4 throughput, replacing the `fmov d,x` thresholds with `ldr d` from a base register does not reduce load-pipe use, because both use the same 3 pipes. Only paired `ldp` loads or vector loads reduce it.

H2 is partly corrected. On the M4 a feature load is not a free L1 hit: it takes one of 3 pipe slots per cycle that the kernel already keeps about 80% busy.

The implementation note that continuations have zero body stack traffic holds for ARM64 only. x86 has 13 `%rbp` push/pop pairs and one spill in 100x100.

## 3. Ranked recommendations

1. Keep 2 accumulators for any flag-compare formulation. This changes nothing and is verified by measurement (1 to 8 sums equal at 100 and 300 trees) and by the bound model on all three targets. The instruction delta is 0. Risk: none.
2. Add the cold-call harness from D4 as the headline latency metric next to the hot chain. It changes no kernel code. It matters because the cold case is 6 to 10 times the hot one on the M4 and scales with trees, so hot-loop wins of 10 to 20% may be invisible or reversed in deployment. Risk: noisy per-call timing on macOS; use means and a timer-only control.
3. For pieces A and C: judge candidate encodings by the per-split compare-port and load-pipe floors, not only instruction count. On the M4, any flag-setting compare costs at least 0.5 cycles per split: 300 cycles (67 ns) at 100 trees and 900 (200 ns) at 300. On Zen 4 and Golden Cove it costs 1.0 cycle per split: 600 and 1800 cycles. Zen 5 costs 0.5. Mask compares cost 0.25 cycles per `fcmgt.2d`, which covers 2 splits (M4, measured). Moving threshold materialization from `fmov` to `ldr d` saves instructions but not M4 load-pipe slots; `ldp` pairs halve them. Evidence strength: measured on the M4, cited for x86.
4. When mask compares land and a 300-tree kernel drops below about 700 cycles (M4) or a 100-tree kernel below about 400 (Zen 4): use 4 lane sums carried as two `<2 x double>`. On ARM64 that is about -1 instruction per 2 trees and +2 at the end. On x86 use 4 scalar sums, costing 2 carried feature slots and +2 final adds, or the vector pair at +1 instruction per 2 trees. Projected chain: M4 345 -> 190 cycles at 300 trees; Zen 4 185 -> 115 at 100. Risk: the `ld1` lane-insert latency is unmeasured, and carried-slot pressure shifts feature residency.
5. For x86, a small code model or base-register addressing (piece A's H1) also removes 37 indirect jumps plus one indirect call per prediction, which matters most on cold calls. The instruction delta is -38 transfers' `movabsq`, about 270 B. Risk: code placement under MCJIT.
6. Cold-call leaf misses (the dominant cold cost). Candidates: collect a batch of tree indices before issuing their leaf loads, so more misses overlap; or `f32` leaves to halve the table lines (a user question, since `f64` is deliberate). This is not verified; see experiment E3. A burst of 100 stores plus 100 loads adds about 200 instructions (+4.5% hot) and trades hot time for cold time.

Bound table in cycles. Each cell is current / projected (1500 to 2500 instructions at 100 trees, 3x at 300). "x86" cells give Golden Cove then Zen 4 where they differ. Projected throughput assumes 1.1 uops per instruction on Golden Cove (assumption).

| Bound | ARM64 M4, 100 | ARM64 M4, 300 | x86-64, 100 (GC / Zen 4) | x86-64, 300 (GC / Zen 4) |
|---|---|---|---|---|
| Throughput, rename width | 445 / 150-250 | 1334 / 450-750 | 869 / 275-460 ; 669 / 250-420 | 2606 / 825-1375 ; 2006 / 750-1250 |
| Throughput, flag-compare port (holds if compares stay scalar) | 300 / 300 | 900 / 900 | 600 / 600 (Zen 5: 300) | 1800 / 1800 (Zen 5: 900) |
| Throughput, M4 load+`fmov` pipes | 393 / 280-400 | 1166 / 850-1200 | - | - |
| Measured hot (M4 only) | 480-515 | 1475-1520 | none | none |
| Chain latency, 2 sums (4 sums) | 135 (85) / same | 345 (190) / same | 130 (80) ; 185 (115) / same | 330 (180) ; 485 (265) / same |
| Front-end, hot | 445 (= rename) / 150-250 | 1334 / 450-750 | 671 (GC fetch) ; 669-1004 (Zen 4 op cache / decode) / 210-350 ; 375-625 | 2003 ; 2006-3008 / 630-1050 ; 1125-1875 |
| Data cache, hot | L1, 0 extra | L1/L2, 0 extra measured | L1/L2 (56 KB touched set incl. pool) | L2 |
| Data and code, DRAM-cold (per call) | about 4,500-6,300 (1.0-1.4 us measured) | about 14,000 (3.2 us measured) | 0.6-1.2 us (ROB model, estimate) | about 3x that |

Active bound per target: the M4 is rename-bound today and becomes compare/load-pipe-bound at 300/900 after a scalar cut. x86 is bound by the compare port, Golden Cove port 0/6 pressure, and decode, both now and after a scalar cut. The chain surfaces only after mask compares, and then 4 sums fix it. Cold calls are bound by leaf-table misses on every target.

## 4. Proposed experiments for the orchestrator

Setup for the probes: add these two entries to `Cargo.toml` temporarily, and revert afterwards. The diff I tested, verbatim (already reverted):

```
@@ -5,3 +5,11 @@ edition = "2024"
 
 [dependencies]
 inkwell = { version = "=0.9.0", features = ["llvm21-1"] }
+
+[[example]]
+name = "d_probe"
+path = "probes/d_probe.rs"
+
+[[example]]
+name = "d_emit"
+path = "probes/d_emit.rs"
```

The chunk sweep applied `sed -i '' "172s/partition(&plan, [0-9]*)/partition(\&plan, N)/" src/sort_policy.rs` for N in 8, 16, 32, 64 and restored 16 (`probes/chunk_sweep.sh`). The probe files are in the worktree under `probes/`: `d_probe.rs`, `d_emit.rs`, `count.sh`, `chunk_sweep.sh`, `chunk_count.sh`, and `ubench/*.c`. `ub5.c` and `ub6.c` were derived from `ub.c` by inline python that added the integer split, `cmp`, and scalar `fcmgt` loops. Their source is in the directory.

- E1, serial rerun of cold calls. `export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21; cargo run --offline --release --example d_probe -- 100x100`, then `-- 300x100`. Compare the per-call tables with the ones above. Expected: hot means about 110 and 350 ns; after 64 MiB, 1.1 to 1.4 us and about 3 us; warm-code control about 0.7 and 1.9 us.
- E2, x86 execution on a Zen 4 or Golden Cove machine: `perf stat -e cycles,instructions` on `scalar_bench 100x100`. On Intel, add `uops_dispatched.port_0,uops_dispatched.port_6,idq.dsb_uops,idq.mite_uops`. Expected: at least 600 cycles per call at 100 trees, port 0 near saturated, most uops from MITE (legacy decode) on Golden Cove.
- E3, leaf-miss overlap. Make a throwaway `src/jit.rs` variant in which `Action::Retire` stores the index to a stack array and the chunk that holds the last retire issues all leaf loads and fadds together. Compare d_probe's hot and 64 MiB rows. Expected: the cold mean falls toward 2 or 3 DRAM latencies (about 300 ns at 100 trees) and hot time rises by about 5%. If the cold number does not move, the ROB-pacing model is wrong.
- E4, repeat the chunk-limit sweep serially: `bash probes/chunk_sweep.sh 8 12 16 24`. Expected: 8 to 16 flat within 2 ns and a rise from 32 on. This confirms that boundaries cost under 0.5 cycles each.
- E5, when piece C produces a mask-compare kernel: run d_probe with 2 and 4 sums at 300 trees, dependent and independent. Expected: dependent minus independent shrinks at 4 sums if the chain is active.

## 5. Questions for the user

1. Which x86 core runs production? The flag-compare floor is 1 per cycle on Zen 4 and Golden Cove but 0.5 on Zen 5, and that changes how much the scalar recodings in pieces A and C can gain.
2. What happens between two predictions in production: roughly how many bytes of other data and code, and how long? If calls are cold, the latency is 0.7 to 3 us, set by leaf-table misses, and tuning for leaf-load overlap or table size outranks cutting instructions. If calls are back-to-back hot, the current ranking stands.
3. Is the target the mean or the p99 per call? Cold p99 on the M4 is 4 to 12 us.
4. Would `f32` leaf values (or a model-level leaf quantization) be acceptable? That halves the leaf-table lines touched by cold calls. This is a question only, since `f64` is deliberate.
5. Can the caller warm the model ahead of a call, for example with one throwaway call or a prefetch hook when a request starts? The warm-code control suggests this would recover about a third of the cold cost at 100 trees.
