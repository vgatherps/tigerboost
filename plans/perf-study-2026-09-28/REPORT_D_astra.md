# Study D: prediction latency and measurement

Keep two scalar sums for now. The current evidence does not show an accumulator bottleneck. Measure calls interleaved with other work before changing that default. Four sums can consume feature capacity or force continuation arguments onto the stack. A smaller instruction count alone also does not guarantee a faster kernel if it adds constant loads.

This study changed no tracked file. The build passed before assembly analysis. All new Rust, Python, LLVM IR, assembly, objects and logs are under `probes/`. The Rust probe uses the existing planner and emitter without changing their code. It checked all 256 rows against `Trees::predict` for eight configurations, including a new 300-tree instance of the existing distinct-threshold generator. No x86 code was executed. Timings below were taken under load.

## 1. Answers and evidence

### D1. Current bounds

The warm 100-tree ARM64 kernel is consistent with instruction throughput as the main limit. The sum dependency is much shorter. On Golden Cove, the current 100-tree kernel should be limited mainly by decode/rename and instruction execution. At 300 trees its roughly 64 KB of code exceeds the 32 KiB L1 instruction cache; instruction fetch from L2 becomes a serious limit. These x86 conclusions are estimates, not measurements. Cold code or cold leaves can change either conclusion.

The cycle tables in section 3 use these inputs:

| Quantity | Apple M4 performance core | Intel Golden Cove client performance core |
|---|---:|---:|
| Maximum instruction supply/rename used in this model | 10 instructions/cycle | 6 instructions/cycle; instructions may expand into several micro-ops |
| L1 instruction fetch | 64 bytes/cycle | 32 bytes/cycle |
| L1 instruction/data capacity | 192/128 KiB | 32/48 KiB |
| Shared cluster/private L2 capacity | 16 MiB | 1.25 MiB |
| Scalar add dependency | 2 cycles; also show 3-cycle sensitivity | 2 cycles for dependent Fast Adder operations |
| L1 load-to-use assumption | 4–5 cycles | 5–7 cycles, depending on consumer/addressing |
| L2 load-to-use estimate | 20 cycles, with roughly 17–36 plausible | 15–16 cycles |

M4 fetch width, scalar `fadd` latency, 4-cycle load-to-ALU latency and 10-wide execution come from [Jie Ge's own M4 microbenchmarks](https://jia.je/hardware/2025/05/21/apple-m4/). That study distinguishes 64-byte L1 lines from 128-byte L2 lines. [Published M4 Pro sysctl output](https://forums.macrumors.com/threads/request-for-some-system-information-about-cpu-caches.2478481/) supplies the Pro's cache capacities. Local sysctl failed with `Operation not permitted`; `probes/sysctl.log` records this sandbox artifact. LLVM reports host CPU `apple-m4`. No local core clock or cycle counter was measured.

[Intel's optimization manual, sections 2.4.1–2.4.3](https://cdrdv2-public.intel.com/821612/248966-Optimization-Reference-Manual-V1-050.pdf) supplies the Golden Cove widths, capacities and Fast Adder distinction: general result latency is three cycles, but dependent adds can forward in two. [uops.info's ADDSD memory-operand measurements](https://uops.info/html-instr/ADDSD_XMM_M64.html#ADL-P) independently show a two-cycle accumulator dependency. A memory-operand latency of about ten cycles is the other input path; it must not be multiplied by the length of the sum chain. [Jie Ge's Golden Cove tests](https://jia.je/hardware/2025/01/10/intel-golden-cove/) measure five-cycle L1 pointer loads and about sixteen-cycle L2 loads. The M4 L2 estimate is consistent with the small-working-set region in [Marian Posăceanu's dependent-load measurements](https://ark.marianposaceanu.com/measuring-apple-m4-cache-latency); those measurements were on a base M4, not this host.

The 100-tree emissions below exactly match the supplied captures. `probes/analyze.py` assembles each file with LLVM 21.1.8 and reads the section sizes. It also counts instructions and explicit stack accesses. `probes/static_summary.txt`, `probes/static_counts.json` and `probes/emitted/*.sections.txt` retain the results.

| Default scalar GP12/FP9, two sums | ARM64 100 | ARM64 300 | x86 100 | x86 300 |
|---|---:|---:|---:|---:|
| Instructions, including C wrapper | 4,453 | 13,338 | 4,015 | 12,033 |
| Text section bytes | 17,812 | 53,352 | 21,458 | 64,082 |
| Instruction bytes excluding alignment gaps | 17,812 | 53,352 | 21,220 | 63,294 |
| Comparison tests | 600 | 1,800 | 600 | 1,800 |
| Planned feature loads | 394 | 1,192 | 394 | 1,192 |
| Emitted feature values loaded | 384 | 1,137 | 384 | 1,137 |
| Chunks / tail transfers | 38 / 37 | 113 / 112 | 38 / 37 | 113 / 112 |
| Explicit stack accesses in continuation bodies | 0 | 0 | 28 | 76 |

The count of 300 trees is an actual emission, not three times the 100-tree count. It has 100 declared features and depth six. `examples/support/mod.rs:39` accepts `Workload::Distinct(trees, features)` internally, although the example CLI only lists three fixed sizes. `probes/emit.rs` calls that generator directly.

ARM64's 600 threshold transfers, 591 `movk` instructions, 600 `fcmp`, 592 `cset`, 8 `cinc`, 492 `orr` and 148 `ubfiz` match the supplied histogram. The x86 737 `movabsq` instructions occupy 7,370 bytes: 599 constant addresses, 100 leaf-table addresses and 38 continuation addresses. One zero threshold uses a register rather than a constant-pool load. This is 34.3% of the text section. At 300 trees, 2,212 `movabsq` instructions occupy 22,120 bytes.

The instruction path within one tree is load feature, compare, form a bit, combine the six bits, load the selected leaf, then add. The six comparisons are independent when their features are ready. Reuse of the flags register does not create a true data dependency between comparisons. The leaf address depends on all six outcomes.

There are five two-input bit combinations per depth-six tree, not six OR operations. LLVM removes the first OR with zero and sometimes uses `cinc`, shifted `orr`, `ubfiz` or x86 `lea`/`add`. The ARM64 probe follows selector registers across tail branches and checks that every leaf depends on six distinct comparisons. Its longest selector paths contain five bit combinations, plus an initial shift on 64 of the 100 trees. The total combine/shift depth is five for 36 trees and six for 64. LLVM has not replaced these paths with a balanced three-level reduction. Evidence is `probes/arm_selector_paths.json` and `probes/analyze.py`.

For the first leaf, compare lines are 69, 79, 95, 119, 147 and 195 in `captures/100x100/aarch64-apple-darwin-scheduled-12gp-9fp-2sums.s`. Its longest selector path uses lines 109, 116, 121, 127, 153 and 215. Lines 254–275 then contain:

```asm
adrp x19, l_leaves_67@GOTPAGE
ldr  x19, [x19, l_leaves_67@GOTPAGEOFF]
ldr  d8, [x19, x12]
fadd d0, d0, d8
```

The x86 first-leaf selector is also serial: capture lines 89, 102, 107, 123, 160 and 233 form one shift followed by five `lea`/`or` combinations. Line 268 consumes that selector in `addsd (%r10,%rdx,8), %xmm0`. This example is not a balanced reduction.

The leaf-base pointer load can overlap selector construction. It is another input to the leaf load, not another dependency after the selector. For ready threshold constants and L1 hits, a complete tree path is roughly 18–23 cycles on M4 and 23–29 on Golden Cove. The estimate includes roughly 5–6 selector operations, compare/bit latency and both loads. It is a dependency estimate, not time per tree multiplied by tree count. Different trees overlap. Source `src/jit.rs:331–347` builds the selector; `356–391` loads and accumulates the leaf.

With two sums, the emitted scalar dependency is 50 or 150 additions, then one reduction and one bias addition. The synthetic scale of one disappears; adding positive zero remains. The end is visible at ARM64 capture lines 4948–4964. At two-cycle scalar add latency this is 104/304 cycles, before the first leaf becomes available. At three cycles it is 156/456. First-leaf readiness and the last tree's leaf path add startup/tail delay; the sum of those small paths is still far below today's instruction-throughput estimate. A non-unit model scale adds a multiply before the existing bias-add dependency; the tables describe this fixture.

The historical timing is 121.151 ns with a 5.568 ns driver floor (`plans/implementation.md:169–173`). At the brief's assumed 4.4 GHz, `(121.151 - 5.568) * 4.4 = 508.6` cycles and `4453 / 508.6 = 8.76` instructions/cycle. This agrees with an 8–10 IPC bound of 445–557 cycles. It does not prove the limiting execution port. The floor performs a score-array load instead of prediction; subtracting it is approximate. The remainder still includes the checked Rust API and other dispatch work. Frequency, core assignment and hardware counters were not measured.

A function need not contain a loop to use a micro-op cache: repeated calls revisit the same instructions. Capacity is the problem here. Golden Cove has about 4,096 cached micro-ops. LLVM's Alder Lake instruction tables model memory `ucomisd` as two micro-ops and `seta` as two; `probes/x86_instruction_tables.txt:25–34` shows them. Even adding only the extra micro-op for each of 599 memory comparisons gives 4,614, already 518 above nominal capacity. Including the model's `seta` expansion gives about 5,214 before other details. At 300 trees those corresponding counts are 13,832 and 15,632. Packing, alignment and surrounding code reduce usable capacity further. These are model-based capacity estimates, not measured DSB occupancy. LLVM's 5,337 total execution micro-ops are not a DSB count: fusion and eliminated moves make those different quantities.

For Golden Cove's legacy path, 21,458/32 is 671 fetch cycles and 4,015/6 is 670 instruction-decode cycles. Extra micro-ops increase allocation demand. Use 4–4.6 effective instructions/cycle for this instruction mix, or roughly 873–1,004 cycles, before unhidden misses and entry/exit delay. At 300 trees the L1 fetch floor is 2,003 cycles. A sequential 64,082-byte pass cannot stay in a 32 KiB L1I. [Chips and Cheese's own Golden Cove fetch test](https://old.chipsandcheese.com/2021/12/02/popping-the-hood-on-golden-cove/) measured about 16 bytes/cycle for code supplied from L2. Applying that rate gives about 4,005 cycles. Partial micro-op/L1 hits can lower it; cold code and competing work can raise it. This is why raw instruction count is insufficient for the x86 300-tree estimate.

The leaves occupy exactly 51,200/153,600 bytes, or 50/150 KiB. Each call loads one eight-byte leaf per tree, hence nominally 100/300 selected cache lines. It does not read the whole table. A selected aligned double does not straddle a 64-byte line. The 800-byte row takes 13–14 such lines, depending on alignment. Tables only promise eight-byte alignment (`src/jit.rs:368`), so table boundaries and adjacent-table sharing can change the total line footprint.

A warm 100-tree model's entire leaf table fits M4 L1D with room for several rows. A 300-tree table does not, but the selected subset may. The full 100-tree table already exceeds Golden Cove L1D, although a repeated small set of leaf choices can fit. Both full tables fit either target's L2. The x86 constant pool adds 4,792/14,392 bytes read per pass. ARM64's separate emission has 100/300 leaf-base pointer loads; their data occupy only 800/2,400 bytes. Real MCJIT relocation may affect that pointer storage, so no separate MCJIT cache footprint was inferred here.

For warm data, emitted load work matters too. ARM64 reads 384 feature values, 100 leaf bases and 100 leaves: about 195 load-issue cycles at three loads/cycle. At 300 trees this is `(1137+300+300)/3 = 579`. Paired `ldp` reduces instruction count, not the number of values. x86 reads 384 features, 599 thresholds and 100 leaves, about 361 cycles; at 300 it is about 1,079. Stack and wrapper traffic add a little. These are optimistic resource floors, not complete runtime estimates.

Independent leaves allow misses to overlap. At today's roughly five M4 cycles per tree, a 20-cycle L2 hit needs about four concurrent misses to hide its latency. A 512-entry Golden Cove reorder buffer can see roughly ten current trees at once; its 16 line-fill buffers are ample for ordinary L2 overlap. `musttail` branches do not serialize the processor or drain that window. Cold DRAM is different. As an explicit sensitivity calculation, 100 independent 400-cycle M4 misses with 8–16 effective concurrent misses take at least 2,500–5,000 cycles; 300 take 7,500–15,000. For an assumed 250-cycle x86 memory latency the corresponding bounds are 1,563–3,125 and 4,688–9,375. These assumed latencies/concurrency levels are scenarios, not measurements of this machine. Code misses, constants, row data and queue contention add to them. Cold memory can outweigh all arithmetic changes.

### D2. After a cut to 1,500–2,500 instructions per 100 trees

Two sums are still sufficient for the pure scalar add dependency on these two cores. Their 104-cycle sum-and-tail floor is below the projected 150–313 ARM64 instruction-supply range, and below the 250–625 x86 range. At 300 trees the comparison is 304 versus 450–938 or 750–1,875. The important qualifications are load count, actual scalar add latency, leaf arrival order, code size and continuation arguments.

If scalar adds take three cycles in the complete kernel, the 156-cycle two-sum floor meets the most aggressive 150-cycle ARM64 instruction floor. Four sums provide margin in that case. For either core at a true 100-cycle kernel, the two-sum floor alone exceeds the budget; four sums would then be justified. It is unnecessary to provision enough sums to saturate every FP add unit: leaves arrive much more slowly than that.

For `S` sums, the current planner emits `S-1` serial `Action::AddSums` operations into sum zero (`src/sort_policy.rs:161–165`). With balanced reduction it would still emit `S-1` additions, but the depth would be `ceil(log2(S))`. Retirement is round-robin (`478–483`). `src/jit.rs:447–450` permits arithmetic reassociation, and `README.md:54–57` states the numerical contract. This permits proposing a different reduction shape, but does not prove that a new shape wins.

| Sums | Longest scalar path, 100 / 300 trees, 2-cycle add | Same with 3-cycle add | Add instructions including synthetic bias, 100 / 300 |
|---|---:|---:|---:|
| 1 | 202 / 602 | 303 / 903 | 101 / 301 |
| 2 | 104 / 304 | 156 / 456 | 102 / 302 |
| 4, current serial reduction | 58 / 158 | 87 / 237 | 104 / 304 |
| 4, proposed balanced reduction | 56 / 156 | 84 / 234 | 104 / 304 |
| 8, proposed balanced reduction | 34 / 84 | 51 / 126 | 108 / 308 |

These are longest possible scalar dependency paths with ready leaves. They exclude feature/selector/leaf startup and fetch. For counts divisible by S, the formula is `(N/S + reduction_depth + 1) * add_latency`. For eight sums use `ceil(N/S)`.

The claim that extra sums cost only registers is false for this implementation. At fixed FP9 the planner counts sums against resident features (`src/sort_policy.rs:502–513`). Increasing FP to compensate makes more live FP values cross continuation boundaries. The common signature contains every sum and the maximum feature-slot count (`src/jit.rs:114–125,428–435`). Two sums plus six feature slots give eight FP arguments. Four sums plus those six slots give ten, and both emitted targets pass excess values through the stack.

| Configuration, GP12 scalar | Planned feature loads 100 / 300 | ARM64 instructions 100 / 300 | ARM64 body stack accesses 100 / 300 | x86 instructions 100 / 300 | x86 body stack accesses 100 / 300 |
|---|---:|---:|---:|---:|---:|
| 1 sum, FP9 | 377 / 1,157 | 4,453 / 13,331 | 0 / 0 | 4,013 / 12,015 | 26 / 74 |
| 2 sums, FP9 | 394 / 1,192 | 4,453 / 13,338 | 0 / 0 | 4,015 / 12,033 | 28 / 76 |
| 4 sums, FP9 | 428 / 1,277 | 4,492 / 13,383 | 0 / 0 | 4,063 / 12,035 | 26 / 34 |
| 4 sums, FP11 | 394 / 1,192 | 4,501 / 13,483 | 42 / 139 | 4,122 / 12,360 | 127 / 400 |

Counts include pushes/pops and explicit stack memory operands, but exclude the call/return address traffic implicit on x86. FP11 keeps the action sequence identical to FP9/two sums except accumulator assignment/reduction; a probe assertion checked this at both sizes. It is a useful isolation experiment, not a recommended fix. `probes/emitted/arm-100-s4-fp11.s:95,204,289` stores/reloads carried doubles through `[sp]`; x86 lines 172–173 and 240 do the same through `%rsp`. The ARM wrapper frame also grows from 160 to 176 bytes.

Four sums at FP9 add 39/45 ARM64 instructions and 48/2 x86 instructions at 100/300 trees. Differences beyond two extra reduction additions come from changed scheduling, loads and allocation. Four sums at FP11 add 48/145 ARM64 instructions and 107/327 x86 instructions. This is evidence against changing only `accumulators` and assuming the rest is free.

Two-lane sums do not remove the work of gathering independently indexed leaves. The isolated `probes/sum_pair.ll` compares scalar updates with construction of a vector pair. `llc -O1` emits:

```asm
; ARM64 scalar updates: two ldr plus two fadd.
; ARM64 vector update:
add     x8, x1, x3, lsl #3
ldr     d1, [x0, x2, lsl #3]
ld1.d   { v1 }[1], [x8]
fadd.2d v0, v0, v1

# x86 AVX2 scalar updates: two memory-operand vaddsd.
# x86 vector update:
vmovsd  (%rdi,%rdx,8), %xmm1
vmovhpd (%rsi,%rcx,8), %xmm1, %xmm1
vaddpd  %xmm1, %xmm0, %xmm0
```

The probe deliberately returns the vector so the update cost is visible; a prediction still needs a horizontal reduction. ARM64 saves no update instruction per pair here. x86 adds one per pair, or 50/150 instructions, before the final horizontal reduction. If both leaves are already in FP registers, packing plus a vector add replaces two scalar adds with two instructions. M4's measured vector add latency is three cycles rather than scalar's two. A single vector sum therefore has a 150/450-cycle recurrence before reduction. Two vector sums shorten it, but pair formation delays the earlier leaf until its partner arrives. Prefer scalar sums unless another change already produces packed leaves.

A full pairwise reduction of all leaves has `ceil(log2(N))` addition depth, but requires retaining/reordering partial results across the schedule and continuations. Its nominal `N-1` additions do not pay for those transfers. No emission here verifies a win. A balanced reduction of four existing sums is the small experiment worth trying if four sums become necessary.

Finally, an instruction cut through constant-pool loads can expose load throughput first. Keep today's 384 feature loads, 100 leaves and 100 leaf-base loads, then add 600 scalar threshold loads: M4's load floor becomes about 395 cycles. That cannot yield a 200-cycle kernel even if its instruction count says otherwise. SIMD or binning could change that load count, but no such implementation is assumed in this study. Reduced-kernel bounds in section 3 show this uncertainty explicitly.

### D3. Continuations and wrappers

There are 37 tail boundaries, not 38. The C wrapper adds a call; the last continuation and wrapper each return. At 300 trees there are 112 tail boundaries. ARM64 capture lines 6–49 have ten paired saves, a row-pointer copy, a direct `bl`, ten paired restores and `ret`: 23 instructions, 92 bytes and a 160-byte frame. Bodies contain only five GP register-to-register copies in total at 100 trees, not a large shuffle at every boundary. No FP register copies or body stack accesses were found. Register assignment already places most carried values correctly.

The x86 premise is different. Capture lines 23–25 contain `movabsq $continuation0,%rax` then `callq *%rax`. Lines 168–171 show `movabsq $continuation1,%rax` then `jmpq *%rax`. All 37 tail transfers are indirect in this emission. The wrapper has 14 instructions, 34 bytes and five pushed GP registers, or 40 saved bytes plus return addresses. Thirteen continuations also push/pop `%rbp`. One body spills a GP value through `-8(%rsp)` at lines 5201 and 5268. Thus 26 save/restore accesses plus two spill accesses occur in continuation bodies. It is not stack-free on x86.

A predicted branch's execution latency is not a serial penalty paid at every boundary. Fetch predicts the next continuation before argument computations finish. Under warm prediction, budget roughly 19–37 branch-resource cycles for ARM64's 37 tails and about 37 for x86. Most overlap useful work already counted in the throughput bound. Source [M4 branch tests](https://jia.je/hardware/2025/05/21/apple-m4/) reports one to two taken branches per cycle; the [Golden Cove branch tests](https://old.chipsandcheese.com/2021/12/02/popping-the-hood-on-golden-cove/) find roughly one for this scale of branch stream. Neither is a measurement of these particular indirect tails.

The ARM wrapper needs 20 eight-byte stores and 20 loads. At two stores/three loads per cycle their issue work is about 10/7 cycles, with overlap. Include call/return and restore readiness for an estimated 10–20 cycles of entry/exit cost. x86's five saves/restores are smaller, but call dispatch is indirect and the bodies add stack traffic; use roughly 10–20 for the wrapper and another 5–20 potentially exposed cycles for body saves/spills. These estimates assume warm stack lines and correct predictions.

Saves at entry and restores hundreds of cycles later are usually served by L1 after stores drain. Do not charge a store-forwarding delay to each wrapper restore. The same applies to most `%rbp` save/restore pairs, which enclose a chunk. The single GP spill and the Rust result stores can expose memory dependencies; only measurement can assign their marginal cost.

For sensitivity, zero to one exposed fetch cycle per ARM64 boundary plus entry/exit gives about 10–60 incremental cycles per 100-tree call. For x86, allow zero to two exposed cycles per indirect boundary plus stack/entry work: about 15–115. These are envelopes, not measured losses, and must not be added to an instruction/byte bound as if all resource use were separate. The 112-boundary versions widen them to roughly 10–135 and 15–280. A benchmark with a different chunk limit is needed to isolate the effect.

At 500 cycles the low end is small; the high end is material. At 200 cycles even ten to twenty entry/exit cycles matter, and repeated fetch bubbles would be costly. Do not call continuation overhead noise. Do not remove continuations on this evidence either: `plans/implementation.md:179` records that the unsplit 100-tree backend delayed the first leaf until 571 comparisons and allocated 3,168 local stack bytes. The present emission loads its first leaf after 30 comparisons. Any boundary experiment must preserve that early leaf issue and check stack traffic.

Cold branch-target state adds unmeasured redirects. A simple sensitivity of one 15-cycle miss per call costs 15 cycles; one at every tail would cost 555/1,680. This is arithmetic to motivate a test, not a prediction that cold calls miss every target.

### D4. What the harness measures

It measures warmed dependent-call latency plus driver/API overhead. That dependence is useful: the next row address depends on the previous score, so independent predictions cannot overlap freely. It does not reproduce unrelated work between arrivals, instruction-cache eviction or the distribution of fresh feature values.

`examples/scalar_bench.rs:11–16,43–50` uses 50,000 calls and `next_row(index, prediction)`. Lines 18–40 verify all rows and the dependent trace before timing. Lines 120–129 warm each variant for 10,000 calls, then run five rounds and reverse variant order after each round. A printed minimum or median is a minimum or median of five *mean call times*, not a percentile of individual call latency. Alternating variant order reduces one source of drift; it does not measure tails.

The 35-row claim is correct but understates repetition. `probes/emission.log` reproduces the trace from the real reference predictions. At 100 trees it visits a 25-row transient and then a 10-row cycle. Of 50,000 measured calls, 49,975 follow that cycle. At 300 trees the new instance has an eight-row transient and a four-row cycle. The historical 13/3 distinct-row values belong to the 500-tree fixtures (`plans/implementation.md:155,165`), not to 300 trees.

The 256 allocated 100-feature rows occupy 204,800 bytes before allocation metadata. The recurring 100-tree rows occupy only 8,000 bytes. For leaf tables aligned to the line size, the ten-row cycle selects 398 distinct 64-byte leaf lines, or 25,472 bytes; all 256 rows select 743, or 47,552 bytes. The 300-tree four-row cycle selects 808 lines, or 51,712 bytes. These are alignment-specific logical footprints, not measured MCJIT cache occupancy. The probe also gives 128-byte group counts. They explain how a model larger than L1 can still behave warmly in the current trace.

Warmup trains code fetch, branch targets and frequently selected data. The trace hides the cost of unfamiliar leaves and many row lines. The 100-tree x86 code can fit L1I but cannot fit entirely in the micro-op cache. Thus “everything hot” means repeated code/data access, not guaranteed residency in every cache. Variant transitions can evict code, but the first few calls are diluted in a 50,000-call mean.

`CompiledModel::predict` checks `row.len() < feature_count` and calls its retained pointer (`src/jit.rs:29–38`). Inkwell 0.9's `execution_engine.rs:488–490,544–548` stores a function pointer and defines `call` as an always-inlined pointer call. There is no per-call LLVM lookup or compilation.

The checked API has more overhead than a comparison in the current executable. `probes/predict.disasm.txt` shows a separate 16-instruction successful path, a 32-byte save frame, one length branch, `blr x8`, result/discriminant stores, restores and return. `probes/driver.disasm.txt` shows the caller's `bl` to that method, result reloads and error check. The method is not inlined into this benchmark. The emitted JIT C wrapper is an additional frame. The result store followed by caller reload also differs from a raw C return. Roughly 10–25 exposed cycles for checked API dispatch/result handling is a reasonable warm estimate, not an isolated measurement. At a 100-cycle kernel it warrants measurement; it is not grounds to remove the row check or expose an unchecked API.

Section 3 gives a concrete new-example design. It times single checked predictions under separate data/code/arrival conditions and reports median/p99. No new production API is needed.

### D5. One timing run under load

`LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 cargo run --offline --release --example scalar_bench -- 100x100` exited zero. The full output is `probes/scalar_bench_100x100.log`. Times include the driver and checked API. Other study processes were active; differences of a few ns do not establish a winner.

| Variant | Minimum ns | Median ns |
|---|---:|---:|
| Reference | 273.50 | 281.06 |
| Tree order, 1 sum | 125.09 | 127.35 |
| Tree order, 2 sums | 122.47 | 132.02 |
| Tree order, 4 sums | 126.81 | 129.08 |
| Scheduled GP4/FP9, 1 sum | 120.06 | 124.98 |
| Scheduled GP4/FP9, 2 sums | 122.55 | 125.21 |
| Scheduled GP4/FP9, 4 sums | 124.17 | 124.76 |
| Scheduled GP6/FP9, 1 sum | 121.12 | 123.84 |
| Scheduled GP6/FP9, 2 sums | 118.17 | 121.68 |
| Scheduled GP6/FP9, 4 sums | 120.52 | 123.22 |
| Scheduled GP12/FP9, 1 sum | 117.64 | 120.15 |
| Scheduled GP12/FP9, 2 sums | 116.33 | 121.01 |
| Scheduled GP12/FP9, 4 sums | 116.08 | 121.72 |
| Scheduled GP20/FP9, 1 sum | 120.47 | 124.99 |
| Scheduled GP20/FP9, 2 sums | 123.93 | 124.86 |
| Scheduled GP20/FP9, 4 sums | 119.50 | 124.39 |
| SIMD2 GP4/FP9, 2 sums | 119.03 | 122.32 |
| SIMD2 GP6/FP9, 2 sums | 120.70 | 122.42 |
| SIMD2 GP12/FP9, 2 sums | 121.76 | 128.30 |
| SIMD2 GP20/FP9, 2 sums | 124.73 | 128.06 |
| Driver floor | 5.57 | 5.85 |

GP12 scalar medians of 120.15/121.01/121.72 ns for 1/2/4 sums do not show a current sum-chain problem. They also change feature residency, so this sweep is not an isolated accumulator-latency experiment.

## 2. Hypotheses and corrections

| Brief claim | Result | Correction supported here |
|---|---|---|
| H1: constant setup dominates code | Partly confirmed | Large static cost is real; 599 x86 constant addresses rather than 600. Replacing setup with loads changes data-port/cache bounds. This study did not test a different code model. |
| H2: feature residency is a small current lever | Partly confirmed | Current plans request 394 loads, but emitted code loads 384 feature values. Sums share the same FP budget. At much lower instruction counts, load issue is no longer automatically a small cost. The historical scheduling timing remains workload-specific. |
| H4: two sums become the next bottleneck | Partly refuted | The 50/150-add recurrence is real. Two-cycle scalar adds make its floor 104/304 including reduction/bias. It need not win at 1,500–2,500 instructions. Three-cycle sensitivity reaches the aggressive ARM floor; cold data and added constant loads can win first. |
| D1: six OR operations in series | Refuted as worded | Five bit combinations plus possible initial shift; no balanced selector reduction in the ARM probe. |
| D2: more sums cost registers only | Refuted | FP9 loses feature capacity; FP11 introduces stack-passed values on both targets. |
| D2: old four-sum sweep regressed | Too broad and stale | Tree order regressed, but old cap-4 four sums were 79.12 versus 79.68 ns for two (`plans/implementation.md:74–80`). It was a different, spilling backend. |
| D3: 38 direct tail boundaries, full frame on both targets | Refuted | 37 tails. ARM uses direct branches and a 160-byte frame; x86 uses indirect tails, a smaller entry frame, body `%rbp` saves and a GP spill. |
| D4: 35 distinct rows describe steady traffic | Incomplete | There is a 25-row transient followed by a ten-row cycle. |

H3, H5, H6 and H7's semantic equivalences were not investigated. This study only uses possible instruction/load reductions as scenarios; it does not certify those transformations.

## 3. Bounds and ranked recommendations

All cells below are cycles. Take the largest applicable resource/dependency bound, then account for exposed startup/tail stalls. Do not add all rows. Current counts are measured static emissions; reduced counts are the brief's projected range. The reduced 300-tree range is 4,500–7,500 instructions, a scaling assumption. No reduced implementation was timed.

| Current kernel bound | ARM64 100 trees | ARM64 300 trees | Golden Cove 100 trees | Golden Cove 300 trees |
|---|---:|---:|---:|---:|
| Instruction throughput | 445–557 at 8–10 IPC | 1,334–1,668 | 873–1,004 at 4–4.6 effective IPC | 2,616–3,008 |
| Sum dependency; add about 20–30 for first-leaf path | 104; 156 at 3-cycle add | 304; 456 at 3-cycle add | 104 | 304 |
| Front end | L1 bytes 278; 10-wide supply 445 | L1 bytes 834; supply 1,334 | L1 bytes 671; about 770–870 allocation/decode floor | L1 byte floor 2,003; roughly 4,005 with L2 code supply |
| Data cache / load issue | Warm issue about 195; L2-only leaves 250–500 | Warm issue about 579; L2-only leaves 750–1,500 | Warm issue about 361; L2-only leaves 188–375 | Warm issue about 1,079; L2-only leaves 563–1,125 |

The data-miss cells assume all selected leaves miss L1 and hit L2, at 20/15 cycles and 4–8 effective concurrent misses. They count leaves only; cold row, base-pointer and threshold lines add work. The warm issue cells count the current data loads. They are alternative inputs to a full miss/overlap model, not additive penalties.

| Projected reduced kernel bound | ARM64 100 trees | ARM64 300 trees | Golden Cove 100 trees | Golden Cove 300 trees |
|---|---:|---:|---:|---:|
| Instruction throughput | 150–313 at 8–10 IPC | 450–938 | 250–625 at 4–6 effective IPC | 750–1,875 |
| Two-sum dependency; add first-leaf path | 104; 156 at 3-cycle add | 304; 456 at 3-cycle add | 104 | 304 |
| Front end | L1 bytes 94–156; supply 150–250 | L1 bytes 281–469; supply 450–750 | L1 bytes 188–469; decode at least 250–417 | L1 bytes 563–1,406; about 1,125–2,813 if supplied from L2 |
| Data cache / load issue | 162–395 in the load scenarios below; cold-leaf estimate unchanged | 479–1,179; cold-leaf estimate unchanged | 162–361; cold-leaf estimate unchanged | 479–1,079; cold-leaf estimate unchanged |

Reduced x86 byte estimates assume 4–6 bytes/instruction, or 6–15 KB at 100 trees and 18–45 KB at 300. A 100-tree reduced kernel could fit the micro-op cache; capacity is no longer ruled out. At 300 trees the instruction count alone exceeds 4,096 and the larger byte cases exceed L1I. A packed instruction may still expand into several micro-ops. These facts need emitted-byte counts from pieces A/C, not an assumed linear speedup.

Reduced load ranges keep today's emitted row reload count. The optimistic end has row loads plus leaves, no leaf-base memory reads or threshold loads. The ARM upper end retains bases and adds one threshold load per split. The x86 upper end retains today's scalar threshold loads. These are explicit alternatives, not a prediction of one implementation. Sharing features more widely or vectorizing loads can lower the counts; packing/extraction and spills can raise them. On M4, constant loads could become the active limit before the sum chain. On x86, decode/fetch generally remains above the scalar chain; at the best reduced 100-tree case, load throughput can become comparable. After unrelated work, memory or code refill can dominate both targets.

1. Keep two round-robin scalar sums on both targets until a reduced kernel demonstrates a chain limit. This requires no production change and has zero instruction delta at 100/300 trees. If a measured M4 kernel approaches 100–150 cycles with warm data, compare four sums and a balanced final reduction. Changing two to four ideally costs two initialization and two final-add instructions per call, not per tree. Actual current deltas are +39/+45 instructions on ARM and +48/+2 on x86 with FP9; FP11 is worse because it adds stack transfers. A balanced four-sum tail keeps three reduction additions and saves one addition level, about two cycles on these cores. Risk is extra feature reloads, stack arguments and changed floating-point rounding. Do not blindly raise the floating budget to compensate.

2. Add a separate single-call benchmark before selecting an optimization from hot-loop timing. Kernel instruction delta is zero on either target, at either size. The change belongs in a new `examples/interleaved_bench.rs`; it does not require modifying `CompiledModel::predict`. It can expose a latency increase larger than the arithmetic savings, but no percentage can be predicted without the intervening workload. Timer cost, core migration and an unrealistic pollution pattern are the main risks. Use the concrete plan below.

3. Measure whether inlining the checked Rust API removes its extra call/frame/result transfers. An experiment can add `#[inline]` to `CompiledModel::predict`, then inspect the caller. Keep the length check and public `Result`. The ARM assembly suggests roughly 9–13 removable instructions per call across the method/caller, plus a 32-byte frame; actual compiler output must decide the number. That is roughly 2–3 instruction-supply cycles at either tree count, with potentially 5–15 more cycles from avoiding calls/result-memory dependencies. There is no measured x86 API emission, so its instruction delta is unknown; expect a constant per-call effect and measure it on x86. This ranks below the measured accumulator/footprint findings because no source-change probe tested it. Risks are caller code growth and a compiler that still retains some transfers. Do not substitute an unchecked API.

4. Ask piece A to include continuation addresses when testing an x86 code-model change. If addresses are within direct-branch range, replacing each `movabsq` plus indirect tail with a direct relative jump saves 37/112 instructions and about 259/784 executed bytes at 100/300 trees. The wrapper's address-plus-call could save one more instruction and about seven bytes. Alignment may change section savings. This is roughly 8/25 issue cycles at 4.6 IPC, plus any predictor benefit. ARM already has direct tails, so this proposal has zero ARM branch delta. `src/jit.rs:84` selects `CodeModel::JITDefault`; both separate emission and actual MCJIT relocation must be validated before recommending a replacement. Risk is invalid address-range assumptions. No code-model change was tested here.

5. Test larger continuation chunks only after lower instruction count is real. A hypothetical 32-test limit would give 19/57 chunks and remove 19/56 tails versus today. That saves at least 19/56 ARM instructions and 38/112 x86 instructions with today's indirect transfer sequence, before changed allocation. Ideal issue savings are about 2/6 ARM cycles and 8/24 x86 cycles. The potential cost is the old delayed-leaf/spill behavior, far larger than those savings. Count the first leaf's position and all stack traffic for every trial. This is an experiment, not a recommendation to remove the continuation design.

The new measurement example should accept `--trees`, `--features`, `--sums`, `--floating`, `--samples`, `--data-bytes`, `--code-bytes`, `--row-state`, `--row-sequence`, `--gap-us`, and `--output`. Define `--row-sequence shuffled` as the default and `dependent` as the current-driver control. Use the existing generator with depth six, scalar GP12, and explicit 100/300 sizes. Compile and validate every kernel before collecting samples. Preallocate output and pollution storage, touch all pages, and keep allocation/logging outside timed intervals.

For each sample, select the next input from a shuffled permutation of all 256 rows, then do the unrelated work. Run one checked `predict`, consume its output, and record one elapsed time. Include a separate mode using the current dependent row rule as the hot control. Single-call timing boundaries must prevent work from crossing the measured region. Do not create dependence by feeding the previous prediction into pollution-buffer addresses: that would silently change the unrelated workload.

Use data pollution sizes 0, 256 KiB, 2 MiB and 64 MiB. A shuffled line-access loop that reads and writes a value in every 64-byte region is a concrete starting point. Sweep these sizes instead of claiming exact eviction from size alone. The 256 KiB case exceeds both L1D capacities; 2 MiB exceeds Golden Cove L2 but fits M4's cluster L2; 64 MiB exceeds both. Complete the pollution accesses before starting the prediction timer. Record pollution duration separately. A random dependent walk is useful as a separate worst case; it is not a stand-in for every application's other work.

Data pollution alone does not evict L1 instructions or the micro-op cache. For a separate code condition, execute a prebuilt 256 KiB collection of unrelated instruction blocks, with real operations whose result is consumed. Report its measured text size and branch count. Do not merely read a 256 KiB byte array. A second, preferred condition runs the maintainer's actual intervening function with its actual working set. Keep a sequential instruction-block control and a branch-heavy control separate so code capacity and branch-target disturbance can be compared.

`--row-state produced` writes the 800-byte input after pollution, which represents freshly computed CPU features. `--row-state untouched` selects the row before pollution and leaves it untouched until prediction. These answer different use cases; neither is inherently correct. Gap trials at 0/10/100 microseconds test clock/scheduling effects separately from explicit pollution. A sleep is not a cache-eviction mechanism.

Report raw ns samples, p50, p90, p99, maximum, and counts for migrations/preemption indicators where available. Use at least 10,000 samples for each condition and randomize condition order across runs. Keep a paired empty timed region and a tiny checked JIT model to show timer/API floors, but do not subtract percentiles. On ARM use the platform clock initially and publish its quantization/overhead; the architectural timer frequency is not the core frequency. A 24 MHz timer has 41.7 ns ticks, which is too coarse to resolve a 100-cycle prediction precisely. If observed quantization is comparable to the call, use a supported cycle-counter or trace facility and report its measurement cost. Do not present tick-sized single-call samples as precise cycle percentiles. A batch timing is a separate control, not a replacement for single-call latency. On x86 use serialized timestamps only after validating their ordering and calibration. Hardware cycle/front-end/cache counters should be collected separately to distinguish the limits. A macOS QoS request can favor a performance core, but is not hard CPU affinity.

Expect the all-row hot control to differ from the ten-row cycle mainly through data locality. Data pollution should increase miss cost; code pollution should matter especially for x86. Four sums should not remove either penalty. A kernel with fewer instructions may have a better hot median but a smaller gain at p99 after pollution. None of these outcomes is a measured result of this study.

## 4. Commands and follow-up experiments

These commands replay the completed static probes without changing tracked files:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo build --offline --release --examples
CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml --bin emit-d -- probes/emitted
python3 probes/analyze.py
python3 probes/validate.py
/opt/homebrew/opt/llvm@21/bin/llc -O1 -mtriple=aarch64-apple-darwin -mcpu=generic probes/sum_pair.ll -o probes/sum_pair.arm.s
/opt/homebrew/opt/llvm@21/bin/llc -O1 -mtriple=x86_64-unknown-linux-gnu -mcpu=generic -mattr=+avx2 probes/sum_pair.ll -o probes/sum_pair.x86.s
/opt/homebrew/opt/llvm@21/bin/llvm-mca -mtriple=x86_64-unknown-linux-gnu -mcpu=alderlake -instruction-tables captures/100x100/x86_64-unknown-linux-gnu-scheduled-12gp-9fp-2sums.s
cargo run --offline --release --example scalar_bench -- 100x100
```

Expect identical baseline captures, the instruction/stack counts above, and successful reference checks. The probe emitter overwrites only its generated files under `probes/emitted`. The sum IR is an instruction-shape probe, not a full predictor or timing test. `llvm-mca` reports warnings about call/return simulation; only its instruction tables were used. Its CPU model contains approximations, as documented in the [LLVM Alder Lake model submission](https://reviews.llvm.org/D130959).

For a serial accumulator comparison, stop other study workloads and run the last command on several fresh processes. Compare minimum and median within each process, not medians across differently loaded runs. Expect no large 1/2/4-sum difference in the current kernel. After pieces A/C produce a reduced implementation, repeat its exact workload with FP9 and preserve the load/stack evidence. A four-sum win should recur without new spills; otherwise the chain hypothesis remains unsupported.

The commands below specify the proposed CLI for `examples/interleaved_bench.rs`. That example is **not implemented** by this study. Run them after the orchestrator accepts and writes the measurement design:

```sh
cargo run --offline --release --example interleaved_bench -- --trees 100 --features 100 --sums 2 --floating 9 --samples 10000 --data-bytes 0 --code-bytes 0 --row-state produced --gap-us 0 --output probes/hot-100.csv
cargo run --offline --release --example interleaved_bench -- --trees 100 --features 100 --sums 2 --floating 9 --samples 10000 --data-bytes 262144 --code-bytes 0 --row-state produced --gap-us 0 --output probes/l1-data-100.csv
cargo run --offline --release --example interleaved_bench -- --trees 100 --features 100 --sums 2 --floating 9 --samples 10000 --data-bytes 2097152 --code-bytes 262144 --row-state produced --gap-us 0 --output probes/interleaved-100.csv
cargo run --offline --release --example interleaved_bench -- --trees 300 --features 100 --sums 2 --floating 9 --samples 10000 --data-bytes 67108864 --code-bytes 262144 --row-state untouched --gap-us 100 --output probes/cold-300.csv
```

Repeat the same conditions at both tree counts and with one/four sums. Separate data-only, code-only and both-polluted trials. Compare current versus reduced kernels on identical rows and pollution seeds. Expected evidence is the change in p50/p99 plus cache/fetch counters, not a predetermined speedup. On a real Golden Cove machine, pin to a verified performance CPU and collect `cycles,instructions,branches,branch-misses,L1-icache-load-misses,L1-dcache-load-misses` with `perf stat` where supported. Use `perf list` to select DSB/MITE events available on that processor. Aggregates over pollution and prediction together cannot attribute misses to prediction; use separate regions or sampling attribution.

For the checked-API experiment, change only `#[inline]` on `CompiledModel::predict`, run `cargo test --offline --release`, rebuild the examples, and compare caller disassembly and the same serial hot/interleaved trials. The success condition is retained row checking and fewer actual call/frame/result instructions. No gain should be assumed from the attribute alone. This study made no such source edit and has no tracked diff to replay.

## 5. Questions that change the recommendation

- Which x86 core and ISA are deployment targets, and is a performance core guaranteed? Golden Cove client is the representative model here; AVX-512 deployment requires a different concrete target.
- What code and data run between predictions, for how long, and do they share the prediction core? This decides whether the hot instruction bound or refill cost is useful.
- Does the CPU just produce the feature row, or does prediction consume a row that has sat in memory? This decides the row-state condition.
- Is the target a median, p99 or deadline, and are preemption and wakeup latency part of it?
- Can representative models and rows provide tree depths, border reuse and selected-leaf distributions? Depth eight quadruples the table bytes versus depth six, which can change the cache recommendation.
- Is the documented reassociated floating-point result contract still sufficient? Balanced/vector reductions can change rounding; exact integer synthetic leaves do not test that sensitivity.

No question above blocks this study. The files and build environment were sufficient. Apple’s full optimization guide required developer access and several web PDF/detail endpoints failed; the report uses accessible primary measurements and labels estimates. The local sysctl permission error was treated as a sandbox artifact. No implementation recommendation was applied.
