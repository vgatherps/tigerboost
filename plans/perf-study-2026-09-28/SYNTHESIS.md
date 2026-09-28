# tigerboost performance study: synthesis and design proposal

Date 2026-09-28, commit da93993. Eight studies (opus 5.5 and gpt-6-astra on each of four
pieces) plus serial timings run by the orchestrator on a quiet M4 Pro. Reports:
`REPORT_{A,B,C,D}_opus.md` here; astra reports in `~/src/tigerboost-astra-{A,B,C,D}/probes/`.
Serial timing outputs in `serial/`. Goal: single-row latency, <300 trees, depth ~6, one row per
call, NEON and AVX2 only (no AVX-512 dependence), thresholds and leaves addressed off a base
register (decided).

## 1. What the kernel is bound by today (measured, M4)

100 trees / 100 features, production default: 121 ns, 4453 instructions, ~490 cycles, IPC ~9,
at 90% of the 10-wide rename limit. Three limits sit close together (opus D microbenchmarks):

| limit, 100 trees | cycles | why |
|---|---:|---|
| rename width | 445 | 4453 instr / 10 |
| load + `fmov d,x` pipes (shared, 3/cycle) | 393 | 384 feature loads + 600 `fmov` + 100 leaf + 100 GOT |
| flag-compare port (`fcmp` 2/cycle) | 300 | 600 scalar compares |

x86 (static, no execution): `ucomisd` is 1/cycle on Zen 4 / Golden Cove, so any scalar kernel
floors at 600 cycles per 100 trees. The 300-tree kernel is 64 KB, over the 32 KB L1I and far over
the uop cache; front end dominates. `JITDefault` = large code model: every constant is
`movabsq` + memory operand, every chunk transfer is `movabsq` + `jmpq *reg`.

Only mask-producing compares escape the compare-port cap: `fcmgt.2d` runs 4/cycle and does 2
splits; `vcmppd ymm` does 4 splits and folds the threshold operand.

## 2. Corrections to the starting hypotheses

- Register file is not the limit; `preserve_nonecc` ARGUMENT registers are. Both targets pass
  only 8 FP arguments (d0-d7 / xmm0-7); x86 passes 12 GP. Everything live at a chunk boundary
  is a `musttail` argument. FP9 with 2 sums fills the 8 FP args exactly; every FP budget above 9
  turns row loads into stack loads+stores; GP≥14 fails register allocation on x86 under the
  large model. (astra B, opus B, opus D, opus A, all independently.)
- Residency is worth ≤6-9% of instructions and none of it is reachable in registers with the
  current chunk signatures. Scheduler is worth 2.5% by count, 2-5% timed on ARM. But loads are
  not free on M4 (pipes ~80% busy), so residency matters again once instruction count drops.
- GP4/FP9 (one active tree, feature-aware admission order) emits fewer instructions than GP12
  and is spill-free on x86; timed equal or better at every emitter level (serial table below).
- Threshold immediates: synthetic dyadic thresholds cost 3 instructions on ARM64; real f32-derived
  borders cost 4 (three `movk`). So R1 below is worth more on real models.
- `early-cse` folds loads from a private constant global back into immediates. The constants
  block must be reached through a pointer ARGUMENT (opaque), not a global. Confirmed by both A.
- Small code model under MCJIT: unsafe as configured. Static small emits 100 absolute
  `R_X86_64_32S` leaf relocations; PIC needs ±2 GB placement the default memory manager does not
  promise. Not needed once constants come through a pointer.
- NaN: `AsIs` ≡ `AsFalse` under strict `>`; NaN threshold ≡ +inf threshold (constant false bit);
  `AsTrue` stays +inf and is false against +inf/NaN borders. Ordered vector compares already give
  the reference result for NaN inputs, so no load-time normalization is needed for AsIs/AsFalse.
  Verified by native runs (opus C 1,000,000 rows incl. specials; astra A 192 cases; astra C 2740).
- Leaf-table permutation is free and executed: the NEON `sli` reduction gives a two's-complement
  3-bit value per lane; store the table with bias +36 (range [-36,27]). Verified natively.
- 2-cycle `fadd` on M4 and Golden Cove; sum chain (104/304 cycles at 2 sums) is hidden under
  every scalar kernel and stays hidden until mask compares land. 1/2/4/8 sums time identically
  today. Four sums at FP9 steal feature slots; at FP11 they spill (argument limit).
- Cold calls are 6-10x the hot number and grow linearly with tree count; leaf-table misses
  dominate and do not overlap (leaf loads ~45 instructions apart, ROB spans ~13 trees).

## 3. Serial measurements (quiet machine, medians, ns per call incl. ~5 ns driver)

Production emitter with opus A's probe diff applied (`opus_probes_A/probe.diff`, all integration
tests pass in every mode; only the three IR-text unit tests fail):

| emitter | 100 trees GP12 | 100 GP4 | 100 tree order | 300 GP12 | 300 GP4 | 300 tree order |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 120.6 | 116.1 | 125.0 | 344.7 | 336.6 | 356.0 |
| + constants block via pointer arg | 102.8 | 103.1 | 116.4 | 284.4 | 279.9 | 323.9 |
| + per-chunk rebased pointer (`ldp` pairs) | 94.7 | 94.1 | 95.3 | 261.1 | 259.1 | 267.8 |
| + inline-asm carry chain (`fcmp t,x; adc`) | 87.8 | 86.8 | 90.1 | 238.1 | 236.8 | 249.9 |

Instruction counts (ARM64 / x86, 100 trees): 4453/4015 → 3189/3330 → 2959/n.a. → 2275/1944.
The x86 carry form is `ucomisd next_up(t); adcq idx,idx` via `llvm.x86.addcarry.64`; astra A's
alternative is `lea; ucomisd; cmovbe` (2558 combined, no `next_up`, no flush-to-zero caveat).
Neither escapes the 1/cycle `ucomisd` cap.

Whole-model generated NEON kernels (opus C `gen.py`; base pointer everywhere; every tree loads its
own six features from the row, no cross-tree reuse; chunks of 4 trees, plain calls):

| model | scalar | tree-major (`fcmgt.2d`+`sli`) | binarize (`tbl`+`cmhi`+`addv`) |
|---|---:|---:|---:|
| 100x100, all thresholds distinct | 111.4 | 93.5 | 102.1 |
| 300x100, all distinct | 329.4 | 263.3 | 296.0 |
| 300x100, 16 borders per feature | 339.6 | 267.4 | 246.1 |
| 300x100, 8 borders per feature | 341.1 | 269.3 | 229.2 |

Cold-call probe (opus D `d_probe`, production kernel, per-call `Instant` samples, 41.7 ns ticks):

| condition | 100 trees p50 / mean | 300 trees p50 / mean |
|---|---:|---:|
| hot | 125 / 122 | 334 / 352 |
| after 4 MiB read | 125 / 138 | 375 / 364 |
| after 16 MiB read | 209 / 349 | 667 / 726 |
| after 64 MiB read | 791 / 1174 | 2000 / 2135 |
| after 64 MiB, code re-warmed, leaves cold | 542 / 561 | 1833 / 1846 |
| after one call of a 2000-tree model | 292 / 300 | 917 / 935 |

## 4. Reading the numbers

On NEON the scalar carry kernel (87.8) and the tree-major kernel (93.5) land in the same place,
~370 cycles for 100 trees, because they hit different caps: scalar is at the 300-cycle compare
floor plus load pipes; tree-major has no compare cap but pays 6 lane loads per tree (600 loads +
300 `ldp q` threshold loads + 100 leaf ≈ 333 load-pipe cycles). Tree-major has headroom the
scalar form does not: features already in registers turn `ld1 {v.d}[1]` (load pipe) into
`mov.d v[1],v[0]` (vector pipe), 22 → 16 instructions per tree and ~40% fewer loads. Scalar
cannot go below 300 cycles per 100 trees on M4 or 600 on Zen 4 / Golden Cove, ever.

On x86 tree-major is the clear pick: 14.4 instructions per tree with folded threshold operands
and `vmovmskpd`, versus 19-25 scalar under a 1/cycle compare cap. llvm-mca (rough): Alder Lake
1206 → 394 cycles, Zen 3 705 → 423 at 100 trees. 300-tree code shrinks from 64 KB to roughly
25-30 KB, inside L1I. Needs `target-cpu=x86-64-v3` (or host features) on both the machine and the
function attributes; SSE2-only still gives 20/tree with `cmpltpd`+`movmskpd`.

Binarize (bins as bytes, `tbl`/`vpshufb` gather, one byte compare per tree) is exact and wins
on NEON only when borders are shared: 229 vs 269 ns at 8 borders/feature, 246 vs 267 at 16,
loses 296 vs 263 when all 1800 thresholds are distinct. On AVX2 it leads tree-major by count in
every case (12-39%) but the `vpshufb` 16-byte-lane gather scales with used features (≤ ~128
features to stay ahead). Its per-feature prologue is a serial phase before any tree starts. It is
a conditional second formulation chosen from border statistics, not the default.

Cold calls: if production calls arrive with cold caches, the hot numbers are not the metric.
Leaf-table misses dominate (0.5-0.8 µs at 100 trees with code warm) and do not overlap. Cutting
instructions per tree helps here too (more trees inside the ROB → more misses in flight), and
tree-major/binarize both shorten the distance between leaf loads. Batching leaf loads (compute
all indices, then load all leaves) is the direct lever; f32 leaves would halve the lines touched.

## 5. Proposed design

1. Constants block through a pointer argument (decided). One private constant global:
   thresholds in plan order, then leaf blocks (padded to 64-entry alignment). Wrapper takes its
   address once; every continuation carries it. ARM64 additionally carries a per-chunk rebased
   pointer so `ldp d`/`ldp q` immediates reach (7-bit scaled range). x86 folds
   `disp(%base,%idx,8)` directly. Leaf offset added (not OR-ed) into the index. Always emit
   wrapper + continuation even for one chunk so the pointer stays opaque to `early-cse`.
   Measured alone: 121 → 95 ns (100 trees), 345 → 261 (300).

2. Tree-major vector compares on both targets, same plan shape.
   NEON per tree: gather 3 pairs (`ldr d`+`add`+`ld1.d[1]` from the row, or `mov.d` from a
   resident register), `ldp q`+`ldr q` thresholds off the rebased pointer, 3 `fcmgt.2d`,
   2 `sli.2d` (intrinsic `llvm.aarch64.neon.vsli.v2i64`), `mov.d`+`fmov`+`add lsl #3`,
   `ldr d`+`fadd`. Leaf table stored with the +36 bias permutation.
   AVX2 per tree: `vmovsd`/`vmovhpd`×2/`vinsertf128` gather, `llvm.x86.avx.cmp.pd.256(x,mem,30)`
   ×2 (4+2 lanes; second narrows to xmm), `vmovmskpd`×2, `shl`+`lea`, `vaddsd mem`. Identity leaf
   order. Generic `fcmp ogt` does NOT fold the threshold operand; the intrinsic does.
   Plan: `Action::Compare` becomes one action per tree carrying lane order, threshold block
   offset, leaf block offset and permutation, and its sum. Feature-major `Load`/`Release`
   bookkeeping survives only as "which features are resident in registers for `ins`".
   Chunks cut at tree boundaries, ~8 trees (48 tests) per chunk; state across a boundary is
   the two pointers, the sums, and ≤6 resident features. This is why the old GP/FP budget
   machinery can go: there are no partial selectors to carry.
   Expected NEON: ~93 ns at 100 trees without residency, lower with it; x86 ~3x by mca.

3. Scheduler: keep tree order with feature-overlap admission (the GP4/FP9 behaviour, which is
   what measures best), drop `RegisterBudget` and `ComparisonMode::Simd2` from the API. Planning
   cost then drops from 85 ms to <1 ms at 300 trees. Residency decisions are made within a chunk
   against the 32 NEON / 16 xmm registers, not across chunks.

4. Sums: keep 2 round-robin scalar sums. Re-measure 4 (as two `<2 x double>` on NEON, four
   scalar xmm on x86) once tree-major lands; the chain becomes visible only below ~700 cycles at
   300 trees on M4.

5. NaN: drop the load-time `uno`+`select` for `AsIs` and `AsFalse` (identical); fold NaN and
   +inf thresholds to constant-false bits at compile time; `AsTrue` either keeps the +inf
   substitution (rare) or uses an unordered-greater compare (`fcmge` reversed / `vcmpnle`).

6. Binarize as a compile-time-selected alternative once real border statistics exist: choose
   it when Σ borders per feature pair is small relative to splits (crossover on NEON between
   8 and ~24 borders per feature at 300 trees; on AVX2 always ahead by count if used features
   ≤ ~128). Needs a new per-feature prologue action and vector bin state across continuations.
   Not first.

7. Measurement: add the cold-call harness (opus D's `d_probe` shape: pollution sizes 256 KiB /
   4 / 16 / 64 MiB, other-model call, per-call p50/p90/p99, timer-only control) beside
   `scalar_bench`. Both hot and cold columns gate every change from here on.

Not recommended: Small code model (unsafe under default MCJIT memory manager; unnecessary with
the pointer); backend `Default`/`instcombine`/`reassociate` (0 to -2%); unlimited chunks (3 KB
of spills persist even after the emitter changes); SIMD2 pairing (loses to scalar once constants
are cheap); raising FP/GP budgets (stack arguments).

Fallback data point: the scalar carry chain (`fcmp t,x; adc x,x,x` via inline asm on ARM64,
`addcarry` intrinsic on x86) measured 87.8 ns and is a small emitter change with leaf-table
reordering, but it is inline asm, capped by the compare port, and diverges from the x86 shape.

## 6. Open questions that change the ranking

1. Deployment: ARM64, x86 AVX2 guaranteed (x86-64-v3), or SSE2 baseline? Host-feature JIT
   acceptable? Zen 4/5 or Intel (Zen 5 has a 0.5-cycle `ucomisd`, changes scalar math)?
2. Hot or cold: what runs between predictions, and is the target p50 or p99? Cold p50 is 6x hot.
3. Real model: distinct thresholds per feature after merging, features used, depth range, NaN
   modes present, are borders f32-derived (they are in CatBoost)?
4. f32 borders/features would allow 4-lane NEON compares and halve binarize prologue and leaf
   lines; a numeric-contract change, so a question, not a recommendation.
5. Inline asm acceptable on ARM64 (only needed for the carry fallback)?
6. Would the caller warm the model (one throwaway call) at request start? Recovers ~1/3 of the
   cold cost at 100 trees.
