# Results of the 2026-09-28 experiments

Apple M4 Pro, LLVM 21.1.8, generic target, offline release builds. Hot numbers are
`scalar_bench` medians of five rounds of 50,000 dependent calls, ns per call including the
~5.5 ns driver. Cold numbers are `cold_bench` per-call percentiles over shuffled rows, 5,000
samples per condition, 41.7 ns timer ticks. Every kernel is checked against `Trees::predict` on all
256 rows before timing. Runs were serial; load average 2 to 3.5 (a build from another project was
running on the machine), so treat differences under about 3% as noise.

## Step 1: constants block through a pointer argument (commit a487af0)

| hot median, ns | 100 trees | 500 trees |
|---|---:|---:|
| before, scheduled GP12/FP9, 2 sums | 120.6 | 572 (historical) |
| after, scheduled GP12/FP9, 2 sums | 92.1 | 433.4 |
| after, scheduled GP4/FP9, 2 sums | 92.0 | 421.0 |

| cold p50, ns (shuffled rows) | 100 trees before → after | 300 trees before → after |
|---|---:|---:|
| hot | 125 → 83 | 334 → 250 |
| after 64 MiB read-modify-write | 791 → 500 | 2000 → 1917 |
| after one call of a 2000-tree model | 292 → 208 | 917 → 708 |

Static, 100 trees: ARM64 4453 → 2963 instructions (300 `ldp` threshold pairs, no continuation stack
traffic); x86-64 4015 → 3330 (no `movabsq` to constants). Cold improved as well as hot: the code
shrank from 17.8 KB to 11.9 KB and the 100 per-tree GOT loads disappeared, which outweighs the
threshold lines the block adds.

## Step 2: tree-major order (commit 97fb5b0), same run for every row

| hot median, ns | 100 trees | 300 trees |
|---|---:|---:|
| scalar, scheduled GP12/FP9, 2 sums (default) | 91.9 | 260.5 |
| scalar, scheduled GP4/FP9, 2 sums | 93.5 | 254.9 |
| scalar, tree order, 2 sums | 96.3 | 266.7 |
| tree-major, 2 trees per chunk | 96.4 | 262.6 |
| tree-major, 4 trees per chunk | 92.5 | 252.4 |
| tree-major, 8 trees per chunk | 86.8 | 236.6 |

Second run, chunk sweep (scalar default measured again in the same run: 94.3 / 260.9):

| tree-major, trees per chunk | 100 trees | 300 trees | ARM64 continuation stack accesses | x86 |
|---|---:|---:|---:|---:|
| 8 | 88.0 | 239.2 | 0 | 4 |
| 16 | 86.5 | 230.6 | 43 | 188 |
| 32 | 89.6 | 241.7 | 277 | 529 |
| 64 | 87.0 | 237.5 | 371 | 753 |

| cold p50 / p90 / p99, ns, 100 trees | scalar default | tree-major, 4 per chunk |
|---|---|---|
| hot | 83 / 125 / 125 | 83 / 125 / 125 |
| after 16 MiB | 209 / 250 / 333 | 250 / 292 / 334 |
| after 64 MiB | 792 / 916 / 2125 | 792 / 917 / 1625 |
| after other-model call | 209 / 250 / 291 | 208 / 209 / 250 |

| cold p50 / p90 / p99, ns, 300 trees | scalar default | tree-major, 4 per chunk |
|---|---|---|
| hot | 250 / 292 / 292 | 250 / 250 / 292 |
| after 16 MiB | 584 / 625 / 750 | 666 / 792 / 1250 |
| after 64 MiB | 2084 / 2250 / 6250 | 2125 / 2291 / 5833 |
| after other-model call | 708 / 750 / 792 | 542 / 584 / 625 |

Static, tree-major 4 per chunk, 100 trees: ARM64 2253 instructions (22.5 per tree; 300 `fcmgt.2d`,
200 `sli`, 238 `ld1` lane loads, 106 `ldp q` + 88 `ldr q` threshold loads), x86-64 2332 (SSE2
`cmpltpd`; the AVX2 `vcmpgt_oqpd` + `vmovmskpd` path is not built yet).

## Reading

- Tree-major at 8 trees per chunk is the largest spill-free configuration and is 6% (100 trees) to
  9% (300 trees) faster hot than the scalar default; 16 per chunk is 8% / 12% faster with a few
  spills. The reviews' prediction (8-12% headroom on NEON because the load pipes become the limit)
  holds.
- Cold is neutral: equal after 64 MiB, better after an other-model call (smaller code), slightly
  worse after 16 MiB at 300 trees (one to two timer ticks).
- The x86 side is untested at runtime; static counts fell 4015 → 2332 and the compare port cap
  (`ucomisd` 1/cycle) is gone, which is where the larger gain is expected.

## Step 3: Rosetta 2 correctness gate and AVX2 emission (commits 716073e, 0a1187e)

`tools/x86_check.sh <workload> [--avx2]` emits every variant for `x86_64-apple-darwin`, links each
with a C driver (`-no_pie`, because the large code model leaves absolute text relocations), runs it
under `arch -x86_64`, and compares all 256 predictions bit for bit with `Trees::predict`. All five
variants pass on `special`, `100x100` and `300x100`, with and without AVX2. Rosetta timing carries
no information (see `ROSETTA_AVX2_RESEARCH.md`); native ARM64 remains the semantic oracle.

| x86-64 static, 100 trees, 100 features | instructions | per tree | compare form |
|---|---:|---:|---|
| scalar default (GP12/FP9, 2 sums) | 3330 | 33.3 | `ucomisd`, 1 per cycle on Zen 4 / Golden Cove |
| tree-major 4 per chunk, SSE2 (generic) | 2332 | 23.3 | `cmpltpd` register operand, shift/or packing |
| tree-major 4 per chunk, AVX2 (`EmitOptions { avx2: true }`) | 1486 | 14.9 | `vcmpgt_oqpd` memory operand, `vmovmskpd` |

### Rosetta 2 wall-clock (translator speed, not x86 speed)

100 trees, dependent calls, best of five rounds of 200,000, ns per call, `arch -x86_64` on the M4 Pro.

| kernel | SSE2 build | AVX2 build | native ARM64, same shape |
|---|---:|---:|---:|
| scalar default (GP12/FP9, 2 sums) | 168.5 | 190.7 | 91.9 |
| tree-major, 4 per chunk | 107.9 | 408.8 | 92.5 |
| tree-major, 8 per chunk | 105.9 | 402.6 | 86.8 |

Rosetta's SSE2 translation lands within 1.2x of native ARM64 for the tree-major kernel and keeps the
scalar-versus-tree-major ordering. Its AVX2 translation is about 4x slower than its own SSE2 build,
which says nothing about a real x86 core and is why no AVX2 speed claim exists.

No native x86 timing exists. The llvm-mca estimate from the study (about 3x over scalar on Alder
Lake, 1.7x on Zen 3) is the only performance indication for this path.

## Step 4: tree-major only (commit 04475f8) and the two emission-order experiments (branch `impl/emission-experiments`, not merged)

The scheduler, register budgets, two-lane threshold pairing, feature Load/Release actions,
comparison-count partitioner and carried slots were deleted (net about 2200 lines); tree-major
emission stayed byte-identical on both targets with and without AVX2.

Two opt-in emission-order changes were then built and measured on the simplified emitter:
issuing a chunk's leaf loads after all its compares (`batched_leaves`), and reusing a feature
already loaded inside the chunk instead of loading it again (`reuse_features`). Result: with
`reuse_features` the final assembly is byte-identical to the default on ARM64, generic x86 and
AVX2, because `early-cse` already merges duplicate loads within a function; the plan-level reuse
is 132 of 600 lane fills (22%) and LLVM was already taking it. With `batched_leaves` the
assembly differs by two to four instructions because the machine scheduler already reorders the
independent leaf loads. Neither option is merged.

Hot medians in that run (ns, 100 / 300 trees): tree-major 8 per chunk 90.7 / 240.1; batched
89.7 / 238.9; reuse 91.6 / 242.4; both 89.0 / 245.4; 16 per chunk 88.2 / 232.6. Cold p50 after
64 MiB for the four identical-or-nearly-identical kernels: 791, 750, 875, 833 ns. That spread on
identical code is the noise band of `cold_bench` on this host: cold differences under about
150 ns at p50 carry no information.

Remaining load counts, 100 trees, 8 per chunk, from the astra final review's classification
(`VERDICT_FINAL_astra.md`): ARM64 feature loads are 216 `ldr d`, 39 `ldp d`, 172 `ld1.d` and 2
`ld1r.2d` (468 feature doubles, no duplicate within a chunk); thresholds 137 `ldp q` + 26 `ldr q`
(300 vectors); leaves 100 `ldr d`; the remaining `ldp` are wrapper restores. x86 SSE2 feature operands
291 `movsd` + 174 `movhpd` + 5 `movupd`; AVX2 274 `vmovsd` + 161 `vmovhpd` + 27 `vbroadcastsd` + 3
`vmovddup` + 3 `vmovupd` + 2 wide operands, with seven doubles re-read within chunks by backend
packing choices. Per tree: ARM64 about 8.3 load slots (4.3 feature, 3 threshold, 1 leaf), SSE2 8.85,
AVX2 7.7. The sum over chunks of distinct features is 468, which is what tree ordering targets.

## Do continuations still help under tree-major?

Chunk-size sweep on the tree-major kernel, 100 / 300 trees, hot medians under a load average of
about 3.5 (indicative), plus exact static counts for 100 trees:

| trees per chunk | hot 100 / 300 ns | ARM64 instr | ARM64 body stack accesses | x86 AVX2 instr | x86 AVX2 body stack accesses |
|---|---:|---:|---:|---:|---:|
| 8 | 91.4 / 254.6 | 2143 | 0 | 1451 | 0 |
| 16 | 88.9 / 242.0 | 2133 | 43 | | |
| 64 | 91.8 / 247.2 | | 371 | | |
| one function | 90.4 / 244.3 | 2459 | 422 | 1868 | 733 |

On the M4 the hot time does not change between eight trees per chunk and no chunking at all,
although the unchunked kernel is 15% larger and makes 422 stack round trips: store-to-load
forwarding hides them. Sixteen per chunk is the fastest point. The chunks therefore do not buy
measured speed on ARM64; they keep the kernel spill-free and smaller, which matters for the cold
case and for x86, where the unchunked AVX2 kernel grows 29% with 733 stack accesses and cannot be
timed here. Keep chunking; 16 per chunk is the better default candidate.

## Fenced single-function kernel with JIT-managed feature lifetimes (branch `impl/fenced-single-function`, not merged)

One function, a compiler fence (`asm sideeffect` with `~{memory}`, row and constants pointers
threaded through as tied operands) at every chunk boundary, no `early-cse`, and an explicit reuse
window W in trees: a loaded, normalized feature value is reused for any lane within the next W
trees of visit order, across fences. 100 trees, 100 features, 8 trees per chunk. Full tables in
`VERDICT_fenced_astra.md`.

| ARM64 | instructions | row loads | threshold loads | stack accesses in kernel | frame bytes |
|---|---:|---:|---:|---:|---:|
| continuation kernel (today) | 2143 | 429 | 163 | 0 | 160 |
| fenced W1 (fence only, no reuse) | 2278 | 430 | 273 | 66 | 304 |
| fenced W8 | 2280 | 389 | 275 | 76 | 320 |
| fenced W16 | 2432 | 298 | 275 | 265 | 672 |
| fenced W32 | 2575 | 204 | 276 | 448 | 1232 |
| fenced W16, shared-feature order | 2299 | 260 | 273 | 169 | 448 |
| fenced W16, one chunk, no fence | 2445 | 74 | 270 | 421 | 1744 |

| x86 AVX2 | instructions | row loads | stack accesses |
|---|---:|---:|---:|
| continuation kernel | 1451 | 470 | 0 |
| fenced W1 | 1434 | 470 | 0 |
| fenced W8 | 1425 | 428 | 0 |
| fenced W16 | 1597 | 326 | 220 |
| fenced W32 | 1708 | 229 | 399 |

Hot medians, ARM64, indicative (ns, 100 / 300 trees): continuation 89.7 / 241.9; fenced W8
92.4 / 244.9; W16 90.7 / 253.7; W32 91.8 / 252.4; W16 shared order 92.4 / 236.0; W16 one chunk
89.9 / 236.4. Cold p50 unchanged within the harness noise band.

What it says:
- The fence works: no load merged across a boundary in IR or assembly on any target.
- Removing the continuation transfers changes nothing measurable: W1 is the continuation kernel
  minus the transfers, and it is not faster.
- Explicit reuse beyond about eight trees does not save loads, it converts them into spills: every
  row load removed past W8 reappears as a store plus a reload of the feature value (W16 spills 58
  feature values and 80 hoisted leaf values on ARM64). The register allocator, not the emitter,
  decides what stays in a register, and the resident working set it can hold is roughly what one
  chunk already reuses. Within-chunk reuse was already there through `early-cse`.
- The ARM64 fenced kernels lost `ldp q` threshold pairing (163 → 273 threshold loads) because the
  per-chunk base is no longer an opaque argument; an implementation artifact, fixable, but it does
  not change the ranking.
- AVX2 fenced W8 is 26 instructions smaller than the continuation kernel with no spills; untimed.

Decision pending: keep continuations (recommended: they are the ABI-enforced residency bound and
cost nothing measurable), or adopt fenced W8 on x86 only.

## Volatile feature loads with emitter-scheduled residency (branch `impl/volatile-residency`, round 1)

One function, no fences, every row feature load `volatile`, Belady residency against a budget K
over the visit order (evict the resident feature whose next use is farthest). Row loads equal the
simulated misses exactly on every target. Tables in `VERDICT_residency_astra.md`.

| row loads, 100 / 300 trees | K8 | K12 | K16 | K20 | K24 | continuation |
|---|---:|---:|---:|---:|---:|---:|
| model order | 423 / 1242 | 375 / 1087 | 338 / 964 | 309 / 866 | 281 / 780 | 468 / 1419 |
| shared-feature order | 347 / 936 | 299 / 789 | 268 / 681 | 248 / 611 | 228 / 550 | |

| ARM64, 100 trees | instructions | row loads | body stack ops | hot ns (100 / 300) |
|---|---:|---:|---:|---:|
| continuation | 2143 | 468 | 0 | 88.9 / 242.1 |
| K8 | 2320 | 423 | 140 | 108.9 / 313.1 |
| K16 | 2299 | 338 | 136 | 102.1 / 287.5 |
| K16 shared | 2267 | 268 | 131 | 102.7 / 285.0 |

x86 AVX2 (static): K8 1681 instructions, 423 row loads, 0 spills; K12 1663 / 375 / 0; K16 1734 /
338 / 89 spills. SSE2: K8 2662 / 423 / 0. Continuation: 1451 / 470 / 0.

What it says:
- The scheduled residency does exactly what the theory says to the load count (−28% at K16
  model order, −43% at K16 shared order, both vs the continuation kernel).
- On ARM64 the single-function kernel spills at every K, K1 included, and what it spills are
  hoisted leaf values (and, with volatile leaves, other hoisted loads), not features: LLVM's
  ARM64 scheduler hoists independent loads across the whole function and the allocator spills
  them. Threshold `ldp q` pairing is also lost (163 → 268 threshold loads) because the base is
  no longer an opaque argument. Net: 13 to 15% slower hot on ARM64 despite fewer row loads.
- On x86 there is no such spilling until K12 to K16, and K8 already beats the continuation
  kernel's load count with zero spills; +15% instructions come from lane inserts that no longer
  fold into `vmovhpd m64`. Untimed natively.
- The round-2 design combines both experiments: fence per chunk (scheduler containment, which
  continuations provided), constants base laundered through the fence (pairing), volatile
  features, Belady residency across fences.

## Cross-boundary feature reuse: six attempts, all rejected on ARM64 time

Each row below changed several things at once, so none isolates one cause. They are recorded as
evidence, not as clean ablations. ARM64 hot medians at 100 / 300 trees against the continuation
kernel measured in the same run. Full reports are the `VERDICT_*` files named.

| attempt | what it bundled | row loads (100 trees) | ARM64 stack ops | hot vs continuation |
|---|---|---:|---:|---|
| fenced window (`VERDICT_fenced_astra`) | one function, memory-clobber fence, no early-cse, reuse window | 298 (W16) | 265 | slower |
| residency round 1 (`VERDICT_residency_astra`) | one function, volatile features, Belady K, no early-cse | 338 (K16) | 136 | +13 to 15% |
| round 2 (`VERDICT_residency_round2_astra`) | + memory-clobber fence per chunk, laundered pointers, visit-order leaves | – | spills at K1 | never faster |
| round 3 (`VERDICT_all_volatile_astra`) | every load volatile and emitter-placed | – | spills every K | slower |
| round 4 (`VERDICT_barrier_residency_astra`) | ordinary loads, no-clobber barrier, Belady K, no early-cse, visit-order leaves | 376 (K8) | 119, spills at K1 | not faster |
| source-order scheduling (`VERDICT_scheduler_options_sol`) | one function, ordinary loads, early-cse on, `-pre-RA-sched=source`, misched off | 99 | 282 | 103.5 vs 90.2 |
| carried vectors (`VERDICT_carried_vectors_astra`) | continuations, features and sums packed in `<2 x double>` arguments | 296 | 11 | 99.2 vs 92.0 |

Common result: every approach cut row loads, and every one paid it back in spills, register
copies, lost `ldp` pairing or pack/extract instructions, because each avoided load was a single
L1-hit instruction. In one function, `early-cse` merges each feature's loads into one value for
the whole kernel (99 loads) and the resulting long live ranges are what spill; scheduler options
reduce but do not remove that. The cheap reuse that pays is grouping trees that share features into
the same chunk (`TreeOrdering::SharedFeatures`), the fastest ARM64 kernel measured: 233.9 ns at
300 trees vs 248.7 in model order. x86 AVX2 with the round-4 barrier at K8 had fewer loads and
instructions than the continuation kernel and no spills; untimed natively.

Single-variable test (`VERDICT_scalar_carry_astra`): carrying up to six already-loaded features as
plain `double` continuation arguments, nothing else changed. ARM64 hot, 100 / 300 trees:

| kernel | instructions | row loads | hot ns |
|---|---:|---:|---:|
| continuation, model order | 2143 | 429 | 88.4 / 239.8 |
| + carry 6 | 2195 | 374 | 88.6 / 239.0 |
| continuation, shared-feature order | 2086 | 332 | 85.3 / 228.2 |
| + carry 6 | 2112 | 279 | 87.1 / 236.8 |

Each carried feature saves one row load and costs about one register move (the allocator shuffles
incoming arguments into place). Neutral in model order, slower in shared order. Decision: the
kernel is continuations + shared-feature order, 8 trees per chunk, 2 sums, with no other option.

## Layout and lane order

Commit `91951d5` places each chunk's thresholds, padding, and leaf blocks together. The ARM64
chunk base addresses thresholds and leaves. A pair's sole shared feature goes to lane 1 on
ARM64 and lane 0 on x86; thresholds move with their features and AVX2 leaves are permuted too.
Counts from [the layout report](VERDICT_layout_astra.md) cover continuations, excluding the C wrapper:

| Target | Instructions, 100 trees before → after | Instructions, 300 trees before → after |
|---|---:|---:|
| ARM64 | 2060 → 1974 | 6247 → 5876 |
| SSE2 | 2295 → 2267 | 6925 → 6859 |
| AVX2 | 1378 → 1334 | 4114 → 3978 |

ARM64 leaf-offset materializations fell from 45 / 267 to zero at 100 / 300 trees;
continuations retained zero stack operations. The maintainer's back-to-back M4 hot measurements
were 89.3 / 232.4 ns for `91951d5`, against 89.6–91.5 / 229.6–234.0 ns for the previous kernel
at `131c251`. These are the layout run's measurements, separate from the earlier 85.3 / 228.2 ns
scalar-carry study based on `6cae781`. No native x86 timing was collected.

## Add-based mask merge

The SSE2 add-based merge reduced continuation instructions from 2267 → 2134 at 100 trees
and 6859 → 6454 at 300 trees. Constant-pool loads fell from 13 → 0 and 38 → 0;
continuation stack operations fell from 6 → 2 and 42 → 10.

After the `Target` descriptor and single chunk-base parameter changes, separately emitted
100x100 and 300x100 kernels have these static counts. Directives and alignment padding are
excluded; stack operations include push/pop.

| Target | Trees | Continuation instructions | Wrapper instructions | Continuation stack operations | Wrapper stack operations |
|---|---:|---:|---:|---:|---:|
| ARM64 | 100 | 1974 | 25 | 0 | 20 |
| ARM64 | 300 | 5877 | 25 | 0 | 20 |
| SSE2 | 100 | 2134 | 15 | 2 | 10 |
| SSE2 | 300 | 6454 | 15 | 10 | 10 |
| AVX2 | 100 | 1334 | 15 | 0 | 10 |
| AVX2 | 300 | 3978 | 15 | 0 | 10 |

Both x86 paths remain byte-identical for both workloads. ARM64 drops the wrapper's base copy
and one continuation general register; its 300-tree continuations gain one instruction, leaving
the total unchanged. Vector-register counts and stack operations are unchanged. Two-tree AVX2
forests at depths 0, 1, and 2 lose the explicit `vzeroupper` and its spill/reload: continuation
instructions fall from 8 → 5, 13 → 10, and 13 → 10, with stack operations 2 → 0 in each case.

## Open

- Native x86 timing of the SSE2 and AVX2 kernels.
- A representative user model.
