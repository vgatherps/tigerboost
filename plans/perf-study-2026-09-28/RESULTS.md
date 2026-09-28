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

Remaining load counts, 100 trees, 8 per chunk: ARM64 316 `ldr d` + 172 `ld1.d` feature loads,
186 `ldp` (threshold pairs), 26 `ldr q`; x86 SSE2 292 `movsd` + 174 `movhpd`; AVX2 274 `vmovsd`
+ 161 `vmovhpd`. These are distinct features per chunk; no further reuse exists within a chunk.

## Open

- Default order: stay scalar, or tree-major 8 (spill-free) / 16 (fastest)? Maintainer decision.
- Native x86 timing of the AVX2 path (needs real hardware).
- x86 execution evidence (Rosetta for correctness; a real host for timing).
