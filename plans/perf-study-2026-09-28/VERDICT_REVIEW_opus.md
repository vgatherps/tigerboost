# Review of SYNTHESIS.md (tigerboost design proposal), reviewer: opus

Worktree `agent-a5c2407fd705af870` at da93993. No tracked file changed (`git status`: only `?? probes/`).
New probes, all under `probes/` in that worktree:

| probe | what it answers |
|---|---|
| `probes/residency/` (cargo crate, calls `sort_policy::plan`) | distinct features per 8-tree window, register fills, gather cost, optimal-eviction bound, GP4 admission order |
| `probes/ubench/pipes.c`, `pipes2.c` | M4 pipe sharing: `ldp q`, `ldp d`, `ld1` lane, `fmov x,d`, `mov x,v.d[1]`, `zip1`, `sli`, `fcmgt` |
| `probes/zip1*.ll` | whether `llc` emits `zip1.2d` for a pair of resident features |
| `probes/sli_depths.c` | exhaustive native check of the `sli` packing and bias for depth 1 to 10, odd depths padded |
| `probes/nan_matrix.c` | every `NanMode` against NaN (quiet, signaling), ±inf, ±0, ±denormal, NaN and +inf thresholds |
| `probes/check_x86.c` + `tm_x86.ll` | first execution of the AVX2 tree-major kernel (via Rosetta 2) against the reference |
| `probes/daz.c` | the x86 `next_up` carry form under MXCSR.DAZ (via Rosetta 2) |

Clock for cycle conversions: 4.5 GHz (my add-chain calibration read 4.48 to 4.51 GHz on the quieter run).
Binarize (SYNTHESIS section 5 item 6 and the shared-border rows) is out of scope per the maintainer and is not reviewed.

## 1. Findings

No blocker. The design is implementable and correct on x86 as far as I can execute it. The main
problem is that the NEON half of the proposal claims a performance advantage the evidence does not
support.

### F1 (major). The NEON performance case for tree-major is not supported. Sections 4 and 5 item 2.

SYNTHESIS lines 100-106 and 151 say tree-major and scalar carry "land in the same place" but that
tree-major "has headroom the scalar form does not", and expect "~93 ns at 100 trees without
residency, lower with it". The measurements say something else.

| NEON, 100 / 300 trees | ns (serial medians) | instructions, 100 trees | source |
|---|---:|---:|---|
| item 1 alone: constants pointer + per-chunk pointer, GP4 | 94.1 / 259.1 | 2959 (GP12) | `serial/A_*_TB_PROBE_base_TB_REBASE_1.txt` |
| tree-major, opus C generator, 4-tree chunks | 93.5 / 263.3 | 2266 (22.7 per tree) | `serial/C_100x100.txt`, `C_300x100.txt`, `s_treemajor_a64_100x100_d6_b0.txt` |
| scalar carry chain + per-chunk pointer, GP4 | 86.8 / 236.8 | 2275 (22.75 per tree) | `serial/A_*_TB_PROBE_carry_TB_REBASE_1.txt`, REPORT_A_opus R1+R2 table |

So on NEON, tree-major as measured buys nothing over item 1 alone and is 7% (100 trees) to 11%
(300 trees) slower than the carry fallback, with the same instruction count per tree. Astra C
(REPORT_C_astra.md:124) already said tree-major at 22 per tree loses to a 3-per-split scalar that
includes its row loads; the synthesis dropped that.

Where each form is capped on the M4. I measured the pipe sharing the synthesis relies on
(`probes/ubench/pipes2.c`, ratios against a dependent add chain, two runs):

| instruction mix | cycles per instruction | reading |
|---|---:|---|
| `ldr d`, `ldr q`, `ld1 {v.d}[1]` | 0.33 | 3 load slots per cycle, lane load takes one slot |
| `ldp d, d` | 0.34 | one slot for two doubles |
| `ldp q, q` | 0.67 | two slots: 3 × 128-bit per cycle, pairing saves no slot |
| `ldp q` alternating with `ldr d` | 0.50 | 2 + 1 slots per pair of instructions |
| `fmov x, d`, `mov x, v.d[1]`, mixed | 0.50 | FP-to-GP moves: 2 per cycle, a separate resource |
| `fmov x, d` alternating with `ldr d` | 0.25 | FP-to-GP moves do not use load slots |
| `fcmgt.2d`, `sli.2d`, `zip1.2d`, mixed | 0.25 | 4 vector pipes, shared |
| `fcmp` alternating with `ldr d` | 0.25 | compare port does not use load slots |

Load-slot count per 100 trees, depth 6:

| kernel | row slots | threshold slots | leaf | total | cycles at 3/cycle | other cap |
|---|---:|---:|---:|---:|---:|---|
| today (GP12) | 360 | 600 (`fmov d,x`) | 200 (leaf + GOT) | 1179 incl. `stp` | 393 | rename 445 |
| scalar carry + per-chunk pointer | 360 | 303 (`ldp d`) | 100 | 763 | 254 | `fcmp` port 300 |
| tree-major, as measured (4-tree chunks) | 504 | 300 | 100 | 904 | 301 | rename 227, vector 147, FP-to-GP 100 |
| tree-major, 8-tree window, tree order | 468 | 300 | 100 | 868 | 289 | |
| tree-major, 8-tree window, GP4 admission order | 426 | 300 | 100 | 826 | 275 | |
| same, 300 trees (per 300) | 1172 | 900 | 300 | 2372 | 791 | scalar carry: `fcmp` port 900 |

Measured tree-major runs at about 398 cycles for 100 trees against a 301-cycle load bound (76%);
the carry kernel runs at about 368 against its 300-cycle compare bound (82%). The best-case
tree-major bound on this workload is 275 cycles, 8% under the scalar 300-cycle floor at 100 trees
and 12% under at 300 trees. That is the whole NEON headroom, and it needs good admission order,
8-tree chunks and equal pipeline efficiency. The synthesis's "333 load-pipe cycles" happens to be
the right slot count for the no-reuse case (it counts `ldp q` as 3 slots per tree, which my
measurement confirms), but its conclusion that residency opens a gap is 3x too large (F2).

Failure scenario: item 2 is built for NEON first, lands at about 93 ns, and the team discovers it
is no faster than item 1 and slower than the 150-line carry variant. The legitimate reasons to
prefer tree-major on NEON are engineering ones (one plan shape for both targets, no inline asm),
and the synthesis should say that instead of claiming speed. Experiment 1 below decides it.

### F2 (major). "22 → 16 instructions per tree and ~40% fewer loads" is wishful on this workload. Section 4 line 105, section 5 item 2.

`probes/residency` plans 100x100 and 300x100 with `Order::TreeOrder`, reads each tree's feature
list back from the `Action::Load`/`Action::Compare` stream, and counts register fills per window of
consecutive trees (a fill is from a register if the feature was already loaded earlier in the
window). It also reorders trees by the `Admit` order of the scheduled GP4/FP9 plan.

| 100x100 | window 4 | window 8 | window 16 |
|---|---:|---:|---:|
| distinct features per window (tree order, typical) | 21-23 | 35-39 | 58-65 |
| fills from registers, tree order | 67 / 600 (11.2%) | 132 (22.0%) | 215 (35.8%) |
| fills from registers, GP4 admission order | 121 (20.2%) | 174 (29.0%) | n/a |
| same at 300x100, GP4 order, window 8 | | 628 / 1800 (34.9%) | |

Register capacity is not the limit: optimal eviction with only 8 resident registers already
reaches the unbounded numbers (174 of 600 at 8, 12, 16, 20 and 24 registers). The limit is how
often a feature repeats inside 8 trees on a 100-feature model with random splits.

What a tree costs then (gather is 9 instructions with no residency: 3 × `ldr d` + `add` +
`ld1 {v.d}[1]`):

| order, window 8 | gather, resident pair = 1 instruction (`zip1`) | resident pair = 2 (`mov.16b` + `mov.d`, what `llc` emits, F5) | per tree total | row load slots per tree |
|---|---:|---:|---:|---:|
| tree order, 100 | 7.68 | 8.14 | 20.7 to 21.1 | 4.68 |
| GP4 order, 100 | 7.26 | 7.92 | 20.3 to 20.9 | 4.26 |
| GP4 order, 300 | 6.91 | 7.71 | 19.9 to 20.7 | 3.91 |

So about 20 instructions per tree, not 16, and total loads fall from 1000 to 826 per 100 trees
(17%), or to 2372 of 3000 at 300 trees (21%), not 40%. The 16 figure is `resident_vec` in
REPORT_C_opus.md:99, which assumes all six features of every tree are already in registers.
Carrying up to 6 features across chunk boundaries (8 FP arguments minus 2 sums) adds at most 66
register fills per 100 trees in GP4 order (72 in tree order), an upper bound that assumes the
planner picks exactly the 6 the next chunk needs.

Also, the measured 93.5 ns kernel already has part of this: LLVM merged repeated row loads inside
each 4-tree chunk (530 row values loaded for 600 fills; 62 `mov.d v[1],v[0]` and 26 `mov.16b`
copies in `s_treemajor_a64_100x100_d6_b0.s`). SYNTHESIS line 78 ("every tree loads its own six
features from the row, no cross-tree reuse") repeats a sentence in REPORT_C_opus.md that the same
report's own gather classification contradicts. The residency headroom above the measured number
is window 4 to window 8 plus ordering: 504 to 426 row slots.

### F3 (major). The x86 carry fallback as named is wrong under MXCSR.DAZ. Section 5 "Fallback data point", section 3.

SYNTHESIS line 182 names the fallback as "`addcarry` intrinsic on x86", which is the
`ucomisd next_up(t); adcq` form. `probes/daz.c`, run under Rosetta 2 (which honors MXCSR),
computes the complemented bit as `!(x < next_up(t))` and compares with `x > t`:

| x, t = 0.0 | default MXCSR | DAZ + FTZ set |
|---|---|---|
| +0.0, -0.0 | matches | says x > t (wrong) |
| ±denormal | matches | says x > t for -denormal and +denormal (reference is false for both under DAZ) |

`next_up(0.0)` is a denormal, which DAZ reads as zero, so every row with a zero (or denormal)
feature against a zero border routes to the wrong leaf, silently, whenever the host process has
DAZ set (for example after loading a library built with `-ffast-math` on x86, which installs
`crtfastmath`). REPORT_A_opus.md flagged this as a caveat; the synthesis kept the form. Astra A's
`lea; ucomisd; cmovbe` select form (2558 instructions, REPORT_A_astra.md A3) has no threshold
adjustment and is exact under any MXCSR. If an x86 scalar fallback is ever needed, use that form.
The ARM64 inline-asm carry form uses the original threshold and is exact (FPCR affects the
reference and the kernel the same way).

### F4 (minor). Item 3 contradicts itself twice. Section 5 item 3 (lines 153-156) and item 2 (line 149).

"Keep tree order with feature-overlap admission (the GP4/FP9 behaviour)" and "planning cost then
drops from 85 ms to <1 ms at 300 trees" do not go together: the GP4 plan is the existing
`schedule` with a 4-register budget, and serial compile times show it costs about 50 ms more than
tree order at 300 trees (`serial/A_300x100_baseline.txt`: tree order 92-105 ms, GP4 146-158 ms).
The <1 ms figure (REPORT_B_opus B6, 0.037 ms) is plain `Order::TreeOrder`. Getting both needs a new
cheap admission heuristic, or accepting ~50 ms. Separately, item 2 says "≤6 resident features"
cross a boundary, item 3 says residency is decided within a chunk and not across chunks. Pick one;
F2 bounds the cross-boundary gain at 66 fills per 100 trees.

### F5 (minor). Resident pairs: `llc` does not emit `zip1` from scalar-inserted vectors. Section 5 item 2, section 4.

Per the maintainer's note I checked `shufflevector <2 x double>` of two scalar-inserted vectors
(`probes/zip1.ll`, `zip1z.ll`, `llc -O1 -mcpu=generic` and `-mcpu=apple-m4`). `llc` emits
`mov.d v0[1], v1[0]` (a destructive lane insert), whether lane 1 is poison or zero. When the
resident feature is used again later, which is the point of keeping it, it adds a `mov.16b` copy:
2 instructions per resident pair (`pair_keep`). `zip1.2d` appears only when both sources are full
`<2 x double>` values: function arguments (`zip1b.ll keep_vec`: two `zip1.2d`, no copies) or splats
(`zip1c.ll`, where `ld1r.2d` needs an `add` because it has no immediate offset). Re-derived: fully
resident tree = 3 gather instructions only if residents are carried as `<2 x double>`; otherwise 6.
F2's table carries both costs. Pair residents with residents (the leaf permutation makes any lane
assignment free) and a resident pair costs 1 (`zip1`) or 2; a resident paired with a row feature
costs 2 (`ldr d` + `mov.d`); a row pair costs 3.

### F6 (minor). Per-chunk pointer: reach for leaves, and `ldp q` saves no load slot. Section 5 item 1 (lines 129-134).

Item 1 carries one per-chunk pointer for thresholds. Leaves sit after all thresholds, so their
offsets from either pointer exceed the 32,760-byte `ldr d` immediate beyond about 64 depth-6 trees:
the 300-tree tree-major kernel spends 136 `mov`/`movk` plus 136 `add ..., lsl #12` on leaf
addresses (`s_treemajor_a64_300x100_d6_b0.s`, 0.9 per tree). Lay the block out per chunk (the
chunk's thresholds, then the chunk's leaf blocks) so one rebased pointer reaches both. Also, by
F1's measurement, `ldp q` costs two load slots, so rebasing to make `ldp q` reach removes 0.75
instructions per tree but no load-pipe work; `ldp d` (scalar thresholds) does halve slots. Mixed
depths: REPORT_A_opus's probe sizes every leaf block to the largest tree (`leaf_block = max`),
which is safe but inflates the region (one depth-8 tree makes every block 2 KB); size blocks per
tree and align each to its own size.

### F7 (minor). Odd depths and pad lanes are unspecified; the generator's padding is unsafe. Section 5 item 2.

`opus_probes_C/gen.py` asserts even depth on NEON and, on AVX2, pads a short group with feature
0.0 against whatever value sits in the constant blob (`vec_from` plus `const_slot`). For a depth-5
tree that pad bit is data-dependent, so the index can set bit 5 and read a leaf from the next
tree's block. The fix is simple and I verified it exhaustively with the real `sli` instruction
(`probes/sli_depths.c`): pad lanes compare against +inf (always false under an ordered compare,
even for NaN or +inf features) and, on NEON, the pad goes in lane 1 of the last vector (the sign
position). Bias is `2^(P-1)` for odd depth and `2^(P-1) + 2^P·2^(P-1)` for even depth, P = ceil(D/2):

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias | 1 | 3 | 2 | 10 | 4 | 36 | 8 | 136 | 16 | 528 |
| index range, bijective onto [0, 2^D) | [-1,0] | [-3,0] | [-2,5] | [-10,5] | [-4,27] | [-36,27] | [-8,119] | [-136,119] | [-16,495] | [-528,495] |

The bias combines safely with an ADD-ed block offset for mixed depths: each tree gets its own
compile-time `block + bias`, the final address stays inside `[block, block + 2^D)`, and `sli`
leaves each lane sign-extended over 64 bits so the `add x, x0, x1, lsl #P` is exact. OR-ing the
offset would break (negative indices), as the synthesis says. Depth-0 trees need a constant-leaf
path; `continuation::partition` asserts `width <= limit` (`src/sort_policy/continuation.rs:59`),
so a per-tree action with the proposed 48-test limit panics for depth above 48 (only theoretical).

### F8 (minor). `AsTrue` via unordered compare needs care the proposal skips. Section 5 item 5 (line 162).

`probes/nan_matrix.c` (324 cases: 3 modes × 12 feature values including signaling NaN × 9
thresholds) confirms the core claim: an ordered compare (`fcmgt.2d`, and `vcmppd` imm 30 which is
the same GT_OQ predicate) gives the reference answer for `AsIs` and `AsFalse` in every case, with no
load-time work. `AsTrue` with an ordered compare fails 21 cases (NaN feature against every threshold
below +inf). `AsTrue` with unordered-greater (`!fcmge(t, x)` on NEON, `vcmpnle` on x86) is exact
only with NaN and +inf thresholds folded; unfolded it fails 15. Two gaps:

- A fold cannot be done through the threshold value on an unordered lane (a NaN feature makes
  unordered-greater true against anything). Realize it by making the leaf table ignore that bit
  (duplicate entries), which the compile-time table build allows for free. On ordered lanes a +inf
  threshold is already a constant false; no fold is needed there.
- One vector compare has one predicate. An `AsTrue` lane cannot share a `fcmgt.2d` or `vcmppd`
  with an `AsIs`/`AsFalse` lane. Either group `AsTrue` lanes together, or keep the load-time +inf
  substitution for `AsTrue` features (today's `fcmp uno` + `select`, `src/jit.rs:244-264`), which
  stays exact in mixed vectors. The "keep the substitution" branch of item 5 is the safe default.

### F9 (minor). x86 details that do not hold as written. Section 4 lines 108-112, section 5 item 2.

- Code size: the AVX2 tree-major 300-tree kernel assembles to 33,292 bytes of `.text`
  (`opus_probes_C/s_treemajor_avx2_300x100_d6_b0.s` through `llvm-mc`), not 25-30 KB, so it is
  just over a 32 KB L1I, and 4,343 instructions exceed Golden Cove's 4K-entry uop cache.
  Continuation transfers under `JITDefault` (`movabsq` + `jmpq *%reg`) come on top.
- "x86 ~3x by mca" (line 151) is Alder Lake only; Zen 3 is 1.7x (705 to 423).
- `target-cpu` on the target machine (`src/jit.rs:78-85`) only affects `emit_assembly`/inspection:
  `compile` discards that machine and MCJIT builds its own generic one through
  `LLVMCreateJITCompilerForModule` (inkwell 0.9 `module.rs:590`; astra A says the same for the code
  model). The function attributes (`src/jit.rs:162-166`) are what govern MCJIT. Setting both keeps
  inspection honest, which is fine.
- "Host features" on an AVX-512 host lets LLVM select EVEX compares into `k` registers, a code
  shape never counted or tested here. That does not strictly create an AVX-512 dependence, but it
  sidesteps the decision in practice. Pin `x86-64-v3` for the AVX2 path, check the host for AVX2 at
  compile time, and fall back to the SSE2 path otherwise.
- SSE2 fallback: legacy `cmppd` accepts predicates 0-7 only, so the GT_OQ (30) intrinsic call must
  become `cmpltpd` with operands swapped (threshold in a register, hence the `movapd` in the
  20-per-tree count), and legacy memory operands need 16-byte-aligned threshold vectors.

### F10 (minor). Smaller misreadings in sections 1-2.

- Line 42: f32-derived borders cost `mov` + two `movk` + `fmov` on Darwin (4), and 5 on Linux
  AArch64 (REPORT_A_opus A2 table), not "three `movk`".
- Line 34: the x86 failure is an argument-count limit, not a budget number: 12 GP arguments fail
  under the large code model, 11 compile (REPORT_A_opus A5; REPORT_B_opus B1 is the same fact
  expressed as budget GP14 = row + 11 selectors + scratch rule at `src/sort_policy.rs:337`). The
  proposal's tree-major boundary (row + constants pointer, 2 GP) is far from it.
- Lines 58 and 122: "leaf misses do not overlap" is REPORT_D_opus's unverified model (its own
  words), and the model itself allows about 13 misses in flight. State it as the hypothesis that
  REPORT_D_opus experiment E3 tests.
- Item 1 "measured alone" (line 134) is ARM64. On x86 the pointer with today's OR selector raises
  continuation stack accesses from 28 to 66 (REPORT_A_astra.md A2 table); it only pays on x86
  together with a selector change or tree-major.
- Item 4: REPORT_D_astra.md:127 measured a 3-cycle vector `fadd` on the M4 against 2 for scalar,
  and argues against packing sums into `<2 x double>`; REPORT_D_opus recommends it. The re-measure
  in item 4 should include 4 scalar sums, not only the vector pair.

### F11 (minor). The NEON ranking compares two different harnesses. Sections 3 and 4.

The carry numbers come from `scalar_bench` (Rust driver, dependent chain that settles into a
10-row cycle at 100 trees and a 4-row cycle at 300, REPORT_D_astra.md:157), the tree-major numbers
from `opus_probes_C/time_predict.c` (C driver, a different dependent chain, arbitrary threshold
values, 4-tree chunks joined by plain calls). REPORT_D_opus also found that the M4 predicts load
values and addresses on short periodic patterns. A 7% gap is within what those differences could
move. The decisive comparison must use one harness, real thresholds and a long row walk.

## 2. Answers by question

### R1. Causal chain

Section 1 holds: the 445 / 393 / 300 cycle bounds, 90% of rename, the 1/cycle `ucomisd` citation,
and the 64 KB x86 300-tree kernel are all supported by REPORT_D_opus and the counts. My
microbenchmark adds that `ldp d` is one load slot (the 393 figure counts it that way, correctly).

Section 2: supported except the minor items in F10. The argument-register limit (8 FP, 12 GP on
x86) is independently confirmed by astra B (td file) and opus B/D (probes). `early-cse` folding a
private global and the small-code-model danger are confirmed by both A reports; astra's nuance
(a load inserted after `early-cse` survives on ARM64 but still costs a `movabsq` per threshold on
x86) does not change the conclusion. The NaN statements are correct for `AsIs`/`AsFalse` (F8).

Section 4: the x86 paragraph is supported by static counts and llvm-mca, and now by one executed
kernel (below), with F9's corrections. The NEON paragraph is not supported (F1, F2).

Where the two models' reports conflict:

| topic | opus | astra | who is right |
|---|---|---|---|
| carry chain needs NaN normalization | no (complemented bit, `fcmp t,x` sets C on unordered) | yes (REPORT_A_astra "carry-based proposals do") | both, for their forms: astra's `ucomisd x,t; rcl` computes the true bit and sets CF on unordered, so it needs normalization; opus's complemented forms do not. `AsTrue` needs substitution in every carry form. |
| tree-major vs scalar on NEON | tree-major wins (REPORT_C_opus) | tree-major 22 loses to a 20-per-tree scalar incl. loads (REPORT_C_astra:124) | astra, on the numbers: carry+rebase is 22.75 per tree and 7% faster (F1) |
| vector sums on NEON | pack 4 sums into two `<2 x double>` (REPORT_D_opus) | prefer scalar; vector `fadd` is 3 cycles (REPORT_D_astra) | unresolved; measure both (F10) |
| M4 L1 line size | 128 B (sysctl, REPORT_D_opus) | 64 B L1, 128 B L2 (REPORT_D_astra) | does not change any conclusion here; line estimates in R2g use 128 B |

### R2a. Plan shape and signatures

With one action per tree that computes the index and retires it, nothing but sums and resident
features crosses a boundary. Concretely: `continuation::partition` never allocates in
`live.indices`, so `Slots.indices` is always empty and `index_slots` in `emit_plan`
(`src/jit.rs:109-113`) becomes 0; `Chunk.tests` becomes the sum of tree depths (48 for 8 depth-6
trees). The parameter list becomes `[row, constants, (per-chunk pointer on ARM64)]` GP, then sums,
then at most 6 features as FP. GP arguments are 2 on x86 (limit 11 under `JITDefault`) and 3 on
ARM64 (24). FP arguments are 2 sums + ≤6 features = 8, exactly the limit; 4 sums as two
`<2 x double>` also take 2 slots. So the argument budget is fine, and `RegisterBudget` and
`ComparisonMode::Simd2` can go. Both are public API (`SortOptions`), so that removal is the
user's call.

One design point the synthesis leaves open: lane grouping (3x2 on NEON, 4+2 on AVX2), pad
placement and the leaf permutation are target-specific, but `plan()` does not know the target.
Keep them in the emitter: the plan should carry tree order, per-tree feature sources (row or
resident `FeatureValueId`) and levels; the emitter builds lanes, the constants block and the
permuted tables per target. Otherwise the planner needs the triple, which REPORT_B_opus
question 2 already raised as an interface change.

### R2b. Residency: answered in F2 and F5

Distinct features per consecutive 8-tree window on 100x100: typically 35-39 in tree order (33 in
GP4 order). Register fills: 22% (tree order) or 29% (GP4 order) of lane fills; 35% at 300 trees in
GP4 order. The "16 instructions per tree" figure is not real for this workload; about 20 is.
Cross-boundary: 8 FP arguments minus 2 sums leaves 6, worth at most 66-72 more fills per 100 trees.

### R2c. Load pipes: answered in F1

Tree-major's next cap on the M4 is the load pipe: thresholds are a fixed 3 slots per depth-6 tree
(48 bytes; `ldp q` does not help), the leaf is 1, and row fills are 3.9-6. Best case on this
workload is 275 cycles per 100 trees against the scalar compare floor of 300 (and 791 against 900
at 300 trees). The claimed extra headroom exists but is 8-12%, not the implied 40% load cut. f32
thresholds would halve the fixed part; that is the user's numeric-contract question.

### R2d. AVX2 emission from inkwell

Nothing is missing. `Intrinsic::find` / `get_declaration` (inkwell 0.9 `intrinsics.rs:45,85`)
declare `llvm.x86.avx.cmp.pd.256` and `llvm.x86.sse2.cmp.pd` (not overloaded) and the overloaded
`llvm.aarch64.neon.vsli` (pass the `<2 x i64>` type); `build_bit_cast` gives `<4 x i1>` to `i4`
which lowers to `vmovmskpd`; `get_host_cpu_name` / `get_host_cpu_features` exist
(`targets.rs:1102,1113`); `InitializationConfig::default()` enables the asm parser, which the ARM64
inline-asm fallback needs under MCJIT. Target settings: F9. I executed the AVX2 tree-major probe
(`tree_major_x86_alt.ll @tree4x2_gtoq`: `vcmpgt_oqpd` memory operands, `vmovmskpd`, identity leaf
order) under Rosetta 2 against the reference with the opus C special-value generator:
`ok 1000000 checks` (`probes/check_x86.c`; the object's disassembly shows `vcmpgt_oqpd 0x80(%rsi),
%ymm1` and `vmovmskpd`). Rosetta 2 on this macOS executes AVX2 even though CPUID does not report
it (`probes/avx2_rosetta.c`). This is correctness only; Rosetta timing means nothing.

### R2e. `sli` math: answered in F7

Depth 4, 6 and 8 biases are 10, 36 and 136, verified exhaustively with the hardware instruction,
along with odd depths (pad in the sign lane) and depths up to 10. ADD-ed block offsets with
per-tree biases are safe for mixed depths.

### R2f. NaN: answered in F8

| mode | NaN feature | ±inf feature | ±0, ±denormal | NaN threshold | +inf threshold |
|---|---|---|---|---|---|
| `AsIs` | ordered GT false = reference | exact | exact (+0 vs -0 equal) | false both | false both |
| `AsFalse` | ordered GT false; reference -inf > t is false for every t | exact | exact | false both | false both |
| `AsTrue` | ordered GT wrong for t < +inf; unordered GT right | exact | exact | unordered GT wrong unless folded | unordered GT wrong for NaN feature unless folded |

x86 imm 30 (GT_OQ) is the same predicate as IR `fcmp ogt` and NEON `fcmgt`, quiet, so signaling
NaN rows raise no trap (the invalid flag is masked by default). The one executed x86 kernel above
covers NaN, ±inf, ±0 and denormals.

### R2g. Cold calls

Nothing in the proposal adds lines per call; the leaf lines, which dominate cold cost, are
unchanged (one per tree). Estimated lines touched per call at 100 trees:

| | code | thresholds | leaves | GOT / const pool | row | total |
|---|---:|---:|---:|---:|---:|---:|
| ARM64 today (128 B lines) | 140 | in code | 100 | 7 (GOT) | 7 | ~254 |
| ARM64 item 1 | 93 | 38 | 100 | 0 | 7 | ~238 |
| ARM64 tree-major | 71 | 38 | 100 | 0 | 7 | ~216 |
| x86 today (64 B lines) | 336 | 75 (pool) | 100 | | 13 | ~524 |
| x86 tree-major | ~174 | 75 | 100 | | 13 | ~362 |

Thresholds move from instruction fetch to sequential data reads, which the hardware prefetcher
handles and which can miss in parallel. Two things could make cold calls worse and should be
checked in the cold harness: a leaf region inflated by max-depth blocks for mixed-depth models
(F6), and on x86 the remaining indirect `jmpq *%reg` per chunk under `JITDefault` (fewer with
8-tree chunks: about 12 at 100 trees instead of 37), which cost front-end redirects when the BTB
is cold.

### R3. Scalar carry fallback

As the primary path it is correctly excluded on x86: `ucomisd` is 1 per cycle, so 600 cycles per
100 trees is a floor, against about 400 for tree-major by llvm-mca. On NEON the performance
evidence favors it today (F1); what excludes it there is engineering: inline asm, a second plan
shape, and a hard 300-cycle ceiling that tree-major can beat by 8-12% at best on this workload.
Risks:

| risk | disqualifying? |
|---|---|
| x86 `next_up` under DAZ | yes for that form (F3, demonstrated). Use astra's `lea; ucomisd; cmovbe` form instead, 2558 instructions, exact |
| ARM64 FPCR.FZ | no: the ARM64 form compares against the original threshold, the reference sees the same FPCR |
| inline asm opacity to the scheduler | no: `fcmp; adc` must stay adjacent anyway, and the measured kernel runs at 82% of its compare bound. It does need the asm parser under MCJIT (enabled by default) |
| plan-order bit mapping | no: the leaf permutation is computed per tree from the order consumers appear in the plan (REPORT_A_opus probe, including shared predicates and mixed depths, passes `tests/differential.rs`). It does tie the table layout to plan order, so tests must cover every order the planner can produce |
| `AsTrue` | needs the load-time +inf substitution in every carry form |

### R5. What the synthesis dropped or misread

- REPORT_C_astra.md:124 (tree-major loses on NEON against a scalar with 3 per split including
  loads) and the matching count in REPORT_A_opus (carry + per-chunk pointer 22.75 per tree). See F1.
- REPORT_C_opus's own gather classification (LLVM already reused 70 fills inside 4-tree chunks),
  against the sentence the synthesis quoted. See F2.
- REPORT_C_opus C5: leaf offsets past the `ldr d` range cost `mov`/`movk` per tree; item 1's
  per-chunk pointer covers thresholds only. See F6.
- REPORT_C_opus `@scalar_masks` (24 per tree: 6 scalar `fcmgt d` + 5 `sli d`). It is a
  non-flag scalar form, so SYNTHESIS line 106 ("scalar cannot go below 300 cycles ... ever") only
  holds for flag compares. It needs no gather (a resident feature costs nothing), keeps the
  feature-major plan, and is capped by 11 vector ops per tree (275 cycles per 100 trees, the same
  as tree-major's best case). Worth one more formulation in experiment 1, since it costs little.
- REPORT_A_astra A3: the x86 select form and its exactness; REPORT_A_astra A2: pointer + old
  selector spills more on x86. See F3, F10.
- REPORT_D_astra: checked `CompiledModel::predict` is not inlined (a 32-byte frame, result store
  and reload, about 10-25 cycles). At 350-cycle kernels that is 3-7%, cheap to test with
  `#[inline]`.
- REPORT_D_astra: the benchmark's row cycle is 10 rows (100 trees) and 4 rows (300 trees). See F11.
- REPORT_B_astra: GP4 with 32-test chunks stays spill-free on both targets; for tree-major that is
  the "~8 trees per chunk" choice, still unmeasured with `musttail` and resident features crossing.

## 3. Ranked experiments (cheapest decisive first)

1. NEON same-harness bake-off: tree-major against carry against item 1 alone.
   Build: extend `opus_probes_C/gen.py` (not the production emitter) with (a) a per-chunk rebased
   pointer argument and a per-chunk layout (thresholds then leaf blocks), (b) 8-tree chunks,
   (c) tree order taken from the GP4/FP9 plan's `Admit` sequence (dump it with
   `probes/residency`), (d) a `carry` formulation (inline asm `fcmp t, x; adc x, x, x`, index seeded
   with the block number, permuted leaves) and a `scalar_masks` formulation, and (e) real
   thresholds and permuted leaves in `consts`/`leafs` with a C reference check. Change
   `time_predict.c` to walk all 256 rows with a stride (for example 97) mixed with the result, so
   no short row cycle forms.
   Gate: every formulation matches the reference on all rows plus special-value rows; timing
   serial on a quiet machine, 100 and 300 trees, hot, plus a 64 MiB pollution variant for cold.
   Expected: tree-major about 83-88 ns and carry about 87 ns at 100 trees; about 230-240 ns both at
   300 trees.
   Kill: if tree-major is not at least 5% faster than carry at both sizes, drop the NEON speed
   argument; choose NEON tree-major only for shape sharing, or keep carry on NEON if the user
   accepts inline asm. If tree-major is more than 5% slower, NEON primary becomes carry.

2. x86 correctness under Rosetta 2 (no x86 hardware needed).
   Build: compile the AVX2 and SSE2 whole-model IR from `gen.py` (with real thresholds, and odd-depth
   padding per F7 for depths 5 and 7) with `llc -mtriple=x86_64-apple-macosx -mattr=+avx2` (and
   without, for SSE2), link with the reference checker, run with `arch -x86_64`. Also run the
   `select` scalar form and, as a negative control, the `next_up` carry form with DAZ set.
   Gate: bit-exact against `Trees::leaf_index` semantics on all rows, including NaN, ±inf, ±0,
   denormals. Expected: pass (the single-tree AVX2 probe already passes 1,000,000 checks).
   Kill: any mismatch that is not a probe bug means the x86 lane or mask composition is wrong.
   Timing under Rosetta is meaningless; do not record it.

3. Cold-call harness in the repo (item 7), before any kernel change.
   Build: promote `d_probe` to `examples/cold_bench.rs` (pollution 256 KiB, 4, 16, 64 MiB, other
   model call, per-call p50/p90/p99, timer-only control, stride row walk). Gate: reproduces
   `serial/D_cold_*.txt` within noise. Kill: none; it is the gate for 4-6.

4. Item 1 in production: constants block through a pointer argument, ARM64 per-chunk pointer,
   per-chunk layout so leaves are in reach (F6), wrapper plus continuation for one-chunk models.
   Build: `src/jit.rs` `emit_plan` (parameters, `Action::Compare` threshold loads, `Action::Retire`
   leaf addressing) from `opus_probes_A/probe.diff`. On x86 do not add the per-chunk pointer (12 GP
   argument failure).
   Gate: `cargo test --offline --release` with the three IR-text unit tests rewritten;
   `tests/differential.rs` incl. NaN modes and mixed depths; `scalar_bench` 100x100 and 300x100;
   `cold_bench`.
   Expected: 121 to 95 ns and 345 to 261 ns hot on ARM64; cold p50 unchanged or better.
   Kill: any test failure, or cold p50 more than 5% worse.

5. Tree-major in production behind an option (only if 1 does not kill it on NEON; on x86 it goes
   ahead regardless once 2 passes).
   Build: per-tree action in `src/sort_policy.rs` carrying tree order and feature sources; lanes,
   pads, biases, permutation and constants layout in `src/jit.rs`; `llvm.aarch64.neon.vsli.v2i64`
   on NEON, `llvm.x86.avx.cmp.pd.256` imm 30 on x86 with `target-cpu=x86-64-v3` function
   attributes and an SSE2 path; `AsTrue` keeps the load-time substitution.
   Gate: all tests, new unit tests for odd depths 1-9, depth 0, a tree repeating a predicate, every
   `NanMode`; x86 output assembled from `emit_assembly` and executed under Rosetta against the
   reference; `scalar_bench` and `cold_bench`. Expected NEON: experiment 1's number. Kill: slower
   than item 1 alone on NEON hot, or any cold regression beyond 5%.

6. x86 hardware timing, whenever an AVX2 host exists: `gen.py` AVX2 tree-major against the
   `select` scalar form with the stride-walk harness, plus `perf stat -e cycles,instructions`.
   Expected: 2-3x on Golden Cove, about 1.7x on Zen 3/4. Kill: under 1.3x on the deployment core
   means x86 does not justify the rewrite either.

7. Residency across boundaries (up to 6 features as FP arguments), only if 5 shows the NEON kernel
   at its load-slot bound. Expected at most 66-72 fewer row loads per 100 trees (about 3% of
   cycles). Kill: under 2% gain.

8. Leaf-load batching and sums, after 5: REPORT_D_opus E3 (collect indices, then issue leaf loads
   together) under `cold_bench`; 4 scalar sums against two `<2 x double>` sums. Kill for batching:
   the 64 MiB row does not move, which also refutes the ROB-pacing model.

## 4. Verdict

The proposal is sound in structure and right for x86: the constants block through a pointer
argument is verified and should land first, the argument-register analysis holds, the NaN
simplification is exact for `AsIs` and `AsFalse` (checked across 324 special-value cases), the
`sli` packing is exact for every depth once pad lanes are specified, the tree-major boundary state
fits the `preserve_nonecc` argument limits with room to spare, inkwell 0.9 can express all of it,
and the AVX2 tree-major kernel now has one executed correctness result (under Rosetta 2). Its
NEON half overstates the case: measured on the same workload, NEON tree-major is no faster than
item 1 alone and 7-11% slower than the carry chain at the same instruction count, and the
residency "headroom" is about a third of what the synthesis claims (about 20 instructions per
tree and 17-21% fewer loads, not 16 and 40%), leaving an 8-12% best-case margin under the scalar
compare floor. Two named options need correcting before anyone implements them: the x86 `next_up`
carry fallback is wrong under DAZ, and `AsTrue` by unordered compare needs grouped lanes and
don't-care table entries rather than threshold folding. Land item 1 and the cold harness, run the
single-harness NEON bake-off and the Rosetta x86 correctness run, and let those two decide whether
NEON shares the tree-major shape for speed or only for simplicity.
