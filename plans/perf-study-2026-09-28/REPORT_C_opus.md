# Study C: SIMD formulation of oblivious trees on NEON, AVX2 (and AVX-512 for information)

Worktree: `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-aaa5211d65c808515` at da93993.
No tracked file was edited, so there is no diff to replay. All probes are untracked under
`probes/` in that worktree. Anchors from the brief checked and correct: `src/jit.rs` 268-350
(`Action::Compare`), 356 (`Action::Retire`), 162 (target attributes); `src/sort_policy.rs` 329
(`operation`), 394 (`select`), 172 (chunk limit 16).

The maintainer's update is applied throughout: NEON and AVX2 decide the ranking, AVX-512 is
listed for information, the SSE2-only case (x86-64 `generic`) is noted, and every formulation
reads thresholds and leaves as immediate offsets from one base register.

## How the numbers were produced

`probes/gen.py` writes LLVM IR for a whole model under one of three formulations, `scalar`,
`treemajor` and `binarize`, for a64, avx2 and avx512. `probes/model.py` reproduces
`examples/support/distinct.rs` exactly: its tree 0 splits match the `Discriminator` list in the
100x100 plan dump. It can also produce a shared-border variant (`TB_BORDERS=k`, each feature
has k borders) and other depths (`TB_DEPTH`). The generator applies the same rules to every
formulation:

- Thresholds, rank vectors and index vectors come from one `%consts` base pointer, and leaves
  come from one `%leafs` blob at offset `tree << depth`.
- Every tree loads its own features from the row, so no formulation gets cross-tree feature
  reuse.
- There are two running sums.
- Trees are grouped into `noinline preserve_nonecc` chunk functions: 4 trees per chunk, or 8
  for binarize so that one AVX-512 permute serves 8 trees. Chunks are joined by plain calls,
  not `musttail`.

`probes/count.py` compiles each file with `llc -O1` and counts instructions and stack
references. The test driver `predict` is left out of the counts. `probes/run_all.sh`
regenerates every count and llvm-mca figure in this report, and `probes/time_all.sh` times the
NEON kernels natively on the M4. The x86 flags are `-mattr=+avx2` for AVX2 and
`-mcpu=x86-64-v4 -mattr=+avx512vbmi` for AVX-512. The a64 flag is `-mcpu=generic`.

Per-tree figures are "all-in": total kernel instructions divided by trees. They include the
binarize prologue, reloads per chunk, and `ret`/`vzeroupper` for each function. Only the
compiled code of the hand-written probes was executed (NEON only). The whole-model files were
compiled and counted. On NEON they were also timed; their constant values are arbitrary, so
the timings say nothing about correctness.

## 1. Answers per question

### C1. Current two-lane formulation

Here is one pair and the two scalar splits it replaces, taken from the captures and using
today's constant addressing.

| Target (capture) | Pair sequence | Pair count | Two scalar splits | Scalar count |
|---|---|---:|---|---:|
| ARM64 `simd2-scheduled-12gp-9fp-2sums.s` lines 88-140 | `adrp`, `ldr x` (GOT), `ldr q` (thresholds); `dup.2d` or `ld1r.2d`; `fcmgt.2d`; `mov.d x, v[1]`, `fmov x, d`; per consumer `and` + `orr` (`sub` at level 0) | 11 | per split `mov`, `movk`, `fmov`, `fcmp`, `cset`, `orr`/`ubfiz` | 12 |
| x86-64 `simd2-...s`, `continuation1` | `unpcklpd` (splat), `movabsq` + `movapd`, `cmpltpd`, `movq`, `punpckhqdq`, `movq`, per consumer `andl` + `orq` | 11 | per split `movabsq`, `xorl`, `ucomisd`, `seta`, `shll`, `orq` | 12 |

Whole kernels agree with this. On ARM64 the kernel shrinks from 4453 to 4051 instructions
(402 saved over 234 pairs, 1.7 per pair). On x86-64 it shrinks from 4015 to 3840 (175 saved,
0.75 per pair). The histograms match the sequence: 234 `fcmgt.2d`, 234 `mov.d`, and 334 `adrp`
(100 for leaves plus 234 for pair constants).

The saving comes entirely from constant materialization. One GOT-relative `ldr q` fetches two
thresholds in 3 instructions, where each scalar threshold also costs 3. Once thresholds are read
from a base register, which is now decided, the pair loses on NEON. Verified in
`probes/simd2_pair.ll`:

| Target, base-register thresholds | Pair (`src/jit.rs` shape) | Two scalar splits |
|---|---:|---:|
| NEON: `ldr q`, `dup.2d`, `fcmgt.2d`, `mov.d`, `fmov`, 2 `and`, 2 `orr` vs `ldp d`, 2 `fcmp`, 2 `cset`, 2 `orr` | 9 | 7 |
| SSE2 (generic x86): `movapd`, `unpcklpd`, `cmpltpd`, `movq`, `punpckhqdq`, `movq`, 2 `andl`, 2 `orq` | 10 | 10 |
| AVX2: `vmovapd`, `vmovddup`, `vcmpltpd`, `vmovq`, `vpextrq`, 2 `andl`, 2 `orq` | 9 | 10 |

This formulation cannot win because the vector holds two thresholds of one feature, and those
two thresholds nearly always feed two different trees. Each lane result must therefore leave
the vector separately (one extract per lane) and be merged into a different index register
(`and` + `orr` per consumer). That is the same per-split tail the scalar path has. The vector
saves only the compare itself (one `fcmgt` instead of two `fcmp`), and the splat plus the
second extract spend that saving again.

In the 100x100 SIMD2 plan dump, 5 of the 234 pairs have a consumer tree in common, which is
2.1%. That agrees with the 99.5% cross-tree figure recorded at `plans/implementation.md:183`
for 500x100.

Pairing only within a tree could keep both bits in one index. Its sequence would be
`ldr q`, `dup`, `fcmgt.2d`, `and.16b` with bit weights, `addp.2d`, `fmov`, `orr`, which is 7 on
NEON: the same as two scalar splits. The `fcmgt`/`and`/`addp`/`fmov` part is verified as the
`mask2` function in `probes/mask_bitcast.ll`. The number of such pairs is also tiny. In the
synthetic 100x100 model only 16 pairs of splits inside a tree use the same feature (52 at
300x100, 76 at 500x100, from `probes/model.py`). With uniform random features the expected
count is 15/F per tree. Even at 2 saved per pair, the best case is 32 of 4453 instructions
(0.7%). Changing the planner cannot rescue it.

### C2. Tree-major lanes (the lanes are the levels of one tree)

The single-tree probes use features 3, 17, 40, 55, 77 and 90 (no two adjacent) and base-register
thresholds. Only the loop body is counted, not `ret`/`vzeroupper`.

| Target | Probe | Per tree | Sequence |
|---|---|---:|---|
| NEON | `tree_major.ll @tree2x3_sli` | 22 | gather 9 (3 `ldr d`, 3 `add`, 3 `ld1.d {v}[1]`), thresholds 2 (`ldp q` + `ldr q`), 3 `fcmgt.2d`, 2 `sli.2d`, index 3 (`mov.d x, v[1]`, `fmov x, d`, `add x, x, x, lsl #3`), leaf 3 (`add`, `ldr d`, `fadd`) |
| NEON, features already in d registers | `tree_major_a64_resident.ll @resident_vec` | 16 | gather is 3 `mov.d v[1], v[0]` |
| NEON, scalar masks | `@scalar_masks` | 24 | 6 `fcmgt d` + 5 `sli d` |
| AVX2 | `tree_major_x86_alt.ll @tree4x2_gtoq` | 14 | gather 7 (`vmovsd`, `vmovhpd` x2, `vinsertf128`, `vmovsd`, `vmovhpd`), 2 `vcmpgt_oqpd mem`, 2 `vmovmskpd`, `shll`, `leal`, `vaddsd mem` |
| AVX2, generic `fcmp ogt` | `tree_major_x86.ll @tree4x2` | 16 | LLVM swaps to `vcmpltpd` with register thresholds (+2 `vmovapd`) |
| SSE2 (x86-64 generic) | `tree_major_sse2.ll` | 20 | 3x(`movsd`+`movhpd`), 3x(`movapd`+`cmpltpd`), 3 `movmskpd`, 2 shift, 2 or/lea, `addsd` |
| AVX-512 (info) | `tree_major_x86.ll @tree8`, v4 | 11 | gather 8 (incl. `vinsertf64x4`), `vcmpgtpd mem, zmm, k`, `kmovb`, `vaddsd` |
| scalar baseline, same addressing | `tree_scalar.ll` | 29 (NEON), 32 (AVX2/SSE2) | NEON: 6 `ldr`, 3 `ldp`, 6 `fcmp`, 6 `cset`, 6 `orr`/`ubfiz`, `ldr`, `fadd` |

Whole-model counts for synthetic trees at depth 6, per tree all-in:

| Model | NEON scalar | NEON tree-major | AVX2 scalar | AVX2 tree-major | AVX-512 tree-major |
|---|---:|---:|---:|---:|---:|
| 100 trees, 100 features, thresholds distinct | 31.9 | 22.7 | 31.6 | 14.4 | 11.7 |
| 300 trees, 100 features, thresholds distinct | 32.9 | 23.4 | 31.6 | 14.5 | 11.7 |
| 300 trees, 100 features, 8 borders per feature | 32.9 | 23.5 | 31.7 | 14.5 | 11.6 |
| 100 trees, depth 8 | 41.9 | 28.8 | 43.2 | 17.5 | 14.4 |

Stack references: 0 in every row. The +0.7 NEON at 300 trees is `mov`/`movk` for leaf offsets
beyond the `ldr d` immediate range. Leaves take 512 bytes per tree, and 300 trees overrun the
32760-byte range. Moving the base once per chunk removes that cost.

Here is what the gather costs on NEON, the maintainer's concern. I classified the
100x100 whole-model output (`s_treemajor_a64_100x100_d6_b0.s`):

- The gather takes 830 of 2266 instructions, 8.3 per tree: 243 `ldr d` and 26 `ldp d` for
  lane 0, 235 `ld1.d` lane loads, 235 `add` for lane-load addresses (`ld1` lane has no
  immediate offset), 62 `ins` and 26 `mov.16b` copies where LLVM reused a loaded feature.
- Scalar spends 6 loads per tree, so the gather adds about 2.3 per tree.
- The compare-and-index part falls from about 21 (3 `ldp`, 6 `fcmp`, 6 `cset`, 6 `orr`) to
  about 10.75 (2.75 threshold loads, 3 `fcmgt`, 2 `sli`, 3 to combine).

The gather therefore eats about a quarter of the saving on NEON, not most of it. The net is
-9 instructions per tree (-29%). On AVX2 the gather is 7 per tree and costs the same whether
the value is in memory (`vmovhpd m64`) or a register (`vunpcklpd`). Feature residency buys
nothing there.

The emitted code may permute the leaf table (H6). Leaves are copied into private constant
globals at `src/jit.rs:356-380`. Each formulation implies this permutation:

- AVX2 and AVX-512 use the identity mapping: bit j comes from lane j, and bits 4-5 come from
  the xmm mask. The existing level-k-to-bit-k convention stays as it is.
- NEON uses `sli`. Vector p holds levels (2p, 2p+1). `r = sli(sli(m0, m1, 1), m2, 2)` leaves in
  each lane a 3-bit two's complement value v = b_a + 2 b_b - 4 b_c. The index
  v_lane0 + 8 v_lane1 falls in [-36, 27] and is one-to-one with the six bits. The table is
  stored with a +36 entry bias, and leaf[ref] goes to slot 36 + v0 + 8 v1. At depth 8 (4
  vectors, 3 `sli`) the bias is 136 and the index falls in [-136, 127].

I executed the NEON kernel natively (`probes/check_sli.c` against `tree_major.ll`). It gave
1,000,000 matching results against the reference rule with this permutation. Rows and
thresholds mixed NaN, both infinities, both zeros, both smallest subnormals, max finite and
forced equality. A negative control (reference using `>=`) is caught at the first model.

LLVM does the natural thing for the lane loads: `ldr d` + `add` + `ld1 {v.d}[1]` on NEON,
`vmovsd` + `vmovhpd` + `vinsertf128` on AVX2. `<6 x double>` compares are unusable (see C4),
so depth 6 must be written as 3x2 lanes on NEON and 4+2 lanes on AVX2.

### C3. Binarize then compare (H7)

Exactness, case by case. Take feature f with the model's distinct thresholds on f, sorted
ascending: b_0 <= ... <= b_{B-1}. Constant-false splits (t = +inf or t = NaN) are folded at
compile time to "bit always 0". Define bin(x) = #{r : x > b_r}.

1. Ordered x (including +-inf). The set {r : x > b_r} is a prefix of 0..B-1: if x > b_r and
   b_{r'} <= b_r then x > b_{r'}. So bin(x) = k with x > b_r exactly when r < k, which means
   x > b_r iff bin > r. This holds even when two entries compare equal, for example -0.0 and
   +0.0 kept as separate borders, because the prefix argument only needs <=. Duplicates waste
   a compare but stay exact.
2. `NanMode::AsIs`, x = NaN. Every ordered `x > b_r` is false, so bin = 0 and every `bin > r`
   is false. That matches the reference at `src/parsed_tree.rs:203-212`, where `NaN > t` is
   false.
3. `NanMode::AsFalse`. The reference maps NaN to -inf. `-inf > t` is false for every t,
   including t = -inf (strict) and t = NaN. So AsFalse behaves exactly like AsIs under strict
   `>`, for direct compares and for bins. The `uno` + `select` at `src/jit.rs:244-262` is dead
   work for AsFalse features.
4. `NanMode::AsTrue`. The reference maps NaN to +inf, and `+inf > t` is true iff t < +inf.
   Constant-false splits are already folded, so every remaining border is below +inf and bin
   should be B. Two exact ways:
   - Substitute +inf at load, as the code does today. The value is then ordered and case 1
     applies.
   - Compare with the unordered-greater predicate: `fcmp ugt` in IR, `vcmpnlepd` on x86. On
     NEON use `fcmge(t, x)` and count the complement as B - #{t_r >= x}, which is B for NaN.
     This needs no select.
5. NaN threshold. It is identical to a +inf threshold: nothing is greater than either. Both
   are constant false. `Trees::new` (`src/parsed_tree.rs:108-151`) does not reject NaN
   thresholds, so the fold has to handle them.
6. Subnormals. Direct and binarized compares both use the same FP compare instruction class
   (`fcmgt`/`vcmppd`), under the same FPCR.FZ / MXCSR.DAZ state as today's `fcmp`/`ucomisd`.
   No new risk.

The byte compare is exact because bins and ranks are unsigned bytes: `cmhi` on NEON,
`vpcmpnleub` on AVX-512. On AVX2 the signed `vpcmpgtb` is exact while every B <= 127. Above
that, XOR both sides with 0x80 (one instruction per bin vector in the prologue, ranks biased
at compile time). An unused lane gets rank 255, and `bin > 255` never holds.

Bins must fit in a byte. When B > 255, split the feature into two virtual features, each
counting one contiguous run of borders (up to 255 each). CatBoost's default is 254 borders
per feature, and a model only needs the borders its splits actually use.

Costs are from the whole-model probes. The prologue groups row-adjacent features in lanes and
loops over ranks. Each rank costs one compare against a threshold vector (padded with NaN) plus
one accumulate. On AVX-512, `select` spelling lowers to `vcmpgtpd` + masked `vpsubq`
(`count_accumulate.ll @acc_select`); `sext`+`sub` gives 3 instructions. The counts are then
truncated to bytes: `uzp1` chain on NEON, `vpmovqb` on AVX-512.

| Target | Prologue fit (from 100x100 and 300x100 worst case) | Per-tree phase |
|---|---|---|
| NEON | about 3 x sum over feature pairs of max B_f, plus about 1 per pair (`fcmgt.2d`, `add.2d`, `ldr q` per rank) | about 9.75: per 2 trees, 3 `ldr q` (2 index, 1 rank), `tbl.16b {4 regs}`, `tbx.16b {3 regs}`, `cmhi.16b`, `and.16b`, `ext`, `zip1`, `addv.8h`, `fmov`, `lsr`, `and`, then 2x(`add`, `ldr d`, `fadd`) |
| AVX2 | about 2 x sum over feature quads of max B_f, plus about 5.5 per quad | about 6.25, plus 7 `vbroadcasti128` per chunk: per 4 trees, 7 `vpshufb ymm, mem`, 6 `vpor`, `vpcmpgtb mem`, `vpmovmskb`, about 6 to extract bytes, 4 `vaddsd mem` |
| AVX-512 (info) | about 2 x sum over feature octets of max B_f, plus about 5.4 per octet | about 3.5: per 8 trees, `vmovdqa64` index, `vpermi2b`, `vpcmpnleub mem`, `kmovq`, byte extracts, 8 `vaddsd` |

The maintainer's question is the cheapest AVX2 byte gather for 6 bins out of up to 128 bytes.
`vpshufb` cannot cross a 16-byte lane. The cheapest sequence I found fills both lanes of 7
ymm registers with the same 16-byte bin chunk (`vbroadcasti128`, once per chunk function). A
`vpshufb` with a memory control then pulls from any chunk into either half, with a zero where
the control's high bit is set. An OR tree combines the pieces, so one ymm gathers 4 trees (32
bytes). That costs 13 instructions per 4 trees, 3.25 per tree (`probes/avx2_bytegather.ll`,
59 instructions for 8 trees including broadcasts). An xmm version gathering 2 trees costs
about 5 per tree (`s_binarize_avx2_*` without `TB_AVX2_YMM`: 275 `vpshufb` + 225 `vpor` for
100 trees). Cross-lane dword permutes (`vpermd`) do not help: each reaches only one source
register. `vpinsrb` from memory costs one instruction per bin, no better than the
tree-major gather.

NEON's per-tree gather is 2 per tree: two index loads and `tbl`+`tbx` per 2 trees. Tree-major
spends 8.3.

Whole-model per tree, all-in, depth 6:

| Model | NEON tree-major | NEON binarize | AVX2 tree-major | AVX2 binarize | AVX-512 tree-major | AVX-512 binarize |
|---|---:|---:|---:|---:|---:|---:|
| 100x100, thresholds distinct (sum B = 600 = splits) | 22.7 | 21.9 | 14.4 | 12.7 | 11.7 | 7.1 |
| 300x100, thresholds distinct (1800) | 23.4 | 21.6 | 14.5 | 11.3 | 11.7 | 6.4 |
| 300x100, 8 borders per feature (sum B = 800) | 23.5 | 15.2 | 14.5 | 8.9 | 11.6 | 4.9 |
| 100x100 depth 8 | 28.8 | 26.2 | 17.5 | 14.2 | 14.4 | 7.7 |

For the worst case, sums over feature groups of max B_f at 100x100 are 353 for pairs, 212 for
quads and 125 for octets. At 300x100 they are 1023, 559 and 329.

Crossover by instruction count:

- NEON: binarize is ahead when 3 P2 + 42 < 12.7 T, where P2 is the sum over feature pairs of
  max B_f. That means P2 < 4.2 T. The worst case has P2 of about 3.4-3.5 T, so binarize
  narrowly leads by count everywhere.
- AVX2: binarize is ahead when 2 P4 + 5.5 F/4 < 7.3 T. The worst case has P4 of about 1.9-2.1 T,
  so binarize leads by 12-22% even when all thresholds are distinct, and by 39% at 8 borders
  per feature.

Measured time on NEON disagrees with the count in the worst case. Timings are on the M4 under
load, median of 7 rounds of 200,000 dependent calls, three alternating runs, including my C
driver (indicative only):

| Model | scalar | tree-major | binarize |
|---|---:|---:|---:|
| 100x100 distinct | 111.8 ns | 94.4-95.0 ns | 102.6-104.1 ns |
| 300x100 distinct | 331.0 ns | 263.0 ns | 297.9 ns |
| 300x100, 8 borders per feature | 331.7 ns | 268.7 ns | 224.4 ns |

Binarize runs at a lower IPC than tree-major on the M4. Its prologue is about 700
vector-pipe operations (`fcmgt.2d` + `add.2d`, 2 lanes each), and M4 has 4 FP/SIMD pipes. The
scalar and tree-major paths spread more work onto integer ALUs and load ports. This is a
hypothesis, not isolated. llvm-mca's `apple-m4` model is Cyclone-derived, dispatch width 6,
and reports `CyUnitV` pressure 82% on the prologue. It agrees in direction: 675 / 579 / 708
cycles at 100x100, 2024 / 1778 / 2063 at 300 distinct, 2034 / 1778 / 1641 at 300 shared. So on
NEON the time crossover lies somewhere between "all thresholds distinct" (tree-major wins by
7-12%) and "8 borders per feature at 300 trees" (binarize wins by 17%).

On x86 the only evidence is llvm-mca, summed over functions with no overlap between them
(rough):

| Model | ADL scalar | ADL tree-major | ADL binarize | Zen3 scalar | Zen3 tree-major | Zen3 binarize |
|---|---:|---:|---:|---:|---:|---:|
| 100x100 distinct | 1206 | 394 | 382 | 705 | 423 | 350 |
| 300x100 distinct | 3618 | 1196 | 1031 | 2118 | 1270 | 962 |
| 300x100 shared 8 | 3618 | 1207 | 781 | 2123 | 1285 | 763 |

Binarize depends on the number of features the model uses, and tree-major does not. On NEON,
`tbl` covers 64 bytes per 4-register table, so table ops per 2 trees = ceil(F_used / 64): 2 at
100 features, 16 at 1000. On AVX2, `vpshufb` ops per 4 trees = number of 16-byte bin chunks
the 24 bins touch, at most min(24, ceil(F_used / 16)): 7 at 100 features, up to 24 at 1000.
For the repo's 500x1000 workload, binarize would lose on both targets. This part is analytic
from the architectural table widths, not probed. The generator supports at most 112 features.

### C4. LLVM feasibility

Each check below is verified with `llc` output in the named probe.

| Question | NEON (generic) | AVX2 | AVX-512 (info) |
|---|---|---|---|
| `bitcast <8 x i1> to i8` after `fcmp <8 x double>` (`mask_bitcast.ll @mask8`) | 4 `fcmgt.2d`, 3 `uzp1`, `and` with a constant, `addv.8h`, `fmov`, `and`. In `tree_major.ll @tree8` the `zext` adds an `str b` / `ldrb` stack round trip. Do not use. | 2 `vcmpltpd`, `vpackssdw`, `vpermq`, `vmovmskps` (2 extra). Use two `<4 x i1>` instead. | `vcmpltpd zmm, k`, `kmovd` (`kmovb` with DQ / x86-64-v4) |
| `<4 x i1>` / `<2 x i1>` to integer | `<2 x i1>`: `fcmgt`, `and` const, `addp.2d`, `fmov` | `vmovmskpd ymm` / `xmm`, exactly as wanted | `kmovd` + `and` |
| `<6 x i1>` to i6 (`@mask6`) | 6 `umov.b` + 5 `bfi`. Unusable. | 6 `vpextrw` + `and`/`shl`/`or` chains. Unusable. | `kmovd` + `andl $63`. Fine. |
| byte mask `icmp ugt <8 x i8>` / `<16 x i8>` to int | `cmhi.8b`, `and`, `addv.8b`, `fmov`. This is the `and`+`addv` wanted. `<16 x i1>` adds `ext` + `zip1` and uses `addv.8h`. | `vpminub`+`vpcmpeqb`+`vpmovmskb`+`not` (unsigned); signed `vpcmpgtb`+`vpmovmskb` | `vpcmpnleub` + `kmovd` |
| Getting the `sli` mask packing | Only via `llvm.aarch64.neon.vsli.v2i64`. Generic `and`/`shl`/`or` gives `add.2d`+`sub.2d`+`sli` (one more instruction). | n/a | n/a |
| Generic `shufflevector <16 x i8>`, constant mask (`shuffle_a64.ll`) | `tbl.16b {v0, v1}` with index from the constant pool (1 or 2 sources); `tbl.8b` for one source. For 3-4 table registers, use `llvm.aarch64.neon.tbl4` / `tbx3` (verified in `s_binarize_a64_*`: `tbl.16b {v2..v5}`, `tbx.16b {v7..v9}`). | 16-byte in-lane only: `vpshufb` | `vpermt2b` (with VBMI), even for `<16 x i8>` |
| Generic `shufflevector <64 x i8>` two sources (`shuffle.ll`) | n/a | n/a | `vmovdqa64` index + `vpermt2b` |
| `<2 x double>` from two row loads | `ldr d` + `add x` + `ld1.d {v}[1], [x]`, 3 per pair (no immediate offset on lane loads); from registers `mov.d v[1], v[0]` | `vmovsd` + `vmovhpd m64` (2 per pair) | same |
| `<4 x double>` from four row loads | n/a | `vmovsd`, `vmovhpd`, `vmovsd`, `vmovhpd`, `vinsertf128` (5); two adjacent features become one `vinsertf128 $1, mem` | `vgatherdpd` from `llvm.masked.gather` is 4 instructions (`kxnorw`, index load, `vxorpd`, gather) but microcoded; not relied on |
| x > t with a memory threshold operand | `ldr q`/`ldp q` separate (no memory operands on NEON) | Generic `fcmp ogt` gives `vcmpltpd` with the threshold in a register (+1 `vmovapd`). `llvm.x86.avx.cmp.pd.256(x, t, 30)` (GT_OQ) folds the load: `vcmpgt_oqpd mem, ymm, ymm`. | Generic `fcmp ogt` folds: `vcmpgtpd mem, zmm, k` |

Target strings:

- NEON needs nothing. Every instruction above (`fcmgt`, `sli`, `tbl`, `tbx`, `cmhi`, `addv`,
  `uzp1`) is base AdvSIMD and appears under `-mcpu=generic`.
- AVX2 needs the function attribute `"target-cpu"="x86-64-v3"` with `"target-features"=""` at
  `src/jit.rs:162`. I verified that this per-function attribute alone switches codegen to
  `vcmpgt_oqpd`/`vmovmskpd` under a generic machine (`probes/attr_probe.ll` with
  `llc -mcpu=generic`). The MCJIT path creates its own engine at `src/jit.rs:49`. I expect the
  per-function subtarget to govern there as well, but I did not run MCJIT.
- AVX-512 as probed is `"x86-64-v4"` plus `"+avx512vbmi"`. VBMI is not part of v4.
- For the host, inkwell 0.9 exposes `TargetMachine::get_host_cpu_name()` and
  `get_host_cpu_features()` (`inkwell-0.9.0/src/targets.rs:1102,1113`). Pass them to
  `create_target_machine` (`src/jit.rs:78-85`, which hard-codes `"generic", ""`) and to the
  function attributes.

Compiling for a feature the executing CPU lacks raises SIGILL at the first such instruction.
This JIT compiles in-process on the machine that runs the code, so host detection cannot
produce foreign instructions. The only risk is if compiled code were ever cached or shipped
elsewhere, which it is not today. The options, and the choice is the user's:

1. Stay generic. On x86 that is SSE2, where tree-major still gives 20 vs 32 per tree and
   binarize is unavailable (no `pshufb`).
2. Use host features at JIT time.
3. Pin a level such as x86-64-v3 and refuse to compile, or fall back, when the host lacks it.

### C5. Other findings, only where concretely cheaper

The first finding is the NaN handling. Under strict `>`, `NanMode::AsFalse` needs no
load-time `uno`+`select`, which is 2 instructions per load of such a feature at
`src/jit.rs:244-262`. `NanMode::AsTrue` can use an unordered-greater compare instead, once
constant-false splits are folded. This applies to every formulation. The synthetic workloads
use only `AsIs`, so the counts above do not change.

The second is the leaf-offset range on NEON, a detail for piece A. With one base register,
leaf tables beyond 32760 bytes (about 64 depth-6 trees) cost `mov`+`movk` per tree. Advancing
the base once per chunk avoids that. The same applies to `ldp q`, whose immediate range is
+-1008 bytes: LLVM paired only 25 of 275 threshold loads.

## 2. Hypotheses touched

H3 is confirmed, with a correction. It is right that the vector axis is wrong (threshold-major
instead of tree-major), and that 99.5% of pairs cross trees (2.1% of 234 pairs share a tree in
100x100). The correction is that the pair's small win today (1.7 per pair on ARM64, 0.75 on
x86) comes only from loading two thresholds with one GOT `ldr q`. With base-register constants
the pair is 2 instructions worse than scalar on NEON, even on SSE2 and 1 better on AVX2.

H5 is confirmed, with details. AsIs and AsFalse are observationally identical under strict
`>`, and a NaN threshold is the same as a +inf threshold (both constant false). Normalizing at
load is not needed by any formulation here. Ordered vector compares (`fcmgt`, `vcmpgt_oqpd`,
`vcmpgtpd`) already return the reference answer for NaN, as the executed NEON check confirms.
AsTrue is exact with an unordered-greater predicate once constant-false splits are folded.
`Trees::new` accepts NaN thresholds, so the fold is required. -0.0/+0.0 and duplicate borders
do not break binarization.

H6 is confirmed and executed. Leaves are private constant globals (`src/jit.rs:356-380`).
The NEON `sli` permutation (bias 36, index range [-36, 27]) matched the reference on 1,000,000
rows with special values.

H7 is partly confirmed. The exactness proof holds (section C3). By instruction count, binarize
beats tree-major on every target even in the worst case. By time on the M4, it loses in the
worst case (-7 to -12% relative to tree-major) and wins with shared borders (+17% at 300 trees
x 8 borders). It also degrades with used-feature count, which tree-major does not. It is a
conditional formulation, not a default.

H2 is touched as a side effect. Tree-major discards cross-tree feature residency. On x86 that
costs nothing, because a lane load from memory is one instruction, the same as a register
insert. On NEON residency is worth up to 6 per tree (22 vs 16, `resident_vec`), but LLVM
already reuses some loads within a chunk.

## 3. Recommendations

Summary table: instructions per depth-6 tree, all-in (prologue, chunk reloads and returns
amortized). The first figure is 100 trees x 100 features with all thresholds distinct (the
worst case for binarize). The second is 300 x 100 with 8 borders per feature. No formulation
gets cross-tree feature reuse, except the current SIMD2 row, which is read from the production
captures (scheduler reuse and today's constant addressing).

| Formulation | NEON | AVX2 | AVX-512 (info only) |
|---|---|---|---|
| Current SIMD2 (captures, 100x100) | 40.5 (scalar default 44.5). With base-register thresholds the pair costs 9 vs 7 scalar, a loss. | Not emitted today: x86 is SSE2, 38.4 (scalar 40.2). Pair 9 vs 10 scalar. | n/a |
| Scalar, base-register thresholds (plain IR, the floor I could verify; piece A's 3-per-split target is this shape) | 31.9 / 32.9 | 31.6 / 31.7 (SSE2 single tree 32) | 31.6 / 31.7 |
| Tree-major lanes | 22.7 / 23.5, of which gather 8.3 | 14.4 / 14.5, of which gather 7 (SSE2 single tree: 20) | 11.7 / 11.6 |
| Binarize then compare | 21.9 / 15.2. Fixed cost about 3 per (feature pair x rank) + 1 per pair. Per-tree phase 9.75. Needs used features <= 112 for 2 table ops per 2 trees. | 12.7 / 8.9 (ymm byte gather). Fixed cost about 2 per (feature quad x rank) + 5.5 per quad. Per-tree 6.25 + 7 broadcasts per chunk. Needs SSSE3/AVX2. | 7.1 / 4.9 |

Ranked by NEON and AVX2:

1. Retire SIMD2 as a direction. This is small and verified. Once thresholds come from a base
   register, `ComparisonMode::Simd2` is 2 instructions per pair worse than scalar on NEON and
   about even on x86 (`probes/simd2_pair.ll`). Pairing within a tree cannot help: at most 16
   pairs at 100x100, and zero saving on NEON. No planner or emitter work should build on it.
   Risk: none.

2. Tree-major lanes on NEON. Verified by count, by native execution for exactness, and by M4
   timing.
   - Emitted code per tree: `ldr d`/`add`/`ld1 {v.d}[1]` lane loads (or `ins` when resident),
     `ldp q`/`ldr q` thresholds off the base register, 3 `fcmgt.2d`, 2 `sli.2d` (through
     `llvm.aarch64.neon.vsli.v2i64`), `mov.d`+`fmov`+`add lsl #3`, then `add`+`ldr d`+`fadd`.
     The leaf table is stored permuted with a +36 bias.
   - Delta: -9.2 instructions per tree against scalar with the same addressing (-29%).
   - At 100 trees: 3195 to 2266 instructions, 111.8 to 94.7 ns (-15%) on my generated kernels.
   - At 300 trees: 9863 to 7024 instructions, 331 to 263 ns (-20%).
   - Against today's production kernel (4453 instructions, which also carries the old constant
     materialization), 2266 is -49%.
   - Risks: no cross-tree feature sharing, already included in the count. The machine
     scheduler spilled scalar at 8-tree chunks (270 stack refs) but tree-major had 0 up to 8
     trees, so keep chunks at 4-8 trees. The NaN modes need the predicate rules from C3/C5.

3. Tree-major lanes on AVX2 and SSE2. Verified by count and llvm-mca, not executed.
   - Emitted code per tree: `vmovsd`/`vmovhpd`/`vinsertf128` gather, 2 `vcmpgt_oqpd mem`
     (target intrinsic, predicate 30), 2 `vmovmskpd`, `shl`+`lea`, `vaddsd mem`.
   - Delta: -17 per tree (-54%) on AVX2, -12 per tree (-37%) on SSE2 generic.
   - At 100 trees: 3158 to 1444 instructions. llvm-mca: Alder Lake 1206 to 394 cycles, Zen 3
     705 to 423.
   - At 300 trees: 9480 to 4343, and 3618 to 1196 (ADL), 2118 to 1270 (Zen 3).
   - This is the same plan structure as NEON, so one planner change serves both targets.
   - Risk: no x86 execution anywhere yet.

4. Binarize then compare, as a second formulation chosen at compile time from border
   statistics. Verified by count; NEON also by timing. Exactness is argued, not executed for
   this formulation.
   - Emitted code: a prologue of row-contiguous feature groups x ranks (`fcmgt.2d`+`add.2d` on
     NEON, `vcmpgt_oqpd`+`vpaddq` on AVX2), truncated to bytes into a 128-byte stack buffer.
     Then, per group of trees (2 on NEON, 4 on AVX2): `tbl4`+`tbx3`, or 7 `vpshufb ymm` + OR
     tree, then one byte compare against a rank vector, mask to index, leaf.
   - On AVX2 it beats tree-major by count in every tested case: -12% at 100x100 distinct, -39%
     at 300x100 with 8 borders. llvm-mca agrees, 1031 vs 1196 and 781 vs 1207 on ADL.
   - On NEON it wins only when borders are shared: -17% time at 300 x 8 borders. It loses 7-12%
     in the all-distinct worst case even though it has fewer instructions.
   - Risks: cost grows with used features, which rules it out for large F such as the 500x1000
     workload. It adds a serial prologue before any tree can start. It needs real model border
     statistics to pick. On AVX2, B_f > 127 needs a bias XOR.

5. Handle NaN modes through predicates rather than a load-time select (AsFalse becomes a plain
   ordered compare; AsTrue uses unordered-greater with constant-false splits folded). This
   saves 2 instructions per load of a non-AsIs feature, whatever the formulation. Risk: low,
   and the case analysis is in C3.

Which formulation per target:

- On NEON, use tree-major lanes now. Consider binarize later, only if real models turn out to
  share borders heavily and use no more than about 112 features.
- On AVX2, use tree-major lanes first, since it is the same planner change as NEON and flat in
  feature count. Add binarize with the ymm byte gather where border statistics and feature
  count favor it. The margin is small in the worst case and large with shared borders.
- On SSE2-only x86 (target-cpu generic), use tree-major lanes (20 vs 32 per tree). Binarize is
  not available there.
- AVX-512 favors binarize strongly (7.1 / 4.9), but it is out of scope for the decision.

The plan structure changes as follows. Today, `Action::Compare { feature, comparisons }`
groups predicates by feature and fans bits out to many trees' index registers
(`src/sort_policy.rs:329-454`). Both new formulations group by tree.

Tree-major needs one action per tree, carrying the tree's lane order (levels paired into
vectors: 3x2 on NEON, 4+2 on x86), its threshold-vector offset in the constant blob, its leaf
offset and permutation, and its sum. Two things become unnecessary:

- `Load`/`Release` residency and the `RegisterBudget` logic. Lanes load directly from the row.
  On NEON a planner could still route features resident in registers into `ins`, which is
  worth up to 6 per tree.
- Carrying per-tree index registers across continuations. A tree finishes inside one action,
  so chunks carry only the sums.

Binarize adds prologue actions (one per group of row-adjacent features: rank count and
threshold-vector offsets) and tree-group actions (2 or 4 trees: byte-slot index vectors, rank
vector, leaf offsets). Chunks carry the sums, and bins sit in a stack buffer reloaded per chunk
(4 instructions on NEON, 7 broadcasts on AVX2).

## 4. Proposed experiments for the orchestrator

All commands run from the worktree root. The probes need no cargo build.

1. Reproduce every count and llvm-mca figure: `sh probes/run_all.sh`. Expect the tables in
   sections C2, C3 and 3 exactly; output from my run is in `scratchpad/c_run_all.txt`.

2. Exactness of the NEON tree-major kernel, run natively:

   ```
   /opt/homebrew/opt/llvm@21/bin/llc -O1 -mtriple=arm64-apple-macosx -mcpu=generic -filetype=obj -o probes/tree_major_a64.o probes/tree_major.ll
   cc -O1 -o probes/check_sli probes/check_sli.c probes/tree_major_a64.o
   probes/check_sli
   ```

   Expect `ok 1000000 checks`.

3. NEON timing on an idle machine, run serially:

   ```
   sh probes/time_all.sh 100 100; sh probes/time_all.sh 300 100; TB_BORDERS=8 sh probes/time_all.sh 300 100
   ```

   Expect tree-major about 15-20% below scalar. Expect binarize above tree-major when all
   thresholds are distinct, and below it with 8 borders.

4. Find the NEON crossover by border sharing. Run `TB_BORDERS=k sh probes/time_all.sh 300 100`
   for k = 2, 4, 16, 32. Expect binarize to beat tree-major below some k between 8 and 32
   (sum B = 100k against 1800 splits).

5. Once tree-major is implemented in `src/jit.rs` and the planner as an opt-in mode, compare
   `cargo run --offline --release --example scalar_bench -- 100x100` (and `500x100`,
   `500x1000`) against the default. Expect roughly the probe's -15 to -20%, plus whatever
   piece A's constant addressing contributes. Confirm 0 stack references in `inspect_code`
   output with chunks of at most 8 trees.

6. If any x86 AVX2 machine is available, build the x86 probes with `llc -filetype=obj` for
   `x86_64-unknown-linux-gnu -mattr=+avx2` and link `time_predict.c`. This would be the first
   x86 execution; llvm-mca predicts tree-major about 3x faster than scalar on Alder Lake and
   1.7x on Zen 3.

## 5. Questions for the user

1. What are the real models' border statistics: for each feature, how many distinct
   thresholds the splits use, and how many features are used in total? These decide whether
   binarize is worth adding on either target. It needs heavy sharing on NEON and at most about
   112-128 used features on both.
2. Which x86 is the deployment target? Is AVX2 guaranteed (x86-64-v3), or must the SSE2
   baseline stay? Is compiling for host features at JIT time acceptable? It is safe for
   in-process MCJIT; SIGILL is only possible if code is cached or moved.
3. What is the depth range? Depth 8 fits all formulations: NEON uses 4 vector pairs and bias
   136, AVX2 uses two ymm, and binarize fills a full byte group. Depths above 8 break the
   one-byte-per-tree binarize layout and the single-ymm/zmm mask.
4. How common are `AsTrue` and `AsFalse` features in real models? That sets the value of the
   predicate-based NaN handling.
5. f32 is a question, not a recommendation. With f32 borders, NEON compares 4 lanes per
   instruction, which would halve the binarize prologue and make tree-major one `fcmgt.4s`
   plus one `fcmgt.2s` per tree. That would change the NEON ranking in binarize's favor.
