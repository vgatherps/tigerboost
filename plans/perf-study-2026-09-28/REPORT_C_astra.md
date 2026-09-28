# SIMD formulation study

Tree-major comparisons are the first SIMD experiment to run. The verified depth-6 index code takes 20 instructions on NEON, 13 on AVX2, and 12 on AVX-512, with six unrelated feature loads included. These are small function bodies, not integrated forest timings. Byte binarization has a larger possible gain on AVX-512 with VBMI. Its setup cost and register lifetime make it a second experiment.

No tracked file changed. No production recommendation was implemented. LLVM is Homebrew 21.1.8. `cargo build --offline --release --examples` passed with `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21`. The standalone ARM64 checks passed 2,740 assertions. There is no new latency measurement and no x86 execution result.

## 1. Answers and evidence

### Current SIMD2 cost

Cross-tree pairing prevents a vector comparison from producing one tree's index directly. It does not make a win impossible. The supplied SIMD2 kernels already have fewer instructions than scalar on both targets. They still lost in the recorded ARM64 timings. Instruction count alone does not price vector loads, register transfers, dependencies, or instruction throughput.

The following counts were recomputed by `probes/capture_stats.py`. Text bytes come from assembling the supplied files with `llvm-mc` and reading their object sections. Counts include wrappers, continuations, and tail transfers. Stack counts include saved registers in wrappers; they are instruction counts, not frame sizes.

| Capture for 100 trees | Instructions | Text bytes | Stack-access instructions | Planned feature loads |
|---|---:|---:|---:|---:|
| ARM64 scalar scheduled | 4,453 | 17,812 | 20 | 394 |
| ARM64 SIMD2 scheduled | 4,051 | 16,204 | 20 | 307 |
| x86 scalar scheduled | 4,015 | 21,458 | 41 | 394 |
| x86 SIMD2 scheduled | 3,840 | 20,435 | 12 | 307 |

Both x86 captures target `generic` with no AVX features. The results are not AVX2 or AVX-512 measurements. Full histograms, border counts, and object section sizes are in [capture_stats.json](capture_stats.json).

There are 234 pairs and 132 scalar comparisons in this workload. Of the pairs, 229 cross trees: 97.86%. The brief's 1,333/1,340 figure is for **500** trees, as `plans/implementation.md:183` says. It is not the pairing rate of these captures. Each plan describes the same 600 distinct thresholds on 99 used features out of 100 declared features.

For an ordinary ARM64 pair that updates two already nonzero selectors, the sequence is:

```asm
; Threshold constant in the supplied JITDefault capture:
adrp    x8, threshold@GOTPAGE
ldr     x8, [x8, threshold@GOTPAGEOFF]
ldr     q1, [x8]
dup.2d  v0, v0[0]
fcmgt.2d v0, v0, v1
fmov    x9, d0
mov.d   x8, v0[1]
and     x9, x9, #16
and     x8, x8, #32
orr     selector_a, selector_a, x9
orr     selector_b, selector_b, x8
```

That is 11 instructions with the feature already loaded, or 12 with its scalar row load. A fresh load/splat can instead be `add` plus `ld1r.2d`, also two instructions. LLVM needs no separate instruction for the IR `sext`; `fcmgt.2d` already produces full masks. The extracts are one `fmov` and one `mov.d`, not two `mov.d` instructions. See the ARM64 SIMD2 capture at lines 83–114 and 127–179. Initial selectors can eliminate ORs; level zero can use subtraction from the full mask. This explains deviations from the regular-pair count.

Two scalar splits use two threshold materializations, each usually `mov; movk; fmov`, plus two `fcmp`, two `cset`, and two selector insertions. That is about 12 instructions before a shared feature load when a shifted OR absorbs bit placement; a separate `ubfiz` can add an instruction per split. See the ARM64 scalar capture at lines 55–84. The observed histogram contains 591 `movk`, so “three instructions for every threshold” is only an approximation.

The x86 SIMD2 capture uses this SSE2 pattern:

```asm
unpcklpd     feature, feature
movabsq     $threshold_pair, address
movapd      (address), masks
cmpltpd     feature, masks
movq        masks, a
punpckhqdq  masks, masks
movq        masks, b
andl        $16, a_low
andl        $32, b_low
orq         a, selector_a
orq         b, selector_b
```

That is 11 instructions before the row load. A `movapd` copy adds one when the original scalar feature must survive the splat. Two scalar splits commonly need two each of `movabsq`, zeroing `xorl`, `ucomisd`, `seta`, and selector combination, plus any explicit shifts: roughly 10–12 instructions before feature loads. Evidence is the x86 SIMD2 capture at lines 72–103 and the scalar capture at lines 92–139. With AVX2 enabled, the isolated current-pair probe uses `vmovddup`, a threshold load, `vcmpltpd`, `vmovq`, `vpextrq`, two ANDs, and two ORs: nine instructions when the threshold base is already supplied. AVX-512 emits the same nine here. See [pairs-avx2.s](pairs-avx2.s) and [pairs-avx512.s](pairs-avx512.s).

Restricting pairs to one tree removes most available pairs. Reconstructing tree levels from the plan gives at most 16 disjoint same-feature pairs in all 100 trees. This is a model bound, independent of scheduling. `operation` selects one feature before it scans consumers (`src/sort_policy.rs:329–391`); a planner-only restriction cannot put unrelated features in its two lanes. The current full-mask extracts also remain unless the emitter changes. On x86, an adjacent two-bit mask reduces nine instructions to six on AVX2 or five on AVX-512 in `pair_same_tree`. At most 48 or 64 instructions disappear across this particular forest before any lost load sharing. On NEON, the generic adjacent-bit probe takes ten, versus nine for the current pair. A resident vector of destination weights can reduce it to seven; loading those weights makes it eight or nine. Even the two-instruction saving has only 16 opportunities here.

In an artificial tree where all six splits use one feature, full tree-major processing can reuse one splat. The depth-6 NEON index body then falls from 20 to 12 instructions before the initial scalar feature load; AVX2 falls from 13 to seven. Those are bounds for a different feature pattern, not expected gains on this model. The useful change is to form a whole tree's index, not merely to exclude cross-tree pairs.

### Tree-major lanes and leaf-table permutation

A tree-major action must name all its level predicates and feature values. `Action::Compare` currently names one feature (`src/sort_policy.rs:90–99`), and the emitter only accepts one or two comparisons (`src/jit.rs:277–321`). Changing the planner's pair preference cannot express this formulation.

The probes load unrelated features at row indices 3, 17, 29, 48, 67, 91, 5, and 81. No contiguous-row discount is assumed. A separate probe supplies each feature in a scalar FP register. Threshold blocks are read through a supplied base pointer, with eight-byte alignment. This assumes the table addressing improvement studied in piece A; it does not silently retain the present per-vector GOT or `movabsq` setup.

| Target and formulation | Depth-6 gather | Depth-6 index body | Depth-8 gather | Depth-8 index body | Depth-6 / depth-8 body bytes |
|---|---:|---:|---:|---:|---:|
| NEON `sli`, row loads | 9 | 20 | 12 | 25 | 80 / 100 |
| NEON `sli`, scalar registers | 3 | 14 | 4 | 17 | 56 / 68 |
| AVX2 explicit compare, row loads | 7 | 13 | 10 | 16 | 75 / 94 |
| AVX2 explicit compare, scalar registers | 4 | 10 | 6 | 12 | 42 / 52 |
| AVX-512 plus DQ, row loads | 8 | 12 | 11 | 13 | 74 / 88 |
| AVX-512 plus DQ, scalar registers | 5 | 9 | 7 | 9 | 41 / 46 |

These counts exclude `ret`, `vzeroupper`, leaf selection, summation, and function alignment. All listed bodies have zero stack accesses. [probe_stats.json](probe_stats.json) records the counts and bytes. The public probe return adds `vzeroupper` on x86; a forest should pay transition costs at its external boundary, with continuation behavior checked after integration.

A NEON pair of arbitrary row values costs `ldr d`, `add` to the other row address, and `ld1.d {v}[1]`. The lane load has no arbitrary immediate offset, so it is not a two-instruction gather. With both values already in registers, `mov.d vA[1], vB[0]` is the assembler spelling of a lane insert. The depth-6 threshold loads become one `ldp q,q` plus one `ldr q`; depth eight uses two `ldp` instructions. Losing adjacent-load pairing adds one or two instructions, respectively.

The NEON reduction from [neon.s:16](neon.s) is:

```asm
; Three comparisons produce [level0,level1], [level2,level3], [level4,level5].
fcmgt.2d v0, v0, v2
fcmgt.2d v1, v1, v3
sli.2d   v0, v1, #1
fcmgt.2d v1, v4, v2
sli.2d   v0, v1, #2
mov.d    x8, v0[1]
fmov     x9, d0
and      x0, x9, #7
bfi      x0, x8, #3, #3
```

It uses no vector bit constants. Emitted bits 0,1,2 represent original levels 0,2,4; emitted bits 3,4,5 represent levels 1,3,5. At depth eight the orders are 0,2,4,6 followed by 1,3,5,7. For every original index `i`, set `new_leaves[permuted(i)] = old_leaves[i]`. This preserves the selected leaf exactly. Private constant leaf globals are created at `src/jit.rs:360–374`; only their copied initializer needs permutation. The source model and `Trees::leaf_indices` retain their original convention. The native harness checks every depth-6 and depth-8 index and the permutation.

A weighted-mask NEON reduction can use three vector ANDs, two ORs, `addp`, and `fmov`: seven reduction instructions if all three weight vectors are resident. Their loads are extra. The tested `sli` reduction uses six instructions, or seven at depth eight. The seven-instruction weighted reduction is verified in `neon.s`, function `weighted_masks`. A second tested `sli` version, `sli6_one_extract`, replaces the two extracts and `bfi` with `ext.16b; sli.2d #3; fmov; and #63`. It also takes 20 instructions for the complete depth-6 index and produces the same permuted index for all 64 masks. It uses one scalar extract, as suggested in the brief; there is no instruction-count gain over the first version. The alternative generic packed-boolean reduction has a compiler problem described below.

AVX2 needs `vmovsd; vmovhpd` for each pair and `vinsertf128` to join two pairs. Four scalar registers instead need two `vunpcklpd` plus one `vinsertf128`. The depth-6 probe uses a four-lane compare and a two-lane compare; LLVM removes the unused upper lanes of the second group:

```asm
vcmpgt_oqpd  (thresholds), ymm_values, ymm_masks
vmovmskpd    ymm_masks, low_bits
vcmpgt_oqpd  32(thresholds), xmm_values, xmm_masks
vmovmskpd    xmm_masks, high_bits
shll        $4, high_bits
orl         low_bits, high_bits
```

See [x86-avx2.s:9](x86-avx2.s). Depth eight uses two YMM comparisons. The natural bit mapping is already the reference mapping; no permutation is needed. The intrinsic's predicate immediate is 30, ordered greater-than. Generic `fcmp ogt` did not produce these folded threshold operands under AVX2.

AVX-512 joins the two 256-bit halves with `vinsertf64x4`, then uses one `vcmpgtpd` to a mask register. An eight-lane result becomes `kmovb` when DQ is enabled. The depth-6 generic probe still uses `kmovd; andb $63; movzbl`, so its measured index count is 12, not the ten-instruction ideal of gather plus compare plus move. See [generic-avx512dq.s:74](generic-avx512dq.s). No lane outside the tree affects its index. The comparison remains ordered.

For the instruction projections below, allow two more instructions per tree for leaf selection and adding its value. On NEON that is an indexed `ldr` plus `fadd` when the leaf base is available. On x86 it can be a base calculation plus an indexed memory `vaddsd`; with an already available leaf base it can be one instruction instead. This is a stated common allowance, not a measurement of an integrated model's address setup. Wrapper cost, scale/bias, continuation transfers, extra base setup, and NaN normalization are outside these local totals.

The brief's scalar target is about three instructions per split before row loads. Under that convention, depth six is 26 with six row loads and the leaf allowance, or 20 with resident features. Depth eight is 34 or 26. These are piece A's proposed targets, not a scalar floor established by this study. If “three per split” is instead achieved including row loads, its corresponding depth-6 total is 20 and NEON tree-major at 22 loses by two. The comparison must use the actual scalar code from piece A.

Hardware gather does not remove the other setup. `gather4` emits four instructions on AVX2: mask initialization, index-vector load, zeroing the destination, and `vgatherqpd`. An eight-lane AVX-512 gather with six active lanes emits five: index load, destination zero, immediate mask, `kmovd`, and `vgatherqpd`. See [gather-avx2.s:13](gather-avx2.s) and [gather-avx512.s:40](gather-avx512.s). A mask destroyed by gather must be recreated for another gather. AVX2's four-lane gather saves only one static instruction over four scalar lane loads; two gathers do not improve the seven-instruction depth-6 scalar gather. AVX-512 could replace eight scalar-gather instructions with five, but those instructions are not equal-cost operations. No Intel throughput number is inferred from an ARM host. Intel documents both gather-specific performance costs and dependence on affected CPUs' microcode in its [gather performance analysis](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/resources/gds-mitigation-performance-analysis.html). Benchmark it on the deployment CPU before choosing it.

### Binarization, exactness, and cost

Normalize NaN thresholds to positive infinity before sorting borders. Merge thresholds that compare identically, including negative and positive zero and repeated positive infinity. Let a used feature have sorted distinct borders `b[0] < ... < b[B-1]`, and define `bin(x)` as the number of true ordered comparisons `x > b[j]`.

For a non-NaN `x`, the true comparisons are exactly a prefix. If `x > b[r]`, at least `r+1` comparisons are true, hence `bin(x) > r`. If `bin(x) > r`, the true prefix includes position `r`, hence `x > b[r]`. Equality at a border stays false. This proof also permits repeated equal borders if ranks and the count both retain them, although removing them saves work.

`Trees::leaf_index` applies each feature's mode before strict comparison (`src/parsed_tree.rs:202–214`). The exact cases are:

| Input or border | Required bin behavior |
|---|---|
| NaN, `AsIs` | Every ordered comparison is false; bin is zero |
| NaN, `AsFalse` | Replaced by negative infinity; bin is zero, including a negative-infinity border |
| NaN, `AsTrue` | Replaced by positive infinity; count only borders strictly below positive infinity |
| Negative infinity | Bin zero |
| Positive infinity | All finite borders and negative infinity count; positive infinity does not |
| Either signed zero | Same result against every border; equality to either zero is false |
| NaN border | Always false; replacing it by positive infinity preserves this |

Thus `AsTrue` does not imply `bin = B` when positive infinity is a border. After canonicalization, that border remains false even for a missing feature in `AsTrue` mode. `AsIs` can stay NaN during ordered SIMD binarization; no replacement is needed. For instructions that require non-NaN inputs, replacing `AsIs` NaNs by negative infinity is valid. Do not assume finite operands or set `ninf`. The native checks cover the table, subnormals, all bit patterns at depths six/eight, and rank comparisons against the actual reference code. No floating-point exception-state contract is claimed.

The 254-border limit in the brief is false as a general CatBoost limit. Its [quantization settings](https://catboost.ai/docs/en/references/training-parameters/quantization) allow `border_count` from 1 to 65,535; 254 is the CPU default. `Trees::new` has no per-feature border cap (`src/parsed_tree.rs:109–145`). An unsigned byte holds bins 0 through 255, so up to 255 distinct borders fit. Above that, never truncate the bin. Use the direct comparison path or a separately costed wider-bin path. These byte estimates do not apply to a wider path.

For cost, write `G_w(f) = ceil(B_f / w)` for the number of vector groups of width `w`. Pad the final border group with positive infinity. It then contributes no extra true comparisons, even for positive infinity. Padding is real constant storage; it never reads beyond the feature row.

| Target | Instructions to compute one bin from a scalar feature | Additional work to store all bins |
|---|---|---|
| NEON | `2*G_2 + ceil(G_2/2) + 3`, with paired threshold loads | One row load and one byte insert per feature |
| AVX2 + POPCNT | `4*G_4` with the explicit ordered-compare intrinsic | One byte insert per feature; combine 128-bit banks and bias each 256-bit bank |
| AVX-512 + DQ + POPCNT | `4*G_8` | One byte insert per feature; combine 128-bit banks into ZMM banks |

The x86 broadcast can include the row load, so do not add a second load to those counts. `x86-extra-avx2.s` emits `vbroadcastsd 136(row), ymm`; `x86-extra-avx512.s` emits a threshold load and comparison against a broadcast row operand. Both match the table's counts at eight borders. Without DQ, the generic AVX-512 eight-bit mask conversion adds one instruction per group in these probes.

NEON sums the negative full masks in 64-bit lanes, applies `addp`, and negates once. This avoids a vector of ones. At eight borders, `negative_bin8` has 13 instructions, versus 15 for the generic zero-extension formulation. Its byte-insert version has 14; the row load makes 15. Evidence is [neon.s:165](neon.s) and [neon.s:332](neon.s). At 16 borders, the count is 23 before the row load and insertion. For large border lists these expressions are instruction estimates assuming bounded live temporaries; the straight-line probes do not establish spill-free behavior for 255 borders. Use bounded comparison groups and measure any continuation overhead. If threshold loads cannot pair, replace `ceil(G_2/2)` by `G_2`.

On AVX2 each group costs one folded compare, `vmovmskpd`, and `popcnt`, plus accumulation between groups. The broadcast and `G_4-1` additions give `4*G_4`. At eight borders that is eight instructions ([x86-avx2.s:93](x86-avx2.s)). AVX-512 uses the same formula with eight lanes and `kmovb`; at eight borders it is four ([generic-avx512dq.s:610](generic-avx512dq.s)). POPCNT is an explicit feature requirement, not something implied by spelling AVX2.

Keeping 100 byte bins needs seven NEON registers, four YMM registers, or two ZMM registers. That storage is not free to construct. The probes for packing use `mov.b` on NEON and `vpinsrb` on x86. `pack64` uses 64 moves/inserts plus three joins on AVX-512. The per-feature table charges the moves/inserts. Seven 128-bit banks need three further joins for four YMM registers, or five for two ZMM registers with the last bank partially filled. AVX2 also needs four bank-wide XORs with 0x80 so signed `vpcmpgtb` implements an unsigned comparison; ranks get the same bias at compile time. This costs seven extra instructions for a 99/100-feature model. AVX-512 needs five joins. NEON also needs one shared bit-weight constant, two setup instructions in the standalone assembly. There may be further base setup and initialization for the chosen continuation layout. For example, explicitly computing a fresh border-block pointer costs one address instruction per feature, about 99/100 more instructions per call here. A model-data base with legal folded offsets may avoid it. That unmeasured layout choice can erase much of the small 100-tree AVX-512 advantage; it is not charged as zero in a claimed full-kernel result.

The per-tree byte lookup costs are:

| Target | Gather and comparison to index, excluding leaf allowance | With two-instruction leaf allowance |
|---|---:|---:|
| NEON, seven bin registers, resident bit weights | 8 | 10 |
| AVX2, four 32-byte banks used by a tree | 12 | 14 |
| AVX2, three such banks used | 10 | 12 |
| AVX-512 BW + VBMI, two bin registers | 5 | 7 |

The NEON sequence in [neon.s:284](neon.s) is:

```asm
ldp      d7, d16, [controls]
tbl.8b   result, {bank0,bank1,bank2,bank3}, v7
ldr      d_rank, [ranks]
tbx.8b   result, {bank4,bank5,bank6}, v16
cmhi.8b  result, result, rank
and.8b   result, result, bit_weights
addv.8b  b_result, result
umov.b   index, result[0]
```

The first control selects features 0–63 and uses 255 for other lanes. The second selects features 64–99 after subtracting 64 at compile time, and uses 255 elsewhere. An out-of-range `tbl` lane gives zero; an out-of-range `tbx` lane preserves the first result. Two dummy level lanes use rank 255 and stay false. The native byte-lookup tests exercise values through 255 and both table groups. The standalone body loads the bit weights itself and therefore has ten instructions, not eight. A forest can keep that one constant resident; that assumption needs a full-kernel register check.

The AVX2 sequence in [x86-avx2.s:143](x86-avx2.s) uses four `vpshufb` with memory control operands, three `vpor`, `vextracti128`, another `vpor`, a signed byte comparison against biased ranks, `vpmovmskb`, and `and $63`. Each bank's two 16-byte halves put their selected bytes in matching output positions. All other shuffle bytes have their high bit set and become zero. ORs combine the disjoint contributions, then the two halves are combined. This respects the 16-byte `vpshufb` boundary. If a tree uses `U` of the four banks, omit absent banks: the cost is `2*U+4` before leaf work, provided both half positions need combination. Some trees can also omit that last two-instruction half combination.

The capture's trees use two banks 19 times, three banks 75 times, and four banks six times, for 287 bank uses. That gives an estimated 1,174 tree instructions including leaf work, before bin setup; three trees could save another two each by using just one half position. The four-bank upper bound alone would overstate this model's gather cost. No assumption that a single `vpshufb` can address 100 bytes is made.

On AVX-512, [x86-avx512.s:186](x86-avx512.s) emits:

```asm
vmovq        control_constant(%rip), xmm_control
vpermi2b     zmm_bank1, zmm_bank0, zmm_control
vpcmpnleub   (ranks), zmm_control, k0
kmovq        k0, rax
andl         $63, eax
```

This is a five-instruction index body with both source banks preserved. LLVM selects `kmovq`, not `kmovb`, because the IR compares 64 bytes and masks the result to six bits afterward. The rank memory operand spans 64 initialized bytes. A narrower compare or explicit mask intrinsic might change that choice; no saving is credited here.

For a crossover, let `N` be trees, `F` used features, and `M = sum(B_f)`. The split count at depth six is `6N`. With the addressing and leaf assumptions above, exact rounded estimates are:

```text
NEON setup = sum_f [2*G_2(f) + ceil(G_2(f)/2) + 5] + 2
AVX2 setup = 4*sum_f G_4(f) + F + 7       for 99/100 features
AVX-512 setup = 4*sum_f G_8(f) + F + 5   for 99/100 features

NEON total = setup + 10*N; compare with 22*N for tree-major.
AVX2 total = setup + sum_t (2*U_t + 6); compare with 15*N.
AVX-512 total = setup + 7*N; compare with 14*N.
```

Thus `M/(6N)` alone is insufficient: per-feature rounding, `F`, and the byte-table banks used by each tree also matter. Ignoring rounding and small shared setup, the necessary comparisons become:

```text
NEON:      1.25*M + 5*F < 12*N
AVX2:      M + F < [15 - (2*average(U) + 6)]*N
AVX-512:   0.5*M + F < 7*N
```

Writing `R = M/(6N)`, NEON needs roughly `R < 1.6 - (2/3)*(F/N)`. AVX2 with three banks per tree needs `R < 0.5 - F/(6N)`; with four banks it needs `R < (1-F/N)/6`. AVX-512 needs `R < (7-F/N)/3`. Small `B_f` values make rounding significant, particularly with eight-lane binarization.

| Model assumption | NEON bin total / direct tree-major | AVX2 bin total / direct tree-major | AVX-512 bin total / direct tree-major |
|---|---:|---:|---:|
| Actual 100-tree capture: `F=99`, `M=600`, no shared borders | 2,322 / 2,200 | 2,020 / 1,500 | 1,260 / 1,400 |
| 300 trees, 100 features, 18 unique borders each, three AVX2 banks per tree | 5,802 / 6,600 | 5,707 / 4,500 | 3,405 / 4,200 |
| 300 trees, 100 features, eight shared borders each, three AVX2 banks per tree | 4,502 / 6,600 | 4,507 / 4,500 | 2,605 / 4,200 |

The first row uses the measured group sums 320, 185, and 114 at widths two, four, and eight. The AVX2 figure retains the possible six extra half-combination instructions. The other rows are explicit assumptions, not representative CatBoost measurements. All numbers are instruction models. The bin approach can still win with distinct borders on AVX-512 because byte selection replaces expensive construction of double vectors. On NEON it loses at this 100-tree size but can amortize the per-feature setup at 300 trees. AVX2 at eight shared borders is essentially tied under the stated three-bank assumption; extra address setup, spills, or bank variation can decide it. Four banks per tree would make that row 5,107 instead of 4,507.

These formulas exclude normalization work for `AsTrue`. Binarization normalizes each used feature once. A direct tree-major path may normalize repeated feature loads more often unless the planner reuses normalized values. This can move the crossover and needs the model's actual NaN modes.

### LLVM IR feasibility and feature selection

All successful `.ll` files were verified, compiled at `-O1`, and assembled for their target. [replay.py](replay.py) contains the exact matrix; [validation.json](validation.json) has 57 command exit statuses. The standalone probes establish local lowering, not behavior of a full `preserve_nonecc` forest.

| IR | Verified lowering |
|---|---|
| AVX2 `fcmp <4 x double>` then `bitcast <4 x i1> to i4` | `vcmpltpd; vmovmskpd` with register operands (`generic-avx2.s:6`) |
| AVX2 eight double predicates packed to `i8` | Two compares, `vpackssdw`, `vpermq`, `vmovmskps`, rather than two independent four-bit masks (`generic-avx2.s:20`) |
| AVX-512 eight predicates, without DQ | Compare, `kmovd`, `movzbl` (`generic-avx512.s`, `fc8`) |
| AVX-512 eight predicates, with DQ | Compare, `kmovb` (`generic-avx512dq.s:20`) |
| NEON four double predicates to `i4` | Two compares, `uzp1.4s`, weight load, AND, `addv.4s`, byte store/reload, stack adjustment (`generic-neon.s:11`) |
| NEON eight double predicates to `i8` | Four compares, three narrowing `uzp1`, weights, AND, `addv.8h`, byte store/reload, stack adjustment (`generic-neon.s:46`) |
| NEON eight byte predicates to `i8` | `cmhi.8b`, weights, AND, `addv.8b`, `umov.b`; no stack access (`generic-neon.s:85`) |
| NEON constant `<16 x i8>` shuffle | Generic IR gives `tbl.16b` with a loaded control vector; two input vectors give a two-register table (`generic-neon.s:452`, `shuffle16_2`) |
| AVX-512 VBMI constant `<64 x i8>` shuffle | One input gives `vpermb`; two give `vpermt2b`, plus control load (`generic-avx512dq.s:327`, `:408`) |
| Two scalar loads inserted in `<2 x double>` on NEON | `ldr d; add; ld1.d [1]` (`generic-neon.s`, `lanes2`) |
| Four scalar loads inserted in `<4 x double>` on AVX2 | Two `vmovsd`, two `vmovhps`, one `vinsertf128` (`generic-avx2.s`, `lanes4`) |

The NEON packed-double `bitcast` really has stack traffic; it is not merely an unattractive disassembly spelling. `fc4` and `fc8` each have two stack-access instructions. The depth-6 masked variant avoids the store/reload but retains stack adjustment. Explicit `sli` avoids both. The byte `icmp` path already gives the desired AND/add reduction.

Generic NEON shuffle lowering is enough for one or two 16-byte banks. Explicit `llvm.aarch64.neon.tbl4.v8i8` and `llvm.aarch64.neon.tbx3.v8i8` make the seven-bank lookup deliberate. Their signatures were checked in installed LLVM's `include/llvm/IR/IntrinsicsAArch64.td:704–754`. `llvm.aarch64.neon.vsli.v2i64` provides the constant-free tree reduction. The scalar-register lane insert requires no intrinsic.

On AVX2, the generic ordered `fcmp` swaps operands into less-than and materializes the threshold vector separately. `llvm.x86.avx.cmp.pd.256(..., i8 30)` plus `llvm.x86.avx.movmsk.pd.256` gives folded threshold operands. The six-level case narrows its second compare to XMM automatically. On AVX-512 generic `fcmp` can use a memory operand already. Generic two-source `shufflevector` also produces `vpermi2b` for the actual bin-gather use, where preserving the two source banks matters. No handwritten machine code is needed.

Inkwell 0.9 has `build_bit_cast` and `build_shuffle_vector` (`builder.rs:2055`, `:3445` in the local cargo registry). Intrinsics can be declared in the module with the signatures in the retained IR. These probes use ordinary comparisons, insertion, extraction, and bit operations already available to this emitter. [LLVM's bitcast definition](https://llvm.org/docs/LangRef.html#bitcast-to-instruction) specifies the vector-bit order; both targets here are little-endian. Unused selector bits must be masked or defined false before a leaf lookup. Do not let poison lanes determine an address.

Use matching CPU/features on both the target machine (`src/jit.rs:78–85`) and each emitted function (`src/jit.rs:162–166`). Updating only one site can leave the intended instructions disabled. The explicit options verified here are:

| Path | `target-cpu` | `target-features` |
|---|---|---|
| NEON | `generic` | `""` on the tested AArch64 triple |
| AVX2 and POPCNT | `generic` | `+avx2,+popcnt` |
| AVX-512 including byte tables | `generic` | `+avx2,+popcnt,+avx512f,+avx512dq,+avx512bw,+avx512vl,+avx512vbmi` |
| Named AVX2 baseline | `x86-64-v3` | `""` |
| Named AVX-512 baseline plus byte permutation | `x86-64-v4` | `+avx512vbmi` |

The last two commands also compiled successfully. VBMI is a separate addition; an AVX-512 path must not assume that every AVX-512 host supplies it. The tree-major double path does not need BW or VBMI; these belong to the byte-bin path. DQ is what changed the eight-bit extraction to `kmovb` in the probes. VL permits narrower AVX-512 operations. A CPU without one of the emitted extensions can fault with SIGILL.

For same-host JIT, `TargetMachine::get_host_cpu_name()` and `TargetMachine::get_host_cpu_features()` return LLVM strings. Inkwell's implementation directly calls `LLVMGetHostCPUName` and `LLVMGetHostCPUFeatures` (`targets.rs:1102–1117` in the registry). Pass those values to machine creation and the function attributes. This ARM host reports `apple-m4`; that is not a CPU name to use for an x86 cross-target emission. The user can choose local-host specialization, a guaranteed deployment baseline, or separately compiled variants selected after feature detection. MCJIT's actual emitted code still needs inspection after making that choice.

### Other formulations

No additional formulation has enough evidence to rank above the two studied here. Hardware gather has a smaller AVX-512 instruction count but no runtime evidence on the target. An f32 change would alter the project's stated numeric contract and is not proposed.

## 2. Hypotheses checked

| Hypothesis | Result and correction |
|---|---|
| H1, threshold cost | Partly confirmed. Captures show the materialization sequences; the exact three-instruction-per-threshold count has exceptions. Base-relative threshold loads and folded x86 operands are verified in standalone probes. This study does not establish MCJIT code-model safety or the best whole-kernel constant layout. |
| H2, feature residency | Load counts confirmed: scalar scheduled 394, tree order 600, SIMD2 307. Their instruction and latency implications are not interchangeable. Tree-major gathering explicitly charges for row loads; resident-feature alternatives are separately counted. |
| H3, SIMD2 cannot win | Refuted as an absolute claim. Both supplied SIMD2 kernels have fewer instructions. Cross-tree mask distribution explains a cost, not a proof of slower execution. The 99.5% figure belongs to 500 trees; here it is 97.86%. Same-tree-only packing has at most 16 opportunities. |
| H5, NaN normalization | Confirmed for selected leaves and scores under the stated comparison rules. `AsTrue` remains false at positive-infinity and NaN borders. Ordered compares already make `AsIs` bins zero without runtime normalization. |
| H6, free bit assignment | Confirmed semantically. Permuting the private copied leaf initializer preserves every lookup. It adds compile-time work and can change locality, but needs no runtime index undo. All masks at depths six/eight were checked. |
| H7, binarize then compare | Exactness confirmed after border canonicalization and byte-range validation. Performance is conditional; setup and table selection matter. Distinct borders do not rule out an AVX-512 win. The asserted universal 254-border cap is false. |

H4's accumulator latency is outside this piece. The study does not endorse a cycle estimate from instruction count or project nanoseconds from these bodies.

## 3. Ranked recommendations

The required comparison below includes the gather and the two-instruction leaf allowance. Row features are unrelated and initially not resident. Counts assume the synthetic model's `AsIs` modes; normalization costs are separate. Current SIMD2 cells are observed whole-forest instructions divided by 100; all other cells are estimates formed from verified local bodies. Their omitted wrapper and address setup must not be treated as measured savings. The bin formulas use `G_w(f) = ceil(B_f/w)` and the bank setup described above.

| Formulation, depth six | NEON | AVX2 | AVX-512 |
|---|---|---|---|
| Current SIMD2 | 40.51/tree in supplied capture | 38.40/tree, current generic SSE2 emission | 38.40/tree, same current generic emission |
| Scalar target from piece A | About 26/tree; 20 with resident features | About 26/tree; 20 with resident features | About 26/tree; 20 with resident features |
| Tree-major lanes | 22/tree; 16 with resident features | 15/tree; 12 with resident features | 14/tree; 11 with resident features |
| Binarize then compare | 10/tree + per feature `2G_2+ceil(G_2/2)+5`; shared setup 2 | `2U+6`/tree, 12 at three banks, 14 at four; per feature `4G_4+1`; bank setup 7 for 99/100 features | 7/tree + per feature `4G_8+1`; bank setup 5 for 99/100 features |

1. On NEON, first test tree-major comparison with the explicit `sli` reduction. It has the smallest demonstrated change to index construction that avoids the generic bitcast's stack accesses. Form one action from six levels, gather three pairs, and permute the leaf initializer in `Action::Retire`. Relative to the stated 26-instruction scalar target, the local saving is four per tree, about 400 at 100 trees and 1,200 at 300. At depth eight it is seven per tree. Losing paired threshold loads reduces the gain. The scalar target may itself improve below the brief's estimate. Keep scalar as the control until complete kernels confirm the difference.

2. On AVX2, first test tree-major comparison with the explicit ordered-compare and movemask intrinsics. It removes per-lane extraction and selector updates and keeps thresholds as memory operands. The estimated saving against the same scalar target is 11 per depth-6 tree, 1,100 at 100 trees or 3,300 at 300. Depth eight saves 16 per tree. Merely using generic vector `fcmp` leaves extra threshold loads in the probe. Relative to that generic tree-major body, the tested intrinsic saves two instructions at depth six, or 200/600 at 100/300 trees; that narrower claim is directly measured in the probes. Runtime risks are lost feature/predicate sharing and the AVX calling-boundary setup. Do not make byte bins the first AVX2 path for 100-feature models: their estimate loses by 520 instructions at the supplied 100-tree case and is effectively tied at the assumed 300-tree shared-border case.

3. On AVX-512, first test direct tree-major comparison, then byte bins if the deployment host has BW and VBMI. Direct comparison saves an estimated 12 instructions per depth-6 tree against the same scalar target, 1,200/3,600 at 100/300 trees. The current generic IR's extra depth-6 mask cleanup is included. Byte bins reduce the estimated totals by another 140 instructions for the actual 100-tree model, or 1,595 for the assumed 300-tree model with eight borders per feature. They also require a new per-feature preparation phase and vector bin state across continuations. The smaller 100-tree difference is easy to erase with setup or spills. The x86 instruction bytes are verified; x86 latency is not.

4. On NEON, test byte bins second when real border statistics support them. At the supplied 100-tree case they add 122 estimated instructions relative to tree-major, so there is no count-based reason to choose them there. At the assumed 300-tree/eight-border case they save about 2,098. This requires all seven bin registers, the shared weights, table controls, and sums to fit the emitted chunks. The current default FP budget of nine cannot represent seven permanent bin registers plus two sums and scratch. Rework the plan's live values explicitly; do not pretend bins fit the existing scalar `FeatureValueId` slots.

Every whole-tree path changes grouping from feature to tree and can lose deduplicated predicates. Every bin path groups border comparisons by feature first and then evaluates trees from ranks. `Slots` currently carries only indices and scalar features (`src/sort_policy/continuation.rs:4–9`), and `emit_plan` builds scalar `f64` feature parameters (`src/jit.rs:114–125`). A bin path needs real vector state in those signatures. Keep bounded chunks; do not return to the unsplit kernel that recorded 3,168 bytes of local stack in `plans/implementation.md:179`. Whole-tree actions should partition at tree boundaries, accounting for six or eight scalar-equivalent tests. The existing literal limit of 16 fits two such trees, but cannot be reused without updating how partitioning recognizes comparison work (`continuation.rs:47–65`).

## 4. Experiments to run

These commands reproduce the evidence without editing tracked files:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo build --offline --release --examples
python3 probes/capture_stats.py
python3 probes/replay.py
```

The replay script verifies IR, compiles the target matrix, assembles every target probe, records disassemblies and body counts, and runs the native reference checks. Compare `sli6_row` with `tree6_row` on NEON: 20 versus 26 index instructions, and no stack adjustment for `sli`. Compare `avx2_tree6` with generic `tree6_row` on AVX2: 13 versus 15. Compare AVX-512 `fc8` with and without DQ: two versus three instructions. Compare `negative_bin8` with `bin8_w2` on NEON: 13 versus 15. See `validation.json` for the exact compiler commands and exit codes.

For a fresh full-model baseline and a serial latency control:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo run --offline --release --example inspect_code -- probes/fresh-baseline 100x100
cargo run --offline --release --example scalar_bench -- 100x100
```

The current benchmark already reports median, minimum, and maximum. Run it without the other study agents' load before interpreting latency. It only benchmarks current variants; it cannot time these proposed formulations. No command in this report pretends the production variants already exist.

After a separately approved implementation, run its correctness gate before trusting assembly:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo test --offline --release
cargo build --offline --release --examples
cargo run --offline --release --example inspect_code -- probes/tree-major-candidate 100x100
cargo run --offline --release --example scalar_bench -- 100x100
```

First integrate one direct tree-major variant while preserving the current chunk boundaries' purpose and scalar control. Compare complete body instructions, text bytes, actual MCJIT bytes, and stack accesses before timing. Expect the local sequences above; reject an apparent instruction win that adds spills or regresses serial prediction latency. Then test a byte-bin variant with measured `B_f` and table-bank distributions. Add an actual 300-tree/eight-border fixture at that time; the current workload parser has no 300-tree option. Repeat serial native benchmarks on the chosen x86 host, including scalar-load versus hardware-gather variants and AVX2 versus AVX-512. No x86 timing can be supplied from this worktree's ARM machine.

Two probe-authoring failures were corrected and retained in `probe_failures.txt`: an invalid Python comprehension in the initial counting script, and an AVX-512 gather intrinsic accidentally included in an AVX2-only compile. The latter made LLVM abort on `gather8`; the AVX2 file now omits that function and the AVX-512 file compiles it with the required features. Neither failure required a production workaround. The final replay has 57 successful commands. No environment blocker remains.

## 5. Questions that change the choice

- Is deployment restricted to this Apple host, to AVX2, or to an identified AVX-512 host with VBMI? Is per-host JIT specialization acceptable?
- How many features are used, and what is the distribution of distinct borders per feature after identical predicates, NaN borders, and signed-zero equivalents are merged?
- How many predicates are shared between trees? A tree-major plan can repeat comparisons that the current feature-major plan shares.
- Is depth mostly six, or do four- and eight-level trees dominate? Does any feature exceed 255 borders?
- Which NaN modes occur in real models? A per-feature normalization phase can change the direct-versus-bin cost.
- Is the `f64` comparison contract fixed? This study assumes it is; an f32 border/input decision would require a separate semantic analysis.

All primary anchors in the brief matched the supplied source. No tracked diff exists to replay because all changes were new files under `probes/`.
