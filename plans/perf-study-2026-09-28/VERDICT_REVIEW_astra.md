# Review of the tigerboost design proposal

status: changes_requested

## Findings

### Major. The residency forecast does not describe this workload

This affects SYNTHESIS sections 4 and 5.2–5.3 (`SYNTHESIS.md:100–106,137–156`). The 16-instruction NEON tree is a valid local sequence when all six inputs already occupy FP registers. It is not a whole-model cost. `REPORT_C_opus.md:98–99` excludes the work that supplies those registers. The proposed eight-tree chunks cannot make that work disappear.

`probes/plan_review.rs` calls the real `plan` with `Order::TreeOrder`, reads the comparisons and their consumers, and counts features in consecutive eight-tree groups. It uses the unmodified generator at `examples/support/distinct.rs:19–50`. Results are in `probes/plan_review.txt:1–16`.

| Tree positions, zero based | Lane fills | Distinct features | Fills that can reuse a prior load in this chunk |
|---|---:|---:|---:|
| 0–7 | 48 | 36 | 12 |
| 8–15 | 48 | 42 | 6 |
| 16–23 | 48 | 39 | 9 |
| 24–31 | 48 | 34 | 14 |
| 32–39 | 48 | 36 | 12 |
| 40–47 | 48 | 38 | 10 |
| 48–55 | 48 | 36 | 12 |
| 56–63 | 48 | 38 | 10 |
| 64–71 | 48 | 37 | 11 |
| 72–79 | 48 | 39 | 9 |
| 80–87 | 48 | 37 | 11 |
| 88–95 | 48 | 35 | 13 |
| 96–99 | 24 | 21 | 3 |
| Total | 600 | 468 summed over chunks | 132 |

A full group has 34–42 distinct features, averaging 37.25. With empty incoming feature state, at most 22% of lane fills reuse an earlier load. This assumes perfect retention until last use. More than 32 distinct features does not itself prove a spill: lifetimes can end before later features arrive. It does disprove keeping the whole group's feature set resident at once.

Both targets have eight FP argument registers. Two scalar sums leave six for features. Even granting six useful carried features at every one of the twelve boundaries, the lower bound is 396 row loads, and only 204/600 fills, 34%, avoid a row load. The probe finds enough intersection between each adjacent pair of windows to reach that counting bound; it does not establish a register allocation that achieves it. GP4 retirement order improves the empty-chunk bound to 426 loads and the optimistic carried bound to 360 (`probes/plan_review.txt:17–32`). That is another order, not the requested TreeOrder result.

The concrete failure is selecting NEON tree-major on a forecast of 16 instructions per tree and roughly 40% fewer total loads, then finding that first-use feature loads, preservation copies and boundary reloads consume the forecast gain. Keep 16 as a local lower bound. Measure actual row reads, lane inserts, copies and spills in the integrated kernel. `REPORT_C_astra.md:75–86,107` also contradicts Opus C's claim that residency buys nothing on x86: its resident gather has four instructions versus seven from memory. Even equal instruction counts would not make memory operands free.

### Major. Tree-major has some M4 headroom, but load issue is its next limit

This affects SYNTHESIS sections 1 and 4 (`SYNTHESIS.md:15–19,100–106`). A depth-six tree reads three 16-byte threshold vectors. The proposed `ldp q,q` plus `ldr q` is two instructions but three load slots. It is not three `ldp q` instructions. Rebasing enables pairing and reduces instructions; it does not halve the vector bytes read.

I checked the contested pairing assumption with `probes/load_pipe.c` and `probes/load_pipe.s`. Each loop has 96 explicit operations; the result is normalized against a dependent integer-add loop. Timings are indicative only, with no core pinning or performance counters. The distinction is large enough to resolve the reports' conflicting accounting (`probes/load_pipe.txt:1–6`).

| Operation | Time relative to one dependent add |
|---|---:|
| `ldr d` | 0.3323 |
| `ldp d,d` | 0.3322 |
| `ldr q` | 0.3321 |
| `ldp q,q` | 0.6793 |
| `fmov d,x` | 0.3321 |

Opus D is supported for paired scalar loads. Astra D's assumption that pairing never reduces load work (`REPORT_D_astra.md:75`) is too broad for `ldp d,d`. Conversely, treating `ldp q,q` as one load slot is wrong. The following bounds use three 16-byte load slots per cycle, one scalar feature read per miss, one leaf read per tree, and one constants-base GOT read per call. They exclude wrapper stack traffic and any spill. Adjacent row-load pairing can improve them slightly.

| Depth-six kernel, 100 trees | Feature load slots | Threshold load slots | Leaf loads | GOT/base loads | Total | Load-slot cycles |
|---|---:|---:|---:|---:|---:|---:|
| Today's approximation, including threshold `fmov` | 384 | 600 | 100 | 100 | 1184 | 394.7 |
| Tree-major, no feature reuse | 600 | 300 | 100 | 1 | 1001 | 333.7 |
| Tree-major, perfect reuse inside TreeOrder chunks | 468 | 300 | 100 | 1 | 869 | 289.7 |
| Above, optimistic six-feature boundary carry | 396 | 300 | 100 | 1 | 797 | 265.7 |
| GP4 tree order, perfect chunk reuse and optimistic carry | 360 | 300 | 100 | 1 | 761 | 253.7 |
| Scalar carry, current 384 feature reads, fully paired thresholds | 384 | 300 | 100 | 1 | 785 | 261.7 |

Today's quoted 393 is an approximate resource model, not an exact recount: `REPORT_D_opus.md:55` reports 1179, while the synthesis's displayed summands give 1184. The fresh baseline still has exactly 4453 instructions (`probes/baseline_counts.txt`). Pairing, immediate `fmov` forms and wrapper traffic explain why a mnemonic total needs classification before being called a port count.

For 600 surviving scalar comparisons, the stipulated M4 compare floor is 300 cycles. The scalar carry row therefore remains compare-limited in this model. Tree-major without reuse has a larger load floor than that. With realistic reuse, its estimated load floor can fall modestly below 300. This supports testing tree-major; it does not support a large guaranteed win. Relative to its no-reuse total, the TreeOrder bounds remove 13.2% or 20.4% of load slots, not 40%.

The next cap is load issue, followed by instruction supply, SIMD lane construction/reduction and dependency latency. Three hundred `fcmgt.2d` instructions have a 75-cycle isolated compare floor at the reported four-per-cycle rate; the two-sum arithmetic floor is 104 cycles. Neither is the first bound in this table. A vector kernel still has a compare bound, so “no compare cap” is shorthand for escaping the scalar flag-compare cap, not unlimited compare throughput.

### Major. Boundary state needs an explicit redesign before budgets can disappear

This affects SYNTHESIS 5.2–5.3 (`SYNTHESIS.md:145–156`). Whole-tree actions can remove carried selectors. Changing only `Action::Compare` and the chunk limit cannot do that safely.

`continuation::partition` finds the next comparison, charges `comparisons.len()`, then absorbs trailing `Release`/`Retire` actions (`src/sort_policy/continuation.rs:47–89`). Its `Slots` vectors retain their lengths after entries become `None` (`:20–35`). `emit_plan` sizes the common signature from those lengths, not the number of live values (`src/jit.rs:109–125`). Thus even tree boundaries with no live selector can leave a historical index slot. Expanding within-chunk feature residency can leave historical feature slots beyond the eight-register FP argument limit, even if only six entries are live at the next boundary.

The new action needs all level predicates/features, lane-to-original-level mapping, target layout offsets, leaf permutation and sum destination. Its execution must finish the tree and update the sum before the boundary. `Chunk.tests` must count the action's scalar-equivalent work, not one action as one test. Eight depth-six trees are 48 tests; eight depth-eight trees are 64. State whether the bound is trees, tests, or both. Zero-depth and fully folded trees also need a bounded action count. `Trees::new` accepts zero, odd and larger depths (`src/parsed_tree.rs:122–134`); `opus_probes_C/gen.py:144` explicitly rejects odd NEON depths. The new path needs defined tails or an existing scalar fallback, without narrowing accepted models.

For an atomic tree action, `Slots.indices` should be absent from carried state and `index_slots` should be zero. Feature boundary slots need their own compact, stable assignment, independent of larger local lifetimes. Every continuation still needs the same function type for `musttail`. Update wrapper arguments, incoming reconstruction and outgoing transfers together (`src/jit.rs:169–225,421–441`).

| Proposed scalar-feature signature | GP arguments | FP arguments with two sums | `sum_offset` | `feature_offset` |
|---|---:|---:|---:|---:|
| x86, row plus constants base | 2 | 2 + resident features, at most 8 | 2 | 4 |
| ARM64, row plus constants base plus chunk threshold base | 3 | 2 + resident features, at most 8 | 3 | 5 |
| x86 with an optional third base | 3 | Same | 3 | 5 |

Resident f64 features are FP arguments. Pointers are GP arguments. The proposal's “two pointers” omits the row on ARM64. Three GP arguments are far below x86's twelve and ARM64's twenty-four; removing selectors eliminates the specific x86 saturation problem. It does not eliminate local vector-register pressure. `probes/abi_aarch64-apple-darwin.s:7–20` and `probes/abi_x86_64-unknown-linux-gnu.s:9–20` confirm three pointer registers and stack placement of the ninth double. The additional 26-pointer probe confirms 24 register arguments on ARM64 and 12 on x86 (`probes/abi_aarch64-apple-darwin.s:30–80`, `probes/abi_x86_64-unknown-linux-gnu.s:32–71`). These counts agree with both B reports (`REPORT_B_opus.md:39–53`, `REPORT_B_astra.md:13–28`).

The failure scenario is a chunk with many short-lived local feature values, six live outgoing features, and a long sparse `Slots.features` vector. Reusing today's maximum-vector-length signature passes some live features on the stack despite the six-live-value assertion. Keep explicit local workspace and boundary limits even if `RegisterBudget` leaves the public API. Preserve bounded `preserve_nonecc` continuations; the scalar reports still show spills without them.

### Major. The leaf mapping must depend on depth and actual lane order

This affects SYNTHESIS 2 and 5.1–5.2 (`SYNTHESIS.md:52–53,129–150`). The signed NEON reduction is correct. A universal +36 bias and an unconditional AVX2 identity permutation are not.

For even depth `d = 2p`, let `b[k]` be the strict comparison bit for original level k. With original-order pairs, repeated `sli` yields

```text
v0 = sum(j=0..p-2, 2^j * b[2j])   - 2^(p-1) * b[2p-2]
v1 = sum(j=0..p-2, 2^j * b[2j+1]) - 2^(p-1) * b[2p-1]
raw = v0 + 2^p * v1
bias = 2^(p-1) * (1 + 2^p)
slot = bias + raw
```

The native LLVM/inkwell probe checks every mask for each requested depth, including concatenated mixed-depth blocks (`probes/intrinsic_review.rs:93–120`, `probes/intrinsic_review.txt:2–4`).

| Depth | Bias | Raw signed index range | Valid table slots |
|---|---:|---|---|
| 4 | 10 | -10 through 5 | 0–15 |
| 6 | 36 | -36 through 27 | 0–63 |
| 8 | 136 | -136 through 119 | 0–255 |

`REPORT_C_opus.md:140–144` is right about the depth-six mapping and depth-eight bias, but its depth-eight upper bound 127 is wrong. The actual generator computes the correct bias (`opus_probes_C/gen.py:142–168`). Astra C uses a different valid reduction: mask each lane to p bits, then concatenate (`REPORT_C_astra.md:88–105`). Its extra AND/bit-insert work must not be mixed with Opus's signed-add count or leaf layout.

For a tree whose block starts at entry `B`, form `B + bias + raw`, using signed i64 arithmetic with checked total layout bounds. This stays within that tree's `2^d` entries. ADD needs no `2^d` block alignment. My mixed-depth test includes a depth-eight block starting at entry 192, which is not 256-aligned, and all 256 indices remain valid. Add the bias exactly once. Do not OR a negative raw index with a block offset or mark its unsigned addition nonwrapping. Align/pad each real block without overlapping the next block; check aggregate byte and LLVM array-length limits. The production emitter currently checks each separate table's `u32` length (`src/jit.rs:361–374`), which does not validate a new combined array.

For a concrete failure, depth eight with only levels 6 and 7 true gives raw -136. Using +36 addresses 100 entries before its block. At depth four, all-false gives raw zero, so +36 selects beyond its 16 leaves.

AVX2 has identity leaf order only when lanes follow original levels. The actual GP4 plan first retires tree 67 with level order `[1,5,0,2,3,4]` (`probes/plan_review.txt:17`). Reusing that split order and identity leaves maps a true original level 1 into bit 0, selecting leaf 1 instead of leaf 2. Preserve the GP4 tree admission order separately from lane order, or compose a corresponding leaf permutation.

### Major. The AsTrue alternative needs complement and per-lane mode handling

This affects SYNTHESIS 2 and 5.5 (`SYNTHESIS.md:48–51,162–164`). Removing normalization for `AsFalse` is correct. `AsIs` already has no load-time normalization (`src/jit.rs:243–264`). NaN and +inf thresholds are constant false under all three modes. These conclusions follow directly from `Trees::leaf_index` (`src/parsed_tree.rs:202–214`).

This table gives the required result for both ordered `fcmgt.2d` and AVX predicate 30, after substituting +inf for an AsTrue feature NaN. Non-NaN rows apply to every `NanMode`.

| Feature value / mode | Threshold -inf | -1 | -0 | +0 | +1 | +inf | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|
| NaN / AsIs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NaN / AsFalse | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NaN / AsTrue | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| -inf / any | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| -0 / any | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| +0 / any | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| +inf / any | 1 | 1 | 1 | 1 | 1 | 0 | 0 |

The probe compares native NEON output with the real `Trees::leaf_indices` in 1089 cases: three depths, three modes, eleven feature values and eleven thresholds, including quiet/signaling NaNs and both smallest subnormals (`probes/intrinsic_review.rs:89–142`, `probes/intrinsic_review.txt:5`). For x86 this is a semantic and emitted-code check, not native execution. The installed `avxintrin.h:1625` defines `_CMP_GT_OQ` as 0x1e, ordered greater-than, non-signaling; `probes/avx2.s` contains `vcmpgt_oqpd`.

For AsTrue without substitution, the correct bit is `!is_constant_false_threshold && (is_nan(x) || x > t)`. NEON can implement the live-threshold part as the bitwise complement of `fcmge(t,x)`. Reversing `fcmge` alone computes `t >= x`; it gives the wrong result at equality and for a NaN feature. AVX can use `_CMP_NLE_UQ` (22), with constant-false threshold lanes suppressed first. `vcmpnle` is not sufficient when a NaN or +inf threshold remains live.

A tree can mix AsIs, AsFalse and AsTrue features in one vector. Its predicate immediate cannot differ per lane. Either normalize the AsTrue lanes before a common ordered comparison, group lanes with matching treatment and permute leaves, or explicitly select the right masks per lane. Price that work. The concrete failure is applying unordered comparison to an entire pair containing one AsIs NaN and one AsTrue NaN: the first lane wrongly becomes true. `REPORT_C_opus.md:173–180` contains the complement rule; the synthesis drops that necessary step. No claim here concerns matching floating-point exception flags.

### Major. The generated timings do not validate the proposed whole-model path

This affects SYNTHESIS sections 3, 4 and 5.2 (`SYNTHESIS.md:72–85,100–119,151`). The generated C kernels use arbitrary constants, different predictions and a different driver from `scalar_bench`. `REPORT_C_opus.md:22–41` explicitly says those timings establish no correctness result. `opus_probes_C/gen.py:31,40–44` allocates constant offsets without filling model values; `:450–466` uses a plain-call wrapper rather than the proposed `musttail` chain. The three different checksums in `serial/C_100x100.txt:1–9` confirm that the timed formulations are not predicting the same model. A dependent driver can consequently visit different rows and leaves.

I reproduced the generator's body counts, including returns, and separately counted the omitted wrapper (`probes/recount.py`, `probes/recount.txt:1–12`).

| Generated kernel | 100-tree body instructions | 100-tree total | 300-tree body instructions | 300-tree total |
|---|---:|---:|---:|---:|
| NEON scalar | 3195 | 3299 | 9863 | 10117 |
| NEON tree-major | 2266 | 2370 | 7024 | 7278 |
| NEON binarize | 2185 | 2282 | 6477 | 6649 |
| AVX2 scalar | 3158 | 3287 | 9480 | 9834 |
| AVX2 tree-major | 1444 | 1573 | 4343 | 4697 |
| AVX2 binarize | 1267 | 1370 | 3400 | 3603 |

The instruction evidence is real. It is evidence for these generated bodies, not for an integrated predictor, cold-call performance, or a threefold improvement over the best scalar carry kernel. The quoted llvm-mca comparison uses the generated ordinary scalar kernel, not Opus A's carry kernel (`REPORT_C_opus.md:264–271`). The proposal should not make that baseline substitution silently.

The 300-tree AVX2 object has 33,276 text bytes, including alignment and its wrapper (`probes/recount/treemajor_avx2_300.sizes.txt:3`). Chunk bodies alone occupy 31,230 bytes without alignment (`probes/code_bytes.txt:2`). Thus “roughly 25–30 KB, inside L1I” is not established by the cited artifact. A smaller `musttail` wrapper or larger chunks might bring the complete code below 32 KiB, but there is little demonstrated margin and large-model transfers also need counting. Count the actual MCJIT object before making an L1I-fit claim.

The concrete failure is deleting the scalar control or selecting tree-major as the default because 93.5 ns and 87.8 ns supposedly identify two resource caps. Those runs differ in constants, driver, wrapper and chunk structure. Their relative speed cannot establish that causal explanation. They remain useful indicative timings and motivation for a controlled experiment.

### Major. Cold-call overlap and footprint are misstated

This affects SYNTHESIS 2, 4 and 5.7 (`SYNTHESIS.md:57–58,121–125,172–174`). Independent leaf misses can overlap. Linear growth with tree count is consistent with a fixed number of concurrent misses; it does not prove no overlap. Opus D itself proposes about thirteen outstanding misses (`REPORT_D_opus.md:74`), contradicting its “not overlapping” sentence. Astra D's explanation is correct here (`REPORT_D_astra.md:77`): `musttail` does not drain the processor's instruction window.

The warm-other-row control is also not “leaves cold.” It warms some leaf lines while warming code and GOT data (`REPORT_D_opus.md:86`). The serial results show large cold sensitivity, but cannot isolate every remaining nanosecond to leaf misses. They support a cold harness, not an exact ROB size or a guaranteed benefit from postponing every leaf load until all indices are ready. That latter change also delays the first leaf and requires retaining indices, the same kind of lifetime expansion chunking exists to avoid (`plans/implementation.md:179`, `src/jit.rs:356–391`).

The constants block adds ARM threshold data reads. For aligned, packed depth-six tables, these are the approximate distinct data lines touched by one call. Features use a produced 800-byte row. Each tree contributes one leaf line. The table uses 64-byte lines; the reports disagree about M4 L1 versus L2 line size, so 128-byte counts follow separately. GOT storage is estimated as contiguous pointer entries; actual MCJIT relocation layout must be inspected.

| Data component, 64-byte lines | Current ARM 100 / 300 | Packed tree-major ARM 100 / 300 | Current x86 100 / 300 | Packed tree-major x86 100 / 300 |
|---|---:|---:|---:|---:|
| Feature row | 13 / 13 | 13 / 13 | 13 / 13 | 13 / 13 |
| Thresholds | 0 / 0 | 75 / 225 | 75 / 225 | 75 / 225 |
| Selected leaves | 100 / 300 | 100 / 300 | 100 / 300 | 100 / 300 |
| Leaf-base GOT entries / new common base | about 13 / 38 | about 1 / 1 | 0 / 0 | 0 / 0 |
| Approximate total | 126 / 351 | 189 / 539 | 188 / 538 | 188 / 538 |

Misalignment can add a row or threshold line. At 128-byte granularity, the analogous ARM totals are about 114/326 before and 146/421 after. The threshold region adds 38/113 lines; the old pointer region uses about 7/19. These are line-touch estimates, not guaranteed cache misses. Wrapper stack lines and code are separate.

For 100/300 depth-six trees, current leaves occupy 51,200/153,600 bytes. The proposed 600/1800 threshold entries followed by 64-entry-aligned leaves give combined blocks of 56,320/168,448 bytes. Padding occupies storage but need not be read. The C AVX2 generator actually aligns each four-lane group to 32 bytes, leaving holes between trees: its threshold regions are 6384/19184 bytes (`probes/recount.txt:5,11`, `opus_probes_C/gen.py:118–124`). That layout touches approximately 100/300 threshold lines, rather than 75/225. A packed 48-byte stride can avoid those holes with honest load alignment; choose and measure the actual layout.

Code gets smaller, which can offset the added ARM data traffic. The fresh current ARM body plus wrapper is 17,812 bytes; the generated tree-major total is 9480 bytes at 100 trees, but uses a different wrapper. A shorter code stream and more threshold data do not imply a known net cold gain.

Leaf permutation is free of a runtime inverse operation, not free of locality effects. On this fixture the ten recurring rows select 398 original versus 392 permuted 64-byte leaf lines; all rows select 743 versus 722 (`probes/plan_review.txt:49–53`). Other leaf distributions can reverse that result. The proposal's f32-leaf aside is outside the fixed f64 decision and is also wrong as a per-call line count: halving table storage does not halve one independently selected cache line per tree. Keep the f64 contract and measure cold calls on the same rows.

The concrete failure is a hot win that regresses cold latency because every ARM call now fetches a threshold region, or because a wider/padded constant layout adds lines. Both columns need a real correctness-checked predictor and identical pollution/row sequences.

### Major. Shared predicates and bin state are missing from the selection rule

This affects SYNTHESIS 4 and 5.2–5.3,5.6 (`SYNTHESIS.md:114–119,145–156,166–169`). The current scheduler shares an exact predicate across consumers, not just feature loads (`src/sort_policy.rs:256–263,370–384`). A whole-tree formulation can repeat that work. Astra C explicitly raised this (`REPORT_C_astra.md:314,357`); the synthesis's default-path rationale drops it.

A new probe constructs 100 identical depth-six trees. The unchanged default plan emits 72 scalar comparisons; a naive tree-major plan executes 600 lanes (`probes/plan_review.rs`, last line of `probes/plan_review.txt`). The proposal's 300/600-cycle scalar floors apply to the distinct-threshold fixture with 600 executed flag comparisons, not to every 100-tree forest. The concrete regression is replacing a shared-predicate model with a tree-only plan based on distinct-border instruction ratios. Keep that workload in the selection experiment.

The two C reports' binarize counts are for different algorithms. Astra C broadcasts one feature against several borders and gathers bins for one tree. Opus C compares adjacent features against their respective borders, and combines four AVX2 trees in one YMM byte gather (`opus_probes_C/gen.py:193–260,267–275,326–355`). My recount reproduces Opus's lower body counts. That is the right count for its formulation; it does not refute Astra's more expensive formulation or make binarize always faster. The byte gather and setup must be evaluated together.

The count-based crossover also needs actual used border counts. `TB_BORDERS=8` supplies eight candidate borders per feature, but random splits do not use all of them (`opus_probes_C/model.py:32–44`). The 300-tree model uses 719 distinct borders, not 800; the 16-border fixture uses 1085, not 1600 (`probes/border_counts.txt:1–3`). Shared-border generation also consumes an extra random draw per split, changing the feature and leaf fixture. The serial 8/16-border wins are real timings of those probes, but do not locate a universal crossover at 8–24 borders per feature. The generator supports at most 112 features in its NEON seven-bank shape; “at most about 128” is not an established AVX2 performance threshold (`REPORT_C_opus.md:273–278`).

Seven NEON byte-bin vectors plus two scalar sums require nine FP/vector arguments if carried independently. That exceeds the eight-register boundary capacity before adding a weight vector. The measured Opus formulation stores bins in a stack buffer and reloads them per chunk (`opus_probes_C/gen.py:408–426,453–455`); the synthesis instead proposes carried vector state. These are different designs. Specify packing the two sums into one vector, a costed bin-buffer path, or another explicit layout before importing the measured count. On AVX2 four packed YMM banks plus two sums can fit six vector argument slots, but the measured gather expands banks into seven duplicated YMM sources inside a chunk. Check that allocation and the moves needed between layouts.

Byte exactness needs a real limit. `Trees::new` imposes no 255-border cap. Above 255 distinct live borders, use a defined direct fallback or separately validated wider/split bins. For signed AVX2 byte compares, values above 127 need the 0x80 bias on both bins and ranks. Inactive tree lanes must remain false; unsigned rank 255 becomes signed -1 without that bias. AsTrue prologue padding must not count NaN padding as true. The literal generator is only a shape probe: it does not populate those controls, and some bin loads overstate 64-byte alignment at 16-byte offsets (`opus_probes_C/gen.py:416–420`). These issues do not invalidate the mathematical bin equivalence. They prevent treating that generator as an implementation template or a correctness-tested model.

### Minor. The AVX2 and SSE2 mechanisms are expressible in inkwell 0.9

This affects SYNTHESIS 5.2 (`SYNTHESIS.md:142–144`). There is no inkwell blocker. `probes/intrinsic_review.rs:1–85` builds and verifies declarations and functions through inkwell, emits AVX2 and SSE2 assembly, and executes the NEON variant with MCJIT. The declarations are:

```llvm
declare <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double>, <4 x double>, i8)
declare i32 @llvm.x86.avx.movmsk.pd.256(<4 x double>)
declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8)
declare i32 @llvm.x86.sse2.movmsk.pd(<2 x double>)
declare <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64>, <2 x i64>, i32)
```

The x86 intrinsics are not overloaded; `Intrinsic::find(...).get_declaration(&module, &[])` works. NEON `vsli` takes the `<2 x i64>` overload type. The compare predicate is an immediate i8; 30 gives ordered non-signaling greater-than. Sources are installed LLVM `IntrinsicsX86.td:288–291,448–449,1161–1164,1265–1266` and `IntrinsicsAArch64.td:453`, under `/opt/homebrew/opt/llvm@21/include/llvm/IR/`. The inkwell helper is in its registry `src/intrinsics.rs:45–105`.

For depth six, explicitly use a four-lane and a two-lane group, or prove that unused lanes of a second four-lane intrinsic are defined and removed. Do not load four threshold doubles from a two-double tail on the assumption that lowering will narrow it. `probes/avx2.s:22–31` confirms the desired folded memory comparisons and movemasks for explicit 4+2 groups. SSE2 uses three two-lane ordered comparisons with operands swapped to `cmpltpd`, then three `movmskpd` and bit assembly (`probes/sse2.s:26–45`). SSE2 cannot use AVX predicate 30 as a legacy three-bit predicate; emit the actual fallback shape. These probes verify instructions and declarations, not a six-unrelated-feature gather timing.

Set compatible CPU/features on the machine and every wrapper/continuation function. Current hard-coded sites are `src/jit.rs:78–85,162–166`. `compile` discards that machine and creates a separate MCJIT engine (`:48–49`), so changing only the inspection machine does not configure executing functions. The function attributes are essential; inspect actual JIT bytes. Use an AVX2-bounded configuration or a verified SSE2 fallback. Do not pass unrestricted host extensions and then credit an AVX-512-dependent result to AVX2. The concrete failure is successful AVX2 inspection alongside generic executing functions, or emitting an unsupported instruction on a baseline host.

### Minor. GP4 and two sums are useful controls, not evidence for every simplification

This affects SYNTHESIS 2 and 5.3–5.4 (`SYNTHESIS.md:31–42,54–56,153–160`). Astra B's GP4 recommendation is better supported than Opus B's “keep GP12 unchanged”: the former compares GP4 directly, and the supplied serial timings mostly favor it. At 100 trees the base-only medians are 103.05 for GP4 and 102.76 for GP12 (`serial/A_100x100_TB_PROBE_base.txt:12,18`), so “equal or better at every level” means within noise, not a strict inequality.

Keeping GP4's feature-overlap admission behavior does not itself make planning sub-millisecond. My indicative 300-tree calls take about 26 ms for GP4, 85 ms for GP12 and 0.05 ms for literal TreeOrder (`probes/plan_review.txt:55,96,137`). `waiting_order`, `select` and `remaining_uses` scan model state (`src/sort_policy.rs:313–325,394–470`). A new whole-tree planner may be much cheaper, but its sub-millisecond figure is a target requiring measurement, not a consequence of deleting a public budget type.

The sentence that no residency improvement is reachable with current signatures is also too strong. Both B reports find 357 loads with ideal eviction at the same six-feature capacity versus 394 in the current plan. My fixed-order simulation reproduces that (`probes/planner_probe_build.txt`, `minimum_same_order ... gp12`). This is a small opportunity, not a recommendation to add a scheduler. Reaching the global 99-load lower bound is what fails under current boundary signatures.

Keep two scalar sums. Their 104/304-cycle dependency calculation is correct for two-cycle fadd, scale one and the current final reduction (`src/jit.rs:387–413`, `src/sort_policy.rs:161–165`). But 1/2/4/8 sums do not literally time identically: the serial 100-tree dependent medians are 117.75/115.45/125.31/126.43 ns (`serial/D_cold_100.txt:5–8`). The sweep changes residency and argument pressure, so it demonstrates no benefit from extra sums, not an isolated equal-cost sum chain.

The suggested 700-cycle threshold for four sums is Opus D's margin heuristic, not a demonstrated bottleneck. Astra D's emitted vector-leaf update also corrects Opus D's proposed one-instruction saving per pair: an independently indexed second leaf needs address construction before `ld1.d`, so the tested NEON sequence is four instructions, like two scalar updates (`REPORT_D_astra.md:110–127`). Packing four sums into two vector arguments can save boundary slots, but does not guarantee cheaper updates or lower latency. The concrete failure is switching sums or deleting scheduling machinery before measuring the actual reduced kernel's limiting resource.

### Minor. Scalar carry is a valid fallback and must remain a timing control

This addresses R3 and SYNTHESIS 5's fallback (`SYNTHESIS.md:181–183`). Excluding carry from the first shared tree-major design is defensible because it retains scalar flag-compare limits and needs a different ARM emission. It is not established as slower on ARM: the supplied carry timings are lower than the generated tree-major timings, although those harnesses are not directly comparable.

The ARM sequence `fcmp t,x; adc index,index,index` appends the complement of `x > t`, including unordered inputs. Its inline-asm operands and flags clobber appear at `opus_probes_A/probe.diff:325–339`. LLVM sees the register dependencies but cannot schedule or cost the two internal instructions independently. This is a performance/maintenance risk, not a demonstrated correctness defect or an automatic disqualification.

The x86 `llvm.x86.addcarry.64` path uses `ULT(x,next_up(t))` as the complement bit (`opus_probes_A/probe.diff:311–323`). It is exact under ordinary gradual-underflow comparisons after +inf/NaN thresholds become constant bits and AsTrue inputs receive their required treatment. MXCSR.DAZ creates an additional correctness problem, beyond the baseline's handling of subnormal features. With `x=0` and `t=0`, the original comparison is false. `next_up(0)` is the smallest positive subnormal; DAZ reads it as zero, so `x < next_up(t)` becomes false and the reconstructed greater-than bit becomes true. The probe records this semantic counterexample against `Trees::leaf_indices` (`probes/intrinsic_review.rs:145–156`). It was not executed on x86. FTZ alone governs results, whereas DAZ is the input-flushing issue here; do not use the two names interchangeably.

If arbitrary caller DAZ state is supported, that x86 transformation is disqualified without a proven alternative. If the execution contract requires DAZ clear, test and document that contract before using it. ARM carry introduces no `next_up` threshold and therefore does not have this additional failure. Neither case authorizes changing process FP state inside prediction.

Both carry paths append bits in actual plan order, potentially different for each tree. The leaf initializer must complement and reverse that order, including repeated consumers and folded bits (`opus_probes_A/probe.diff:119–141`). The probe's block seed relies on padding every tree to the maximum leaf block (`:125–127,265–269`); it is not the synthesis's mixed-size ADD layout. A copied seed rule with differently aligned mixed-depth blocks can select another tree's leaves. These are testable mapping requirements. Astra A's `select(bit,index+mask,index)` remains a lower-risk x86 scalar control: it keeps level bits and avoids `next_up`, reaching 2458 instructions with the shared block and ADD offsets, not merely the 2558 threshold-only count cited in the synthesis (`REPORT_A_astra.md:235,331–334`).

### Minor. Several causal claims need narrower wording

This completes R1's claim-by-claim check. Sections 1, 2 and 4 contain paragraphs and bullets rather than numbered claims, so the table identifies each by its source lines. Findings above contain the concrete failure cases for the major claims.

| SYNTHESIS claim | Verdict against reports and code |
|---|---|
| Section 1, lines 12–19: present M4 instruction, load and compare limits | The 4453 count reproduces. The three bounds are plausible estimates; measured timing alone does not identify the binding port. 121 ns, approximately 5 ns driver and assumed 4.4 GHz imply about 509 cycles, not 490. The approximately 490 figure comes from a different timing/calibration in Opus D. Keep frequency and driver assumptions beside the estimate (`REPORT_D_astra.md:65`, `REPORT_D_opus.md:60`). Load arithmetic needs the corrections above. |
| Section 1, lines 21–24: x86 scalar port, 64 KB code and large model | The baseline contains 600 `ucomisd`, 737 `movabsq` and 37 indirect tails at 100 trees (`probes/baseline_counts.txt`). Both D reports agree on 64,082 bytes at 300. Front-end dominance is a target-specific inference, not an x86 measurement. The 600-cycle floor applies to 600 executed `ucomisd` on the named cores, not all scalar formulations or shared-predicate models. |
| Section 1, lines 26–27: mask compares escape the flag limit | Supported as the studied mechanism. NEON emission contains `fcmgt.2d`; AVX2 emission folds thresholds (`probes/neon.s`, `probes/avx2.s`). The four-per-cycle M4 rate is Opus D's indicative microbenchmark, not a throughput guarantee for the entire compare/reduce sequence. |
| Section 2, lines 31–35: argument-register limits | Supported for scalar carried state on these triples. Both B reports and the new ABI probe agree. Local register pressure remains relevant, and budget values do not imply every model necessarily uses all available slots. |
| Section 2, lines 36–38: residency at most 6–9%, none reachable | The quoted range is a fixture-specific saving relative to today's default, with Opus B using emitted loads and Astra B also discussing larger plan-level savings from tree order. It is not a universal upper bound. “None reachable” is false: the same six-slot fixed-order minimum is 357 versus 394 loads. |
| Section 2, lines 39–40: GP4 beats GP12 | Supported as the next control. Astra B's direct GP4 comparison and serial counts outweigh Opus B's recommendation to leave GP12 unchanged. Tiny timing reversals remain within the measured ranges. |
| Section 2, lines 41–42: f32-derived borders cost four instructions | Supported for the ordinary Darwin large-model examples, not every representable border. “Three movk” is wrong: the reproduced sequence is `mov`, two `movk`, `fmov` (`probes/f32_border.s:7–10`; `REPORT_A_opus.md:54–62`). Zero and encodable FP immediates are exceptions. |
| Section 2, lines 43–44: private-global loads fold; opaque argument works | Both A reports agree. `src/jit.rs:91–95` runs function-only early-cse. The one-chunk wrapper requirement is sound for this pipeline; retaining a private global also preserves the existing engine ownership (`src/jit.rs:22–25`). Future interprocedural specialization would need rechecking. |
| Section 2, lines 45–47: small code model | Supported as a reason to keep JITDefault. Astra A verifies both absolute 32-bit static relocations and PIC reach requirements (`REPORT_A_astra.md:69–89`). Changing `prepare_module` alone also does not change the MCJIT factory. Pointer data addressing does not itself remove large-model indirect tail jumps; Opus D's contrary claim at `REPORT_D_opus.md:171` is wrong. |
| Section 2, lines 48–51: NaN equivalence | Correct for leaf decisions with the AsTrue qualifications above. The citation overstates native coverage: Astra A's 192-case check is `semantics.py` (`REPORT_A_astra.md:208`), not native x86 execution. Native NEON and reference-code checks now independently cover the stated cases. |
| Section 2, lines 52–53: free permutation and +36 | Correct at depth six, with no runtime inverse permutation. Different depths, lane orders and locality require the corrections above. |
| Section 2, lines 54–56: sums | The two-cycle scalar-add scenario and 104/304 arithmetic are supported. No evidence favors more sums today. “Time identically” and a universal point where the chain appears are overstated; Astra D's address-inclusive vector-sum cost is the better comparison. |
| Section 2, lines 57–58: cold growth and no overlap | Cold sensitivity is supported. “No overlap” is false as an inference and conflicts with both the source dependencies and Opus D's own concurrent-miss model. |
| Section 4, lines 100–106: equal NEON timings prove different caps and large residency headroom | Not established by these drivers. Counts support modest load headroom after reuse; no-reuse tree-major is already above the scalar 300-cycle compare floor. A matched experiment must decide. |
| Section 4, lines 108–112: AVX2 wins, threefold mca gain, fits L1I, SSE2 works | Intrinsic lowering and SSE2 feasibility are supported. Counts reproduce, but use a generated scalar baseline and omit wrappers. Threefold is the Alder Lake model result; the cited Zen 3 ratio is about 1.67. The reproduced 300-tree object exceeds 32 KiB. No native x86 winner is established. |
| Section 4, lines 114–119: binarize exact and conditionally faster | Mathematical equivalence is supported with constant-false thresholds, per-mode treatment, correct padding and byte-range validation. The complete generated model was not correctness-tested. Opus's cheaper algorithm explains the disagreement with Astra's estimates. The 8/16-border timings support a conditional experiment, not a universal feature-count/crossover rule. |
| Section 4, lines 121–125: cold metric, overlap, batching and f32 lines | Measuring cold calls is supported. Existing misses already overlap; postponing all leaf loads is an untested tradeoff. f32 is outside the fixed decision and would halve table bytes, not necessarily lines touched per call. |

The serial table is mostly transcribed correctly. One small correction is 100-tree generated tree-major: the median of the three reported medians is 93.56 ns, conventionally 93.6 to one decimal, not 93.5 (`serial/C_100x100.txt:2,5,8`). Serializing study jobs does not establish an idle machine: the retained start/end load averages are approximately 2.5–2.7 (`serial/00_load_start.txt:1`, `serial/99_load_end.txt:1`). All timings in this review are indicative only. Static counts and emitted instruction shapes are stronger evidence.

Section 5.1's constants pointer is the best-supported first implementation step and respects the fixed decisions. Sections 5.2–5.3 need the explicit mapping, boundary and workload qualifications above. Section 5.4 should retain two sums pending evidence. Section 5.5 is sound with the corrected mode handling. Section 5.6 should remain conditional and later. Section 5.7 should come before performance selection, with shuffled rows, a timer-only control, separate data/code interference and both produced/untouched row cases from Astra D (`REPORT_D_astra.md:248–258`). Its checked-API overhead experiment was also dropped: `CompiledModel::predict` has a separate checked call path, so keep the API identical between timing variants before attributing small gains to kernels (`src/jit.rs:29–38`, `REPORT_D_astra.md:163–165`).

## Validation performed

The required `cargo build --offline --release --examples` passed. `cargo test --offline --release --lib --tests` passed 30 unit and 20 integration tests (`probes/baseline_tests.txt`). The standalone intrinsic probe passed native NEON/reference checks and emitted AVX2/SSE2 code. The plan probe uses production planning and the existing workload generator. Generator recounts were assembled with installed LLVM tools. No x86 code was executed. No network, formatter, commit, or tracked-file edit was used. New probes are under `probes/`; this review is the only new root report.

Reproduce the review probes from this worktree:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml --bin plan-review
CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml --bin intrinsic-review
python3 probes/abi_review.py
python3 probes/recount.py
python3 probes/border_counts.py
python3 probes/load_pipe.py
cc -O2 probes/load_pipe.c probes/load_pipe.s -o probes/load_pipe
probes/load_pipe
```

## Ranked experiments

Run these in order. These are proposed changes for the orchestrator, not changes made to tracked files in this review. Keep a common correctness corpus and the same hot/cold drivers throughout. Use `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21`, offline builds and LLVM 21. Do not promote an instruction-count win that regresses a required timing condition. No experiment requires AVX-512, f32 data, removing continuations or a runtime scheduler.

### 1. Finish the small semantic and ABI probes

Extend `probes/intrinsic_review.rs` and `probes/abi_review.py` before changing the production plan. Add mixed `NanMode` lanes, odd depths, zero-depth trees, all-constant-false lanes, repeated features, arbitrary lane orders, and mixed block offsets. Exercise three pointers plus two sums plus six carried features, then a deliberate seventh carried feature as a stack-placement control. Keep direct AVX2 4+2 threshold loads and the real SSE2 fallback.

The gate is exact leaf-index agreement with `Trees::leaf_indices`, no out-of-range table slot, verified IR and assembled objects on both targets. On x86 hardware, run predicate-30 and carry tests under DAZ clear/set; restore the caller's original FP state in the test harness. The new ARM tests already give 336 exhaustive masks and 1089 special-value cases. Expected biases are 10/36/136; a ninth FP argument goes to the stack. Any wrong index kills that encoding until its mapping or predicate is corrected. Inkwell declaration failure would kill the claimed emission mechanism, but the current probes pass.

These small kernels have no useful forest latency forecast. Hot and cold measurements at this step establish unchanged baseline controls only; do not rank production directions from the index-probe timing.

### 2. Make the existing controls comparable, including cold calls

Build a probe harness with `examples/scalar_bench.rs::measure`, the real `support::workload`, and the cold sampling shape described in `REPORT_D_opus.md:132–137`. It must validate the exact constants, permuted leaves and predictions before timing. Use the checked `CompiledModel::predict` path for every integrated variant, or explicitly separate a raw-call comparison for every variant. Keep both the dependent trace and a shuffled all-256-row trace. Add timer-only samples, data pollution at 256 KiB/4/16/64 MiB, another-model/code interference, and produced versus untouched input rows.

The gate is identical predictions and row/leaf traces for every variant being compared, plus timer controls and raw samples. Expected current two-sum controls are approximately 121/345 ns in `scalar_bench` for 100/300 trees. Serial single-call hot p50 is 125/334 ns; 64 MiB p50 is 791/2000 ns and mean is 1174/2135 ns (`serial/D_cold_100.txt:13,18`, `serial/D_cold_300.txt:13,18`). These are sanity ranges, not precision targets. Timer quantization is about 41.7 ns. A changed checksum, variant-dependent input trace or unexplained timer floor kills the comparison, not the kernel. This step is required before interpreting the old C timing advantage.

### 3. Integrate only the decided constants block, with GP4 and GP12 controls

Change `src/jit.rs::emit_plan` in the implementation worktree: own one private constant block, pass its base through a real wrapper and every continuation, use checked per-tree ADD offsets, and add the ARM chunk threshold pointer. Keep current comparisons and bounded chunks. Account for the extra GP arguments while the old selectors still exist; do not put the ARM extra pointer into saturated x86 GP12 signatures. First compare existing GP4/FP9 and GP12/FP9 plans without a scheduler rewrite.

Add dedicated tests for model-drop ownership, one-chunk opacity after early-cse, mixed depths, last-table access, empty/constant models, and identical incoming/outgoing signatures. Run the existing release tests and both-target assembly checks. Expected 100-tree GP12 ARM instructions are 4453 → 3189 → 2959, with zero continuation-body stack traffic in the cited fixture; x86 shared-base GP12 is 3330, but still needs its own spill accounting. Indicative hot GP12 medians are 120.63 → 94.73 ns and 344.74 → 261.07 ns (`serial/A_100x100_TB_PROBE_base_TB_REBASE_1.txt:18`, corresponding 300-tree file). Cold timing has no verified improved number: expect the new threshold-line footprint and require measured hot/cold columns from experiment 2. Unexpected extra materialization, x86 allocation failure, new spills or a cold regression large enough to defeat the deployment objective rejects that layout/configuration.

### 4. Compare one complete tree-major path with scalar carry on the same models

Implement an opt-in whole-tree action in `src/sort_policy.rs`, tree-boundary partitioning in `src/sort_policy/continuation.rs`, and target-specific comparison/reduction in `src/jit.rs::emit_plan`. Use original level order initially, two scalar sums, actual model constants, no speculative cross-boundary residency, and a 2/4/8-tree chunk sweep. Keep both the plain scalar constants path and Opus A's scalar carry as controls in the experiment. On x86 also keep Astra A's select-plus-ADD form if DAZ rules out carry.

New tests must cover exact per-tree selection at depths 0–8, mixed depths and modes, duplicate predicates with multiple consumers, odd tails, model drop, no partial selector crossing and no hidden signature slots. Emit and inspect actual MCJIT bytes as well as separate assembly. Expect the local sequences already demonstrated, approximately 22 NEON or 14 AVX2 instructions per depth-six tree before complete integration costs. The reproduced C whole-body controls are 2266/7024 NEON and 1444/4343 AVX2 instructions at 100/300 trees; they are reference shapes, not all-in ceilings. ARM carry's cited controls are 2275/6780 instructions and indicative GP4 hot 86.80/236.79 ns. Tree-major's old 93.56/263.30 ns figures are only comparison targets until the drivers match.

Run both hot traces and all cold conditions from experiment 2. On native AVX2/SSE2 hardware, validate before timing; llvm-mca is not that gate. Reject tree-major as the default on a target if it repeatedly loses to a semantically valid scalar control, adds body spills, or regresses the shared-predicate control materially. A win on one target need not force the same default on the other. Reject an eight-tree chunk size if four trees avoid spills or issue leaves earlier at better latency.

### 5. Add measured feature retention and tree admission separately

Only after experiment 4, add within-chunk feature lifetimes to the whole-tree plan and cap compact cross-boundary scalar features at six. Keep the emitter's sum count and chunk shape fixed. First test literal TreeOrder with local reuse; then feature-overlap admission; then optional cross-boundary retention. Changes belong in whole-tree planning and the incoming feature-slot mapping, not an unrelated planner rewrite.

Tests must check that every retained value still names the right row feature and NanMode, releases precede reuse of a slot, and every boundary stays within eight FP arguments. Compare actual loads with the counted bounds: 468 row reads with ideal local retention, 396 with optimistic six-feature carry for 100-tree TreeOrder; GP4 retirement-order bounds are 426/360. A bound is not an expected exact emission. The corresponding unpaired load floors are approximately 290/266 and 275/254 cycles, before other costs. The 16-instruction resident tree is not the expected whole-model result. Measure planner time too; sub-millisecond is a target, not an existing GP4 result.

Use the same hot/cold harness. Require recurrent latency benefit and no body spills or added cold footprint that erases it. If preserving/copying vectors spends the saved lane-load work, stop adding residency complexity. If feature-overlap admission worsens repeated-predicate models or adds material compile time without prediction gain, keep the simpler order.

### 6. Test extra sums or earlier leaf issue only when their limit is visible

If the integrated vector kernel is fast enough for the sum dependency to matter, compare two versus four scalar sums in `Action::Retire`/the whole-tree action and `Action::AddSums`. For ARM, separately test four sums packed into two vectors, including both leaf addresses and the final horizontal reduction. Keep feature capacity fixed where possible and explicitly report any changed residency. A separate cold experiment may group a small bounded set of indices and issue their leaf loads earlier; do not start by storing every index and delaying every leaf to the end of the forest.

New tests cover the permitted reassociated arithmetic contract, non-integer/cancelling leaves, scale/bias, feature-slot limits and identical selected leaves. Expected two-cycle scalar arithmetic floors at 300 trees are 304 cycles for two sums and 156 for four with a balanced final reduction; whole-call gains can be zero. Vector leaf gathering is four ARM instructions per independently indexed pair in Astra D's probe, not three. There is no justified promised 300 ns cold target for batching.

Use dependent and independent hot controls plus 64 MiB and code-interference cold calls. Reject extra sums if they merely trade feature loads for stack transfers or do not improve latency. Reject grouped leaf issue if earlier readiness does not improve cold latency, or if its hot regression is unacceptable. A kernel above the measured load/front-end floor needs no speculative accumulator change.

### 7. Build binarize only for models whose measured statistics justify it

After the direct path has a controlled baseline, add an explicit per-feature prologue and tree-group actions. Use the Opus adjacent-feature formulation if reproducing its lower count; do not substitute Astra's per-feature-broadcast setup and retain Opus's forecast. Define byte-range fallback, signed-byte bias, inactive lanes, mixed modes, row alignment and actual vector state through continuations. Test the bin buffer and a carried-bin layout as different implementations.

New tests cover strict equality, signed zero, infinities, NaN thresholds, every mode, 127/128/255/256 distinct borders, short feature rows with no vector overread, and bank boundaries. Use the same models and leaf tables in direct and bin paths. Gate on native x86 execution for an AVX2 choice, not only byte counts. Report used borders and group maxima from the actual model. For the supplied 300-tree generators the counts are 719/1085/1800 used borders for the 8-candidate/16-candidate/distinct cases. Opus's indicative NEON targets are about 229/246/296 ns against tree-major 269/267/263 ns; the matched correct implementation may differ. The distinct 100-tree AVX2 body count control is 1267 versus 1444, only 177 instructions of margin before integration.

Run the common hot/cold harness, including all threshold/control lines and prologue stores/reloads. Any byte overflow, false inactive-lane bit or boundary spill rejects that implementation. If the complete prologue and bin state erase the advantage, reject binarize for that model class and retain direct comparisons. Do not use an unconditional “at most 128 features” selection rule.

## Overall verdict

Revise the proposal before implementation. The constants pointer, bounded continuations, two sums and tree-major intrinsic mechanisms are well supported. The NEON residency forecast, mixed-depth/lane mapping, AsTrue wording, bin-state ABI and cold-call explanation need correction. Tree-major is a strong experiment, especially on AVX2, but the cited evidence does not yet establish it as the best complete predictor on either target. Keep scalar carry and shared-predicate models in the controlled hot/cold comparison, and select the path from those results.
