Reviewed HEAD `97fb5b0260c69bcb7aef2ca55c7a6213f28460d0`. No blocker or major finding. Two minor reporting fixes remain. No prediction, permutation, or continuation failure was found.

| Severity | Location | Concrete failure and requested fix |
| --- | --- | --- |
| minor | `examples/cold_bench.rs:287`, `examples/cold_bench.rs:469`, `examples/cold_bench.rs:512` | Running the same workload with `--order scalar` and `--order tree-major` produces reports with no order identifier. `Report` drops the selected order, the printed header omits it, and CSV contains only `condition,ns`. Once results are separated from their command lines, the two treatments cannot be identified from the artifacts. Carry the order into `Report` and both outputs; test that the selected order survives reporting. This matters when investigating the reported cold regression. |
| minor | `VERDICT.md:72` | The `ld1 (ld1.d)` count of 238 conflates two instructions. Regenerated assembly contains 235 `ld1.d` lane loads and 3 `ld1r.2d` broadcasts. Report these separately, or explicitly label 238 as their combined count. The total instruction count is correct. |

The following instruction counts come from a fresh `inspect_code` run on `100x100`. `probes/review/analyze.py` counts instructions within functions, excluding directives, labels, constants, and alignment padding. Object section sizes come from assembling those files with LLVM 21 `llvm-mc` and reading them with `llvm-size`; they are separately emitted code, not a disassembly of MCJIT memory.

| Target and order | Total instructions | Continuation instructions | Wrapper instructions | Continuation stack accesses | Total/tree | Continuations/tree | Text section bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARM64 tree-major, 4 trees/chunk | 2253 | 2227 | 26 | 0 | 22.53 | 22.27 | 9012 |
| ARM64 scalar default | 2963 | 2937 | 26 | 0 | 29.63 | 29.37 | 11852 |
| x86 tree-major, 4 trees/chunk | 2332 | 2317 | 15 | 0 | 23.32 | 23.17 | 15669 |
| x86 scalar default | 3330 | 3315 | 15 | 38 | 33.30 | 33.15 | 17475 |

| Tree-major instruction | ARM64 count | x86 count |
| --- | ---: | ---: |
| `fcmgt.2d` | 300 | 0 |
| `sli.2d` | 200 | 0 |
| `ld1.d` | 235 | 0 |
| `ld1r.2d` | 3 | 0 |
| `ldr q` | 88 | 0 |
| `ldp q` | 106 | 0 |
| `cmpltpd` | 0 | 300 |
| `movmskpd` | 0 | 0 |
| `ucomisd` | 0 | 0 |

For the requested ARM64 per-tree listing, tree 8 in `_continuation2` is a clean 22-instruction example. The source is `probes/review/assembly/aarch64-apple-darwin-tree-major-4-2sums.s:241`. Below are its actual instructions, retaining relative order while omitting interleaved work for the other three trees. `x20` is the row, `x21` the constants block, and `x22` this chunk's threshold base. The tree's features in level order are 50, 53, 27, 2, 84, 7.

| # | Instruction | Work |
| ---: | --- | --- |
| 1 | `ldr d5, [x20, #216]` | Load level 2 into lane 0. |
| 2 | `add x9, x20, #16` | Address level 3 for a lane load. |
| 3 | `ldr d4, [x20, #400]` | Load level 0 into lane 0. |
| 4 | `ldp q2, q7, [x22]` | Load threshold pairs for levels 0–1 and 2–3. |
| 5 | `add x8, x20, #424` | Address level 1. |
| 6 | `ld1.d { v5 }[1], [x9]` | Insert level 3 in lane 1. |
| 7 | `ld1.d { v4 }[1], [x8]` | Insert level 1 in lane 1. |
| 8 | `ldr d8, [x20, #672]` | Load level 4 into lane 0. |
| 9 | `fcmgt.2d v5, v5, v7` | Compare levels 2–3. |
| 10 | `fcmgt.2d v2, v4, v2` | Compare levels 0–1. |
| 11 | `add x8, x20, #56` | Address level 5. |
| 12 | `ld1.d { v8 }[1], [x8]` | Insert level 5 in lane 1. |
| 13 | `ldr q10, [x22, #32]` | Load thresholds for levels 4–5. |
| 14 | `sli.2d v2, v5, #1` | Retain bit 0; insert the second pair at bit 1. |
| 15 | `fcmgt.2d v5, v8, v10` | Compare levels 4–5. |
| 16 | `sli.2d v2, v5, #2` | Retain bits 0–1; insert the final pair at bit 2. |
| 17 | `mov.d x8, v2[1]` | Extract signed lane 1. |
| 18 | `fmov x10, d2` | Extract signed lane 0 as integer bits. |
| 19 | `add x8, x10, x8, lsl #3` | Compute `raw = v0 + (v1 << 3)`. |
| 20 | `add x8, x21, x8, lsl #3` | Scale the signed element offset by eight bytes. |
| 21 | `ldr d2, [x8, #9184]` | Load the leaf, folding `(leaf_base + bias) * 8` into the displacement. |
| 22 | `fadd d0, d0, d2` | Accumulate into sum 0. |

This is nine feature/address/insert instructions, two threshold loads, three compares, two inserts of comparison masks, two extracts, two leaf-address instructions, one leaf load, and one addition. Here `leaf_base = 600 + 8*64 = 1112`, `bias = 36`, and `(1112 + 36)*8 = 9184`. Chunk transfer and final sum reduction are outside the listing.

The lane-load `add` is required by that chosen `ld1.d` form, which has no arbitrary immediate-offset addressing. A separate `ldr dTmp, [row, #offset]` followed by a lane insert replaces two instructions with two instructions; it does not establish a saving. It could change execution pressure and needs measurement. Across the full kernel there are 27 `mov.16b` copies and 62 vector-to-vector `mov.d` inserts in addition to 100 lane-1 extracts. For example, `_continuation0` preserves feature 23 in `v0`, copies it to `v8`, and later also uses it in tree 3. These copies interact with LLVM's reuse and scheduling; reversing insertion order is an experiment, not a demonstrated deletion. The `fmov` in the listing is an integer extraction required by the current leaf-address construction, not a redundant floating conversion.

Threshold pairing already works: 106 `ldp q` instructions supply 212 vectors and 88 `ldr q` supply the rest. Further pairing depends on register pressure and load scheduling. It cannot be inferred just from adjacent source slots. All these are possible follow-ups, not failures of the pinned implementation.

On x86, the same tree in `continuation2` uses `movsd` plus `movhpd` to form each feature vector, then loads thresholds with `movupd`. SSE2 `cmpltpd` compares threshold < feature, preserving ordered strict greater-than semantics after operand reversal. The first insert becomes `paddq m1,m1; psubq m0,m1`: for masks 0/−1, `2*m1 - m0` equals `(m1 << 1) | (m0 & 1)`. The final insert is `psllq $2`, `pand` with `[3,3]`, and `por`. `movq`, `pshufd $238`, and another `movq` extract the lanes; `leaq (...,...,8)` forms `raw`; `addsd 9184(constants,raw,8),sum` folds the leaf load into the addition.

That example needs 22 tree-specific instructions: six feature loads/inserts, three threshold loads, three compares, five mask operations, three extracts/shuffles, one `lea`, and one memory `addsd`. The chunk additionally materializes and loads the `[3,3]` mask once, then materializes the next continuation's address and jumps. Actual whole-kernel averages are in the table because feature reuse, copies, first-sum initialization, and the return tail vary. The planned AVX2 path using a `vcmpgt_oqpd` memory operand and `vmovmskpd` is outside this task; its absence is not a defect.

The cold-call evidence does not establish why the median increased. The original logs agree with `VERDICT.md`; their means and upper percentiles make a uniform slowdown particularly doubtful. Copies are retained as `probes/review/cold-original-scalar.log` and `probes/review/cold-original-tree-major.log`. These are the original measurements, not a new cold run.

| Row sequence, after 64 MiB traffic | Scalar mean ns | Tree-major mean ns | Scalar p50 ns | Tree-major p50 ns | Scalar p90 ns | Tree-major p90 ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shuffled | 666.6 | 650.5 | 500 | 583 | 1000 | 917 |
| dependent | 654.0 | 654.4 | 500 | 583 | 1000 | 917 |

`examples/cold_bench.rs:347` disturbs the data buffer before `row_buffer.copy_from_slice` at line 355. Thus the timed row is freshly written and warm in both treatments. The other-model condition likewise copies the row after calling the other model. The pollution loop does not explicitly evict instructions. Its 64-byte `Line` type is a traffic unit, not proof of the host's physical cache-line size.

The table below counts aligned 64-byte address regions for comparability with that harness. It does not claim a particular hardware cache-line geometry. The code guarantees 64-byte alignment for constants and only eight-byte alignment for feature rows.

| Call footprint, 100 depth-6 trees | Scalar default | Tree-major, 4 trees/chunk |
| --- | --- | --- |
| Row bytes referenced | Same 99 distinct features within an 800-byte row; 13–14 regions depending on base alignment | Same regions |
| Emitted ARM64 row loads, counting each loaded double once | 384 | 533, after LLVM eliminates some of the 600 source loads |
| Threshold storage read | 600 doubles, 4800 bytes, 75 regions | Same size and region count |
| Threshold arrangement | First-use slots follow scalar scheduling; one tree spans 2–6 regions, mean 3.77 | Six consecutive slots/tree; 50 trees span one region and 50 span two, mean 1.50 |
| Leaf storage allocated | 100 blocks × 64 doubles = 51200 bytes | Same; depth 6 needs no padding expansion |
| Leaves read in one prediction | One double/tree, exactly 100 distinct 64-byte regions | Same count; permutation changes which region in each block is selected |
| ARM64 constants total | 56000 bytes | 56000 bytes |
| ARM64 text bytes | 11852 | 9012 |
| x86 text bytes | 17475 | 15669, plus 400 bytes of per-chunk mask constants |

The smaller number of threshold regions per tree does not mean scalar reads thresholds randomly during execution: scalar slots were assigned in first-use order. Both traverse the threshold prefix according to their own plans. The measurements above come from the production plan dumps and assembly, with no assumed tree schedule. A permutation changes leaf addresses and their reuse across rows, even though it does not increase this workload's per-call leaf-region count. Odd-depth forests would have larger tree-major leaf storage, but that cannot explain this all-depth-6 result.

If the median effect survives a controlled measurement, different overlap of threshold misses and dependent leaf loads is more plausible than an increased cold-data footprint. Tree-major processes four whole trees per continuation and does more warm-row loading; scalar scheduling spreads work across active trees and reuses features. Either can change when leaf requests become ready. The vector compare → `sli` → extraction → address → leaf-load dependency also differs from scalar index construction. Assembly alone cannot assign the extra time to these causes. Smaller code and fewer instructions are consistent with the improved other-model condition, but do not prove an instruction-cache explanation there.

At the reported clock granularity, one tick is approximately 41.7 ns. The p50 gap is about 83.3 ns, two ticks. It is too coarse to treat that difference as a precise latency estimate, and too large to dismiss automatically as one rounding step. Separate runs, changed distribution shape, timer overhead, and machine state remain confounders. The lower p90/p99 and nearly unchanged means matter alongside the worse median.

One measurement would distinguish a real effect from this uncertainty: a randomized paired scalar/tree-major experiment in one process, with identical rows, a fresh full 64 MiB disturbance before each individual timed call, and the existing row copy after disturbance. Under matched core/frequency conditions, collect per-call CPU cycles and data-cache refill counts with hardware-counter tooling, then report the paired difference with uncertainty. Cycles remove the coarse wall-clock binning; refill counts distinguish extra misses from different handling of a similar miss workload. Timing a hot batch after one disturbance would answer a different question.

The implementation matches the pinned design and AMENDMENT 1 as follows.

| Task item | Verified implementation | Deviation or omission |
| --- | --- | --- |
| 1, option | `src/sort_policy.rs:33` adds `TreeMajor { chunk_trees: NonZeroUsize }` and documents ignored comparisons. `plan:152` validates budgets only for `Scheduled`. | None. Defaults are unchanged. |
| 2, plan | `tree_major:189` creates fresh predicates in level order, appends the specified `+inf` predicate for odd depth, uses `Admit`/`Retire` only for depth 0, and assigns `TreeId % sum_count`. `partition_trees:38` counts whole trees, padding tests, and retains the tail. | None. Original `partition` is unchanged in `git show HEAD`. |
| 3, emission | `src/jit.rs:118`, `199`, `541`, `810`, `831` implement slot assignment, chunk bases, paired loads, NaN normalization, target-specific mask accumulation, signed addressing, simulated permutation, and bijection checks. | None. The vector/signature test uses two depth-6 trees, one per chunk, so the existing sum-count clamp permits two sums. This is a justified fixture adjustment. |
| 4, harnesses | `300x100`, the three exact benchmark labels, `--order scalar\|tree-major`, and `tree-major-4-2sums` are present. Scalar remains default; the other-model compile remains default at `examples/cold_bench.rs:436`. | No pinned requirement omitted. Order provenance in cold outputs needs the minor fix above. |
| 5, tests | All requested new tests and differential configurations 1/3/8 are present. Benchmark tests retain their old cases and add the requested variants/options. | No required test omitted. Additional review coverage is described below. |
| AMENDMENT 1 | `src/sort_policy/tests.rs:206` evaluates real action predicates with feature normalization, strict comparison, and original level bits, then adds the original leaf. The mixed 0/1/3-depth replay test starts at line 271. | None. Padding is ignored; no wildcard substitute was added. Existing tests in this file are unchanged. |

Exactly the ten source paths allowed after AMENDMENT 1 changed in HEAD. No invented policy, public helper, feature deduplication pass, or altered scalar action arm appears in the diff. LLVM's existing `early-cse` pass can still remove redundant loads after emission. That explains why machine code has fewer than 600 tree-major row loads without violating the instruction to emit each real feature load.

For independent scalar verification, the parent's source was read with `git show HEAD^:<path>` into `probes/review/baseline` and built there. No stash or checkout was used. All eight pre-existing inspection variants are byte-identical between that build and HEAD. The scheduled control hashes match the original report:

| Target | SHA-256 of `scheduled-12gp-9fp-2sums.s`, parent and HEAD |
| --- | --- |
| ARM64 | `72e4dc156d62f3e8dd4514aae7bb69d76680a7c4340ae3c55481f052f5c673f0` |
| x86 | `2c478e5055074ade4f81b5f19e7fd38bb219f87b1362af9ef6882ea6b2bac6bd` |

The signed arithmetic is exact for the emitted model domain. `raw_of` at `src/jit.rs:810` creates precisely the 0/−1 masks produced by `fcmp ogt` followed by sign extension. For pair `j > 0`, it shifts the new mask left by `j` and keeps only the previous accumulator's low `j` bits. These are the specified `vsli(a,b,j)` operands in the same order at lines 647–655, and the same `shl`/`and`/`or` operands at lines 665–675. No scalar truth-bit conversion intervenes.

LLVM integers have fixed-width bitvector shift semantics here. `build_left_shift` calls `LLVMBuildShl`; the emitted shifts have neither `nsw` nor `nuw`, as the probe also checks. A negative left operand therefore does not itself make the operation poison. Shift counts must be below 64, which they are. Rust's `checked_shl` checks the shift count, not signed numeric overflow, but every shifted value here also fits i64. Thus its bit result agrees with LLVM's. The final `build_int_add` is plain addition as required. In particular, negative `v1` is retained: it is shifted as signed two's-complement bits, never zero-extended from a smaller integer.

Inductively, after `p` pairs each lane ranges over every signed p-bit value exactly once as its own comparison bits vary. Holding lane 1 fixed, varying lane 0 gives a consecutive interval of length `2^p`; incrementing lane 1 moves that interval by `2^p`. Those intervals abut without overlap. This proves the raw values cover one consecutive interval of length `2^(2p)`. It verifies the simulation's bounds; the implementation itself still uses the required simulation, not a replacement permutation formula.

| Depth | Pairs | Each lane's full-pattern range | Full-pattern raw range | Bias | Block entries | Reachable slot offsets for odd depth |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 1 | −1 through 0 | −3 through 0 | 3 | 4 | 2 through 3 |
| 2 | 1 | −1 through 0 | −3 through 0 | 3 | 4 | All |
| 3 | 2 | −2 through 1 | −10 through 5 | 10 | 16 | 8 through 15 |
| 4 | 2 | −2 through 1 | −10 through 5 | 10 | 16 | All |
| 5 | 3 | −4 through 3 | −36 through 27 | 36 | 64 | 32 through 63 |
| 6 | 3 | −4 through 3 | −36 through 27 | 36 | 64 | All |
| 7 | 4 | −8 through 7 | −136 through 119 | 136 | 256 | 128 through 255 |
| 8 | 4 | −8 through 7 | −136 through 119 | 136 | 256 | All |

For odd depth, the last comparison is lane 1's sign bit and is always zero. Therefore only the upper half of permuted slots is reachable. `permuted_leaves:853` recognizes exactly those original patterns through `pattern < tree.leafs.len()`. The low d bits already equal the original index; all other entries retain initialized `0.0`. For even depth every pattern is reachable. Hence `raw + bias` lies in the tree's block, and adding `leaf_base` cannot address another tree. Checked block sizing and the LLVM array-length limit precede initialization. In particular, a padded depth-31 block cannot reach the `1u32 << 32` unwrap: its block length already fails the LLVM array-length check. Depth 0 never calls `raw_of` or `permuted_leaves`.

Depth 1 performs no `sli`: the real lane is 0/−1, the padded lane is zero, and bias 3 selects offset 3 for false or 2 for true. Depth 8 uses shifts 1, 2, and 3, then `v1 << 4`. For example, lane 1 = −8 contributes −128, not a large positive selector. The minimum combined raw value is −136 and the maximum 119; bias 136 covers all 256 leaves. Exhaustive production-kernel tests exercise all reachable patterns at both endpoints and all intermediate depths.

NaN handling in `EvaluateTree:599` is the same as `Load:402` and `Trees::leaf_index` at `src/parsed_tree.rs:202`. Modes belong to each feature and are applied separately before vector construction. Neither feature normalization nor `fcmp ogt` has fast-math flags that discard NaNs. The exact outcomes for a NaN feature are:

| Feature mode | Value compared | NaN threshold | −inf threshold | Finite threshold, including either signed zero | +inf threshold |
| --- | --- | --- | --- | --- | --- |
| `AsIs` | NaN | false | false | false | false |
| `AsFalse` | −inf | false | false | false | false |
| `AsTrue` | +inf | false | true | true | false |

For non-NaN features, modes do nothing. A NaN or +inf threshold is always false. At a −inf threshold, every non-NaN value except −inf is true. At either zero threshold, only strictly positive values are true; both signed zero feature values are false. At a finite threshold, ordinary strict numerical order applies, with −inf false and +inf true. NaN in one vector lane cannot change the other lane's comparison.

The padding predicate is stored with level 0's feature as required, but emission intentionally reuses the normalized lane-0 value of the last pair. Whether that value is NaN, +inf, −inf, or finite, ordered `value > +inf` is false. Different modes on level 0 and the last real level therefore cannot activate padding.

Repeated features and repeated exact predicates remain separate level bits and separate predicate IDs. The slot-scan assertion tests ID reuse, not equal feature/threshold contents. Threshold slots remain consecutive, every real level is emitted, and `original_index` preserves both bits. Common-subexpression elimination may reuse a load or comparison without merging its uses in mask construction. Both the committed tests and the review probe compare these cases with `Trees::predict`.

Exactness here means exact comparison results and selected leaves. `README.md:56` explicitly permits reassociation and does not promise bitwise reference agreement or identical overflow behavior for arbitrary forest sums. Tree-major retains the existing accumulator/reduction behavior and reassociation flags. The exact-equality tests use leaf values and totals for which that arithmetic distinction cannot hide a wrong selection.

Chunk behavior follows tree IDs, not predicate count. `partition_trees` cuts immediately before `Admit` or `EvaluateTree` when a nonzero tree ID is a multiple of the limit. It does not cut at the corresponding `Retire`, so a constant tree stays intact. It initializes one chunk even for an empty forest, whose sole action is `ReturnBias`.

| Forest depths and tree limit | Trees per chunk | `Chunk.tests` per chunk |
| --- | --- | --- |
| `[0,1,0,2,0,3,0]`, limit 2 | `[0,1]`, `[2,3]`, `[4,5]`, `[6]` | 2, 2, 4, 0 |
| `[0,0,0,0,0]`, limit 2 | `[0,1]`, `[2,3]`, `[4]` | 0, 0, 0 |
| `[]`, any positive limit | One chunk containing the return | 0 |
| `[1,0,3,0,8,0]`, limit 8 or `usize::MAX` | All six trees in one chunk | 14 |

Every incoming `Slots` is empty. Thus `index_slots` and `feature_slots` are zero, and the parameter construction at `src/jit.rs:229` gives exactly `pointer_count + sum_count`: three pointers on ARM64, two on x86, plus the clamped number of sums. Every continuation is created from the same `FunctionType` at line 236. Calls use the same convention, and each inter-continuation call is `musttail` immediately followed by returning its result at line 801. The C wrapper uses an ordinary call, as required by its different signature. No tree action follows the final `AddSums`/return tail, so partitioning retains that whole tail in the last chunk, including when the last chunk contains only constant trees. Threshold-free chunks use slot zero as their ARM64 base, which is valid even for the empty forest's one-entry constants allocation.

The added tests use production paths, with the following limits.

| Test or extension | Production path and independent check | What it cannot establish alone |
| --- | --- | --- |
| `raw_of_permutation_is_a_bijection_for_depths_1_to_8`, `src/jit.rs:903` | Calls production permutation, `plan`, and `compile`; predicts all reachable patterns and compares directly with `Trees::predict`. Unique leaf values identify each selected leaf. It also checks unreachable entries are zero. | A table-only test could miss emitter errors, but this is not table-only. Native execution covers the host target. |
| `tree_major_plan_shape`, `src/jit.rs:934` | Calls `plan`, checks exact action structure, IDs, padding, sums, chunks, tests, and ignored comparison mode. | Does not execute emission. Constant trees only appear at the front of this fixture. |
| `tree_major_matches_reference_on_special_values`, `tests/scalar_jit.rs:24` | Calls `compile` and the independent reference for all 343 three-feature rows, six threshold rotations, eight depths, and three modes; additionally tests an exact repeated predicate. | Its main fixtures give every feature the same mode, and thresholds omit explicit −0.0. Differential tests give mixed modes, but not an exhaustive lane-mode matrix. |
| `tree_major_emits_vector_compares_only`, `src/jit.rs:1017` | Calls `emit_plan`, LLVM verification, and `emit_assembly` for both targets; checks signatures, comparisons, intrinsic calls, and load alignment. | Presence/count assertions cannot prove arithmetic correctness or native x86 execution. The other tests supply semantic coverage. |
| `tree_major_dropped_model_and_odd_depth_last_tree`, `tests/scalar_jit.rs:95` | Compiles 70 trees, drops the plan and model, and checks both extreme rows against saved reference values. | Only two patterns per tree. Allocator reuse is not forced, so this alone would not prove ownership. `CompiledModel` retains the JIT function/engine and emission owns constant initializers. |
| `tree_major_replay_matches_reference`, `src/sort_policy/tests.rs:271` | Calls production `plan`; replay validates predicate identity and uses reference level bits, with mixed feature modes and NaN rows. | It intentionally does not emulate `raw_of` or the emitted vector arithmetic, and cannot test the JIT. This independence satisfies AMENDMENT 1. |
| `seeded_forests_preserve_every_trees_leaf_index`, `tests/differential.rs:15` | Adds three production `plan`/`compile` configurations. Disjoint exact integer bit fields prevent one tree's wrong leaf from hiding in another tree's sum. Includes empty and mixed-depth forests with mixed modes. | Seeded coverage is not a deterministic assertion of every chunk edge case. |
| Cold benchmark tests, `examples/cold_bench.rs:568` | Parse valid/default/invalid/repeated options; compile both orders; compare predictions with reference; execute both benchmark orders. | Do not inspect order identification in reports or CSV. |
| Scalar benchmark tests, `examples/scalar_bench.rs:185` | Compile all three tree-major variants for fixed workloads and verify all rows plus dependent traces. The distinct-workload test adds the requested `300x100` parsing/model case. | The latter test retains only scalar/SIMD configurations; actual benchmark runs verify tree-major on `100x100` and `300x100`. |

No existing assertion was removed or weakened. The cold tests wrap their original bodies in an order loop; the scalar and differential tests extend existing case lists. The replay arm uses strict comparison and original level bits, so it does not merely repeat the emitter's arithmetic.

The additional probe is `probes/review/main.rs`. It runs the public `compile` API and compares with `Trees::predict`. It also includes the unchanged production `src/jit.rs` to expose private IR for verification and execute the generic `shl`/`and`/`or` arm after retargeting that already-emitted IR to the ARM64 host. This second execution checks generic arithmetic and the two-pointer signature; it is not execution of x86 machine code.

The probe covers 150 boundary configurations across empty, all-constant, and interleaved forests; tree limits 1/2/3/8/`usize::MAX`; sums 1/2/5; and both comparison modes. It checks chunk membership, tests, empty incoming slots, final-tail placement, matching continuation types, parameter counts, and `musttail` counts on both emitted targets. It also covers all 441 combinations of two lane modes and two thresholds chosen from NaN, ±inf, −1, ±0, +1; all 49 feature-value pairs; mixed-mode odd-depth padding; every reachable pattern at depths 1–8; and repeated features/predicates. All 47,146 native/generic prediction comparisons passed. This closes deterministic review gaps without modifying tracked tests. Native x86 execution remains outside the available ARM64 environment.

For maintainability, the feature-load/NaN-selection sequence is duplicated at `src/jit.rs:388` and `586`. A small private helper would be reasonable when this code next changes, because a NaN-mode correction should not need two edits. It is not necessary for correctness here, and a refactor should preserve the now-verified scalar assembly. The threshold-address duplication likewise follows the existing scalar rule exactly. No new dead public API or unused implementation was found; clippy passes. The unreachable `TreeMajor` label arm in the old scalar benchmark order loop is needed to keep the enum match exhaustive; the actual variants use `tree_major_variants`.

The new `unwrap`/`expect` calls are guarded by internal invariants on valid models: nonempty evaluated trees have at least one pair, padding is only the second lane, intrinsic lookup uses pinned LLVM 21, checked slot bounds precede address emission, and permutation sizing is bounded by the constants-block checks. `raw_of`'s checked shifts and additions fit the reachable domain discussed above. Oversized constants return existing compile errors before permutation. No user-triggerable panic on an otherwise compilable valid model was found. Error strings identify the failed count/base invariant, and the added safety comments explain the unsafe GEP bounds. No unrelated cleanup is needed for acceptance.

All review builds used offline cargo with `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'`.

| Review command | Final result |
| --- | --- |
| `cargo test --offline --all-targets` | 72 tests passed |
| `cargo test --offline --release --all-targets` | 72 tests passed; repeated after restoring HEAD's release artifacts |
| `cargo test --offline --doc` | 1 test passed |
| `cargo clippy --offline --all-targets -- -D warnings` | Passed |
| `cargo build --offline --release --examples` | Passed |
| `cargo run --offline --release --example inspect_code -- probes/review/assembly 100x100` | Both targets and all variants emitted |
| `cargo run --offline --release --manifest-path probes/review/Cargo.toml --target-dir target` | 47,146 native/generic prediction checks passed |
| `cargo run --offline --release --example scalar_bench -- 100x100` | All kernels verified; timing completed |
| `cargo run --offline --release --example scalar_bench -- 300x100` | All kernels verified; timing completed |
| `git diff --exit-code HEAD --` and `git diff --check` | Passed; tracked files unchanged |

Probe setup initially omitted the parent's `README.md` include; adding its exact `git show` contents resolved that build. Building the copied parent package into the shared release target then caused a cache collision: a subsequent HEAD example build saw the parent's `Order` enum. Clearing only the `tigerboost` release artifacts and rebuilding HEAD resolved it. The final release suite and hot benchmarks ran afterward. Neither failure was a source defect. Logs and assembly are retained under `probes/review/`; no commits or formatter runs were made.

The new hot runs were serial and remain indicative. Two trees per chunk offers no useful advantage over the default at 100 trees and remains slower than the GP4 control in both workloads; the larger chunks did better in this run.

| Variant | 100 trees, median ns | 300 trees, median ns |
| --- | ---: | ---: |
| Scheduled GP12/FP9, sums 2 | 93.28 | 260.25 |
| Scheduled GP4/FP9, sums 2 | 92.05 | 254.41 |
| Tree-major, 2 trees/chunk, sums 2 | 93.33 | 259.93 |
| Tree-major, 4 trees/chunk, sums 2 | 89.71 | 249.58 |
| Tree-major, 8 trees/chunk, sums 2 | 83.56 | 233.89 |

Accept with fixes. The opt-in planner, signed permutation, padding/NaN behavior, continuation signatures, and emitted code satisfy the pinned design; scalar emission is unchanged. Correct the instruction-count label and preserve the chosen order in cold benchmark outputs. The tests and added probes found no correctness defect. Hot results support the larger chunks for these workloads, while the cold p50 result remains an unresolved measurement rather than proof of a larger data footprint or a universal slowdown.
