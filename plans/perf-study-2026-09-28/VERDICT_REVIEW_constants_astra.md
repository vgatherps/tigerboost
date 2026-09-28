One minor finding. No blocker or major finding in the implementation.

Minor, `src/jit.rs:884`: `simd_pair_loads_both_thresholds` does not check which addresses supply the two lanes on the non-native target. Two distinct loads with reversed threshold addresses satisfy lines 884–891. On this ARM64 host, changing only the x86 pair addresses from predicate `p` to `p ^ 1` still passes this test. For its `[1.37, 2.61]` model and input `2.0`, that x86 code would select leaf 2 instead of leaf 1. The mutation is confined to `probes/wrong_x86_pair.rs`; the passing run is in `probes/wrong-x86-pair-test.log`. Extend the two-target assertions through each load's GEP, checking the pointer parameter and the slot corresponding to that lane's `Comparison::predicate`. The production code has the correct mapping; an independent probe checked every scalar and SIMD threshold operand on both targets.

The diff against `master` at `da939936e80b02c7ae562bc5f8f6178de34536b7` modifies only `src/jit.rs`. The planner, model, examples, and `tests/*.rs` are unchanged. No production deviation from pinned items 1–5 was found.

| Task item | Verified implementation |
| --- | --- |
| 1 | `src/jit.rs:112` checks padding, accumulated entries, LLVM array length, and bytes before constructing the initializer. One private constant global, alignment 64; predicate order, padding, contiguous tree-order leaf blocks, and the empty-forest zero entry match the task. |
| 2 | `src/jit.rs:105`, `:164`, `:193` select the architecture from the supplied triple, use two or three pointer parameters, attach the requested attributes, and always emit the wrapper and continuations. The pass pipeline is unchanged. |
| 3 | `src/jit.rs:142`, `:349`, `:393` use the minimum predicate ID per chunk, load the correct scalar slots, and insert the first and second thresholds into lanes 0 and 1. No comparison or consumer-bit semantics changed. |
| 4 | `src/jit.rs:466` adds the leaf base with `add nuw`, then loads through `constants`, aligned to 8. Per-tree globals are removed. Sum, scale, bias, and `ReturnBias` operations are unchanged. |
| 5 | `src/jit.rs:227`, `:269`, `:517` pass and reconstruct pointers, indices, sums, and feature values at the shifted positions. The next chunk's base is computed from `constants`, not the current `chunk_base`. |
| 6 | All six requested tests were added, and all three existing IR tests were updated. All use production paths. The minor finding above concerns the SIMD test's cross-target address coverage. |

The test assertions were checked individually.

| Test in `src/jit.rs` | Production path and assertion coverage | What it does not establish by itself |
| --- | --- | --- |
| `constants_block_layout_matches_plan`, line 607 | `plan`, `emit_plan`, `compile`. Checks the entire initializer, shared predicate deduplication, depths 0/1/3, padding, tree order, linkage, constness, alignment, and three reference predictions. The unaligned leaf bases exercise addition rather than OR. | Predictions run on the native target; initializer correctness alone does not establish every comparison's address. |
| `one_chunk_model_keeps_wrapper_and_opaque_loads`, line 638 | `plan`, `prepare_module` on both targets. After `early-cse`, checks two functions, the wrapper call, three comparison operands and three retirement operands supplied by aligned loads, eight total loads, and no direct global reference inside the continuation. | A load from the wrong slot can pass these shape checks. |
| `pointer_parameters_have_bounds`, line 661 | `plan`, `emit_plan`, `emit_assembly`. Checks pointer count and positions, every pointer attribute, both chunks' exact bounds, every threshold GEP base/index, and the ARM64 transfer base. Checks x86 memory comparisons and forbidden materialization instructions on both targets. | Uses scalar tree order; it does not cover scheduled nonmonotone IDs or SIMD lanes. ARM64's assembly load-presence assertion alone could be satisfied by a feature or leaf load, but the IR assertions check threshold addresses. |
| `dropped_model_and_last_tree_leaf`, line 780 | `plan`, `compile` in both orders. Drops both plans and the model before predictions. Checks reference values for first/last leaves in 70 depth-6 trees, beyond the requested immediate reach. | All trees select the same leaf number per row, so permuting equal-depth tree bases could preserve the total. Dropping without overwriting freed storage is also insufficient alone to prove ownership. The code inspection below supplies that proof. |
| `empty_and_constant_forests_still_predict`, line 815 | `plan`, `emit_plan`, `compile`. Checks the zero sentinel, three constant leaves, bias despite NaN scale for the empty forest, and the scaled/biased constant sum. | Does not inspect pointer attributes; the review probe checked both targets' zero-comparison bounds. |
| `simd_pair_loads_both_thresholds`, line 845 | `plan`, `emit_plan`, `compile`. Both targets have two aligned scalar loads feeding the correct insertion lane numbers and a vector comparison. Native predictions cover all four outcomes across both threshold orders, equality, and NaN. | Non-native addresses are unchecked, as demonstrated above. |
| `scheduled_order_shares_loads_and_comparisons_before_llvm_passes`, line 931 | `plan`, `emit_plan`. Load totals now include thresholds; scalar/vector comparison counts, mask extension, extraction counts, and the forbidden narrow extraction remain asserted. | Aggregate counts do not identify each load's source; that limitation existed before this change. |
| `ir_has_readonly_rows_constant_leaves_and_only_reassociation`, line 1010 | `plan`, `emit_plan`. Replaces the obsolete one-function/no-call assertions with the required wrapper/continuation shape and loaded operands. Preserves row bounds, memory attributes, arithmetic flags, and every forbidden-IR check. | Checks pre-pass IR; the dedicated one-chunk test supplies post-pass coverage. |
| `continuations_follow_plan_ranges_with_identical_musttail_signatures`, line 1062 | `plan`, `emit_plan`, `emit_assembly`. Preserves identical types, calling convention, `noinline`, chunk comparison counts, next-continuation calls, immediate returns, and wrapper checks. Adds threshold loads to the expected total. | Matching types cannot detect swapping two values of the same type; unchanged differential tests exercise carried indices and features. |

None of the three pre-existing IR tests was weakened. `emits_predict_for_aarch64_and_x86_64` is unchanged. The separate integration tests passed unchanged, including both continuation carry tests and reference comparisons after model drop.

Ownership is sound. `src/jit.rs:128` creates `constants` in the same module handed to `create_jit_execution_engine` at line 48. The initializer copies LLVM scalar constants; emitted addresses never point into `Trees`, `Plan`, `values`, or `leaf_base`. In the installed Inkwell 0.9.0 source, `module.rs:590` transfers module ownership to LLVM, `execution_engine.rs:327` clones the engine reference into `JitFunction`, and `execution_engine.rs:467` disposes the engine only when its last reference drops. `CompiledModel` retains that `JitFunction` and only a copied feature count.

The bounds are exact. Let `N` be the checked entry count and `m` the chunk's minimum predicate ID. The pointer attributes are `8*N` bytes for `constants` and `8*(N-m)` for `chunk_base`; each threshold offset `p-m` reaches slot `p`. A chunk without comparisons uses `m=0`, including empty and constant-only forests. The final chunk receives its own base and emits no transfer past the chunk list. The review probe checked these attributes across every chunk of the scalar and SIMD workloads on both targets, plus empty, constant-only, and one-chunk models. For example, the scalar workload has 7000 entries; chunk 0 starts at predicate 30 and has 55760 remaining bytes, while the final chunk starts at 186 and has 54512.

`Trees::new` validates exactly `2^depth` leaves (`src/parsed_tree.rs:123`); comparisons set only those depth bits. Thus `index < tree.leafs.len()` and `leaf_base + index < N`. For every emitted model, `N <= u32::MAX`, so the i64 `add nuw` cannot overflow. Larger valid models return an error before emission. The `u32` rejection is reachable in principle, for example with two valid depth-31 trees on a 64-bit host. The `isize` check is correct but its error is unreachable after a successful `u32` check on this 64-bit host: the maximum accepted byte count is 34359738360. It would constrain a 32-bit host. These huge-allocation error paths were inspected, not exercised with enormous allocations.

Parameter reconstruction and transfer agree: indices start at `pointer_count`, sums at `pointer_count + index_slots`, and features after the sums. Every continuation uses the same function type and `preserve_nonecc`; the wrapper retains its single-pointer C ABI. Module verification passed before and after optimization. The review's post-`prepare_module` IR retains opaque threshold and leaf loads for one-chunk models on both targets. Scalar and SIMD operand tracing also checked each lane against its actual planned predicate, including nonmonotone scheduled IDs.

No leftover per-tree global helper, unused new import/helper, valid-model panic introduced by the diff, or error-style mismatch was found. The new `unwrap` sites extract types or instructions just constructed; nonempty chunks are guaranteed by `continuation::partition`. New safety comments explain pointer bounds. Error strings follow the existing lower-case `leaf table exceeds ...` form with `constant block` as the subject.

All commands used `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'`.

| Validation command | Exit | Result |
| --- | ---: | --- |
| `cargo build --offline --release --examples` | 0 | Both examples built |
| `cargo test --offline --all-targets` | 0 | 60 passed |
| `cargo test --offline --release --all-targets` | 0 | 60 passed |
| `cargo test --offline --doc` | 0 | 1 passed |
| `cargo clippy --offline --all-targets -- -D warnings` | 0 | No warnings |
| `cargo run --offline --release --example inspect_code -- probes/asm 100x100` | 0 | All eight target/configuration outputs regenerated |
| `cargo run --offline --release --example scalar_bench -- 100x100` | 0 | GP12/FP9/two sums: 101.00 ns; GP4/FP9/two sums: 95.60 ns |
| `cargo run --offline --release --manifest-path probes/review/Cargo.toml --target-dir target` | 0 | Real emitter operand/bounds checks, optimized IR dumps, and baseline assembly regeneration |
| `git diff --check` | 0 | No whitespace errors |

The Q4 counts below use `scheduled-12gp-9fp-2sums.s`. Baseline assembly was regenerated from the exact `da93993` JIT source under `probes/`, using the unchanged planner and workload. Both baseline files match the implementor's saved baseline byte for byte. All eight current instruction totals match `VERDICT.md`. `probes/count_assembly.py` and `probes/assembly-counts.json` preserve the counting method and results.

| Assembly measure | Baseline | Current |
| --- | ---: | ---: |
| ARM64 instructions, wrapper included | 4453 | 3224 |
| ARM64 continuation `sp` memory accesses | 0 | 0 |
| ARM64 `movk` instructions | 591 | 0 |
| x86 instructions, wrapper included | 4015 | 3330 |
| x86 `movabsq $.LCPI` instructions | 599 | 0 |
| x86 continuation pushes | 13 | 14 |
| x86 continuation pops | 13 | 14 |
| x86 continuation `rsp` memory operands | 2 | 10 |
| x86 continuation `rbp` memory operands, excluding `lea` | 6 | 0 |
| x86 register-syntax total used in `VERDICT.md` | 34 | 38 |
| x86 actual explicit stack operations | 28 | 38 |

The six baseline `rbp` operands read threshold or leaf tables: each base comes from `movabsq` of `.LCPI` or `.Lleaves`, not a stack address. Thus 34 to 38 reproduces the report's stated syntax rule, while actual explicit stack traffic rises by ten instructions. ARM64 has no continuation stack traffic. All four regenerated ARM64 files contain zero `movk`; all four x86 files contain zero `movabsq $.LCPI`. No register-allocation failure occurred.

| ARM64 continuation loads | `ldp d,d` instructions | Single `ldr d` instructions | Values loaded |
| --- | ---: | ---: | ---: |
| Thresholds, based on `x22` | 34 | 532 | 600 |
| Features, based on `x20` | 24 | 336 | 384 |
| Leaves | 0 | 100 | 100 |

The threshold loads currently cost 566 instructions. There are 58 double-load pairs across continuation bodies, but only 34 are threshold pairs; feature pairs and wrapper restores must not inflate that count.

First-use layout would materially increase pairing on this workload. Its 600 predicate IDs are each compared once. The 38 chunks contain 37 groups of 16 comparisons and one group of 8. Mapping each predicate to its first-use slot, and deriving chunk bases from those mapped slots, would give consecutive local offsets from 0 through at most 120 bytes. That makes up to 300 threshold pairs available within the requested `ldp` offset range, compared with 34 now. Pairing all 532 remaining singles would save 266 instructions, giving an arithmetic estimate of 2958 total. Allow roughly 265–266 fewer instructions; this explains the supplied 2959 probe result without claiming to have reproduced its code generation. Backend scheduling can affect the exact count. No first-use implementation was made: it would change the task's explicitly pinned PredicateId-order layout.

Accept with fixes. The implementation matches the pinned constant-block design, preserves model behavior, and passes every gate. Fix the minor cross-target SIMD test gap before considering its lane-address coverage complete. The first-use layout is a separate performance change, not an omission from this implementation; the current ARM64 load counts explain its likely benefit. The x86 stack increase is real and reported above, alongside the lower overall instruction count.
