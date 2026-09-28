# Study A: backend instruction cost

Use a carried constant pointer to remove ARM64 threshold materialization. Try `select(bit, index + mask, index)` first on x86. The source-level `select` probe passes the release suite and cuts the 100-tree kernel from 4,015 to 3,159 x86 instructions. Combining that emitter shape with a threshold-pointer IR probe gives 3,105 ARM64 and 2,558 x86 instructions. These are instruction counts, not latency guarantees.

The baseline counts are correct. Several explanations in the brief need correction: ARM64 leaf addresses use a GOT load; x86 leaf loads fold into `addsd`; there are 37 continuation-to-continuation transfers; small-model ELF code also uses absolute 32-bit relocations. A larger chunk limit spills with the original selector. With table constants and the source select emitter, limit 64 has no explicit data spills at 100 or 300 trees.

## 1. Answers and evidence

### A1. Every baseline instruction is accounted for

The two original files contain 4,453 and 4,015 instructions. The table counts each instruction once. A folded memory operand is part of its comparison or accumulation instruction. The instruction-classification CSVs record the source line and category for every instruction:

- `probes/aarch64-apple-darwin-classified.csv`
- `probes/x86_64-unknown-linux-gnu-classified.csv`

`python3 probes/audit.py` reproduces the classifications. The assembly paths below use the prefix `captures/100x100/`.

| Category | ARM64 | ARM64 / split | ARM64 / tree | x86 | x86 / split | x86 / tree |
|---|---:|---:|---:|---:|---:|---:|
| Threshold materialization | 1,781 | 2.968 | 17.81 | 600 | 1.000 | 6.00 |
| Feature loads | 360 | 0.600 | 3.60 | 384 | 0.640 | 3.84 |
| NaN normalization | 0 | 0.000 | 0.00 | 0 | 0.000 | 0.00 |
| Comparisons | 600 | 1.000 | 6.00 | 600 | 1.000 | 6.00 |
| Selector construction | 1,240 | 2.067 | 12.40 | 2,110 | 3.517 | 21.10 |
| Leaf addresses | 200 | 0.333 | 2.00 | 100 | 0.167 | 1.00 |
| Separate leaf loads | 100 | 0.167 | 1.00 | 0 | 0.000 | 0.00 |
| Accumulation and sum zeros | 105 | 0.175 | 1.05 | 105 | 0.175 | 1.05 |
| Continuation transfers, copies and stack overhead | 43 | 0.072 | 0.43 | 101 | 0.168 | 1.01 |
| C wrapper | 23 | 0.038 | 0.23 | 14 | 0.023 | 0.14 |
| Other: final continuation return | 1 | 0.002 | 0.01 | 1 | 0.002 | 0.01 |
| Total | 4,453 | 7.422 | 44.53 | 4,015 | 6.692 | 40.15 |

The workload has 600 splits and 100 trees. Thus the whole kernel costs 7.422 instructions per split or 44.53 per tree on ARM64; x86 costs 6.692 per split or 40.15 per tree. These ratios include the wrapper and continuation overhead.

On ARM64, 591 thresholds use `mov` + `movk` + `fmov`, eight use an immediate `fmov`, and the zero threshold uses `fcmp d, #0.0`. That is 1,781 instructions, not 1,800. The remaining `fmov` copies a carried feature in continuation 31. See `aarch64-apple-darwin-scheduled-12gp-9fp-2sums.s:55-85`, `:358`, `:4082`, and `:4555`.

The 394 planned `Action::Load`s become 384 actual feature reads on both targets. ARM64 uses 336 `ldr` and 24 `ldp` instructions, so the row-load instruction count is 360. Ten repeated reads disappear within chunks. x86 uses 384 `movsd`s. The original tree-order capture has 562 actual feature reads, with 519 ARM64 row-load instructions. Thus scheduling removes 178 actual reads here, or 159 ARM64 row-load instructions. The plan-level difference of 206 is not the emitted instruction difference.

ARM64 selector construction has 592 `cset`, eight `cinc`, 492 `orr`, and 148 `ubfiz`. x86 has 600 each of `xorl` and `seta`, 340 `shll`, 395 `orq`, 15 `orl`, 70 `addl`, 48 `leal`, and 42 `leaq`. These totals include index scaling that LLVM moves into bit construction.

Each ARM64 tree has `adrp` plus a GOT pointer `ldr`, then a leaf `ldr`. For example, the first retirement uses lines 254, 257, and 260 of the ARM64 capture. Each x86 tree has one `movabsq` for its table and a memory-source `addsd`; there is no separate leaf-load instruction. See the x86 capture at lines 266-268 and 288-289. All 100 leaf reads are counted in the 102 accumulation instructions on x86.

Accumulation also includes three zero-register initializations on each target. The 102 additions are 100 tree contributions, one sum reduction, and the bias addition. Scale is one in this fixture. `src/jit.rs:356-413` emits these operations. The final continuation return is the one instruction in “other.”

The continuation category includes register placement and stack overhead. On x86, one selector store is charged there; the memory-source `orq` that reloads it stays in selector construction. This avoids double-counting that instruction. The two stack accesses remain visible in the separate stack metric.

There is no NaN normalization in this workload. All feature modes are `AsIs` in `examples/support/distinct.rs:50`. For other modes, `Action::Load` emits `fcmp uno` and a select of negative or positive infinity before any split comparison, at `src/jit.rs:243-263`.

### A2. Thresholds should come through a real pointer argument

ARM64's generic backend accepts these dyadic constants as cheap integer materializations. Most need two integer moves and a transfer to an FP register. LLVM 21's `AArch64TargetLowering::isFPImmLegal` explicitly permits a two-instruction integer materialization for this target. It also accepts the small set of immediate FP constants. Arbitrary `f64` thresholds need not have the same cost. See [LLVM 21.1.8 AArch64 lowering, lines 11193-11233](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/AArch64/AArch64ISelLowering.cpp).

x86 has no corresponding arbitrary FP immediate operand. LLVM puts the value in a constant pool. Its x86 target selects `CodeModel::Large` when the JIT flag is true and no explicit code model was supplied. That explains the absolute `movabsq` before the memory-source compare. The relevant code is `getEffectiveX86CodeModel`, lines 201-212 of [LLVM 21.1.8 X86TargetMachine.cpp](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/lib/Target/X86/X86TargetMachine.cpp).

These full-kernel probes keep the original OR-based selector:

| Probe | ARM64 instructions | ARM64 text bytes | ARM64 body stack accesses | x86 instructions | x86 text bytes | x86 body stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4,453 | 17,812 | 0 | 4,015 | 21,458 | 28 |
| Private threshold global, then early-CSE | 4,453 | 17,812 | 0 | 4,015 | 21,458 | 28 |
| Small, static ELF relocation | 4,454 | 17,816 | 0 | 3,259 | 16,520 | 8 |
| Small, PIC | 4,454 | 17,816 | 0 | 3,371 | 16,852 | 20 |
| Threshold pointer | 3,242 | 12,968 | 0 | 3,458 | 18,135 | 66 |

Text bytes are assembled executable-section sizes, including alignment padding. They exclude constants, relocation records, and unwind data. “Body stack accesses” includes continuation `push`/`pop` and explicit stack memory operands, but excludes the separate C wrapper. Counts are in `probes/counts.json`; every listed assembly file has an assembled `.o` and a `.sections.txt` beside it.

#### Small code model

Changing only `CodeModel::JITDefault` to `CodeModel::Small` at `src/jit.rs:84` changes inspection output, not the engine created at line 49. `create_jit_execution_engine` calls `LLVMCreateJITCompilerForModule` with only the optimization level. This is visible at inkwell 0.9 `src/module.rs:569-609`. The target machine returned by `prepare_module` is discarded by `compile`.

The one-line source probe passed 51 release tests including the doctest. Its benchmark also passed all reference comparisons. The native timing therefore does not demonstrate small-model MCJIT: the factory still chooses its own code model. The exact tested source diff is in section 3.

Inkwell has a second factory, `create_mcjit_execution_engine_with_memory_manager`, at `src/module.rs:647-724`. It writes its explicit `code_model` argument into `LLVMMCJITCompilerOptions.CodeModel` and calls `LLVMCreateMCJITCompilerForModule`. It requires a supplied memory manager. The ordinary factory does not expose that choice. Use explicit `Small` if testing this path; do not assume that the word `Default` has the same effect in two different factories.

`probes/mcjit.cpp` separately tests `EngineBuilder::setCodeModel(Small)` with LLVM's default memory manager on ARM64. All 256 rows pass. Captured native objects contain 4,453 instructions with Large and 4,454 with Small. The objects and disassemblies are `probes/native-{large,small}.{o,dis}`. The same probe also calls `LLVMCreateMCJITCompilerForModule` with `options.CodeModel = LLVMCodeModelSmall`, the C-API path used by inkwell's explicit-options helper. It passes all 256 rows and also captures 4,454 instructions; see `probes/mcjit-capi.log` and `probes/native-small-capi.{o,dis}`. This proves that LLVM honors the explicit option on the available execution target. It does not prove x86 placement or execution safety.

For x86, a small model removes 599 threshold-address `movabsq`s and 38 wrapper/tail target-address materializations. The zero threshold had a register zeroing instruction, not a constant-pool address. Small static code also folds all 100 leaf addresses directly into memory operands. Remaining differences come from allocation and selector lowering.

```asm
# Baseline
movabsq $.LCPI1_0, %rax
ucomisd (%rax), %xmm2
# Small
ucomisd .LCPI1_0(%rip), %xmm2
```

The placement condition in the brief is incomplete. `probes/small-static-relocations.txt` contains 638 `R_X86_64_PC32` relocations and 100 `R_X86_64_32S` relocations. The first group, including constant references and unwind relocations, needs signed 32-bit relative reach. The second group addresses leaf data with absolute signed 32-bit displacements. Putting code and data near each other at arbitrary high virtual addresses does not satisfy those absolute relocations. Low-address placement is required for this static output, unless the relocation/emission scheme changes.

With `-relocation-model=pic`, the probe has 738 `R_X86_64_PC32` relocations and no `R_X86_64_32S`. Leaf bases then use RIP-relative `lea`. This avoids the low-address requirement, but code, constants and referenced data must still satisfy relative reach. The default `SectionMemoryManager::MemoryMapper` contract says its near-allocation request is not a guarantee; see `/opt/homebrew/opt/llvm@21/include/llvm/ExecutionEngine/SectionMemoryManager.h:53-61`. Explicit section placement or validation is still required. Inkwell's MCJIT-options helper exposes a code model but no relocation-model parameter. The C++ `EngineBuilder` route can set both; this study does not introduce that integration.

Small ARM64 emission changes leaf address pairs from `adrp`/GOT `ldr` to `adrp`/`add`, so it removes 100 GOT pointer reads despite the one-instruction increase. The available under-load timings do not establish a latency win. No x86 object was executed. The default large model with a carried data pointer avoids making this placement change.

#### Direct private global

`probes/global/*.raw.ll` replaces each immediate threshold with a constant-index GEP and ordinary aligned load from a private constant array. Running the existing `function(early-cse)` replaces those loads with constants. The resulting instruction streams match the baseline.

A second experiment inserts the same loads after the existing early-CSE pass and sends that IR straight to `llc`. ARM64 then emits 3,317 instructions and uses `adrp`/GOT-load once per continuation. x86 emits 4,016 instructions: it still uses an absolute address per threshold, now named `.Lconstants+offset`. See `probes/global/*.backend.s`. Thus the verified folding culprit in the current pipeline is early-CSE. It is false that the backend invariably turns this load back into an immediate.

A late IR rewrite can preserve the loads without changing read-only attributes. It adds a pass-order dependency and does little for large-model x86. The pointer argument below works with early-CSE left in place.

#### Pointer carried through continuations

`probes/variants.py` adds a private constant array and a second pointer argument to each continuation. The C wrapper takes its address once. Every `musttail` call passes the same pointer in the same argument position. It keeps the row and data parameters read-only, aligned and dereferenceable, and keeps `memory(read)` on the functions. There is no volatile access or fake side effect.

The continuation body receives a pointer, not a known global initializer. The current function-only pass pipeline cannot replace its loads with literal values. `noinline` preserves the real computation boundary. A future interprocedural constant-propagation or specialization pass would need a new check; `noinline` alone is not a promise against those passes.

```llvm
; Before, from the first split
%split = fcmp ogt double %feature, 0x3FF8518000000000
; After, in the continuation body
%threshold_ptr_0 = getelementptr inbounds double, ptr %constants_base, i64 0
%threshold_0 = load double, ptr %threshold_ptr_0, align 8
%split = fcmp ogt double %feature, %threshold_0
```

```asm
; ARM64 before, from continuation 0
mov  x8, #89610197663744
movk x8, #16376, lsl #48
fmov d0, x8
fcmp d2, d0
; Table probe, first two thresholds share a load
ldp  d1, d3, [x21]
fcmp d2, d1
; A later threshold uses ldr dN, [x21, #offset].
```

```asm
# x86 table probe
ucomisd 80(%r13), %xmm5
```

The probe lays out thresholds in emitted comparison order. For a real plan with repeated predicates, reuse a slot for each existing `PredicateId`; otherwise equal threshold loads at different offsets lose a sharing opportunity. This distinct-threshold fixture does not measure that tradeoff.

One extra GP register is live across the whole chain: `x21` or `r13` in these probes. ARM64 gains two wrapper instructions for the global address; x86 gains one. The planner is unchanged in these experiments. A real emitter change must include the persistent pointer in the planner's GP accounting, rather than assume the logical budget proves physical safety.

ARM64 uses 536 single threshold loads and 32 paired loads for 600 values. That replaces 1,781 materialization instructions with 568 loads. The two wrapper instructions leave a net saving of 1,211. The unsigned scaled `ldr` immediate covers indices 0 through 4095 inclusive: 4,096 doubles, not 4,095. This covers both the 600- and 1,800-threshold probes. Pair-load offsets have a smaller signed range, so the 32 paired loads near the start do not extend across the entire table.

On x86, the threshold-only pointer probe saves 557 instructions, including the changed allocation. The extra persistent GP register increases body stack accesses from 28 to 66 with the old selector. Combining the pointer with the source `select` shape removes that stack traffic and saves 601 instructions relative to that selector alone.

The array can remain a module-private constant global. The existing retained `JitFunction` keeps the engine, code and globals alive (`src/jit.rs:22-25`). This keeps source-model-drop behavior. A Rust `Box<[f64]>` would instead need an owning field in `CompiledModel` that outlives every call; moving the box is fine, freeing it before the code is not. There is no reason to add that second ownership scheme for this proposal.

ARM64 ranks the pointer first for savings, the late global rewrite second, and Small last. The direct global with the current pass order saves nothing. x86 Small static saves 756 instructions and Small PIC saves 644, versus 557 for the pointer with the old selector. By integration risk, the pointer ranks first: it preserves the current code-placement contract. Small needs a separate engine/allocator decision. The direct global gives no useful x86 win in either tested pass order.

### A3. `select` gives a useful x86 sequence without NaN changes

The smallest tested change that helps both targets is a conditional index increment:

```llvm
%bit = fcmp ogt double %feature, %threshold
%candidate = add i64 %index, 8
%updated = select i1 %bit, i64 %candidate, i64 %index
```

The mask is `1 << consumer.level`. Its bit has not previously been set for this tree, so adding it has the same result as OR-ing the true bit. This holds for shared predicates too: consumers update their own distinct level bits. The source probe uses the original `i1` comparison result; SIMD mask insertion stays on its original path.

For a bit at level three, `probes/bits-x86_64-unknown-linux-gnu.s:ogt_select` emits:

```asm
leaq    8(%rdi), %rax
ucomisd %xmm1, %xmm0
cmovbeq %rdi, %rax
```

The original microprobe emits `xor` / `ucomisd` / `seta` / `shl` / `or`. The select form saves two instructions in this case. It also avoids temporary Boolean registers for later tree levels. `cmovbe` chooses the old index for unordered comparisons, so NaNs remain false without normalization.

The exact source edit and the table transformation applied to its output give these whole-kernel counts:

| Probe | ARM64 instructions | ARM64 text bytes | ARM64 body stack accesses | x86 instructions | x86 text bytes | x86 body stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| Source select, 100 trees | 4,315 | 17,260 | 0 | 3,159 | 19,545 | 2 |
| Source select, 300 trees | 12,872 | 51,488 | 0 | 9,451 | 58,410 | 2 |
| Threshold pointer + source select, 100 | 3,105 | 12,420 | 0 | 2,558 | 16,177 | 0 |
| Threshold pointer + source select, 300 | 9,269 | 37,076 | 0 | 7,650 | 48,466 | 0 |

The 100-tree source capture is `probes/selector-source/*scheduled-12gp-9fp-2sums.s`. The 100/300-tree IR and assembly are in `probes/selector-emitter-{100,300}/`. On x86 the source probe saves 856 instructions at 100 trees and 2,582 at 300. It has one `rbp` save/restore pair and no explicit selector spill. ARM64 saves 138 and 466 instructions.

An earlier rewrite of already-optimized IR selected only updates that still had an `or` node. That emitted 4,365/3,153 instructions at 100 trees and 13,029/9,440 at 300. Those numbers are in `probes/selector-select*/`; they are not substituted for the actual source probe. The first contribution to each selector takes a different path through early-CSE when the emitter itself uses select.

Other verified microprobes are in `probes/bits.ll`, generated by `probes/bits.py`. Each sequence below omits only `ret`; ABI result copies are shown when present. Operands are already in registers.

| IR form | ARM64 | x86 |
|---|---|---|
| Ordered compare, OR at bit 0 | `fcmp; cset; orr` | `xor; ucomisd; seta; or` |
| Ordered compare, ADD at bit 0 | `fcmp; cinc` | `xor; ucomisd; seta; add` |
| Ordered compare, OR at bit 3 | `fcmp; cset; orr ... lsl #3` | `xor; ucomisd; seta; shl; or` |
| Ordered compare, ADD at bit 3 | `fcmp; cset; add ... uxtw #3` | `xor; ucomisd; seta; lea (...,8)` |
| Ordered compare, select between `index+8` and `index` | `fcmp; add #8; csel` | `lea 8(index); ucomisd; cmovbe` |
| `2*index + zext(ordered compare)` | `fcmp; cset; orr ... lsl #1` | `xor; ucomisd; seta; lea (...,2)` |
| `index + zext(fcmp ult threshold, feature)` | `fcmp; cinc` | `ucomisd; mov index,result; adc $0,result` |
| `2*index + zext(fcmp ult threshold, feature)` | `fcmp; cset; orr ... lsl #1` | `xor; ucomisd; setb; lea (...,2)` |
| `2*index + zext(icmp ult index,31)` | `cmp; cset; orr ... lsl #1` | `xor; cmp; setb; lea (...,2)` |

`nnan ogt` produced the same sequences as `ogt` in these probes. LLVM does emit `adc` from ordinary IR for the unshifted `ult` addition. It did not emit `rcl` or an `adc index,index` form for the shift-and-append probes.

`probes/bits-inline.ll` does emit the two-instruction compare/append sequence through inline asm:

```asm
movq    %rdi, %rax
ucomisd %xmm0, %xmm1
rclq    %rax
```

Here `%xmm0` is the feature and `%xmm1` is the threshold. In AT&T syntax this compares threshold against feature, so CF is set when threshold is below feature, or when the operands are unordered. The first `movq` is the isolated function's input-to-return-register copy. The inline asm declares a flags clobber, is not marked `sideeffect`, and has no memory operand.

There are two costs behind this apparent win. First, unordered must be eliminated correctly. Second, x86's reversed comparison needs the threshold in an XMM destination; a threshold in memory cannot occupy that destination. A table threshold therefore needs an extra `movsd`. It loses the folded threshold operand used by the current compare and by the select proposal. Rebuilding every selector as shift-and-append would also require remapping leaf-table bits to the scheduler's actual comparison order. The two-instruction register-only sequence is not a two-instruction total-cost forecast.

`probes/semantics.py` checks 192 combinations of feature values, thresholds and `NanMode`, including quiet/signaling NaN encodings, both infinities and both signed zeros. For leaf decisions, `AsIs` NaNs can become negative infinity. NaN thresholds can become positive infinity. `AsTrue` must still replace a feature NaN with positive infinity; its comparison against a positive-infinity or NaN threshold remains false. Both operands are then non-NaN and reversed `ult` has the correct Boolean result. This does not justify adding `nnan` to the current unnormalized `AsIs` comparisons.

The whole-kernel ADD-only IR probe emits 4,312 ARM64 instructions but 3,954 x86 instructions with 126 body stack accesses and 21,536 text bytes. x86 text grows from 21,458 bytes. Do not promote that change on instruction count alone. The exact microprobe win depends on the bit position and on the backend's knowledge of non-overlapping bits.

Three instructions are a common ARM64 update cost, not a universal lower bound: bit-zero increment needs `fcmp; cinc`, and an initial Boolean needs `fcmp; cset`. For a shifted update to an arbitrary existing selector, the tested `orr` and select forms each need three. The full source select probe also uses `csel`, `cinc` and constant masks. `fccmp` does not preserve six independent split results in the selector. The Boolean-AND microprobe emitted `fcmp; cset; fcmp; csel` at O1, not `fccmp`. A conjunction can benefit from conditional compares in other lowering contexts, but replacing six required index bits with one combined condition changes the leaf decision.

### A4. Contiguous leaves help x86 when offsets use ADD

A shared base can remove each tree's address materialization. It does not make a nonzero tree offset free. ARM64's register-indexed `ldr` cannot also carry an arbitrary independent displacement. x86 can put both a scaled index and a displacement in its memory operand.

The probes compare three exact forms: OR the table offset into the completed index; ADD that offset before the GEP; or seed the offset into the tree's first selector contribution and carry it through continuations. `probes/seed_offsets.py` traces each completed selector through the real continuation arguments to its first contribution. It checks the depth-six table shape and refuses to seed a selector contribution shared by two trees. The input generator supplies distinct predicates. It does not invent a different comparison order.

For the combined block, the first 600 entries are thresholds, 40 entries pad to a multiple of 64, and tree `t` starts at entry `640 + 64*t`. At 300 trees, leaves start at 1856. The total 100-tree block is 56,320 bytes. There is one persistent pointer, not separate threshold and leaf pointers.

These measurements use the original OR selector so the offset forms can be compared directly:

| Probe | ARM64 instructions | ARM64 text bytes | ARM64 body stack accesses | x86 instructions | x86 text bytes | x86 body stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| Threshold pointer only | 3,242 | 12,968 | 0 | 3,458 | 18,135 | 66 |
| Leaf pointer only, OR offset | 4,437 | 17,748 | 0 | 4,184 | 21,804 | 90 |
| Leaf pointer only, ADD offset | 4,390 | 17,560 | 0 | 3,993 | 21,053 | 91 |
| Shared thresholds/leaves, OR offset | 3,229 | 12,916 | 0 | 3,523 | 18,242 | 30 |
| Shared thresholds/leaves, ADD offset | 3,189 | 12,756 | 0 | 3,330 | 17,475 | 38 |
| Shared thresholds/leaves, seeded offset | 3,199 | 12,796 | 0 | 3,414 | 17,799 | 46 |

The targeted one-load leaf lookup is present, but address savings do not all survive as kernel savings. Offset construction, register allocation and lost combines matter. OR with an offset is especially poor on x86 because it prevents folding that offset into a displacement. Its incoming selectors are LLVM arguments without a low-six-bit range proof.

With the tested source select emitter and ADD offsets, x86 removes exactly 100 more instructions at 100 trees and 299 at 300, relative to threshold pointer plus select. The kernel becomes 2,458 / 7,351 instructions and has zero body stack accesses. The ADD offset folds into the leaf `addsd` memory operand. ARM64 becomes 3,051 / 9,235, a further saving of only 54 / 34 instructions. These are the `probes/both-add-emitter-{100,300}/` artifacts.

```asm
# x86 before, from the baseline capture
movabsq $.Lleaves_67, %r10
addsd   (%r10,%rdx,8), %xmm0
# Shared block: the actual instruction includes the tree's displacement.
# See probes/both-add-emitter-100/x86_64-unknown-linux-gnu.s.
addsd	39424(%r13,%rax,8), %xmm0
```

On ARM64 the desired terminal operation is still one `ldr` from the shared base and the completed index. The nonzero block offset has to reach that address through a preceding base/index operation or an earlier selector seed. The seeded OR-selector probe saves 43 instructions beyond thresholds alone at 100 trees and 132 at 300. It is a valid alternative to measure, not evidence that all 200/600 leaf-address instructions disappear at zero cost. With select introduced by the post-pass rewrite, seeding helps ARM64 but hurts x86 relative to ADD offsets; see `probes/both-seeded-select-{100,300}/`.

H6 holds for leaf decisions. `src/jit.rs:364-374` copies leaf values into private globals. Permuting index bits and applying the inverse permutation to the leaf table leaves the chosen value unchanged. `probes/semantics.py` tests all 720 permutations of six bits against all 64 indices. This study does not implement a different bit assignment in the full kernel.

For mixed depths, concatenation needs actual per-tree offsets. OR seeding additionally needs each tree's block aligned to its own `2^depth` size. The fixed `64*t` formula is a depth-six probe, not a proposed general interface. ADD offsets do not need that OR-alignment condition.

### A5. Chunk safety depends on selector emission too

There are 38 chunks, 37 tail transfers, one wrapper call and one final continuation return. The original `src/sort_policy.rs:172` passes 16 to `continuation::partition`. `probes/runner.rs` applies the same partition function to the same plan with another limit. `probes/plan_copy.rs` differs from the original planner only in the path used to find its continuation module. The action order and budgets stay fixed.

Every ARM64 tail boundary has one `b`. Chunks 1, 11, 13, 19, 28 and 31 also have one plain register copy each: five GP copies and one FP copy. These copies are conservatively charged to carried-value placement; they are not all adjacent to the tail branch or claimed to be mandatory ABI moves. All other boundaries have no copy. There is no continuation-body stack traffic. The wrapper has ten `stp`, a row-pointer move, `bl`, ten `ldp`, and `ret`: 23 instructions and a 160-byte save frame.

Every x86 tail boundary has `movabsq $continuationN` and `jmpq *reg`. Chunks 8, 9, 10, 18, 21, 22, 23, 24, 28, 29, 32, 34 and 35 also save/restore `rbp`. Chunk 34 spills a selector to `-8(%rsp)` at capture line 5201 and reads it through `orq` at line 5268. The wrapper has five pushes, one target-address `movabsq`, one row-pointer move, one call, five pops and a return: 14 instructions. Its five pushes occupy 40 bytes; the call also places a return address on the stack.

`probes/*-boundaries.csv` lists every chunk, including the final return-only chunk. It separates branch, target-address materialization, plain register copies, push/pop and explicit stack memory operations.

The matrix uses separate target emission at O1 with the existing OR selector. “One base” is the OR-offset contiguous threshold/leaf variant from A4. It keeps all read-only attributes.

| Comparison limit | Chunks | ARM64 original instructions / stack | ARM64 one base instructions / stack | x86 original instructions / explicit stack / push-pop | x86 one base instructions / explicit stack / push-pop |
|---|---:|---:|---:|---:|---:|
| 16 | 38 | 4,453 / 0 | 3,229 / 0 | 4,015 / 2 / 26 | 3,523 / 6 / 24 |
| 32 | 19 | 4,552 / 118 | 3,334 / 123 | 4,000 / 83 / 34 | 3,462 / 47 / 32 |
| 64 | 10 | 4,712 / 369 | 3,513 / 392 | 3,978 / 158 / 18 | 3,442 / 128 / 18 |
| Unlimited | 1 | 5,026 / 944 | 3,799 / 921 | 4,120 / 607 / 12 | 3,551 / 538 / 2 |

For the baseline unlimited case, there is no wrapper/continuation split. Its ARM64 raw stack count includes 20 C-save instructions; the local-body count is 924 after excluding those saves. Its x86 raw count includes 12 C-save pushes/pops; the remaining 607 are explicit stack accesses. The one-base unlimited probe needs a C wrapper to pass the data pointer, so its displayed stack count already excludes the wrapper.

The unlimited baseline allocates 3,168 local stack bytes beyond its 160-byte ARM64 save frame. This reproduces the cause recorded in `plans/implementation.md:179`; constants alone do not remove LLVM's long-lived merged loads and delayed consumers.

| Probe | ARM64 instructions | ARM64 text bytes | ARM64 body stack accesses | x86 instructions | x86 text bytes | x86 body stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| One base + post-pass select, limit 32 | 3,115 | 12,460 | 0 | 2,597 | 15,859 | 4 |
| One base + post-pass select, limit 64 | 3,057 | 12,228 | 0 | 2,566 | 15,491 | 14 |
| One base + post-pass select, limit 1000000 | 3,227 | 12,908 | 365 | 2,753 | 16,476 | 446 |

These extra checks use the post-pass select rewrite. ARM64 has no body stack accesses at 32 and 64 here; x86 still uses the stack. The follow-up uses the actual source select emitter with one threshold/leaf block and ADD offsets:

| Probe | ARM64 instructions | ARM64 text bytes | ARM64 body stack accesses | x86 instructions | x86 text bytes | x86 body stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| Source select + shared block, limit 16 | 3,051 | 12,204 | 0 | 2,458 | 15,677 | 0 |
| Source select + shared block, limit 32 | 2,985 | 11,940 | 0 | 2,408 | 15,093 | 2 |
| Source select + shared block, limit 64 | 2,905 | 11,620 | 0 | 2,376 | 14,789 | 4 |
| Source select + shared block, limit 1000000 | 3,062 | 12,248 | 367 | 2,598 | 15,836 | 444 |
| Source select + shared block, 300 trees, limit 64 | 8,766 | 35,064 | 0 | 7,035 | 43,587 | 2 |

At 100 trees, the actual source emitter has no explicit data spills at either 32 or 64 on either target. x86 has one `rbp` save/restore pair at 32 and two pairs at 64. The 300-tree limit-64 follow-up emits 8,766 ARM64 instructions with no stack access and 7,035 x86 instructions with one `rbp` save/restore pair and no explicit data spill. Both ARM64 sizes pass all 761 reference rows.

Keep 16 with the committed OR selector. With the changed emitter, 64 is a useful serial-timing candidate on both targets. Its measured reductions are 146/469 instructions on ARM64 and 82/316 on x86 at 100/300 trees relative to the same emitter at 16. That is per-fixture evidence, not a universal physical-register guarantee. Unlimited still spills.

### A6. Keep the current pass and optimization choices for now

With pointer thresholds and the source select emitter, `llc -O2` produces the same assembly instructions and text size as O1. This is an actual emission check, not an inference from the historical single-function result. Additional passes and CPU settings give:

| Probe | ARM64 instructions | ARM64 text bytes | ARM64 body stack accesses | x86 instructions | x86 text bytes | x86 body stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| O1, generic, early-CSE | 3,105 | 12,420 | 0 | 2,558 | 16,177 | 0 |
| O2, generic, early-CSE | 3,105 | 12,420 | 0 | 2,558 | 16,177 | 0 |
| O1 + instcombine | 3,124 | 12,496 | 0 | 2,632 | 16,418 | 6 |
| O1 + reassociate | 3,105 | 12,420 | 0 | 2,558 | 16,177 | 0 |
| O1, apple-m4 / x86-64-v3 | 3,104 | 12,416 | 0 | 2,558 | 16,177 | 0 |
| O1, apple-m4 / x86-64-v4 | 3,104 | 12,416 | 0 | 2,558 | 16,161 | 0 |

`instcombine` makes both targets larger and introduces x86 body stack traffic. `reassociate` is unchanged on this emitter shape. With the older OR selector and the combined constant block, reassociate instead raised x86 from 3,523 to 3,585 instructions and from 30 to 84 stack accesses; ARM64 stayed at 3,229. Pass effects depend on the IR shape.

The CPU probes replace the function's `target-cpu` attribute, which controls that function's effective CPU. Changing only the target-machine argument would leave the explicit `generic` function attributes at `src/jit.rs:162-166`. For this scalar table/select shape, `apple-m4` saves one ARM64 instruction. x86-64-v3 does not reduce instructions or text. x86-64-v4 keeps the instruction count and saves 16 text bytes. These results do not justify requiring AVX2 or AVX-512 for this scalar change. SIMD target selection belongs to piece C.

## 2. Hypotheses checked

| Hypothesis | Result | Correction or limit |
|---|---|---|
| H1, thresholds dominate | Partly | ARM64 threshold materialization is 1,781/4,453 instructions, about 40%. x86 thresholds are 600/4,015, about 15%; selector construction is larger at 2,110. A base pointer removes repeated address/materialization work. Small model alone does not give ARM64 table loads and does not configure the current MCJIT factory. |
| H2, feature residency is a small instruction lever | Partly | Plan loads are 394 versus 600. Emitted feature reads are 384 versus 562. ARM64 load pairing makes the instruction difference 159. An unlimited-residency claim must count physical registers and spills; this study does not establish such a policy. |
| H5, normalize NaNs once | Confirmed for leaf decisions | `AsIs` can use negative infinity; NaN thresholds can use positive infinity. `AsTrue` still uses positive infinity and stays false against positive infinity. Reversed carry tests need both operands non-NaN. |
| H6, leaf bit assignment can change | Confirmed for leaf decisions | The table must be permuted consistently with the actual comparison-to-bit mapping. A different mapping can change emitted costs and continuation state; it is not automatically a free optimization. |

H3, H4 and H7 are not tested here. This report does not derive execution width, compare latency, or a throughput bound from static instruction counts.

## 3. Ranked recommendations and tested diffs

The 300-tree numbers below are actual emissions from the same distinct-threshold generator at depth six, with 100 declared features. Limit 16 makes 113 chunks; the final limit-64 row has 29. They are better evidence than multiplying the 100-tree total by three. They remain projections for that synthetic workload family, not for an unknown user model.

| Emission shape | ARM64 at 100 | ARM64 at 300 | x86 at 100 | x86 at 300 |
|---|---:|---:|---:|---:|
| Committed baseline | 4,453 | 13,338 | 4,015 | 12,033 |
| Threshold pointer only | 3,242 | 9,733 | 3,458 | 10,315 |
| Source select only | 4,315 | 12,872 | 3,159 | 9,451 |
| Threshold pointer + source select | 3,105 | 9,269 | 2,558 | 7,650 |
| Above + contiguous leaves with ADD offset | 3,051 | 9,235 | 2,458 | 7,351 |
| Above at chunk limit 64 | 2,905 | 8,766 | 2,376 | 7,035 |

1. On ARM64, pass a private threshold block through continuations. In `emit_plan`, create the global once; add the pointer to `continuation_type`, `arguments`, wrapper entry and tail transfers; replace scalar threshold literals in `Action::Compare` with aligned GEP/load pairs. Keep the existing memory attributes and 16-test limit. A normal threshold changes from three materialization instructions to one load, and some pairs use one `ldp` for two thresholds. The measured net saving is 1,211 instructions at 100 trees and 3,605 at 300 before changing selectors. Risk is extra data-load traffic and one persistent GP register. The x86 old-selector version spills more, so pair it with recommendation 2 there.

2. On x86, emit `select(bit, index + (1 << level), index)` for scalar consumers in `Action::Compare`. This is the smallest source change tested through the release suite. A common noninitial update changes from `xor; ucomisd; seta; shl; or` to `lea; ucomisd; cmovbe`, saving two instructions. Actual whole-kernel savings are 8.56 per tree at 100 trees and 8.61 at 300; ARM64 saves 1.38 and 1.55 per tree. Risk is the dependent selector-update chain and target-specific conditional-move cost. These savings use one consumer per predicate. With many consumers of one predicate, reusing one materialized Boolean may cost less than several conditional updates; re-emit that workload. Native ARM64 timing is only indicative; x86 execution is still needed. Combining it with recommendation 1 removes the old-selector table probe's x86 spill penalty.

3. After those two changes, try one contiguous leaf block on x86 with ADD offsets. In `Action::Retire`, replace per-tree globals with a GEP into the shared block. Keep a model-derived tree offset and add it to the selector before addressing. x86 folds that offset into `addsd disp(base,index,8)`, removing about one instruction per tree. It saves 100/299 instructions in the tested combination. ARM64 saves only 54/34, so it ranks behind the threshold and selector work there. Do not substitute OR offsets on x86: that measured worse. Mixed depths and offset overflow require real checked layout logic before an implementation.

4. After the combined emitter changes, test a chunk limit of 64. The planner change would be the literal passed to `continuation::partition` at `src/sort_policy.rs:172`. The probe calls that same function with 64 through the local harness. It removes 28 tail boundaries at 100 trees and 84 at 300, and changes within-chunk load merging/allocation. ARM64 tail transfers change from 37/112 `b` instructions to 9/28; x86 changes the same number of `movabsq; jmpq` pairs. The net savings are 1.46/1.56 instructions per tree on ARM64 and 0.82/1.05 on x86. The resulting totals are in the table. Risk is renewed spilling on other models, depths or budgets. Serial timings and those model shapes must decide the limit; it is not justified for the committed OR emitter.

The before/after instruction sequences are in A2-A5. The whole-kernel tests use the existing sums, strict comparison semantics and scheduler action order. No recommendation is installed in the tracked tree.

Small-model PIC is a separate integration experiment, not the first implementation recommendation. It saves x86 instructions without a carried threshold pointer, but requires an explicit MCJIT configuration and placement contract. The unsafe static-small result is not a deployment recommendation. Keep O1/early-CSE. Keep the current chunk limit with the committed OR selector; measure the larger candidates after changing selector emission. The new 100/300-tree limit-64 captures have no explicit data spills on either target.

Only `src/jit.rs` was edited in tracked source. These are both tested diffs, reproduced verbatim. Each was applied separately and restored before the next source experiment.

```diff
diff --git a/src/jit.rs b/src/jit.rs
index 7efa22a..31a9193 100644
--- a/src/jit.rs
+++ b/src/jit.rs
@@ -81,7 +81,7 @@ fn prepare_module<'ctx>(
             "",
             BACKEND_OPTIMIZATION,
             RelocMode::Default,
-            CodeModel::JITDefault,
+            CodeModel::Small,
         )
         .ok_or("could not create target machine")?;
     let module = emit_plan(context, plan)?;
```

```diff
diff --git a/src/jit.rs b/src/jit.rs
index 7efa22a..433dffe 100644
--- a/src/jit.rs
+++ b/src/jit.rs
@@ -321,30 +321,18 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
                         _ => unreachable!("comparison pack must contain one or two predicates"),
                     };
                     for (comparison, bit) in comparisons.iter().zip(bits) {
-                        let scalar = comparisons.len() == 1;
-                        let bit = if scalar {
-                            builder.build_int_z_extend(bit, index_type, "bit")?
-                        } else {
-                            bit
-                        };
                         for consumer in &comparison.consumers {
-                            let level_bit = if scalar {
-                                builder.build_left_shift(
-                                    bit,
-                                    index_type.const_int(consumer.level as u64, false),
-                                    "level_bit",
-                                )?
-                            } else {
-                                builder.build_and(
-                                    bit,
-                                    index_type.const_int(1u64 << consumer.level, false),
-                                    "level_bit",
-                                )?
-                            };
                             let index = indices[consumer.tree.0]
                                 .expect("comparison requires an active tree");
-                            indices[consumer.tree.0] =
-                                Some(builder.build_or(index, level_bit, "leaf_index")?);
+                            let mask = index_type.const_int(1u64 << consumer.level, false);
+                            let updated = if comparisons.len() == 1 {
+                                let candidate = builder.build_int_add(index, mask, "candidate_index")?;
+                                builder.build_select(bit, candidate, index, "leaf_index")?.into_int_value()
+                            } else {
+                                let level_bit = builder.build_and(bit, mask, "level_bit")?;
+                                builder.build_or(index, level_bit, "leaf_index")?
+                            };
+                            indices[consumer.tree.0] = Some(updated);
                         }
                     }
                 }
```

The table and layout proposals were probed in generated LLVM IR, not implemented in tracked Rust. `probes/variants.py` is the exact full-module rewrite used for the threshold and leaf experiments. `probes/selector_variants.py` contains the earlier post-pass select and ADD experiments; `probes/seed_offsets.py` contains the seeded-offset experiment. All retain the real model constants and action-derived code. The complete before/after IR is stored beside each assembly file. Verbatim full-module IR diffs are also saved as `probes/thresholds-<triple>.ir.diff` and `probes/both-add-100-<triple>.ir.diff`; they include the literal tables and are large. There is no omitted tracked-source table implementation to replay.

## 4. Validation and experiments to run serially

The initial command `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 cargo build --offline --release --examples` passed before assembly was used. A normalized instruction comparison between `llc -O1 -code-model=large` on the captured baseline IR and the supplied assembly matches both baseline instruction streams. The default `llc` comments differ; instruction operands do not.

`cargo test --offline --release` initially exited 101 in rustdoc because the README doctest linked the wrong zstd location. The documented README workaround fixes it:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
export RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'
cargo test --offline --release
```

That command passed for the baseline, the Small inspection probe, and the source select probe: 30 unit tests, 20 integration tests across three binaries, and one doctest. The select source probe also passed `cargo build --offline --release --examples`, both-target inspection, and the existing scalar benchmark's reference checks. Logs are `probes/{test-baseline-fixed,test-prepare-small,test-selector-source,build-selector-source,emit-selector-source,bench-selector-source}.log`.

Each ARM64 full-model IR variant was checked against `Trees::predict` on 256 generated rows, five rows filled with exceptional values, and 500 rows with one feature replaced by NaN, infinity or signed zero: 761 rows per variant. Logs are `probes/native-checks*.log`. The dedicated semantic and leaf-permutation checks are `probes/semantics.log`. LLVM MC assembled both targets' full kernels and all bit-insertion snippets. x86 was never executed.

The exact bench command was:

```sh
cargo run --offline --release --example scalar_bench -- 100x100
```

| Source state | Minimum ns/call | Median ns/call |
|---|---:|---:|
| Committed emitter | 120.83 | 121.58 |
| Small only in `prepare_module` | 117.86 | 120.41 |
| Scalar select emitter | 115.12 | 118.11 |

These were separate runs under machine load, with the existing dependent driver included. Their ranges overlap. The Small preparation run still used the default MCJIT code model. The raw-pointer IR and C++ probe timings use different drivers; their full minimum/median output is retained in the logs and is not mixed into this comparison.

To reproduce the constant-pointer experiment without editing tracked files:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml -- dump 100 16 probes/replay-baseline
python3 probes/variants.py probes/replay-baseline probes/replay-baseline
python3 probes/variants.py probes/replay-baseline probes/replay-thresholds thresholds
python3 probes/audit.py probes/replay-baseline/aarch64-apple-darwin.s probes/replay-thresholds/aarch64-apple-darwin.s
```

For a serial same-driver comparison of the retained validated IR:

```sh
for round in 1 2 3 4 5; do
  target/release/probe check 100 probes/baseline/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/thresholds/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/thresholds-emitter-100/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/both-add-emitter-100/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/both-add-emitter-100/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/thresholds-emitter-100/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/thresholds/aarch64-apple-darwin.ll
  target/release/probe check 100 probes/baseline/aarch64-apple-darwin.ll
done
```

The expected static counts are 4,453, 3,242, 3,105 and 3,051. The experiment should test whether their latency order follows instruction count once other machine activity is removed. It may not: replacing immediate construction adds memory traffic, and selector dependencies change.

To replay the actual source select diff and its gates, `python3 probes/run_selector_source.py` applies it, saves its diff, runs the suite/build/inspection/benchmark, and restores the original bytes in `finally`. The inspection example requires a fresh directory: first change the script's output directory from `probes/selector-source` to a new path, or run in a fresh worktree with these probe files. `python3 probes/run_selector_source.py --ir-only` captures 100/300-tree IR from that already-tested source change and restores the source. It does not rerun the gates. No format tool is involved. To reproduce the final 300-tree chunk experiment:

```sh
python3 probes/run_selector_source.py --chunks-only --300 --only-64
python3 probes/variants.py probes/selector-emitter-300-64 probes/replay-combined-300-64 both-add
target/release/probe check 300 probes/replay-combined-300-64/aarch64-apple-darwin.ll
python3 probes/audit.py probes/replay-combined-300-64/aarch64-apple-darwin.s probes/replay-combined-300-64/x86_64-unknown-linux-gnu.s
```

Expected totals are 8,766/7,035 instructions. The first command restores `src/jit.rs` before returning.

For the chunk check, use the same model and replace 16 with 32, 64 or 1000000 in the runner command, then emit the unchanged and `both` variants. Compare instruction count, stack accesses and text bytes, not only the branch count:

```sh
CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml -- dump 100 32 probes/replay-32
python3 probes/variants.py probes/replay-32 probes/replay-32
python3 probes/variants.py probes/replay-32 probes/replay-both-32 both
python3 probes/audit.py probes/replay-32/aarch64-apple-darwin.s probes/replay-both-32/aarch64-apple-darwin.s
```

Expected ARM64 totals are 4,552/3,334, with 118/123 body stack accesses. To check a backend setting against the retained table/select IR:

```sh
/opt/homebrew/opt/llvm@21/bin/llc -O2 -asm-verbose=false -code-model=large probes/thresholds-emitter-100/aarch64-apple-darwin.ll -o probes/replay-default-aarch64.s
/opt/homebrew/opt/llvm@21/bin/opt -S '-passes=function(early-cse,instcombine)' probes/thresholds-emitter-100/x86_64-unknown-linux-gnu.ll -o probes/replay-instcombine.ll
/opt/homebrew/opt/llvm@21/bin/llc -O1 -asm-verbose=false -code-model=large probes/replay-instcombine.ll -o probes/replay-instcombine-x86_64.s
```

Expected counts are unchanged at 3,105 for the first command and worse at 2,632 for the instcombine case. The equivalent target-cpu probes must update the function attributes, not just `llc -mcpu`.

On a native x86 host, run the same corpus/reference check and dependent benchmark for the original emitter, scalar select, threshold pointer plus select, and ADD-offset leaves. Keep the large model first:

```sh
CARGO_TARGET_DIR=target cargo build --offline --release --manifest-path probes/Cargo.toml
target/release/probe check 100 probes/baseline/x86_64-unknown-linux-gnu.ll
target/release/probe check 100 probes/selector-emitter-100/x86_64-unknown-linux-gnu.ll
target/release/probe check 100 probes/thresholds-emitter-100/x86_64-unknown-linux-gnu.ll
target/release/probe check 100 probes/both-add-emitter-100/x86_64-unknown-linux-gnu.ll
```

These commands require that host's LLVM 21 prefix. Expected static totals at 100 trees are 4,015, 3,159, 2,558 and 2,458. Then test Small PIC with a memory manager that checks section placement and inspect relocation distances before execution. The present machine cannot provide that last gate.

The attempted `git checkout -- src/jit.rs` failed because the sandbox could not create the linked worktree's external `.git/worktrees/.../index.lock`. The single Small edit was restored directly and checked with `git diff`. The select probe restored its saved original bytes. This is recorded as a sandbox artifact; no runtime code was altered to hide it. The final restored build passed. The final status and diff checks are recorded in `VERDICT.md`.

## 5. Questions that could change the ranking

- Is deployment ARM64, x86-64-v3, x86-64-v4, or several of these? x86 execution evidence is still missing.
- Must MCJIT allocations work at arbitrary virtual addresses, or can code/data be placed in a checked region? That decides whether a small-model experiment is viable.
- What are the real model's depths, repeated predicates and threshold bit patterns? The dyadic distinct-border fixture makes most ARM64 immediates exactly three instructions; other thresholds may already use pool loads.
- How often do feature rows contain NaNs, and which `NanMode`s are used? Select needs no new normalization. Carry-based proposals do.
- Are model constants hot across calls, and how many compiled models compete for cache? A threshold table trades instructions for loads.
- Does the real model have mixed depths or many constant trees? That changes the leaf-block layout and the value of offset seeding.
