I could not write the report to `/private/tmp/.../scratchpad/REPORT_A_opus.md`: the harness refused the Write ("Subagents should return findings as text, not write report files"), so the full report is below. Probe scripts, `.ll` probes, the instruction classifier and every emitted capture are in the worktree under `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-a4e7cb7f48ea806f7/probes/`. The tested diff is saved at `probes/probe.diff` and also appears verbatim in the appendix. `git status --short` shows only `?? probes/`: every tracked file I touched (`src/jit.rs`, `src/sort_policy.rs`, `examples/support/mod.rs`, `examples/inspect_code.rs`) was reverted with `git checkout --`.

Summary of the ranked recommendations:
1. Put thresholds and leaves in one constants block and read them through a pointer argument. On ARM64, also pass a second pointer that moves to each chunk's thresholds so `ldp` can load two at once. ARM64 drops from 4453 to 2959 instructions (−34%), x86-64 from 4015 to 3330 (−17%). At 300 trees: 13338 → 8945 and 12033 → 9930. Plain IR, all integration tests pass.
2. Build each leaf index with a carry chain on top of 1. On x86-64 that is `llvm.x86.addcarry.64` plus `ucomisd` against the next double above the threshold; on ARM64 it is inline asm `fcmp t, x; adc`. Each split costs 2 instructions, and the leaf tables are reordered at compile time to match. x86-64 goes 3330 → 1944 and ARM64 2959 → 2275; at 300 trees 5758 and 6780. Tests pass on ARM64. The x86 intrinsic path is checked only in emitted assembly, never run.
3. Emit `AsFalse` features like `AsIs`, since the two give identical results. Saves 4 instructions per such feature load on ARM64 and 7 on x86-64. It is zero on these workloads because every feature is `AsIs`.
4. After 1 and 2, raise the chunk limit to 32 or 64 on ARM64 only (−65 / −138 instructions, no stack traffic) and keep 16 on x86-64.
5. Do not switch MCJIT to `CodeModel::Small`. It is unsafe as configured and adds only about 37 instructions on top of 1 and 2. Backend `Default`, extra `instcombine`/`reassociate` and host CPUs change counts by 0 to 2%.

---

# Study A: backend instruction cost of the scalar kernel (100x100)

Worktree at da93993. All anchors in the brief matched: `emit_plan` at `src/jit.rs:100`, the Compare arm at 268-350, Retire at 356-392, the transfer at 421-442, `prepare_module` at 71-98, and the chunk literal at `src/sort_policy.rs:172`. A fresh default emission is byte-identical to `captures/100x100`.

Timings were taken on the M4 Pro while other agents were running, so they are indicative only. Instruction counts, code bytes and stack-access counts are the evidence. Tool: `probes/classify.py` puts every instruction line into exactly one category.

## 1. Answers per question

### A1. Where the instructions go

The category totals reconcile exactly to the file totals.

| Category | ARM64 | x86-64 | Mnemonics |
|---|---:|---:|---|
| Threshold constant | 1781 | 599 | ARM64: 591 `mov #imm`, 591 `movk`, 599 `fmov` (591 `fmov d,x`, 8 `fmov d,#imm`). x86: 599 `movabsq $.LCPI` |
| Feature load | 360 | 384 | ARM64: 336 `ldr` + 24 `ldp` off `x20` (384 values). x86: 384 `movsd N(%r12)` |
| NaN normalization | 0 | 0 | every feature is `NanMode::AsIs` in this workload |
| Comparison | 600 | 600 | `fcmp` / `ucomisd` (x86 uses the threshold as a memory operand) |
| Bit insertion | 1240 | 2109 | ARM64: 592 `cset`, 492 `orr ..lsl`, 148 `ubfiz`, 8 `cinc`. x86: 600 `xorl`, 600 `seta`, 394 `orq`, 340 `shll`, 70 `addl`, 48 `leal`, 42 `leaq`, 15 `orl` |
| Leaf address | 200 | 100 | ARM64: `adrp l_leaves_N@GOTPAGE` + GOT `ldr`. x86: `movabsq $.Lleaves_N` |
| Leaf load | 100 | (folded) | ARM64: `ldr d,[x,xidx]`, index already scaled by 8 in the shifts. x86: folded into `addsd (%r,%i,8)` |
| Accumulation | 102 | 102 | 100 leaves + `AddSums` + bias add |
| Sum zero / bias zero | 3 | 4 | `movi` / `xorpd`,`xorps` (one x86 zero is the 0.0 threshold) |
| Transfer | 43 | 102 | ARM64: 37 `b`, 5 `mov`, 1 `fmov`. x86: 37 `movabsq $continuationN` + 37 `jmpq *%r`, 13 push/13 pop of `%rbp`, 1 spill store, 1 reload |
| Wrapper | 23 | 14 | ARM64: 10 `stp` + 10 `ldp` (callee saves around the `preserve_nonecc` call), `mov`, `bl`, `ret`. x86: 5 push, 5 pop, `movabsq`, `movq`, `callq`, `retq` |
| Return | 1 | 1 | |
| **Total** | **4453** | **4015** | |

Per split: ARM64 costs 6.0 instructions (2.97 threshold + 1 `fcmp` + 2.07 bit insertion). x86-64 costs 5.5 (1 `movabsq` + 1 `ucomisd` + 3.5 bit insertion).

Per tree, on top of its splits: ARM64 costs 7.6 (3.6 shared feature loads + 2 leaf address + 1 leaf load + 1 `fadd`); x86-64 costs 5.8. A depth-6 tree therefore costs about 44 instructions on ARM64 and 39 on x86-64. Threshold materialization plus bit insertion is 68% of ARM64 and 67% of x86-64.

NaN normalization, for `AsFalse` or `AsTrue` features, appears at `Action::Load` as `fcmp uno` + `select` (`src/jit.rs:244-264`). Probed in `probes/nan.ll`:
- ARM64: `mov x,#0xfff0..`, `fmov`, `fcmp d0,d0`, `fcsel` (4 per load).
- x86-64: `movabsq`, `movsd`, `movapd`, `cmpunordsd`, `andpd`, `andnpd`, `orpd` (7 per load).

### A2. Threshold constants

**Why ARM64 uses `mov`/`movk`/`fmov`.** AArch64 builds an f64 constant through a general register when that takes at most two 16-bit moves. For the `generic` CPU there is no literal fusion, so the limit is two. The synthetic thresholds are `i/8192 - 4` (`examples/support/distinct.rs:15-17`): only their top 32 bits are non-zero, so all of them fit in two moves. Real CatBoost borders are float32; widened to f64 they usually have three non-zero 16-bit chunks.

`probes/thr.ll` compares the two kinds:

| Target, code model | Synthetic (2 chunks) | f32 border (3 chunks) |
|---|---|---|
| aarch64-darwin, large (= JITDefault) | `mov`,`movk`,`fmov` (3) | `mov`,`movk`,`movk`,`fmov` (4) |
| aarch64-darwin, small | 3 | `adrp` + `ldr d,[x,lCPI@PAGEOFF]` (2) |
| aarch64-linux, large | 3 | `movz` + 3 `movk` + `ldr` (5) |
| x86-64, large | `movabsq` + `ucomisd (%rax)` (1 extra) | same |
| x86-64, small | `ucomisd .LCPI(%rip)` (0 extra) | same |

On real borders, today's ARM64 kernel pays 4 instructions per threshold, not 3. That is an estimated +600 instructions per 100 trees; I probed the per-constant cost but did not emit a whole kernel with f32 borders.

x86-64 gets `movabsq` because `JITDefault` means the large code model for MCJIT on x86-64. `prepare_module` passes `JITDefault` (`src/jit.rs:84`), and MCJIT is created through `LLVMCreateJITCompilerForModule`, which takes no code model (inkwell `module.rs:569-596`).

**a. Small (or Default) code model.** Emitting the default plan with `CodeModel::Small` gives:
- x86-64: 3259 instructions. The `movabsq` in front of thresholds, leaves and continuations all disappear.
- ARM64: 4454. `adrp`+`add` replaces `adrp`+GOT `ldr` for leaves; the synthetic thresholds do not change.

It is unsafe under MCJIT as configured:
- With `RelocMode::Default`, the x86-64 small model emits 32-bit absolute data addresses (`addsd .Lleaves_67(,%rdi,8)`, and in the probe `movl $.Lconstants, %r13d`). Those only work if the data sits below 2-4 GB. MCJIT's section memory comes from `mmap` and has no such guarantee.
- With PIC, code and data must be within ±2 GB of each other, which the default MCJIT memory manager does not promise.
- I believe RuntimeDyld checks that range only with an `assert`, so release LLVM would truncate silently. I did not verify that in LLVM source.

inkwell 0.9 exposes a code model only through `create_mcjit_execution_engine_with_memory_manager` (`module.rs:646-723`), which needs a custom `McjitMemoryManager`. I did not execute the small model under MCJIT.

**b. Private constant global with a constant index (`probes/thr_global.ll`).** `function(early-cse)` folds every such load back into an immediate; no loads remain. Even without that fold it would not help x86 under the large model, which emits one `movabsq $.Lthresholds+8k` per threshold anyway. The way to stop the fold without fake side effects or `volatile` is to make the address opaque where it is loaded, by passing it as an argument. That is option c. Dropping `constant` from the global also stops the fold, but it gives up the read-only annotation the project keeps on purpose.

**c. Base pointer argument (`probes/thr_base.ll`, then the whole kernel with `TB_PROBE=base`).** The wrapper passes the address of one private constant global as a second parameter to every continuation. The global holds the thresholds in first-use order, then one 64-entry leaf block per tree. The parameter is marked `readonly noalias align 64 dereferenceable(N)`. Emitted code:
- ARM64: `ldr d,[x21,#imm]` or `ldp`.
- x86-64: `ucomisd 128(%r13),%xmm3` with no extra instruction, and `addsd 39424(%r13,%rcx,8),%xmm0` for leaves.

Whole kernel: ARM64 4453 → 3189, x86-64 4015 → 3330. All integration tests pass. The only failures are the three unit tests in `src/jit.rs` that assert today's IR text.

On ARM64, only 32 of 568 threshold loads became `ldp`. The per-chunk count explains it: chunks 0-3 pair fully and chunks 4-37 not at all. `ldp d` has a 7-bit scaled offset (−512 to +504 bytes), so it cannot reach thresholds further than 504 bytes from the pointer. Passing a third pointer that each transfer moves to the next chunk's first threshold (`TB_REBASE=1`, one `add x22,x22,#128` per transfer) fixes it: 297 `ldp` + 6 `ldr` for 600 thresholds, 2959 instructions in total, still plain IR.

Register cost is one general register, two with the ARM64 per-chunk pointer. On x86-64 this runs into the argument limit described in A5.

Ownership options:
- A module-private global keeps today's arrangement: the engine owns it like `leaves_N`, and the wrapper computes its address once per call (ARM64 `adrp` + GOT `ldr`, x86 `movabsq`).
- A Rust-owned `Box<[f64]>` passed from `CompiledModel::predict` as a second argument removes the last data relocation from the module. That matters only for recommendation 5.
- Either way, a one-chunk model must still be emitted as wrapper plus `continuation0`, or early-cse folds the loads back. The probe forces that split.

**Ranking of a, b, c:**

| Option | ARM64 saved | x86-64 saved | Risk |
|---|---:|---:|---|
| c, plus ARM64 per-chunk pointer | 1494 | 685 | low |
| a | 0 synthetic (2 per f32 border) | 756 | high under MCJIT unless an arena memory manager is written |
| b | 0 (early-cse folds it back) | 0 | none, and no effect |

### A3. Bit insertion

Probed in `probes/bits.ll`, `probes/carry_x86.ll` and `probes/carry_a64.ll`, with thresholds read from a pointer so they do not obscure the result.

| IR shape | ARM64 | x86-64 |
|---|---|---|
| today: `or idx, shl(zext(fcmp ogt),k)` | `fcmp`,`cset`,`orr lsl` (3; `cinc` for bit 0) | `xorl`,`ucomisd`,`seta`,`leaq` or `shll`+`orq` (4-5) |
| same with `add` | identical | identical |
| doubling chain `idx*2 + zext(fcmp)`, including `ult`/`ule` | rewritten back to `cset`+`orr` (3) | rewritten to `set*`+`lea`/`or` (4) |
| `select c, idx\|bit, idx` | `fcmp`,`orr #bit`,`csel` (3) | `movq`,`orq`,`ucomisd`,`cmovbe` (4) |
| `llvm.x86.addcarry.64(zext i8 (fcmp ult x, next_up(t)), idx, idx)` | n/a | `ucomisd 8(%rdi),%xmm1` + `adcq %rsi,%rsi` (2) |
| ARM64 inline asm `"fcmp ${1:d}, ${2:d}\nadc $0, $3, $3"`, constraints `"=r,w,w,r,~{cc}"` | `fcmp d4,d0` + `adc x8,x1,x1` (2) | n/a |

What the probes show:
- Using `add` instead of `or` changes nothing.
- No plain IR form produces `adc` or `rcl`; LLVM turns the doubling chain back into shift and or.
- On x86-64, the intrinsic `llvm.x86.addcarry.64` gives 2 instructions per split.
- On ARM64 there is no IR route from a floating-point compare's carry flag to `adc`, so 2 per split needs inline asm. The plain-IR floor is 3 (`fcmp`, `cset`, `orr` with the shift folded in).
- `fccmp` combines several comparisons into one flag, but a leaf index needs one bit per split, so it does not help.

How the carry forms work. Both build the index most significant bit first, one doubling per split in plan order. The bit they add is the complement of `x > t`:
- **ARM64.** `fcmp t, x` sets the carry flag when `t >= x` or the operands are unordered. That is exactly `!(x > t)`, including NaN `x` and NaN `t`, so no threshold adjustment is needed.
- **x86-64.** `ucomisd T(%base), %xmm_x` sets the carry flag when `x < T` or unordered. With `T = next_up(t)`, that equals `!(x > t)` for every non-NaN `x`. For NaN `x` the flag is 1, which is the right answer for `AsIs` and `AsFalse`.
  - A `+inf` or NaN border can never be exceeded, so it becomes a constant bit at compile time.
  - Caveat: if the host sets denormals-are-zero in MXCSR, `next_up(0.0)` is a denormal and reads as 0, which breaks `x > 0` for denormal `x`. ARM64 has the same caveat with FPCR.FZ.
- **Leaf tables.** The bits are complemented and their positions follow plan order, so each leaf table is reordered at compile time: the original index is the sum over plan-order split j of `(1 - bit_{d-1-j}(p)) << level_j`. That is H6 in practice.
- **Seed.** Each tree's index starts at its block number, so after d doublings it is the absolute slot in the constants block.

Validation:
- `TB_PROBE=carry` (ARM64 inline asm, run natively) passes.
- `TB_PROBE=carrygen` is the x86 formulation written in generic IR so it runs on ARM64; it also passes.
- Both pass `tests/differential.rs` (NaN, ±inf, every NaN mode, mixed depths), `model_reference` and `scalar_jit`, plus the exact-equality check in `scalar_bench` on 100x100.
- The lowering of the x86 intrinsic is checked only in emitted assembly.

Whole kernel with the carry chain (`TB_PROBE=carry`): ARM64 2501 (2275 with the per-chunk pointer), x86-64 1944.

### A4. Leaf lookup

| Form | ARM64 per tree | x86-64 per tree |
|---|---|---|
| today | `adrp` + GOT `ldr` + `ldr` + `fadd` (4) | `movabsq` + `addsd (%r,%i,8)` (2) |
| base pointer, index built from 0 | `add`/`mov #off` + `ldr` + `fadd` (about 3.5) | `addsd disp32(%r13,%i,8)` (1) |
| base pointer, seeded with the tree's block number | `mov w8,#31` + `ldr d,[x21,x9,lsl #3]` + `fadd` (3) | `movl $0,%r` + first `adcq $2K,%r` + `addsd (%r13,%r8,8)` (2) |

H6's seeding idea works as stated: after d doublings the index lands on the tree's block start. The seed costs one instruction per tree. On x86 it is `movl $0` rather than `xorl`, because `xorl` would overwrite the flags between `ucomisd` and `adcq`.

Net per tree: ARM64 saves 1 instruction, x86-64 saves nothing. The leaves share the threshold base register. ARM64's `ldr` with an immediate reaches 4095 thresholds; leaves use a register index with no reach limit. The worst case in scope, 300 trees at depth 8 (2400 thresholds), fits.

### A5. Continuation transfer and chunk size

Cost per chunk boundary today:
- ARM64: one `b`, plus about 0.2 register moves.
- x86-64: `movabsq` + `jmpq *%rax`, plus 13 push/pop pairs of `%rbp` spread over the 37 transfers. `preserve_nonecc` keeps `%rbp` callee-saved on x86-64.
- Wrapper: ARM64 23 instructions, 20 of them register saves and restores; x86-64 14.

Chunk-limit sweep (`TB_CHUNK`; each cell is instructions / stack accesses outside the wrapper; "unlimited" puts the whole kernel in one function):

| Limit | default ARM64 | default x86 | carry ARM64 | carry + per-chunk ptr ARM64 | carry x86 |
|---|---|---|---|---|---|
| 8 | 4518 / 0 | 4121 / 0 | 2567 / 0 | 2392 / 0 | 2010 / 0 |
| 16 | 4453 / 0 | 4015 / 28 | 2501 / 0 | 2275 / 0 | 1944 / 2 |
| 32 | 4552 / 118 | 4000 / 117 | 2448 / 0 | 2210 / 0 | 1902 / 26 |
| 64 | 4712 / 369 | 3978 / 176 | 2389 / 0 | 2137 / 0 | 1909 / 92 |
| unlimited | 5026 / 944 | 4120 / 619 | 2557 / 315 | 2555 / 319 | 1995 / 446 |

The spilling that chunking exists to prevent is still real: an unlimited kernel spills on both targets even after the carry change. With thresholds and leaves off a pointer plus the carry chain, ARM64 has no stack traffic up to 64 tests per chunk. x86-64 starts spilling at 32.

**x86-64 compile failure.** Continuations with 12 general-register arguments fail to compile under `JITDefault`: `ran out of registers during register allocation in function 'continuationN'`.
- It happens with the unmodified emitter at GP budget 16 or 20 on 100x100 (13 or 17 index slots plus the row pointer).
- It also happens with base pointer + per-chunk pointer + 9 index slots. Eleven arguments compiled; twelve did not.
- With `CodeModel::Small`, which emits direct `jmp continuationN`, all of these compile.
- Likely cause: the large model's indirect `jmpq *%reg` needs a free register, and `preserve_nonecc` has only 12 general argument registers.
- The default GP12 plan carries at most 9 indices here, so today's default is fine. `scalar_bench`'s GP16 and GP20 kernels would not compile on an x86-64 host. Piece B should use 11 general arguments as the x86-64 ceiling.

### A6. Backend settings

Instruction totals, ARM64 / x86-64, all emitted:

| Setting | default kernel | base pointer | base pointer + carry |
|---|---|---|---|
| `Less` (current) | 4453 / 4015 | 3189 / 3330 | 2501 / 1944 |
| `OptimizationLevel::Default` | byte-identical | byte-identical | byte-identical |
| passes `early-cse,instcombine,reassociate` | 4484 / 4061 | 3220 / 3349 | unchanged |
| target-cpu `apple-m4` / `x86-64-v3` | 4450 / 3996 | 3185 / 3301 | 2497 / 1902 |
| target-cpu `apple-m4` / `x86-64-v4` | 4450 / 3988 | 3185 / 3293 | 2497 / 1901 |
| `CodeModel::Medium` | not supported on AArch64 (aborts) | | |

### Timings

Indicative, loaded machine, ARM64, scheduled GP12/FP9 with two sums. Median ns per call, including a 5.5-5.9 ns driver.

| Variant | Instructions | ns |
|---|---:|---:|
| baseline | 4453 | 124.1, 122.0 |
| base pointer | 3189 | 105.0 |
| base pointer + per-chunk ptr | 2959 | 95.0 |
| carry in generic IR (x86 semantics), no per-chunk ptr | (not emitted) | 101.0 |
| carry in generic IR + per-chunk ptr | (not emitted) | 94.2 |
| base pointer + carry, inline asm | 2501 | 96.6, 97.7 |
| base pointer + carry, inline asm + per-chunk ptr | 2275 | 89.9 |

Time falls less than instruction count. At 89.9 ns the kernel runs about 370 cycles for 2275 instructions (about 6 per cycle), against about 8.7 per cycle at baseline.

The per-chunk pointer removed 265 load instructions and 226 instructions in total, and bought 5-8 ns. The inline-asm carry chain removed 684 instructions and bought about 5 ns (95.0 → 89.9). About 790 load instructions remain. My guess is that load issue is the next limit on ARM64; piece D should test that, it is not measured here.

## 2. Hypotheses touched

- **H1: partly confirmed, with corrections.**
  - The 1781 of 4453 figure is right, but it is a synthetic artifact. f32 borders cost 4 instructions on Darwin under `JITDefault` and 5 on Linux AArch64.
  - The x86 `movabsq` per threshold is a large-code-model effect, and the small model does fold it. But the small model is unsafe under MCJIT: it emits 32-bit absolute addresses.
  - A pointer argument gets the same folding safely. On ARM64, pairing thresholds into `ldp` needs a pointer that moves per chunk.
- **H4: not visible yet.** With the carry chain, sums 1/2/4 measured 96.7/96.6/100.8 ns. Piece D owns this.
- **H5: confirmed, with corrections.**
  - `AsIs` and `AsFalse` behave identically (`src/parsed_tree.rs:202-214`). `AsTrue` differs.
  - A NaN threshold behaves exactly like `+inf`: the bit is always false.
  - `-0.0` and `+0.0` compare equal, so zero borders are safe.
  - The carry forms handle NaN `x` directly, so `AsIs` and `AsFalse` need no normalization at all.
  - The x86 `next_up` trick needs `+inf`/NaN borders turned into constants and assumes denormals are not flushed to zero.
- **H6: confirmed.** The carry probe reorders every leaf table and passes the differential tests.
- **H2: context only.** After the changes, feature loads are 16% of the ARM64 kernel and 20% of x86. Tree order against scheduled order measured 92.8 against 89.9 ns in the carry + per-chunk pointer run.

## 3. Ranked recommendations

Totals emitted on 100x100 and 300x100. The 300x100 workload was added to `Workload::parse` only for this probe.

| Variant | ARM64 100 | ARM64 300 | x86 100 | x86 300 | x86 text bytes, 100 / 300 |
|---|---:|---:|---:|---:|---:|
| baseline | 4453 | 13338 | 4015 | 12033 | 21458 / 64082 |
| R1 base pointer | 3189 | 9701 | 3330 | 9930 | 17475 / (not measured) |
| R1 + ARM64 per-chunk ptr | 2959 | 8945 | n/a | n/a | n/a |
| R1 + R2 | 2501 | 7500 | 1944 | 5758 | 13209 / 39586 |
| R1 + R2 + ARM64 per-chunk ptr | 2275 | 6780 | n/a | n/a | n/a |

The per-chunk pointer rows are ARM64 only. On x86-64 it fails to compile under `JITDefault`.

### R1. Read thresholds and leaves off a pointer argument, with a per-chunk pointer on ARM64

**What changes in `src/jit.rs`:**
- Build one private constant `[N x double]` global: thresholds in first-use order, then leaf blocks.
- Add a pointer parameter after `row` on every continuation (`readonly noalias align 64 dereferenceable(N)`); the wrapper passes the global's address.
- `Action::Compare` loads the threshold through `gep base, slot` instead of using `const_float`.
- `Action::Retire` loads through `gep base, (block start + index)` instead of creating `leaves_N`.
- On ARM64, add a third pointer that each transfer moves to the next chunk's first threshold, and address thresholds relative to it.
- Always emit wrapper plus continuation, so the pointer stays opaque to early-cse.

ARM64 before, one split and one leaf:
```
mov x8,#89610197663744 ; movk x8,#16376,lsl #48 ; fmov d0,x8 ; fcmp d2,d0 ; cset w12,gt ; orr x14,x14,x12,lsl #1
adrp x19,l_leaves_67@GOTPAGE ; ldr x19,[x19,l_leaves_67@GOTPAGEOFF] ; ldr d8,[x19,x10] ; fadd d0,d0,d8
```

ARM64 after (`probes/out/dbgb`, `continuation5`):
```
ldp d3,d6,[x22] ; ldp d8,d9,[x22,#16] ; fcmp d4,d3 ; cset w8,gt ; ... orr x11,x11,x24,lsl #3 ; ... add x22,x22,#128 ; b _continuation6
```

x86-64 after: `ucomisd (%r13),%xmm2` and `addsd 39424(%r13,%rcx,8),%xmm0` replace the `movabsq` pairs.

**Saving.** ARM64: about 2.5 per split plus 0.5 per tree. x86-64: 1 per split plus 1 per tree. At 100 trees ARM64 goes 4453 → 2959 (−34%) and x86-64 4015 → 3330 (−17%); at 300 trees 13338 → 8945 and 12033 → 9930. On f32 borders the ARM64 saving grows by about 600 per 100 trees. ARM64 time moved from about 122 to 95 ns (indicative).

**Risk: low.**
- Plain IR with honest read-only annotations.
- Costs one or two general arguments, and the three IR-text unit tests need rewriting.
- On x86-64, keep at most 10 index slots when a base pointer is carried, and do not use the per-chunk pointer under `JITDefault`.
- `TB_PROBE=base` passed all integration tests. The per-chunk pointer passed `scalar_bench` verification but was not run through the test suite.

### R2. Build leaf indices with a carry chain, on top of R1

**What changes:**
- x86-64: `fcmp ult x, T` with `T = next_up(t)` stored in the block, zero-extended to `i8`, then `llvm.x86.addcarry.64(c, idx, idx)`, taking field 1.
- ARM64: inline asm `"fcmp ${1:d}, ${2:d}\nadc $0, $3, $3"` with constraints `"=r,w,w,r,~{cc}"`, no side effects.
- `Admit` seeds the index with `block start >> depth`.
- Leaf blocks are reordered for complemented bits in plan order; `+inf` and NaN borders become constant bits.

Before and after, per split:
```
x86 before: movabsq $.LCPI2_0,%rax ; xorl %ecx,%ecx ; ucomisd (%rax),%xmm3 ; seta %cl ; shll $3,%ecx ; orq %rdi,%rcx
x86 after:  ucomisd 128(%r13),%xmm3 ; adcq %rsi,%rsi
ARM64 R1:   fcmp d4,d3 ; cset w8,gt ; orr x11,x11,x8,lsl #3
ARM64 R2:   fcmp d9,d4 ; adc x8,x8,x8
```

**Saving on top of R1.**
- x86-64: about 2.3 per split. 3330 → 1944 at 100 trees, 9930 → 5758 at 300. Code size 21.5 → 13.2 KB at 100 trees and 64 → 39.6 KB at 300, against a 32 KB L1 instruction cache.
- ARM64: 1 per split. 2959 → 2275 at 100 trees, 8945 → 6780 at 300. Time 95 → 90 ns (indicative).
- The same semantics in plain IR gives no ARM64 saving, so on ARM64 this only pays with inline asm.

**Risk: moderate.**
- The x86 path has never run on x86 hardware. Its lowering is checked in emitted assembly, and its logic is checked by running the generic-IR version.
- Denormals-are-zero breaks the x86 `next_up` trick.
- The ARM64 path is inline asm, which LLVM cannot see into.
- The leaf reordering becomes something tests must cover. The existing differential test already covers NaN modes, mixed depth and a predicate shared across levels, and passes.

### R3. Treat `AsFalse` features as `AsIs`

Skip the `fcmp uno`/`select` for `AsFalse` in `Action::Load`. The result is exact (H5) for today's `fcmp ogt` and for the carry forms. It saves 4 instructions per `AsFalse` feature load on ARM64 and 7 on x86-64, which is zero on these workloads. Risk is very low. The per-load cost comes from the `probes/nan.ll` probe; I did not emit a kernel with this change.

### R4. Per-target chunk limit, after R1 and R2

On ARM64, 32 saves 65 instructions and 64 saves 138, with no stack traffic. On x86-64 keep 16. Re-check on 500x100 before changing.

### R5. Leave MCJIT on `JITDefault`

The small model saves only 37 `movabsq` on top of R1 and R2. It also cuts x86 text to 11120 bytes with the per-chunk pointer and removes the 12-argument compile failure. Doing it safely needs `create_mcjit_execution_engine_with_memory_manager` with an arena memory manager, and a module with zero data references (constants passed from Rust, including bias, scale and NaN replacement constants). Consider it only for an x86 deployment where code bytes matter.

### Not recommended

Backend `Default`, extra `instcombine`/`reassociate`, host-specific target CPUs (0 to 2% effect), and a constant global without a pointer argument.

## 4. Proposed experiments for the orchestrator

Apply `probes/probe.diff` with `git apply` in a scratch worktree and `export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21`. With no switches set, emission is byte-identical to today.

1. **Timing, serially, on 100x100 and 500x100.** Run `cargo run --offline --release --example scalar_bench -- 100x100` five times: with no switch, `TB_PROBE=base`, `TB_PROBE=base TB_REBASE=1`, `TB_PROBE=carry TB_REBASE=1`, and `TB_PROBE=carrygen TB_REBASE=1`. Compare the "Scheduled GP12/FP9, sums 2" median. Expect roughly 121 → 102 → 93 → 88 → 92 ns; these are scaled from my loaded runs.
2. **Tests for each mode.** `TB_PROBE=<mode> [TB_REBASE=1] cargo test --offline --release --tests --no-fail-fast`. Expect the three IR-text unit tests to fail and all integration tests to pass. I ran `base`, `carry` and `carrygen` without `TB_REBASE`.
3. **Load-issue hypothesis.** Run `TB_PROBE=carry TB_REBASE=1` with `TB_CHUNK=32` and `64` (2210 and 2137 instructions) and check whether time follows load count or total count.
4. **Realistic borders.** Round thresholds through f32 in `dyadic` (`examples/support/distinct.rs`) and re-emit. Expect the default ARM64 kernel to grow by about 600 instructions and `TB_PROBE=base` to stay unchanged.
5. **x86 hardware, if available.** Run `TB_PROBE=carry scalar_bench 100x100` there, without `TB_REBASE`.
6. **x86 register ceiling.** Add a GP16 scheduled variant to `inspect_code` and emit for x86-64. Expect `ran out of registers ... continuation20`; with `TB_CM=small` it should compile.

## 5. Questions for the user

1. Is production ARM64 or x86-64 (AVX2 or AVX-512)? R2 splits by target, and on x86-64 code size matters more at 300 trees.
2. Are real borders float32-derived, as in CatBoost? If so, today's ARM64 cost per threshold is 4 instructions and R1 is worth more than measured here.
3. Is inline asm acceptable in the ARM64 path? Without it, 3 instructions per split is the floor for building the index.
4. Can the host process ever run with flush-to-zero or denormals-are-zero set? That decides whether the x86 `next_up` formulation is exact.
5. Is a Rust-owned constants block passed as `predict(row, constants)` acceptable? It is only needed for R5.

## Appendix: tested diff (reverted; also in `probes/probe.diff`)

Switches: `TB_PROBE=base|carry|carrygen`, `TB_REBASE=1`, `TB_CHUNK=<n>|unlimited`, `TB_OPT=default|aggressive`, `TB_CM=small|medium`, `TB_PASSES=<pipeline>`, `TB_CPU_A64`, `TB_CPU_X86`, and `TB_GP` (inspect_code only). The diff also adds the `300x100` workload.

```diff
diff --git a/examples/inspect_code.rs b/examples/inspect_code.rs
index edec357..c8fa8a3 100644
--- a/examples/inspect_code.rs
+++ b/examples/inspect_code.rs
@@ -44,7 +44,7 @@ fn main() -> Result<(), Box<dyn Error>> {
                 "scheduled-12gp-9fp-2sums",
                 SortOptions {
                     order: Order::Scheduled(RegisterBudget {
-                        general: NonZeroUsize::new(12).unwrap(),
+                        general: NonZeroUsize::new(std::env::var("TB_GP").map_or(12, |v| v.parse().unwrap())).unwrap(),
                         floating: NonZeroUsize::new(9).unwrap(),
                     }),
                     accumulators: NonZeroUsize::new(2).unwrap(),
diff --git a/examples/support/mod.rs b/examples/support/mod.rs
index 694a2f0..34b20bc 100644
--- a/examples/support/mod.rs
+++ b/examples/support/mod.rs
@@ -22,6 +22,7 @@ impl Workload {
             "wide" => Ok(Self::Wide),
             "100x100" => Ok(Self::Distinct(100, 100)),
             "500x100" => Ok(Self::Distinct(500, 100)),
+            "300x100" => Ok(Self::Distinct(300, 100)),
             "500x1000" => Ok(Self::Distinct(500, 1000)),
             _ => Err(
                 "workload must be mixed, reuse-heavy, disjoint, wide, 100x100, 500x100, or 500x1000",
diff --git a/src/jit.rs b/src/jit.rs
index 7efa22a..14c6862 100644
--- a/src/jit.rs
+++ b/src/jit.rs
@@ -74,22 +74,33 @@ fn prepare_module<'ctx>(
     triple: &TargetTriple,
 ) -> CompileResult<(Module<'ctx>, TargetMachine)> {
     let target = Target::from_triple(triple)?;
+    PROBE_X86.with(|x| x.set(triple.as_str().to_string_lossy().starts_with("x86_64")));
+    let cpu = probe_cpu();
     let machine = target
         .create_target_machine(
             triple,
-            "generic",
+            &cpu,
             "",
-            BACKEND_OPTIMIZATION,
+            match std::env::var("TB_OPT").as_deref() {
+                Ok("default") => OptimizationLevel::Default,
+                Ok("aggressive") => OptimizationLevel::Aggressive,
+                _ => BACKEND_OPTIMIZATION,
+            },
             RelocMode::Default,
-            CodeModel::JITDefault,
+            match std::env::var("TB_CM").as_deref() {
+                Ok("small") => CodeModel::Small,
+                Ok("medium") => CodeModel::Medium,
+                _ => CodeModel::JITDefault,
+            },
         )
         .ok_or("could not create target machine")?;
+    PROBE_X86.with(|x| x.set(triple.as_str().to_string_lossy().starts_with("x86_64")));
     let module = emit_plan(context, plan)?;
     module.set_triple(triple);
     module.set_data_layout(&machine.get_target_data().get_data_layout());
     module.verify()?;
     module.run_passes(
-        "function(early-cse)",
+        &std::env::var("TB_PASSES").unwrap_or("function(early-cse)".into()),
         &machine,
         PassBuilderOptions::create(),
     )?;
@@ -97,6 +108,13 @@ fn prepare_module<'ctx>(
     Ok((module, machine))
 }
 
+fn probe_cpu() -> String {
+    let key = if PROBE_X86.with(|x| x.get()) { "TB_CPU_X86" } else { "TB_CPU_A64" };
+    std::env::var(key).unwrap_or("generic".into())
+}
+
+thread_local! { static PROBE_X86: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
+
 fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Module<'ctx>> {
     let model = plan.model();
     let module = context.create_module("tigerboost");
@@ -105,7 +123,73 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
     let index_type = context.i64_type();
     let zero = index_type.const_zero();
     let chunks = plan.chunks();
-    let split = chunks.len() > 1;
+    // PROBE: TB_PROBE=base (thresholds+leaves off one base pointer argument) or
+    // TB_PROBE=carry (base + MSB-first complemented-bit carry chain). Scalar plans only.
+    let all_scalar = plan.actions().iter().all(|a| match a {
+        Action::Compare { comparisons, .. } => comparisons.len() == 1,
+        _ => true,
+    });
+    let probe = std::env::var("TB_PROBE").ok().filter(|_| all_scalar);
+    let use_base = probe.is_some();
+    let carry = probe.as_deref().is_some_and(|p| p.starts_with("carry"));
+    // carrygen: x86 semantics (ult against next_up, complemented bits) in generic IR, runnable here.
+    let generic = probe.as_deref() == Some("carrygen");
+    let x86 = PROBE_X86.with(|x| x.get()) || generic;
+    let split = chunks.len() > 1 || use_base;
+    // TB_REBASE: a third pointer argument re-pointed at each chunk's first threshold (ldp range).
+    let rebase = use_base && std::env::var("TB_REBASE").is_ok();
+    let lead = 1 + use_base as usize + rebase as usize;
+    let mut slot_of = vec![usize::MAX; plan.predicates().len()];
+    let mut block: Vec<f64> = Vec::new();
+    let mut bit_order: Vec<Vec<usize>> = vec![Vec::new(); model.trees().len()];
+    for action in plan.actions() {
+        if let Action::Compare { comparisons, .. } = action {
+            for c in comparisons {
+                if slot_of[c.predicate.0] == usize::MAX {
+                    slot_of[c.predicate.0] = block.len();
+                    let t = plan.predicates()[c.predicate.0].threshold;
+                    if carry && x86 {
+                        // +inf and NaN borders never split; the emitter uses a constant bit for them.
+                        block.push(if t.is_nan() { f64::INFINITY } else { t.next_up() });
+                    } else {
+                        block.push(t);
+                    }
+                }
+                for consumer in &c.consumers {
+                    bit_order[consumer.tree.0].push(consumer.level);
+                }
+            }
+        }
+    }
+    let leaf_block = model.trees().iter().map(|t| t.leafs.len()).max().unwrap_or(1);
+    let leaf_region = block.len().div_ceil(leaf_block) * leaf_block;
+    block.resize(leaf_region, 0.0);
+    for (tree, order) in model.trees().iter().zip(&bit_order) {
+        let depth = tree.leafs.len().trailing_zeros() as usize;
+        for p in 0..leaf_block {
+            block.push(if !carry || p >= tree.leafs.len() {
+                tree.leafs.get(p).copied().unwrap_or(0.0)
+            } else {
+                assert_eq!(order.len(), depth, "probe: one consumer per level");
+                let original: usize = order
+                    .iter()
+                    .enumerate()
+                    .map(|(j, &level)| (1 - ((p >> (depth - 1 - j)) & 1)) << level)
+                    .sum();
+                tree.leafs[original]
+            });
+        }
+    }
+    let constants = use_base.then(|| {
+        let values: Vec<_> = block.iter().map(|&v| f64_type.const_float(v)).collect();
+        let global = module.add_global(f64_type.array_type(values.len() as u32), None, "constants");
+        global.set_linkage(Linkage::Private);
+        global.set_constant(true);
+        global.set_alignment(64);
+        global.set_initializer(&f64_type.const_array(&values));
+        global
+    });
+    let block_bytes = (block.len() * size_of::<f64>()) as u64;
     let index_slots = chunks
         .iter()
         .map(|c| c.incoming.indices.len())
@@ -116,10 +200,10 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
         .map(|c| c.incoming.features.len())
         .max()
         .unwrap();
-    let sum_offset = 1 + index_slots;
+    let sum_offset = lead + index_slots;
     let feature_offset = sum_offset + plan.sum_count();
     let row_type = context.ptr_type(AddressSpace::default());
-    let mut parameter_types = vec![row_type.into()];
+    let mut parameter_types = vec![row_type.into(); lead];
     parameter_types.resize(sum_offset, index_type.into());
     parameter_types.resize(feature_offset + feature_slots, f64_type.into());
     let continuation_type = f64_type.fn_type(&parameter_types, false);
@@ -159,15 +243,42 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
         }
         // LLVM 21: Ref=1 in each of four two-bit memory locations, including globals.
         attribute(AttributeLoc::Function, "memory", 0b01_01_01_01);
-        for (name, value) in [("target-cpu", "generic"), ("target-features", "")] {
+        if use_base && function != wrapper {
+            attribute(AttributeLoc::Param(1), "readonly", 0);
+            attribute(AttributeLoc::Param(1), "noalias", 0);
+            attribute(AttributeLoc::Param(1), "align", 64);
+            attribute(AttributeLoc::Param(1), "dereferenceable", block_bytes);
+            if rebase {
+                attribute(AttributeLoc::Param(2), "readonly", 0);
+                attribute(AttributeLoc::Param(2), "noalias", 0);
+                attribute(AttributeLoc::Param(2), "align", 8);
+            }
+        }
+        let cpu = probe_cpu();
+        for (name, value) in [("target-cpu", cpu.as_str()), ("target-features", "")] {
             function.add_attribute(
                 AttributeLoc::Function,
                 context.create_string_attribute(name, value),
             );
         }
     }
-    let arguments = |row: inkwell::values::PointerValue<'ctx>| {
-        let mut args = vec![row.into()];
+    let chunk_first: Vec<usize> = chunks
+        .iter()
+        .map(|c| {
+            plan.actions()[c.actions.clone()]
+                .iter()
+                .filter_map(|a| match a {
+                    Action::Compare { comparisons, .. } => Some(slot_of[comparisons[0].predicate.0]),
+                    _ => None,
+                })
+                .min()
+                .unwrap_or(0)
+        })
+        .collect();
+    let arguments = |row: inkwell::values::PointerValue<'ctx>,
+                     extra: &[inkwell::values::PointerValue<'ctx>]| {
+        let mut args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> = vec![row.into()];
+        args.extend(extra.iter().map(|&b| inkwell::values::BasicMetadataValueEnum::from(b)));
         args.resize(sum_offset, index_type.get_undef().into());
         args.resize(feature_offset + feature_slots, f64_type.get_undef().into());
         args
@@ -177,7 +288,24 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
         // The first chunk has no carried values and initializes its sums to zero.
         let call = builder.build_call(
             functions[0],
-            &arguments(wrapper.get_first_param().unwrap().into_pointer_value()),
+            &arguments(
+                wrapper.get_first_param().unwrap().into_pointer_value(),
+                &{
+                    let mut extra: Vec<_> = constants.map(|g| g.as_pointer_value()).into_iter().collect();
+                    if rebase {
+                        // SAFETY: inside the constants block.
+                        extra.push(unsafe {
+                            builder.build_in_bounds_gep(
+                                f64_type,
+                                extra[0],
+                                &[index_type.const_int(chunk_first[0] as u64, false)],
+                                "tbase",
+                            )?
+                        });
+                    }
+                    extra
+                },
+            ),
             "prediction",
         )?;
         call.set_call_convention(PRESERVE_NONE_CALL_CONVENTION);
@@ -189,13 +317,16 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
         let function = functions[chunk_number];
         builder.position_at_end(context.append_basic_block(function, "entry"));
         let row = function.get_first_param().unwrap().into_pointer_value();
+        let base = use_base.then(|| function.get_nth_param(1).unwrap().into_pointer_value());
+        let tbase = if rebase { Some(function.get_nth_param(2).unwrap().into_pointer_value()) } else { base };
+        let slot_bias = if rebase { chunk_first[chunk_number] } else { 0 };
         indices.fill(None);
         loaded_features.fill(None);
         for (slot, tree) in chunk.incoming.indices.iter().enumerate() {
             if let Some(tree) = tree {
                 indices[tree.0] = Some(
                     function
-                        .get_nth_param((1 + slot) as u32)
+                        .get_nth_param((lead + slot) as u32)
                         .unwrap()
                         .into_int_value(),
                 );
@@ -225,7 +356,14 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
             .collect();
         for action in &plan.actions()[chunk.actions.clone()] {
             match action {
-                Action::Admit(tree) => indices[tree.0] = Some(zero),
+                Action::Admit(tree) => {
+                    indices[tree.0] = Some(if carry {
+                        let depth = model.trees()[tree.0].leafs.len().trailing_zeros();
+                        index_type.const_int(((leaf_region + tree.0 * leaf_block) >> depth) as u64, false)
+                    } else {
+                        zero
+                    })
+                }
                 Action::Load { value: id, feature } => {
                     // SAFETY: Trees validates the feature index and row byte span;
                     // CompiledModel::predict checks that the caller supplies that span.
@@ -274,11 +412,76 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
                     let threshold = |comparison: &Comparison| {
                         f64_type.const_float(plan.predicates()[comparison.predicate.0].threshold)
                     };
+                    let load_threshold = |comparison: &Comparison| -> CompileResult<FloatValue<'ctx>> {
+                        let slot = slot_of[comparison.predicate.0];
+                        // SAFETY: slot < block length, which is the dereferenceable span.
+                        let address = unsafe {
+                            builder.build_in_bounds_gep(
+                                f64_type,
+                                tbase.unwrap(),
+                                &[index_type.const_int((slot - slot_bias) as u64, false)],
+                                "threshold_ptr",
+                            )?
+                        };
+                        let loaded = builder.build_load(f64_type, address, "threshold")?;
+                        loaded.as_instruction_value().unwrap().set_alignment(8)?;
+                        Ok(loaded.into_float_value())
+                    };
+                    if carry {
+                        let [single] = comparisons.as_slice() else { unreachable!() };
+                        let t = load_threshold(single)?;
+                        for consumer in &single.consumers {
+                            let index = indices[consumer.tree.0].expect("active tree");
+                            let never = plan.predicates()[single.predicate.0].threshold;
+                            let never = !(never < f64::INFINITY);
+                            let next = if generic {
+                                let c = if never {
+                                    context.bool_type().const_int(1, false)
+                                } else {
+                                    builder.build_float_compare(FloatPredicate::ULT, feature, t, "not_split")?
+                                };
+                                let bit = builder.build_int_z_extend(c, index_type, "bit")?;
+                                let doubled = builder.build_left_shift(index, index_type.const_int(1, false), "doubled")?;
+                                builder.build_or(doubled, bit, "leaf_index")?
+                            } else if x86 {
+                                let c = if never {
+                                    context.bool_type().const_int(1, false)
+                                } else {
+                                    builder.build_float_compare(FloatPredicate::ULT, feature, t, "not_split")?
+                                };
+                                let k = builder.build_int_z_extend(c, context.i8_type(), "carry")?;
+                                let intrinsic = inkwell::intrinsics::Intrinsic::find("llvm.x86.addcarry.64").unwrap();
+                                let decl = intrinsic.get_declaration(&module, &[]).unwrap();
+                                let r = builder.build_call(decl, &[k.into(), index.into(), index.into()], "adc")?;
+                                builder
+                                    .build_extract_value(r.try_as_basic_value().unwrap_basic().into_struct_value(), 1, "leaf_index")?
+                                    .into_int_value()
+                            } else {
+                                let ty = index_type.fn_type(&[f64_type.into(), f64_type.into(), index_type.into()], false);
+                                let asm = context.create_inline_asm(
+                                    ty,
+                                    "fcmp ${1:d}, ${2:d}\nadc $0, $3, $3".to_string(),
+                                    "=r,w,w,r,~{cc}".to_string(),
+                                    false,
+                                    false,
+                                    None,
+                                    false,
+                                );
+                                builder
+                                    .build_indirect_call(ty, asm, &[t.into(), feature.into(), index.into()], "leaf_index")?
+                                    .try_as_basic_value()
+                                    .unwrap_basic()
+                                    .into_int_value()
+                            };
+                            indices[consumer.tree.0] = Some(next);
+                        }
+                        continue;
+                    }
                     let bits = match comparisons.as_slice() {
                         [single] => vec![builder.build_float_compare(
                             FloatPredicate::OGT,
                             feature,
-                            threshold(single),
+                            if use_base { load_threshold(single)? } else { threshold(single) },
                             "split",
                         )?],
                         [first, second] => {
@@ -358,6 +561,21 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
                         .take()
                         .expect("retirement requires an active tree");
                     let tree = &model.trees()[tree_id.0];
+                    if let Some(base) = base {
+                        let index = if carry {
+                            index
+                        } else {
+                            let start = leaf_region + tree_id.0 * leaf_block;
+                            builder.build_int_add(index, index_type.const_int(start as u64, false), "leaf_slot")?
+                        };
+                        // SAFETY: index addresses this tree's leaf block inside the constants span.
+                        let address = unsafe { builder.build_in_bounds_gep(f64_type, base, &[index], "leaf_ptr")? };
+                        let leaf = builder.build_load(f64_type, address, "leaf")?;
+                        leaf.as_instruction_value().unwrap().set_alignment(8)?;
+                        let accumulator = &mut sums[sum.0];
+                        *accumulator = reassociate(builder.build_float_add(*accumulator, leaf.into_float_value(), "sum")?)?;
+                        continue;
+                    }
                     let leaf_count = u32::try_from(tree.leafs.len())
                         .map_err(|_| "leaf table exceeds LLVM array length limit")?;
                     let leaf_type = f64_type.array_type(leaf_count);
@@ -419,10 +637,22 @@ fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Mod
             }
         }
         if let Some(next) = chunks.get(chunk_number + 1) {
-            let mut args = arguments(row);
+            let mut extra: Vec<_> = base.into_iter().collect();
+            if rebase {
+                // SAFETY: inside the constants block.
+                extra.push(unsafe {
+                    builder.build_in_bounds_gep(
+                        f64_type,
+                        tbase.unwrap(),
+                        &[index_type.const_int((chunk_first[chunk_number + 1] - slot_bias) as u64, false)],
+                        "next_tbase",
+                    )?
+                });
+            }
+            let mut args = arguments(row, &extra);
             for (slot, tree) in next.incoming.indices.iter().enumerate() {
                 if let Some(tree) = tree {
-                    args[1 + slot] = indices[tree.0].expect("carried index must be live").into();
+                    args[lead + slot] = indices[tree.0].expect("carried index must be live").into();
                 }
             }
             for (sum, value) in sums.iter().enumerate() {
diff --git a/src/sort_policy.rs b/src/sort_policy.rs
index fcc41e0..2c6bd50 100644
--- a/src/sort_policy.rs
+++ b/src/sort_policy.rs
@@ -169,7 +169,8 @@ pub fn plan(model: &Trees, options: SortOptions) -> Result<Plan<'_>, Insufficien
     } else {
         Action::ReturnScaled(SumId(0))
     });
-    plan.chunks = continuation::partition(&plan, 16);
+    let limit = std::env::var("TB_CHUNK").map_or(16, |v| v.parse().unwrap_or(usize::MAX));
+    plan.chunks = continuation::partition(&plan, limit);
     Ok(plan)
 }
```

Other observations:
- In default mode, the README doctest fails to link: `/usr/local/lib/libzstd.dylib` is x86_64 while the build is arm64. This is an environment problem, not caused by any probe; all other default-mode tests pass.
- `CodeModel::Medium` aborts on AArch64, which does not support it.