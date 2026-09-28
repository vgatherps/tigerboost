# Study A: backend instruction-cost audit of the emitted kernel

Read `STUDY_CONTEXT.md` first. Then this.

## Question in one sentence

Where do the 4453 (ARM64) and 4015 (x86-64) instructions of the 100x100 scalar kernel go,
which of them are avoidable through a different LLVM IR shape or backend setting, and what
exact emission changes get rid of them?

## Read list

Primary anchors (read these; reading anything else in the repo to understand code is fine):

- `src/jit.rs` whole file. `emit_plan` at line 100. The `Action::Compare` arm is lines
  268-350, `Action::Retire` (leaf table global, gep, load, fadd) lines 356-392, the
  continuation transfer (argument marshalling and `musttail`) lines 421-442, target machine
  and pass setup in `prepare_module` lines 71-98.
- `captures/100x100/aarch64-apple-darwin-scheduled-12gp-9fp-2sums.s` and
  `captures/100x100/x86_64-unknown-linux-gnu-scheduled-12gp-9fp-2sums.s` with their
  `.plan.txt` dumps. The SIMD variants are there for reference; piece C owns them.
- `plans/implementation.md` lines 84-94 (earlier spill findings and the `Default`/`None`
  backend experiments) and 161-179 (continuation results).

## Questions

A1. Classify every instruction of both scalar captures into categories and reconcile the
category totals to the file totals: threshold constant materialization; feature load (from
the row); NaN normalization (none in this workload; note where it would appear); comparison;
bit insert into the selector (`cset`/`orr`/`ubfiz`, `xorl`/`seta`/`shll`/`orq`); leaf address
materialization (`adrp`/`add`, `movabsq`); leaf load; accumulation; continuation transfer
(argument moves, `b`/`jmpq`, any `ldp`/`stp`/`push`/`pop`); wrapper prologue/epilogue;
other. Give per-split and per-tree costs derived from the table.

A2. Threshold constants (H1). Explain from the emitted code and from LLVM's lowering why
ARM64 gets `mov`/`movk`/`fmov` per threshold and x86-64 gets `movabsq` + memory-operand
`ucomisd`. Then evaluate these ways to make each threshold cost one load (ARM64) or zero
extra instructions (x86-64, folded memory operand):

  a. `CodeModel::Small` (or `Default`) instead of `JITDefault` for the MCJIT engine and
     for `emit_assembly`. State what MCJIT requires for the small model on x86-64 to be
     safe (section placement within 2 GB, relocation kinds) and whether inkwell 0.9 /
     LLVM 21 MCJIT honors it. If you can test it (change `prepare_module`, run the test
     suite and `scalar_bench 100x100`), do.
  b. A private constant global holding every threshold, loaded with a constant index
     (`gep` + `load`). Will `early-cse` or the backend fold that load back into an
     immediate? Probe it (emit and look). If it folds, what stops the folding without
     losing the read-only annotations that the project keeps on purpose (see
     `PLAN.md`, "Observed compiler limitation": no fake side effects, no volatile).
  c. A base pointer argument: the wrapper passes the address of a constants block
     (thresholds, and possibly leaf tables) as a second parameter carried through every
     continuation, so every threshold is `ldr d, [xbase, #imm]` on ARM64 (12-bit
     scaled immediate, so up to 4095 f64 at 8-byte stride before a second base is
     needed) and `ucomisd imm(%rbase), %xmm` on x86-64. Count the register cost and the
     instruction savings. Say how the block would be owned (the `CompiledModel` owns the
     leaf globals today through the engine; a Rust-side `Box<[f64]>` passed by pointer
     is a different ownership story; a module-private global whose address is taken once
     in the wrapper is another).
  Rank a, b, c by instructions saved and by risk, for each target.

A3. Bit insertion. Today each split on ARM64 is `fcmp` + `cset` + `orr x, x, x, lsl #k` (or
`ubfiz` for the first bit) and on x86-64 `xorl` + `ucomisd` + `seta` + `shll` + `orq`.
Find cheaper sequences that LLVM will emit from IR, and check them by probe:

  - x86-64: after NaN normalization (H5, verify it yourself for the cases that matter here)
    the comparison can be operand-reversed so the carry flag holds the result
    (`ucomisd t, x` sets CF when t < x), letting `adc`/`rcl`/`setb` build the index with
    fewer instructions. Does LLVM produce any of these from `icmp`/`select`/`add` IR
    patterns, or only from inline asm?
  - ARM64: `fcmp` + `cinc`/`csinc`/`csel` forms, or `fcmp` + `cset` with the shift folded
    into `orr` (already done). Is three per split the floor without SIMD? Is
    `fccmp` useful for anything here?
  - Would building the index with `add` instead of `or` (the bits never overlap) change
    the lowering?
  Report the verified emitted asm for each candidate.

A4. Leaf lookup. Today: one private global per tree, `adrp` + `add` (ARM64) or `movabsq`
(x86-64) for the address, then a scaled-index load. Evaluate one contiguous leaf block for
the whole model addressed off one base register, with the tree's block offset folded into
the selector's initial value (H6: the tree's 64-entry block starts at a multiple of 64, so
the offset can be OR-ed or used as the initial index instead of zero). Target: one
`ldr d, [xbase, xidx, lsl #3]` / one `movsd (%rbase,%ridx,8), %xmm` per tree. Count the
saving and any interaction with A2c (same base pointer for thresholds and leaves?).

A5. Continuation transfer. Count what each of the 38 chunk boundaries costs on each target
(argument shuffling moves, the branch, any stack traffic) and what the C wrapper costs
(prologue/epilogue, the `bl`/`callq`). With thresholds and leaves off a base register (fewer
live values), is a larger chunk limit (32, 64, unlimited) safe from the spills that motivated
chunking? Emit at limits 16/32/64 and unlimited by changing the literal at
`src/sort_policy.rs:172`; report instruction totals and stack-access counts (grep for
`sp`-relative accesses inside continuation bodies) per limit. Piece B asks the same from the
planner side; you answer from the emitted code side. Do not spend more than a quarter of your
effort here.

A6. Backend settings. With constants coming from a table (after A2), re-check whether backend
`OptimizationLevel::Default` or additional IR passes (`instcombine`, `reassociate`) change the
emitted code. `plans/implementation.md:92-94` records `Default` as byte-identical and `None`
as regressing under the old single-function kernel. Only report a difference if you emitted
it.

## Deliverable

Report at `probes/REPORT_A.md` (or the path your launch instructions name), per the report
contract in `STUDY_CONTEXT.md`. Section 3 must contain, for each recommended emission change:
the IR change in `src/jit.rs` terms, the emitted instruction sequence before and after (asm
snippets from probes), instructions saved per split or per tree, projected kernel totals at
100 and 300 trees for ARM64 and x86-64, and the diff you tested (reverted from tracked files
before finishing).
