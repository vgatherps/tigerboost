# tigerboost performance study: shared context

Read this first. It is shared by four study briefs (A backend cost, B scheduler/residency,
C SIMD formulation, D latency chain and measurement). Your piece brief names which one you do.

## What the project is

`tigerboost` is a model-specialized LLVM JIT for single-row prediction over CatBoost-style
oblivious (symmetric) trees. Rust crate, LLVM 21 through inkwell 0.9 / MCJIT. Read
`README.md` and `PLAN.md` at the repo root for the API and the design as built; read
`plans/implementation.md` for the measurement history. Source is small:

- `src/parsed_tree.rs` (217 lines): validated model `Trees`, reference evaluator `Trees::predict`,
  leaf index rule: at tree level k, strict `feature > threshold` sets bit k of the leaf index.
- `src/sort_policy.rs` (572 lines): compile-time planner. Produces a flat list of `Action`s
  (Admit / Load / Compare / Release / Retire / AddSums / Return) from either tree order or a
  "scheduled" order that tries to share feature loads across trees under logical register
  budgets (`RegisterBudget { general, floating }`, default GP12/FP9, two accumulator sums).
- `src/sort_policy/continuation.rs` (95 lines): partitions the action list into chunks of at
  most 16 comparison tests (limit is the literal `16` at `src/sort_policy.rs:172`).
- `src/jit.rs` (704 lines): emits one LLVM function per chunk, `noinline preserve_nonecc`,
  chained by `musttail` calls carrying live selectors, features and sums as arguments; a C
  `predict(const double*) -> double` wrapper calls the first chunk. Backend: target-cpu
  "generic", empty target-features, `early-cse` only at IR level, backend opt level `Less`,
  code model `JITDefault`.
- `examples/scalar_bench.rs`, `examples/inspect_code.rs`, `examples/support/{mod,distinct}.rs`:
  timing harness and assembly emission for synthetic workloads. `100x100` = 100 depth-6 trees
  over 100 declared features with random feature reuse and globally distinct thresholds.

## The goal this study serves

Single-row prediction latency for SMALL forests: fewer than 300 trees, depth about 6 (assume
4 to 8), one feature vector per call (no batching), features arrive as a flat `f64` row.
Compile time is paid once per model and is secondary. Deployment architecture is NOT pinned:
development and all measurements so far are on an Apple M4 Pro (ARM64 NEON); production is
plausibly x86-64 (AVX2 baseline, AVX-512 possible). Treat both as targets and give per-target
answers. Where a recommendation differs by target, say so.

## Current measured state (Apple M4 Pro, LLVM 21.1.8, generic target)

From `plans/implementation.md` lines 161-194 (median ns per call, includes about 5 ns of
benchmark driver): 100 trees / 100 features scalar default 121 ns; 500 / 100 scalar 572 ns;
SIMD2 opt-in is slower than scalar in every case.

Rough cycle math the orchestrator did (treat as an estimate, redo it if your piece needs it):
121 ns minus 5 ns driver at roughly 4.4 GHz is about 510 cycles for the 4453-instruction ARM64
scalar kernel, so about 8.7 instructions per cycle. That is close to the core's width. The
kernel is throughput bound on instruction count today. Instruction count (and on x86, code
bytes) is the metric that matters, not logical register budgets.

## Shared artifacts

Emitted assembly and plan dumps for the `100x100` workload are in the directory named in your
piece brief (`captures/100x100/`). Files: `<triple>-<variant>.s` and `.plan.txt` for
`aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`, variants `tree-order-1`,
`scheduled-12gp-9fp-2sums` (the production default), `simd2-scheduled-6gp-9fp-2sums`,
`simd2-scheduled-12gp-9fp-2sums`. These are separate target emissions from the same LLVM
setup as MCJIT (not disassembled MCJIT bytes); `plans/implementation.md:177` records that
actual MCJIT captures matched in shape. Regenerate with
`cargo run --offline --release --example inspect_code -- <fresh-dir> 100x100`.

Mnemonic histogram the orchestrator counted with `grep -E '^\s+[a-z]' file.s | grep -v '^\s+\.'
| awk '{print $1}' | sort | uniq -c` (recount yourself; these are counts, not anchors):

```
aarch64 scheduled-12gp-9fp-2sums, 4453 total:
 600 fmov  600 fcmp  597 mov  592 cset  591 movk  536 ldr  492 orr  148 ubfiz
 102 fadd  100 adrp   37 b     34 ldp    10 stp     8 cinc   3 movi   2 ret   1 bl
aarch64 simd2-scheduled-12gp-9fp-2sums, 4051 total:
 798 ldr  460 orr  435 and  366 fmov  334 adrp  234 mov.d  234 fcmgt.2d  168 ld1r.2d
 168 add  132 mov  132 fcmp  131 movk  125 cset  102 fadd   64 ubfiz  59 dup.2d  38 b ...
x86_64 scheduled-12gp-9fp-2sums, 4015 total:
 737 movabsq  600 xorl  600 ucomisd  600 seta  395 orq  384 movsd  340 shll  102 addsd
  70 addl  48 leal  42 leaq  37 jmpq  18 pushq  18 popq  15 orl ...
```

Plan statistics (from the `.plan.txt` dumps, `grep -c ': Load'`): scheduled default 394 feature
loads, tree order 600, SIMD2 default 307; 600 comparison tests; 38 chunks.

## Orchestrator hypotheses (H1 to H7)

These are hypotheses from a first read, not facts. Confirm or refute each one that touches
your piece, with evidence. When one is wrong, record the correction and continue; a wrong
hypothesis is never a reason to stop.

- H1 Threshold constants dominate. On ARM64 each split's threshold is materialized as
  `mov` + `movk` + `fmov d, x` (three instructions, about 1800 of 4453). On x86-64 every
  constant-pool reference is `movabsq $.LCPI, %reg` followed by `ucomisd (%reg), %xmm` because
  `CodeModel::JITDefault` selects the large code model; a small code model or a base-register
  addressing scheme would fold the operand (`ucomisd .LCPI(%rip), %xmm`, zero extra
  instructions) and `ldr d, [xbase, #imm]` (one load) on ARM64.
- H2 Feature residency is a small lever. The scheduler saves 206 loads versus tree order
  (394 vs 600) on 100x100, about 5% of instructions, and each avoided load is one cheap
  L1-hit `ldr` on a hot 800-byte row. The logical budgets (GP12/FP9) are far below what
  `preserve_nonecc` makes physically available, but even unlimited residency caps the saving
  at 500 instructions here. `plans/implementation.md:175` measured scheduling worth only
  2.8 to 4.4% over tree order under the same backend.
- H3 The current SIMD2 formulation (one feature splatted against two thresholds) cannot win
  because 99.5% of pairs target two different trees (`plans/implementation.md:183`), so both
  lanes are extracted (`mov.d`), masked (`and`) and OR-ed into separate selectors. The
  vector axis is wrong: it is threshold-major instead of tree-major.
- H4 The accumulator chain is hidden today but will not stay hidden. Two round-robin sums
  give a dependent `fadd` chain of trees/2 (50 adds for 100 trees, 150 for 300). At about
  3 cycles per `fadd` that is 150 to 450 cycles, under the current throughput bound but
  comparable to it once instruction count drops two to three times.
- H5 NaN handling can be normalized at load time. Under strict `x > t`: NaN compares false
  against every t, and negative infinity also compares false against every t including
  negative infinity, so `NanMode::AsIs` is observationally identical to `AsFalse`. A NaN
  threshold is identical to a positive-infinity threshold (nothing exceeds either). Therefore
  every feature can be made NaN-free once at load and every later comparison may assume
  ordered operands. This unlocks carry-flag and mask tricks and integer binarization.
  Check this against every special value: NaN, +-inf, -0.0/+0.0, NaN thresholds, and the
  `AsTrue` mode (+inf is not greater than +inf).
- H6 Leaf-index bit assignment is free. Leaf tables are copied into private constant
  globals at compile time, so any permutation of the table is allowed, which means any
  mapping from comparison lanes to index bits is allowed. The model's "bit k for level k" rule
  is a reference-evaluator convention, not a constraint on emitted code.
- H7 Binarize-then-compare may beat direct comparison. For feature f with B_f distinct
  sorted borders, bin_f = count of borders b with x > b (SIMD compares, no per-lane extract).
  Then `x > border_r` is exactly `bin_f > r`, an integer byte compare, and a tree's six
  splits become one 8-lane byte compare whose mask IS the leaf index (AVX-512 `k` mask /
  `vmovmskps`-style, or NEON `and` + `addv`). Cost is a function of the sum of B_f versus the
  number of splits. On the synthetic `distinct` workloads every threshold is unique, so the
  sum of B_f equals the split count (worst case). Real CatBoost models share borders
  heavily (at most 254 per feature). The real model's border statistics are unknown.

## Known and deliberate (do not report as findings)

- Features, thresholds, leaves and sums are `f64`. No f32 path. (If f32 borders would change
  a recommendation, list it as a question for the user, not a defect.)
- LLVM 21 via inkwell 0.9 and MCJIT is the backend. Cranelift was tried and dropped. No
  hand-written machine code; inline asm or LLVM target intrinsics from IR are fair to propose.
- No runtime scheduler, interpreter, or work queue. Everything model-dependent is decided at
  compile time.
- The continuation/musttail design exists because LLVM's global load merging and machine
  scheduling produced 3 KB of stack traffic in a single-function kernel
  (`plans/implementation.md:179`). You may critique it, but engage with that cause.
- Target-cpu "generic" is the current setting, not a requirement. Whether to specialize to the
  host (or to x86-64-v3/v4) is a question for pieces A and C.
- JSON/CBM model loading, categorical features, probability transforms are out of scope.
- x86-64 code is emitted and assembled here but never executed; there is no x86 timing.
- The synthetic workloads are synthetic. No representative user model exists yet.

## Environment and commands

Build needs the LLVM prefix exported; the crate builds in a few seconds with a warm cache:

```
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo build --offline --release --examples
cargo test --offline --release            # if you change src/jit.rs or the planner
cargo run --offline --release --example inspect_code -- <fresh-dir> 100x100
cargo run --offline --release --example scalar_bench -- 100x100
```

LLVM tools for probes: `/opt/homebrew/opt/llvm@21/bin/{llc,llvm-mc,llvm-objdump,opt}`. Hand
written `.ll` probes compiled with `llc -O1 -mtriple=aarch64-apple-darwin -mcpu=generic` or
`-mtriple=x86_64-unknown-linux-gnu -mattr=+avx2` (or `+avx512f,+avx512bw,+avx512vbmi`) are the
cheapest way to check what instruction sequence a given IR shape produces.

Timing caveat: several study agents run on this machine at the same time. Any `scalar_bench`
timing you take is indicative only; report min and median and say it was taken under load.
Static instruction counts, code bytes and stack-access counts from emitted assembly are the
trustworthy evidence. Where a timing experiment would settle a question, write down the exact
command and the expected outcome; the orchestrator runs those serially afterwards.

## Edit policy

You work in your own git worktree. Tracked-file edits are allowed only as throwaway probes to
answer a question (for example, changing an emission pattern in `src/jit.rs` and re-emitting
assembly). Before finishing, revert every tracked-file change (`git checkout -- <path>`) and
put the diff you tested, verbatim, into your report so it can be replayed. Put probe files
under `probes/` at the worktree root. Do not run `cargo fmt` or `rustfmt`. Do not commit.
Do not edit `README.md`, `PLAN.md` or anything under `plans/`.

## Report contract

Write the report to the path your piece brief names. Structure:

1. Per question: the answer, then the evidence (file:line, capture line numbers, counts,
   probe asm snippets). If you could not answer, say what blocked you.
2. Hypotheses touched by your piece: confirmed / refuted / partly, with the correction.
3. Ranked recommendations. Each one states: what changes (in terms of emitted instructions
   and planner/emitter code), estimated instruction delta per split or per tree, projected
   effect at 100 and 300 trees, per target, and the risk. Small verified wins rank above
   large unverified ones.
4. Proposed experiments for the orchestrator: exact commands, what to compare, expected result.
5. Questions for the user that would change a recommendation (deployment target, real model
   border statistics, depth range, and so on).

Prose rules: write like a person. Plain sentences. No AI vocabulary (comprehensive, seamless,
robust, leverage, pivotal, crucial, delve, showcase, underscore, landscape, testament), no
filler, no bold-label bullet lists, no emojis, no em-dash chains. Say what a thing does or say
nothing. Explain in observable terms with real identifiers (`Action::Compare`,
`preserve_nonecc`, `fcmgt.2d`), never invented shorthand; a sentence that only parses if the
reader has this brief open is a sentence to rewrite. Tables for numbers.
