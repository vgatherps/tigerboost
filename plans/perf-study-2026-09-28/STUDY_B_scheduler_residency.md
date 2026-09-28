# Study B: does the scheduler earn its complexity? Residency, budgets, chunking

Read `STUDY_CONTEXT.md` first. Then this.

## Question in one sentence

Given what the machine actually offers under `preserve_nonecc` and what a feature reload
actually costs, how much can feature residency ever save, is the planner's budget model
(GP12/FP9) leaving most of that on the table, and is the 16-test chunking the right size?

The maintainer's own complaint that started this: "it claimed shared residency was tiny vs
what it could be". Take that seriously and quantify it both ways: what residency could be at
realistic budgets, and what it would be worth in instructions.

## Read list

Primary anchors (read these; reading anything else in the repo to understand code is fine):

- `src/sort_policy.rs` whole file. Budget check in `plan` lines 143-174; `Planner` state
  lines 225-238; `operation` (per-feature operation formation under the GP/FP counts) lines
  329-392; `select` (feature choice ranking) lines 394-454; `run` (admission, eviction loop
  at 502-514, load/reuse at 515-527, release-at-zero-uses 549-555, retirement 556-559).
- `src/sort_policy/continuation.rs` whole file (95 lines).
- `src/sort_policy/tests.rs` only as needed to see how plans are inspected in tests.
- `README.md` "Scheduling" section; `plans/rolling-scheduler.md`.
- `plans/implementation.md` lines 143-159 (resource-aware planning results) and 161-179
  (continuations; note line 175: scheduling is worth 2.8 to 4.4% over tree order under the
  same backend).
- `captures/100x100/aarch64-apple-darwin-scheduled-12gp-9fp-2sums.plan.txt` and
  `...-tree-order-1.plan.txt` (plans are target independent) plus the matching `.s` files
  for spill checks.
- `src/jit.rs` lines 100-225 only, to see how chunk `incoming` slots become function
  parameters (the physical consequence of the plan's live sets).

## Questions

B1. Physical registers. Under `preserve_nonecc` (LLVM 21) on AArch64 and on x86-64, which
general and vector registers are argument registers and which are caller-saved inside the
continuation bodies? Cite LLVM 21 source (AArch64CallingConvention.td / X86CallingConv.td
in the installed or upstream tree; the installed headers are under
`/opt/homebrew/opt/llvm@21`) or the LangRef. Translate that into the largest GP/FP budget
that could be register-resident with no spills on each target. Compare with GP12/FP9.

B2. Residency curve. The scheduled plan issues 394 loads for 100 features (tree order:
600; lower bound: 100 if every feature stayed resident). Produce the load count, chunk
count, and maximum simultaneous resident features for budgets FP = 9, 12, 16, 24, 32 crossed
with GP = 12, 20, 28 (write a small probe under `probes/`, for example a test or an example
that calls `plan` and counts `Action::Load`; do not edit tracked tests). For at least three
points on that curve, emit assembly for both targets and count stack accesses inside the
continuation bodies. Is there a budget where loads approach 100 with no spills?

B3. What is residency worth? Each avoided load is one `ldr`/`movsd` from an L1-hot row.
Express the whole possible saving (600 minus 100 loads) as a fraction of the kernel at 100 and
300 trees, and compare it to the threshold constant cost that piece A is chasing (about
three instructions per split on ARM64 today). Then answer plainly: is any scheduling for
residency worth its planner complexity, or would tree order plus a cheap backend be within
noise? Use `plans/implementation.md:175` as measured evidence and say whether the
per-target answer differs (x86-64 has 16 xmm registers without AVX-512, so residency is
tighter there).

B4. Heuristic quality. Read `select`, `operation` and the eviction loop. Do they achieve the
reuse the budget allows, or do they thrash? Concretely: at FP 9 with two sums, how many
features can be resident at once by the planner's own arithmetic (`sum_count + resident +
!loaded + comparisons <= floating`), and how often is a feature evicted then reloaded
(count reload events in the plan: a `Load` of a feature index that was loaded earlier)?
Does `waiting_order` (overlap-first admission) actually cluster trees by shared features,
measured on 100x100 as the average number of trees that consume each loaded feature value
before it is released? Report the distribution.

B5. Chunking. 16 tests per chunk gives 38 chunks. From the plan side: how many live
selectors and features cross each boundary (the `incoming` slot counts), and how does that
scale with the limit at 16/32/64/unlimited (change the literal at `src/sort_policy.rs:172`
as a probe, revert after)? Piece A counts the emitted transfer cost; you provide the
crossing-state counts and say what limit the plan structure suggests. Also: is the chunk
boundary placed well (after trailing releases and retirements, per `continuation.rs:60-63`),
or would cutting at retirement points only (so no selector ever crosses) be cleaner?

B6. Planner cost. Time `plan` alone (not LLVM) at 100 and 300 trees, 100 features, default
budget. The scans in `waiting_order`, `uses`, `remaining_uses` are quadratic-ish. Report the
numbers; recommend nothing unless it is over roughly 100 ms, since compilation is once per
model.

## Deliverable

Report at `probes/REPORT_B.md` (or the path your launch instructions name), per the report
contract in `STUDY_CONTEXT.md`. Section 3 must give one of: keep the scheduler with new
budgets (state them per target, with the B2 evidence); simplify it (say which parts go);
or replace it with tree order plus a specific cheaper trick (say which). State the projected
instruction delta at 100 and 300 trees for the option you rank first. Include the residency
curve table from B2 and the reuse distribution from B4.
