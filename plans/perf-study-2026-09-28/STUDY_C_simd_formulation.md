# Study C: which SIMD formulation fits oblivious trees on NEON, AVX2 and AVX-512

Read `STUDY_CONTEXT.md` first. Then this.

## Question in one sentence

The current two-lane "one feature against two thresholds" packing loses to scalar; which
vector formulation of a depth-6 oblivious tree actually cuts instructions per tree on each
target, by how much, and can LLVM IR (via inkwell) be made to emit it?

## Read list

Primary anchors (read these; reading anything else in the repo to understand code is fine):

- `src/jit.rs` lines 268-350 (the `Action::Compare` arm: scalar path and the two-lane path
  with `sext` to full masks and per-lane `extractelement`), lines 356-392 (leaf lookup).
- `src/sort_policy.rs` lines 329-392 (`operation`: how pairs are formed, width 2 only when
  two distinct predicates on the same feature fit) and 394-454 (`select`).
- `plans/portable-simd.md`; `plans/implementation.md` lines 96-102 (first SIMD results),
  119-123 (why SIMD lost on the reuse-heavy fixture) and 181-194 (full-width masks;
  line 183: 1,333 of 1,340 pairs target different trees).
- `captures/100x100/*-simd2-scheduled-12gp-9fp-2sums.s` and `.plan.txt` for both targets,
  and the scalar `...-scheduled-12gp-9fp-2sums.s` files as the baseline.
- `src/parsed_tree.rs` lines 170-216 (reference semantics you must preserve exactly:
  strict greater-than, NaN modes, bit k for level k in the reference; H6 says the emitted
  code may permute leaf tables).

## Questions

C1. Current formulation cost. From the SIMD2 captures, give the instruction sequence and count
for one pair on each target (splat `ld1r`/`dup`, `fcmgt.2d`, `sext`, two `mov.d` extracts,
`and`, `orr`; the x86-64 equivalents) versus two scalar splits. Explain in one paragraph why
the 99.5% cross-tree pairing rate makes this formulation structurally unable to win, and
whether any planner change (pairing only within a tree) could rescue it. Quantify the best
case for the rescued version.

C2. Tree-major lanes (lanes = levels of one tree). Gather the tree's six feature values into
vector lanes, compare against a constant threshold vector (memory operand on x86-64, `ldr q`
on ARM64, or a table off a base register, see piece A), and turn the lane masks into the leaf
index directly:
  - AVX2: two `vcmppd ymm` (4 lanes each) then `vmovmskpd` gives 4 index bits per compare;
    shift and or.
  - AVX-512: one `vcmppd k, zmm` (8 lanes) then `kmovb` gives the index; depth 8 fits.
  - NEON: three `fcmgt.2d`; then either `and` with per-lane bit constants and `addp`/`orr`
    reduction, or `sli` (shift-left-insert) chains that pack masks without constants, then
    one lane-combine and `fmov x, d`.
  Under H6 the lane-to-bit mapping is free (permute the leaf table at compile time), so pick
  whatever mapping the reduction makes cheapest and say which permutation it implies.
  Cost the gather honestly: lane loads from the row (`ld1 {v.d}[1], [x]`, `vmovsd` +
  `vmovhpd` + `vinsertf128`, or `vgatherqpd` whose throughput on Intel is poor), and the
  cheaper path when a feature already sits in a scalar register (`ins`, `vunpcklpd`). Give
  instructions per tree at depth 6 (and 8) on NEON, AVX2, AVX-512, versus the scalar floor
  piece A is heading to (about 3 per split plus leaf lookup).

C3. Binarize then compare (H7). Per feature f with B_f distinct sorted borders, compute
bin_f = count of borders b with x > b, using SIMD compares (NEON 2 lanes per instruction and
accumulate; AVX2 4 lanes; AVX-512 8 lanes plus `popcnt` of the k-mask), and keep all bins as
bytes in vector registers (100 features is 7 NEON registers or 2 zmm). Then a tree is: gather
its six bins (`tbl` on NEON from up to 4 source registers = 64 bytes, so 100 features needs
`tbl` + `tbx` or two `tbl`; `vpermb`/`vpermi2b` on AVX-512 VBMI; on AVX2 `vpshufb` is limited
to 16-byte lanes, cost that out), one byte compare against a constant rank vector
(`cmhi.8b` / `vpcmpub k`), mask to index (`and` + `addv` on NEON, `kmovb` on AVX-512,
`vpmovmskb` on AVX2), leaf load, add. Prove exactness: `x > border_r` iff `bin_f > r` for
sorted borders; NaN under AsIs/AsFalse gives bin 0 (all splits false); AsTrue gives +inf,
bin = B_f unless a border is +inf; a NaN threshold behaves as +inf (H5, verify). Give:
  - per-feature binarization cost as a function of B_f on each target;
  - per-tree cost on each target;
  - the crossover versus C2 as a function of the sum of B_f over the model relative to the
    split count (6 x trees). Evaluate at the worst case (all thresholds distinct: sum equals
    split count, as in the synthetic workloads) and at a plausible shared-border case
    (state your assumption, e.g. 300 trees, 100 features, 8 borders per feature).
  Bins must fit a byte: CatBoost caps borders at 254 per feature; say what happens above.

C4. LLVM feasibility. For each formulation and target, say what IR produces the intended
instructions with target-cpu generic or with a named feature set. Verify by writing `.ll`
probes under `probes/` and running `llc` (see `STUDY_CONTEXT.md` for flags). Specifically check:
  - `bitcast <8 x i1> to i8` / `<4 x i1> to i4` after a vector `fcmp`/`icmp`: on x86 with
    AVX2/AVX-512 this should become `vmovmskpd`/`kmovb`; on AArch64 report the sequence it
    becomes and whether it is the `and`+`addv` you want.
  - `shufflevector` of `<16 x i8>` with a constant mask on AArch64: does generic lowering
    give `tbl`, or is `llvm.aarch64.neon.tbl1` needed?
  - `vpermb`/`vpermi2b` from generic `shufflevector` on `<64 x i8>` with `+avx512vbmi`.
  - Building a `<2 x double>` from two scalar loads: does LLVM emit `ldr d` + `ld1 {v.d}[1]`
    or `ldr` + `ins`? What about `<4 x double>` on AVX2 from four scalar row loads?
  Also state exactly what `target-cpu`/`target-features` strings `emit_plan` would need
  for AVX2 and AVX-512 paths (today it sets "generic" and "" at `src/jit.rs:162`), how to get
  the host's features from inkwell (`TargetMachine::get_host_cpu_name` /
  `get_host_cpu_features`), and the consequence of compiling for features the deployment host
  lacks (SIGILL). This is a user decision; state the options.

C5. Anything else. Only if it is concretely cheaper than C2/C3 on some target; do not pad.
(Comparing in f32 is a user question, not a recommendation: CatBoost stores borders as f32
but this project is f64 by decision.)

## Deliverable

Report at `probes/REPORT_C.md` (or the path your launch instructions name), per the report
contract in `STUDY_CONTEXT.md`. Section 3 must contain one table: rows = formulation
(current SIMD2, scalar floor from piece A's target, tree-major lanes, binarize then compare),
columns = NEON, AVX2, AVX-512, cells = instructions per tree at depth 6 plus any fixed
per-feature cost, with the gather cost included and the assumption on feature reuse stated.
Then the ranked recommendation per target, the plan-structure change each implies (the
current `Action::Compare` groups predicates by feature; tree-major formulations group by
tree), and the verified `llc` output for each instruction sequence you rely on.
