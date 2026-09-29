## Final review of tigerboost at master e466d0e

I found no blocker and no major issue. On native ARM64, and under Rosetta for both SSE2 and AVX2, every probed model picked the same leaves as `Trees::predict`. What's left is two small simplifications, one latent inconsistency, and one stale results table.

Worktree: `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-a36b9d2772d94900b`, HEAD e466d0e. `git status --short` shows only `?? probes/`. I touched no tracked file.

### Correctness: no disagreement found

The fuzzer is `probes/fuzz/src/main.rs`. It builds models only through the public API (`Trees::new`, `compile`, `compile_plan`, `emit_assembly_with`) and runs x86 code through `tools/x86_check.c` under Rosetta. It generates:
- 0 to 100 trees, including 0/1/2/7/8/9/15/16/17, so forests smaller than a chunk and chunk edges are hit.
- 1 to 12 features, all three NaN modes, depth 0 to 12 (16 or 18 in the deep runs), odd depths, and repeated features within a tree.
- Thresholds from NaN, ±inf, ±0, subnormals, MIN/MAX, reused per-feature thresholds, and random values. Rows use those values, their `next_up`/`next_down`, and NaN.
- Leaves that are random integers below 2^36, scale in {±1, 2, 0.5, 0.25}, and bias in {0, −0, 3, −1024}. Every sum is therefore exact, and a wrong leaf shows up as a bit mismatch.

The probe profile keeps overflow checks and debug assertions on, including inside tigerboost. About half the models contain swapped lane pairs, counted through the public `ordered_predicates`.

| Run | Models | Rows each | Mismatches | Models with swaps (ARM64 / x86 pairs) |
|---|---:|---:|---:|---|
| Native ARM64, `compile` and `compile_plan` | 10,300 | 256 | 0 | 5,469 (6,756 / 7,004) |
| Native ARM64, depth up to 18 | 300 | 256 | 0 | 117 |
| Rosetta SSE2 | 1,200 + 120 at depth 16 | 256 | 0 | 581 + 52 (x86 688 + 40) |
| Rosetta AVX2 | 1,200 + 120 at depth 16 | 256 | 0 | 581 + 52 (x86 688 + 40) |

The empty-forest wrapper checks out (`probes/fuzz/src/bin/special.rs`, log `probes/special.log`):
- Native, with 0 or 3 features, it returns the exact bias bits for signaling NaN (both signs), a quiet NaN with payload, −0.0, the smallest subnormal and +inf.
- A short row gives `RowLengthError`, same as the reference, and a longer row is accepted.
- SSE2 and AVX2 assembly carries the same bits as a `.quad`.

For non-empty forests whose output is NaN, the NaN payload or sign can differ from the reference. There were 31 of 240 such cases, all NaN on both sides. The cause is LLVM folding the constant scale or bias. README already says bitwise agreement is not promised, so this is not a finding. No FMA contraction appears on any target.

Existing gates: 67 tests pass in both debug and release with `--all-targets`, the doctest passes, and clippy with `-D warnings` is clean.

### Findings

1. **Minor, complexity: ARM64 carries a third pointer only to compute the next chunk's base** (`src/jit.rs:290`, `:329-341`, `:360-382`, `:433-439`). In each continuation the `constants` parameter is used for one thing: `arguments(row, constants, chunk_number + 1)` builds the next `chunk_base` from it. A single base parameter does the job. On ARM64 it moves by `origins[next] - origins[this]` at each chunk transfer; on x86 its origin is 0, so it never moves.
   - I prototyped it (`probes/onebase`, diff in `probes/onebase.diff`): 29 lines added, 42 removed.
   - Gone: `pointer_count = 2 + usize::from(arm64)`, the `Param(2)` attribute block, the `if arm64` in `arguments`, and the three `chunk_base.unwrap_or(constants)`.
   - SSE2 and AVX2 assembly is byte-identical at 100 and 300 trees.
   - ARM64 loses the wrapper's `mov x22, x21` and one carried register. The per-chunk `add x22, x21, x8` becomes `add x21, x21, x8`, stack traffic is unchanged, and instruction count is within one.
   - `tests/jit.rs`, `tests/differential.rs` and the `x86_check` example tests (including the Rosetta gate) pass on the prototype. Three unit tests fail only on their pointer-count and parameter-index assertions, which need updating.
   - Moving the base on x86 too is worse: +20 / +37 SSE2 instructions at 100 / 300 trees. Keep the x86 origin at 0.

2. **Minor, latent: the AVX2 leaf slot is the only address not written chunk-relative** (`src/jit.rs:555-558`). It uses `leaf_base[tree.0]`; the two-lane path (`:653`), constant trees (`:665`) and thresholds (`:510`, `:579`) all subtract `chunk_origin`. It is correct today only because `chunk_origin` is 0 on x86. Any change that gives x86 a nonzero origin would make AVX2 read the wrong leaves while the SSE2 path stays correct. The fix is to subtract `chunk_origin` there too.

3. **Minor, complexity: the emitted merge and `raw_of` are chosen by different switches.**
   - The merge branch is picked by whether the `vsli` intrinsic was declared (`src/jit.rs:396-410`, `:606`). `raw_of` and the leaf permutation are picked by `packing` (`:211-215`).
   - The two agree only because `vsli` is left undeclared exactly when no ARM64 tree has two pairs, and a one-pair tree never merges.
   - Fix: declare `vsli` when `packing == MaskPacking::Insert` and branch on `packing`.
   - Also, the `MaskPacking::Insert if pair == 0 => mask` arm in `raw_of` (`:809`) is redundant: the general formula at pair 0 is `(mask << 0) | (acc & 0) = mask`.

4. **Minor, stale text: `plans/perf-study-2026-09-28/RESULTS.md:285-290`.** The layout table still gives SSE2 as 2267 / 6859 instructions. The add-merge in de6693e made that 2134 / 6454, with 0 constant-pool loads (was 13 / 38). Those numbers exist only in `VERDICT_polish_astra.md:40`, whose evidence links point at deleted `/tmp` files. PLAN says RESULTS "records the measurements and experiments", so add a short add-merge section.

5. **Nit: target detection is duplicated.** The ARM64 test appears in both `emit_plan` (`src/jit.rs:208-209`) and `ordered_predicates` (`:754`). `prepare_module` (`:159`) chooses AVX2 only for triples starting with `x86_64-`, so `x86_64h-apple-darwin` silently gets the SSE2 kernel. A shared helper fixes both.

6. **Nit:** with AVX2, a final chunk with no 4-lane compare always gets the explicit `vzeroupper` (`src/jit.rs:696-705`), even when no chunk ever touched a ymm register (for example, a forest of trees of depth 2 or less). It costs one instruction and is correct.

### Were the previous review items implemented correctly?

Yes, all of them. I checked them against de6693e:
- The add-based merge is right. Each lane value is minus that lane's pattern bits, and the `raw_of` `Add` mode matches the emitted `shl`/`add`. `emits_vector_compares_only` guards against the constant-pool load coming back at depth 6. The same assertion in `pointer_parameters_have_bounds` (depth 3) would have passed before the change too, so it guards nothing, which is harmless.
- Unified threshold and leaf addressing is done, apart from finding 2.
- The swapped-pair AVX2 kernel runs under Rosetta in `cargo test --all-targets` (100x100 has 54 x86 swaps), and the README gate list has that command.
- The following are all done:
  - the AVX2 identity shortcut and computing `raw_of` once per pattern
  - `usize` swap masks
  - the `constant_bits` helper
  - the partition test using `plan_with`
  - the fixed stale test premises
  - one `plan.txt`
  - benchmark headers showing the real sum count
  - the PLAN and README layout text
  - the empty-forest wrapper

### Assembly: nothing cheap left

Each tree now costs:

| Target | Per-tree instructions |
|---|---|
| ARM64 | `fcmgt`/`sli`, two lane moves, `add`, `add`, `ldr`, `fadd` |
| SSE2 | `cmpltpd`, `paddq`/`psllq`, `movq`, `pshufd`, `movq`, `lea`, `addsd` with a memory operand |
| AVX2 | `vcmpgt_oqpd` with a memory operand, `vmovmskpd`, `shl`, `lea`, `vaddsd` with a memory operand |

Continuation stack traffic is zero on ARM64 and AVX2; SSE2 has one `push rbp` / `pop rbp`. The remaining cost is the ARM64 C wrapper saving and restoring all callee-saved registers, 20 stack operations per call. That comes with calling `preserve_nonecc` code from C, and avoiding it would need LLVM's global interprocedural register allocation option. I don't recommend that.

### Ranked list (value per effort)

| Rank | Change | Effect | Cost |
|---:|---|---|---|
| 1 | Subtract `chunk_origin` in the AVX2 leaf slot (`src/jit.rs:555-558`) | Removes the one address that breaks if x86 ever gets a nonzero origin | 1 line |
| 2 | One chunk-base pointer on every target (finding 1) | Removes the 2-vs-3 pointer split and a duplicate attribute block. x86 byte-identical, ARM64 one register freer | −13 lines, 3 test assertions updated |
| 3 | Branch the merge on `packing`, and drop the redundant `raw_of` arm | Emitter and simulation read from one switch | ~5 lines |
| 4 | Add the add-merge SSE2 counts to RESULTS.md | Docs match the code | ~6 lines |
| 5 | One target-detection helper that also accepts `x86_64h` | Removes duplication and a silent SSE2 fallback | ~6 lines |

### Verdict

The crate is correct as far as I could probe. About 13,000 random forests selected exactly the reference leaves: native ARM64 plus about 1,300 each under Rosetta for SSE2 and AVX2, with about half containing swapped pairs, depths up to 18, all NaN modes and every special value. The newest code holds up: the add-based merge and its `raw_of` mode, the chunk-relative addressing, and the empty-forest wrapper, which keeps exact bias bits including signaling NaN and −0 and still checks row length. All items from the second review pair are in and correct. What remains is small. The AVX2 leaf slot should be written chunk-relative like every other address. ARM64 can drop the pointer it carries only to find the next chunk's base, with x86 assembly unchanged. The merge choice should key off `packing`. RESULTS.md needs the add-merge numbers.

Everything is in `probes/`:
- `fuzz/`: the fuzzer (`tb-probe-fuzz native|sse2|avx2 <start> <count>`, env `TREES`, `MAXDEPTH`, `WORK`) and `src/bin/special.rs`
- `native-10k.log`, `native-deep.log`, `sse2.log`, `avx2.log`, `sse2-deep.log`, `avx2-deep.log`, `special.log`: results
- `onebase/` and `onebase.diff`: the one-pointer prototype
- `asm100/`, `asm300/`, `asm100-onebase/`, `asm300-onebase/`: emitted assembly