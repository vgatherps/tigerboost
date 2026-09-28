tigerboost review at dd903f7

Worktree: /Users/samschetterer/src/tigerboost/.claude/worktrees/agent-a61814a6d776fa414. `git status --short` shows only `?? probes/`. No tracked file was changed, nothing was committed, and fmt was not run.

I found no correctness bug. The new adversarial probe agrees bit for bit with `Trees::predict` natively on ARM64 and under Rosetta for both x86 paths. The biggest remaining waste is in the ARM64 leaf addressing, which costs one extra instruction per tree once the leaf offsets pass 32 KB (finding 1).

## Findings

**1. Major (performance): ARM64 leaf loads lose their immediate offset past 32 KB.**
- Location: jit.rs:650-668. In the assembly, see `_continuation7` onward of `probes/asm/aarch64-apple-darwin-tree-major-8-2sums.s`, e.g. `mov w10, #37856` / `ldr d8, [x8, x10]`.
- The leaf address is `constants + (leaf_base+bias)*8 + raw*8`. `ldr d` has a scaled 12-bit immediate, which reaches only 32760 bytes. Beyond that, every tree gets one extra instruction: either `mov wN, #imm` or `add xN, x8, #K, lsl #12`.
- The `add ... lsl #12` form sits on the chain from compare result to leaf load.
- Size of the cost:

| model | extra instructions | share of all instructions |
|---|---:|---:|
| 100x100 (100 trees) | 47 | 2.2% of 2143 |
| 300x100 (300 trees) | 291 (91 `mov w` + 200 `add lsl #12`) | 4.5% of 6521 |

- This can be fixed in the IR. It is the same problem the per-chunk threshold base already solves for thresholds. Either lay constants out per chunk as [chunk thresholds | chunk leaves], so the existing `chunk_base` reaches both within a few KB, or pass a second opaque per-chunk leaf pointer.
- x86 is unaffected, because disp32 absorbs any offset.

**2. Minor (test gap, now filled by the probe): the shipped x86 gate misses cases.**
- Location: examples/x86_check.rs:43 and :100-106.
- The `special` workload has depths {0,1,2,3,5,6,8} but only 9 distinct rows, because row r is `VALUES[(r+f)%9]`. The depth-8 tree therefore visits at most 9 of its 256 leaves.
- Every standard workload is depth 6.
- As a result, neither the x86 SSE2 path nor the AVX2 path ever runs these shapes under the shipped gates:
  - depth 4 (a single 4-lane group);
  - depth 7 (4+4 lanes, with the padding lane in the upper group);
  - depth 9 and above.
- The randomized `tests/differential.rs` uses depths 0..6 only (`next(&mut random, 7)`).
- My probe covers these shapes (see Part A). It found no bug, so this is a coverage gap, not a defect.

**3. Minor: `compile_with(.., EmitOptions { avx2: true })` is not checked against the host.**
- Location: jit.rs:83-98, 129.
- On an x86 host without AVX2, a safe API call produces a kernel that dies with SIGILL. This is a crash, not memory corruption. The README warns about it.
- `compile_with` always targets the host, so it could return `Err` when `avx2 && !EmitOptions::from_host().avx2`.
- On a non-x86 triple the option is silently ignored. The test `avx2_off_is_byte_identical` covers that.

**4. Minor: `emit_assembly_with` returns a string with a trailing NUL.**
- Location: jit.rs:120.
- `buffer.as_slice()` includes LLVM's terminator byte. `examples/x86_check.rs:190-194` and its test at :351 work around it with `trim_end_matches('\0')`.
- `inspect_code` writes the NUL into every `.s` file, so ugrep treats them as binary.
- Fix it at the source and delete the workarounds.

**5. Minor (waste, can be fixed in IR): one `vzeroupper` before every inter-chunk `jmpq` on AVX2.**
- That is 12 per call at 100 trees. The next chunk receives only xmm0/xmm1 scalars.
- The per-function attribute `"target-features"="-vzeroupper"` on every continuation except the last removes it. I verified this with llc (`probes/vzu.ll`).
- The last continuation must keep it before returning to C.

**6. Minor (waste, a TargetMachine setting, not IR): x86 inter-chunk transfer is `movabsq $continuationN, %rax; jmpq *%rax`.**
- Cause: `CodeModel::JITDefault` resolves to the large code model.
- `-code-model=medium` gives a direct `jmp` (verified with llc).
- The risk under RuntimeDyld is untested. The x86 JIT cannot run on this host.
- On ARM64 the continuation calls are already direct `b`. The only large-model artifact there is a GOT load of `l_constants` in the wrapper.

**7. Minor (claim versus code): the SSE2 x86 path loads a constant-pool vector per chunk.**
- For trees with 3 or more pairs (depth 5 and up), each chunk does `movabsq $.LCPI2_0` + `movdqa`, the `{3,3}` mask for `pand`.
- On all x86 paths, a non-zero bias is also loaded from the constant pool, not from the constants block.
- `pointer_parameters_have_bounds` (jit.rs:1565-1571) asserts "no `movabsq $.LCPI`" only on a depth-3 model, where it happens to hold. Real depth-6 models contradict it.

**8. Minor (tests that cannot fail):**
- `emit_options_from_host_matches_llvm` (jit.rs:1219-1229) repeats the body of `from_host` and compares the two.
- On this ARM64 host, `tree_major_host_options_match_default` compares `compile` with itself.
- `avx2_leaf_blocks_are_unpermuted` builds its expected ARM64 blocks with the production `permuted_leaves`.

**9. Minor (docs): README static counts are recorded at 4 trees per chunk.**
- At the default of 8 trees per chunk, 100x100 is: ARM64 2143, SSE2 2300, AVX2 1451. At 4 per chunk the counts are 2253 / 2332 / 1486, matching the README.

## Part A: correctness

**Probe.** `probes/adversarial` is a Cargo crate that uses the public `compile` and `emit_assembly_with`. `probes/tools/run_x86.sh` reuses `tools/x86_check.c` unchanged. Cases:
- Every leaf pattern for depths 1-14.
- Depths 1-9 with 3 features reused across levels. Thresholds are NaN, ±inf, ±0, ±subnormal, ±1 and MAX, over 4 rotations of the three `NanMode`s. Rows are the full 13³ grid: NaN, -NaN, ±inf, ±1.5, -1, ±subnormal, ±0, MIN_POSITIVE, 1.
- For each mode, one feature at every level with an identical predicate repeated at two levels, plus all-NaN-threshold and all -inf-threshold trees.
- A mixed forest of depths [0,1,2,3,4,5,6,7,8,0,7,3] with an identical predicate pair inside one lane pair. Leaves use disjoint bit fields, so every reordered sum is exact. It runs with scale 2 and bias -3, under `chunk_trees` {1,2,3,5,8,64} × `accumulators` {1,2,3,64}.
- Native only: an empty forest (NaN scale, bias -0.0) and an all -0.0 constant forest.

**Results.**
- 407,691 native predictions match bit for bit.
- 366 x86 kernels (SSE2 and AVX2, `x86_64-apple-darwin`) pass under Rosetta.
- A mutation check (`vcmpgt`→`vcmpge`, `cmpltpd`→`cmplepd`) fails as it should, so the gate catches a wrong kernel.

**A1.** I found no breaking input.
- `fcmp ogt`, `vcmpgt_oq` and SSE `cmpltpd` (thresholds on the left) are all false on NaN, like `>` in the reference.
- NaN replacement happens before the lane insert, including for the padding lane.
- The padding threshold is +inf, so its bit is always 0, even for `AsTrue` (+inf > +inf is false).
- An empty forest gives sums=1 and `ReturnBias`. If `accumulators` exceeds the forest, it is clamped by `min`. If `chunk_trees` exceeds the forest, there is one chunk.

**A2.**
- `raw_of` computes the same as `vsli`, and the same as `(m<<p)|(prev & (2^p-1))`. The x86 backend lowers the first step as `paddq`/`psubq`, which is equal because m0 ∈ {0,-1}.
- The `shl` of negative i64 values carries no nsw flag. Rust `checked_shl` only checks the shift amount.
- raw ∈ [-bias, 4^p - bias), so the slot stays inside a block of length 2^(2p). That block length is doubled only for odd depth on the 2-lane path (jit.rs:208).
- Padding entries are unreachable and set to 0.0.
- AVX2: `vmovmskpd` bit i is lane i, which is level `first_level+i`. The shift is folded into `leal (%rcx,%rax,8)` with `shll $7`.
- The probe confirms all of this through depth 14 on all three paths.

**A3.**
- `dereferenceable` sizes are exact: row bytes, the whole block, and block minus chunk base.
- `constants` and `chunk_base` are both `noalias` although they alias. This is legal, because `noalias` only constrains memory that is modified, and none is.
- The `add nuw` values are bounded by u32 entries.
- `raw_of`/`permuted_leaves` would panic at depth ≥ 31, but the u32 length check (jit.rs:220) returns `Err` first. Depth 31 on the 2-lane path doubles to 2^32 entries.
- The `isize` byte check (jit.rs:222-225) cannot trigger on 64-bit targets once the u32 check has passed.
- All loads that claim `align 8` are emitted as `ldr q`/`ldp q`, VEX memory operands, or `movupd`, none of which fault on 8-byte alignment. Legacy SSE `cmpltpd` correctly does not fold them.
- Dropping `Trees`/`Plan` leaves nothing referenced; the tests cover it.

**A4.** See finding 3. `from_host` does an exact token match on `+avx2`. The v3 target has no instructions beyond AVX2 in the output: no shlx, andn, fma, movbe or lzcnt. There is also no FMA contraction, because `reassoc` does not imply `contract`.

**A5.** The gate is honest: `memcmp` on every row, and the reference is `Trees::predict`. It is exact only because the workload leaves are small integers. Coverage is the gap described in finding 2.

**A6.** See findings 2, 7 and 8.

## Part B: what can be deleted or folded, ranked by lines removed per risk

The first nine items leave emitted code unchanged.

1. `emit_assembly` (jit.rs:100-107) has no caller outside the unit tests. About 8 lines, no risk.
2. `Chunk.tests` (continuation.rs:9-10, 14-16, 19-25, 36-39) is read only by tests and the `inspect_code` Debug dump. About 12 production lines plus test assertions; no risk.
3. `Admit`/`Retire` can become one `Constant { tree, sum }` action. `Admit` is a no-op in the emitter (jit.rs:419). This simplifies sort_policy.rs:34-38 and 111-113, continuation.rs:20, jit.rs:676-698, and the replay state machine. Drop the `build_int_nuw_add(zero, const)` too. About 20 lines, low risk.
4. `PredicateId` and `predicate_slots` are the identity map: tree-major pushes predicates in action order, and `constants_block_layout_matches_plan` asserts `first_use == 0..n`. Replacing them with a `Range<usize>` (or a first index) in `EvaluateTree` deletes:
   - the slot-assignment loop (jit.rs:172-194);
   - the minimum-over-slots in `chunk_first_slot` (249-264);
   - the consecutive-slot asserts (431-437, 535-536);
   - the `PredicateId` type.
   About 40 lines, low risk.
5. The two `EvaluateTree` arms duplicate the gather loop (jit.rs:453-474 and 564-585). The leaf load + `reassociate(fadd)` tail appears three times (511-525, 660-674, 682-698). Two helpers would save about 35 lines, low risk.
6. `evaluated_trees` (jit.rs:173, 180, 208, 238) is always `!selections.is_empty()`. About 6 lines.
7. The trailing-NUL workaround in x86_check.rs:190-194 and :351 goes away once finding 4 is fixed at source. About 6 lines.
8. Duplicated example code:
   - The `<trees>x<features>` parser appears twice: cold_bench.rs:34-45 and x86_check.rs:24-40, including the magic `10_922` (which is `GRID/DEPTH` from distinct.rs). Moving it into `Workload::parse` deletes the four named sizes in support/mod.rs:23-26 and lets `scalar_bench` accept any size. About 25 lines.
   - The splitmix `random` is duplicated: cold_bench.rs:82-88 and support/distinct.rs:7-13.
   - `check_variants_natively` (x86_check.rs:141-156) duplicates the native loop in `write_outputs`.
9. Tests that mirror production:
   - The tautological `from_host` test (finding 8).
   - The label-string asserts in scalar_bench.rs:175-185.
   - The plan replay interpreter (sort_policy/tests.rs:51-127, about 77 lines). It is redundant with the JIT-versus-reference differential tests unless a plan-level oracle is wanted.
   - The asm string parsing in `avx2_tree_major_emits_folded_compares` (about 135 lines) could shrink to opcode counts.
10. `TreePath` could become a resolved `bool` (jit.rs:41-54). About 10 lines, cosmetic.

These change emitted code, so they are listed separately:

11. The generic x86 shift/and/or path (jit.rs:615-629) is the default x86 path and cannot simply be deleted. Moving non-AVX2 x86 onto the movmsk formulation (the AVX2 arm already has a 2-lane SSE2 variant) would:
    - delete the generic branch and the x86 leaf permutation;
    - cut about 3 of the 5 pack instructions plus 3 extract instructions per tree.

    This needs a Rosetta correctness run. There is no x86 timing.
12. The public items `TreeId`, `SumId`, `PredicateId`, `Action` and `Chunk` are consumed only by `inspect_code` Debug output. Making them `pub(crate)` would need a `Debug` on `Plan`, which is an API change. `emit_assembly` is the only public function with no external user.

## Part C: continuation1 (trees 8-15), 100x100, 8 trees per chunk, 2 sums

### ARM64: 169 instructions, 21.1 per tree

| class | count | per tree |
|---|---:|---:|
| feature load (`ldr d` 21, `ldp d` 2, `ld1.d` lane 17) | 40 | 5.0 |
| feature address (`add x, x20, #off` 15, `mov x10, x20` 1) | 16 | 2.0 |
| lane insert from register (`mov.d v[1], v[0]`) | 7 | 0.875 |
| register copy (`mov.16b`) | 2 | 0.25 |
| threshold load (`ldp q` 10, `ldr q` 4) | 14 | 1.75 |
| compare `fcmgt.2d` | 24 | 3 |
| mask pack `sli.2d` | 16 | 2 |
| extract (`mov.d x, v[1]`, `fmov x, d`) | 16 | 2 |
| leaf address (`add ..., lsl #3` ×2) | 16 | 2 |
| leaf load | 8 | 1 |
| `fadd` | 8 | 1 |
| transfer (`add x22` next base, `b`) | 2 | 0.25 |

- Chunks 0-11 range from 158 to 174 instructions.
- Chunks 7 and later add 8 leaf-offset instructions each (finding 1).
- Over the whole file, `mov.16b` averages 0.53 per tree.
- There is no stack traffic in any continuation.

### x86 AVX2: 116 instructions, 14.5 per tree

| class | count | per tree |
|---|---:|---:|
| feature load `vmovsd` | 25 | 3.125 |
| feature load + lane insert `vmovhpd m` | 17 | 2.125 |
| lane insert from register `vunpcklpd` | 7 | 0.875 |
| half combine `vinsertf128` | 8 | 1 |
| threshold load + compare `vcmpgt_oqpd m` | 16 | 2 |
| mask extract `vmovmskpd` | 16 | 2 |
| leaf index (`shll`, `leal`) | 16 | 2 |
| leaf load + add `vaddsd m` | 8 | 1 |
| transfer (`movabsq`, `vzeroupper`, `jmpq`) | 3 | 0.375 |

There are no `vmovapd` copies. Other chunks sometimes build lanes with `vbroadcastsd`+`vblendpd`, which costs 2 instructions per upper lane. That is a backend choice.

### x86 SSE2: 183 instructions, 22.9 per tree

| class | count |
|---|---:|
| feature loads (42 row operands) | 42 |
| `unpcklpd` | 7 |
| `movapd` | 2 |
| threshold `movupd` | 24 |
| `cmpltpd` | 24 |
| pack (`paddq`, `psubq`, `psllq`, `pand`, `por`) | 40 |
| extract (`movq` ×2, `pshufd`) | 24 |
| `leaq` | 8 |
| `addsd m` | 8 |
| transfer + constant pool (`movabsq` ×2, `movdqa`, `jmpq`) | 4 |

### C2: operations not needed for the computation

| waste | where it comes from | avoidable? |
|---|---|---|
| ARM64 `add x, x20, #off` before `ld1.d` lane loads (2 per tree) | `LD1` single-lane has only base or post-index addressing | No. The alternative `ldr d` + `ins` is also 2 instructions. It disappears only for adjacent features (`ldp d`), and the backend already exploits post-increment once. |
| ARM64 `mov.16b` and `mov.d v[1],v[0]` | A feature shared inside the chunk is reused after the loads are combined, and lane insert is destructive | A backend choice. Not loading shared features once would trade copies for loads. |
| ARM64 extract + `add lsl` | inherent | A vector-side `ushl` + `addp` + `fmov` is also 3 instructions, so no gain. |
| ARM64 leaf address: 2 adds + `ldr`, +1 past 32 KB | finding 1 | The extra one can be fixed in IR. |
| ARM64 wrapper | `preserve_none` | 10 `stp` + 10 `ldp`, a GOT `adrp`/`ldr` and 2 `mov`, about 26 per call. Inherent to `preserve_none`. The GOT load comes from the large code model. |
| x86 `vzeroupper` ×12 per call | finding 5 | Can be fixed in IR. |
| x86 `movabsq` + indirect `jmpq` ×12, wrapper `movabsq` + `callq *%rax` | large code model (finding 6) | Code-model setting. |
| x86 wrapper 5 `push`/`pop` | `preserve_none` | Inherent. |
| SSE2 constant-pool `movabsq` + `movdqa` per chunk, and the bias from `LCPI` | finding 7 | Would go away with the movmsk path (Part B item 11). |

The scale-1 multiply is folded away, but `+0.0` bias is kept. That is required, because x + 0.0 is not x when x = -0.0.

### C3: load-port work per tree

Convention: every `ldp` counts as 2 slots. 48 feature lanes are needed per chunk; 42 are loaded, and the other 6 reuse a value already loaded in the chunk.

| target | feature | threshold | leaf | total per tree |
|---|---:|---:|---:|---:|
| ARM64 | 5.25 (57%) | 3.0 (32%) | 1.0 (11%) | 9.25 |
| x86 AVX2 (all folded memory operands counted) | 5.25 (64%) | 2.0 (24%) | 1.0 (12%) | 8.25 |
| x86 SSE2 | 5.25 (57%) | 3.0 (32%) | 1.0 (11%) | 9.25 (+1 constant-pool load per chunk) |

If ARM64 `ldp` counts as one operation instead, ARM64 is 7.75 per tree (5.0 / 1.75 / 1.0).

### C4: correctness hazards in the machine code

I found none.
- Every lane is written before it is read.
- `vzeroupper` appears only before function exits. No ymm value is live across them.
- No emitted instruction depends on the 8-byte alignment claim in a way that could fault.
- Execution confirms this: the probe runs all three paths bit for bit.

## Verdict

The crate is correct for every input I could construct. The permutation, the padding, the NaN modes, the chunking and the accumulator clamping all agree with `Trees::predict` bit for bit through depth 14 on native ARM64, x86 SSE2 and AVX2. The real test gap was x86 coverage of depths 4, 7 and 9 and up, plus the 9-row `special` workload; the probe fills it with no failures. The next concrete performance fix is the ARM64 leaf addressing past 32 KB (about 4.5% of instructions at 300 trees), which a per-chunk layout or base pointer handles the same way it already does for thresholds. Smaller wins that can be made in IR or config are the per-chunk `vzeroupper` and the code-model indirect jumps on x86. On simplicity, several hundred lines can go without changing emitted code; the cheapest are `emit_assembly`, `Chunk.tests`, the `PredicateId` identity map, `Admit`/`Retire`, the three copies of the leaf-add tail, and the example parser duplicated in two places. None of these block shipping.

## Files

All under `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-a61814a6d776fa414/probes/`:
- `adversarial/` (Cargo.toml, src/main.rs): the differential probe. Run it with `target/release/adversarial probes/x86`.
- `tools/run_x86.sh`: runs every exported variant under Rosetta.
- `tools/classify_arm.sh`, `tools/arm_all.sh`, `tools/classify_x86.sh`: the instruction classifiers used for Part C.
- `vzu.ll`: the llc check for `-vzeroupper` and the medium code model.
- `asm/`, `asm300/`: regenerated 100x100 and 300x100 assembly.
- `x86/`: the 366 exported variants.
- `mut/`: the mutation check.
