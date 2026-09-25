# Cranelift load/reuse experiment

Historical evidence. User switched to LLVM before the dependent-call timing correction; this is not an active backend gate. The saved timing data measures independent-call cost only. The runs' `method.txt` files predate the fuller warmup description below, so they do not independently establish that protocol.

2026-09-24. High-pressure `speed` kernels can spill and run slower than `none`. This happens with ordinary loads too. Removing immutable annotations is not the remedy: `readonly + can_move` improved placement and reduced spilling in the measured comparisons.

These are steady-state batch costs, not isolated single-row latency. Calls have independent inputs; hardware can overlap them. Tiny kernels measure below 1 ns/call, making that distinction important. No production setting or scheduling limit is justified by these probes alone.

## Reproduce

Reproduction requires historical revision `ae90c35`, which retains the probe and its dependencies. These commands do not apply to the current tree; saved experiment artifacts remain unchanged.

```sh
cargo test --offline --all-targets
cargo run --offline --release --example cranelift_probe -- sweep
cargo run --offline --release --example cranelift_probe -- forward 64 readonly movable speed 1
```

Each sweep creates a new directory under `target/cranelift-probe/`, refusing existing names. It saves the lockfile, per-variant input/output IR, post-allocation backend listing, settings, emitted byte count, compilation time, summary CSV and all nine timing samples. The listing is Cranelift's output, not an independent decoding of machine bytes.

Reported run: `1790262268541946000-97673`; repeat: `1790262303707075000-98081`. Both remain under `target/cranelift-probe/`. An earlier exploratory run predates the per-batch warmup and is not used below.

## Environment and method

- Apple M4 Pro; `aarch64-apple-darwin`, macOS 15.6 / 24G84, Darwin 24.6.0 / ARM64_T6041. CPU identity was later verified with an approved `sysctl` query.
- `rustc 1.93.0-nightly`, commit `c90bcb9571b7aab0d8beaa2ce8a998ffaf079d38`, LLVM 21.1.3. Rust release profile.
- Direct Cranelift crates 0.128.2; supporting codegen-meta/shared/control/entity/bforest/bitset/srcgen/assembler crates 0.128.4; `regalloc2` 0.13.5. Experiment Cargo.lock SHA-256: `b8463c053beb806a7d562ae4d76785f12ce06d8061cb1a2d8046415b5436afa4`; this saved snapshot predates the LLVM dependencies.
- Native ISA defaults, CPU-feature inference disabled. Backtracking allocator; alias analysis, verifier and register-allocation checker enabled. NaN canonicalization disabled. AArch64 optional ISA flags all false; complete settings saved per variant.
- 192 variants: immediate/forward/reverse reuse; 4/16/64/128 features; ordinary/readonly; fixed/movable; none/speed; 1/4 requested sums. Every input is used twice. Four sums assign uses round-robin, then combine them left-to-right at return; this changes addition order explicitly.
- 32 deterministic dyadic rows shared by paired variants. Warmup: eight row passes initially, one before each timed batch. Nine rounds rotate/reverse variant order. Each timed batch makes 16,384 checked calls, black-boxes inputs/results, and uses one timer around the batch. No overhead subtraction. Median and min/max below describe batch means, not latency percentiles.
- Compilation time includes ISA/module setup, verification, register-allocation checking, IR/listing capture and executable finalization. One compilation per variant; these are diagnostic compilation costs, not stable production compile-time estimates.

## Results

All rows in this table use `readonly + can_move`; `notrap + aligned` apply throughout. Times are ns/call. The byte and compile columns show `none → speed`.

| Sequence / features | Requested sums | none median | speed median [min, max] | Emitted bytes | Compile µs |
|---|---:|---:|---:|---:|---:|
| Immediate / 16 | 1 | 3.588 | 2.416 [2.401, 2.469] | 264 → 200 | 94 → 114 |
| Immediate / 16 | 4 | 3.560 | 1.895 [1.877, 2.045] | 276 → 148 | 108 → 87 |
| Immediate / 128 | 4 | 23.903 | 14.109 [13.845, 14.661] | 2068 → 1044 | 680 → 493 |
| Forward / 64 | 1 | 20.976 | 22.669 [22.491, 25.274] | 1032 → 1104 | 334 → 546 |
| Forward / 64 | 4 | 12.670 | 9.232 [9.224, 9.959] | 1044 → 788 | 330 → 352 |
| Forward / 128 | 4 | 23.272 | 25.297 [25.101, 29.218] | 2068 → 1680 | 699 → 810 |
| Reverse / 128 | 4 | 23.252 | 45.550 [43.800, 52.628] | 2068 → 2416 | 663 → 1445 |

The repeat retained the high-pressure regressions: forward/128/four sums was 23.539 (`none`) versus 25.688 (`speed`); reverse was 24.602 versus 46.743. This is evidence of a problem in these kernels, not a claim that spill traffic alone explains every timing difference.

For forward/64/one sum at `speed`, ordinary/fixed measured 26.708 ns; ordinary/movable 26.845; readonly/fixed 26.805; readonly/movable 22.669. The first three emitted identical backend listings. Both flags together changed placement. All four optimized 128 feature reads down to 64.

## Loads versus spills

Counts below are static instructions in the inspected AArch64 listings, excluding frame setup and callee-save pairs.

| speed kernel | Feature reads | q spill stores | q reloads | Callee-save d pairs |
|---|---:|---:|---:|---:|
| Forward / 64 / one sum / ordinary fixed | 64 | 36 | 69 | 4 |
| Forward / 64 / one sum / readonly movable | 64 | 34 | 34 | 4 |
| Forward / 64 / four sums / readonly movable | 64 | 0 | 0 | 0 |
| Forward / 128 / four sums / readonly movable | 128 | 8 | 10 | 4 |
| Reverse / 128 / four sums / readonly movable | 128 | 100 | 102 | 4 |

Forward/64/one sum/readonly/movable contains this sequence; omitted lines compute other input contributions:

```asm
ldr d3, [x0]          // feature[0], caller memory
fadd d2, d2, d3
str q3, [sp, #528]    // preserve feature[0] in a 16-byte spill slot
...                  // other features contribute to the result
ldr q3, [sp, #528]    // reload the earlier value, not feature[0]'s address
fadd d2, d2, d3
```

The prologue's `stp d14, d15, [sp, #-16]!` and epilogue's matching `ldp` preserve caller registers; they are not the pressure spills above. The frame-pointer/link-register pair is separate too. Corresponding `none` listings reread features directly and have no q spill traffic.

## Why source order is insufficient

Pinned source [classifies loads as pure](https://github.com/bytecodealliance/wasmtime/blob/881e9a384c38afece75ea1d616ce014ba11a24a7/cranelift/codegen/src/inst_predicates.rs) only with readonly, notrap and can_move. [Alias analysis](https://github.com/bytecodealliance/wasmtime/blob/881e9a384c38afece75ea1d616ce014ba11a24a7/cranelift/codegen/src/alias_analysis.rs) can independently merge ordinary repeated loads. The [egraph pass](https://github.com/bytecodealliance/wasmtime/blob/881e9a384c38afece75ea1d616ce014ba11a24a7/cranelift/codegen/src/egraph.rs) removes pure computations from the instruction layout, then [reconstructs them from their consumers](https://github.com/bytecodealliance/wasmtime/blob/881e9a384c38afece75ea1d616ce014ba11a24a7/cranelift/codegen/src/egraph/elaborate.rs).

That matches these IR dumps: ordinary loads remain ahead of the pure additions; immutable movable loads are emitted with their consumers. With four sums, the optimized IR completes one sum's chain before beginning the next, despite round-robin source emission. Forward/64 partitions into groups of 16 feature values and avoids spills. Reverse reuse shares values across different sums and keeps more values live.

Immediate/four sums has another limitation: pairs of accumulator chains are identical, so Cranelift merges them. Two distinct chains survive and each is doubled. The report therefore says four *requested* sums; it does not claim four surviving independent chains. No artificial constants or dependencies were added to prevent this valid optimization.

## Gate

Seven tests pass: explicit use sequences, input/offset/count validation, executable ownership, all 144 pattern/flag/optimization/sum combinations at 16/64/128 features, each feature perturbed independently, deterministic varied rows, interleaving and no-overwrite artifacts. A plain original-order scalar sum is the oracle; dyadic rows also satisfy the hand-derived `2 * sum(row)` result exactly.

Keep valid immutable annotations and normal speed optimization as the initial real-model baseline. Do not add motion barriers or disable alias analysis from this evidence. Compare accumulator counts and inspect actual leaf-index/leaf-load dataflow next. These sum kernels lack comparisons and dependent leaf loads; only native ARM64 was executed. The real-model benchmark still needs a single-row latency measurement that distinguishes independent-call throughput from dependency latency.
