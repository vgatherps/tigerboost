status: complete

The amended mode is implemented and all required gates passed. It remains opt-in. Native ARM64 W16 took 253.65 ns versus continuation's 241.87 ns at 300 trees, a 4.9% regression. Rosetta SSE2 W16 regressed from 304.08 to 354.30 ns at 300 trees. Spills remain. Timings are indicative.

Tracked changes are confined to `src/jit.rs`, `tests/differential.rs`, `tests/scalar_jit.rs`, `examples/scalar_bench.rs`, `examples/cold_bench.rs`, `examples/inspect_code.rs`, and `examples/x86_check.rs`. `tools/x86_check.sh` already consumes the variant manifest and needed no edit. Reports, measurement programs, prepared IR, assembly and logs are under [probes/fence-amended](probes/fence-amended). The accepted earlier raise is preserved in [probes/fence-raise/VERDICT-before-amendment.md](probes/fence-raise/VERDICT-before-amendment.md). No commits, formatting commands, stashes or checkouts were made. Existing untracked task/run logs were retained.

1. `EmitOptions` defaults to `avx2: false`, `fenced: false`, window 8. `from_host` leaves fenced mode disabled and the window at 8. With fenced mode disabled, different windows produce identical assembly on both triples with AVX2 enabled and disabled. All 18 pre-existing inspection files match the saved before-edit files byte for byte.
2. Fenced mode emits one C-convention `predict`, no wrapper or continuations. Every boundary calls empty side-effecting asm with exactly `=r,=r,0,1,~{memory}`; extracted pointer outputs replace row/constants for subsequent chunks. Entry row alignment/bounds and load alignment remain. The prohibited alias/read-only/function-memory attributes are omitted. `prepare_module` skips `run_passes`; backend optimization remains `Less`. Both vector paths use the same feature map and load helper. Expiration uses the loading tree's visit position; the map spans chunk boundaries and stores normalized values. Constant trees, reductions and scale/bias retain their existing emission.
3. The five requested scalar variants, optional `--fenced [W]`, inspection variants and AVX2 twins, and both x86 correctness variants are present. Inspection also includes W1 and one-chunk W16 for the static measurements. The cold benchmark changes only the measured kernel's emission options; its other-model disturbance still runs the existing 2000-tree, 100-feature continuation kernel.
4. Tests cover disabled-option byte identity, one-function shape, exact fence constraints and pointer threading, attributes, prepared-IR and assembly load counts, differential reference checks with windows 1/8/64 and both orders, dropped-model lifetime, and cold CLI options. Tests use production emitters. The measurement probe includes the production `src/jit.rs` to access private `prepare_module`; no public inspection API was added.

Anchors were checked against HEAD `5a915cd405d539c6c83d6aba1b777fc85729e528`, the current emitter, planner/continuation partitioner, harnesses, and registry inkwell 0.9.0 signatures. Original line numbers have shifted. Environment: arm64 macOS 15.6, build 24G84; LLVM 21.1.8; `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21`; `RUSTDOCFLAGS=-C link-arg=/opt/homebrew/lib/libzstd.dylib`.

| Gate | Result |
| --- | --- |
| Before-edit release `inspect_code`, `100x100` | Existing successful snapshot `/tmp/tb-fence-before-77319`, 18 files |
| `cargo build --offline --release --examples` | Pass |
| `cargo test --offline --all-targets` | Pass, 68 tests |
| `cargo test --offline --release --all-targets` | Pass, 68 tests |
| `cargo test --offline --doc` | Pass, 1 test |
| `cargo clippy --offline --all-targets -- -D warnings` | Pass |
| After-edit release `inspect_code`, `100x100` | Pass; [assembly and plans](probes/fence-amended/assembly) |
| `cmp` every pre-existing output | Pass; [18 identical files](probes/fence-amended/byte-identity.txt) |
| `tools/x86_check.sh 100x100` | Pass; six variants, 256 rows each |
| `tools/x86_check.sh special --avx2` | Pass; six variants, 256 rows each |
| `scalar_bench` release, `100x100` and `300x100` | Pass; all variants verified and measured |
| `cold_bench` release, `100x100 --samples 3000` | Pass |
| Cold with `--fenced` (W8) and `--fenced 16` | Pass, 3000 samples per condition and row sequence |
| Additional Rosetta hot probes | Pass, all 28 target/workload/variant combinations; rows and dependent traces checked |
| `git diff --check` | Pass |

Gate logs are in [probes/fence-amended](probes/fence-amended). Rosetta links retained the existing `-no_pie` deprecation/unaligned-pointer warnings; correctness checks passed.

Static counts below use `100x100`, depth 6, two sums, eight trees/chunk except the one-chunk variant. Targets are `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`, generic/SSE2 or `x86-64-v3`/AVX2. Total instructions include the C entry and all continuations, excluding alignment padding and data. A paired load counts as one memory instruction; folded x86 compare/add memory operands count as loads. Row counts include machine reloads from the row. Leaf counts exclude stack reloads of previously loaded leaves.

The stack column follows the requested syntactic rule: any SP/RSP/RBP memory operand or push/pop, excluding C saves/restores. This includes a few RBP-based non-stack loads, detailed below. C bytes show prologue + epilogue, including return and the sole-function AVX2 `vzeroupper`; parentheses give their combined instruction count. Frame bytes are maximum explicit stack storage plus any used red zone, including the active wrapper frame, excluding implicit call return addresses. [Counts and per-store evidence](probes/fence-amended/static-counts.json), [assembly classifier](probes/fence-amended/static_counts.py), and [object disassembly](probes/fence-amended/objects) make the totals reviewable. Independent syntactic stack counts agree for all 21 kernels.

ARM64:

| Variant | Instructions | Row loads | Threshold loads | Leaf loads | Stack | C bytes (instructions) | Frame bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuation | 2143 | 429 | 163 | 100 | 0 | 40+44 (21) | 160 |
| Fenced W1 | 2278 | 430 | 273 | 100 | 66 | 24+28 (13) | 304 |
| Fenced W8 | 2280 | 389 | 275 | 100 | 76 | 24+28 (13) | 320 |
| Fenced W16 | 2432 | 298 | 275 | 100 | 265 | 24+28 (13) | 672 |
| Fenced W32 | 2575 | 204 | 276 | 100 | 448 | 24+28 (13) | 1232 |
| Fenced W16 shared | 2299 | 260 | 273 | 100 | 169 | 24+28 (13) | 448 |
| Fenced W16 one chunk | 2445 | 74 | 270 | 100 | 421 | 24+28 (13) | 1744 |

x86 SSE2:

| Variant | Instructions | Row loads | Threshold loads | Leaf loads | Stack | C bytes (instructions) | Frame bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuation | 2300 | 470 | 300 | 100 | 6 | 9+10 (11) | 48 |
| Fenced W1 | 2264 | 470 | 300 | 100 | 0 | 1+2 (3) | 8 |
| Fenced W8 | 2311 | 428 | 300 | 100 | 14 | 5+6 (7) | 88 |
| Fenced W16 | 2485 | 326 | 300 | 100 | 254 | 17+18 (15) | 440 |
| Fenced W32 | 2567 | 229 | 300 | 100 | 414 | 17+18 (15) | 904 |
| Fenced W16 shared | 2408 | 282 | 300 | 100 | 149 | 14+15 (15) | 264 |
| Fenced W16 one chunk | 3097 | 104 | 300 | 100 | 886 | 17+18 (15) | 3336 |

x86 AVX2:

| Variant | Instructions | Row loads | Threshold loads | Leaf loads | Stack | C bytes (instructions) | Frame bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuation | 1451 | 470 | 200 | 100 | 0 | 9+10 (11) | 40 |
| Fenced W1 | 1434 | 470 | 200 | 100 | 0 | 9+13 (12) | 40 |
| Fenced W8 | 1425 | 428 | 200 | 100 | 0 | 10+14 (14) | 48 |
| Fenced W16 | 1597 | 326 | 200 | 100 | 220 | 17+21 (16) | 440 |
| Fenced W32 | 1708 | 229 | 200 | 100 | 399 | 17+21 (16) | 1440 |
| Fenced W16 shared | 1464 | 282 | 200 | 100 | 110 | 17+21 (16) | 320 |
| Fenced W16 one chunk | 2147 | 102 | 200 | 100 | 717 | 17+21 (16) | 3752 |

SSE2 continuation's six counted stack instructions include two leaf loads using RBP as an index; the remaining four are temporary RBP push/pop instructions in two continuations. SSE2 W16 shared includes four RBP-based threshold loads; its actual stack count is 145. AVX2 W32 includes one RBP-based threshold load, giving 398 actual stack instructions. AVX2 one-chunk includes seven RBP-indexed leaf loads, giving 710 actual stack instructions. Other stack counts need no such adjustment. Fenced x86 frames using a red zone are SSE2 W8/W16/W32/shared/one-chunk and AVX2 W16/W32/shared/one-chunk; each listed extent includes 128 red-zone bytes except SSE2 W8 (64) and AVX2 W32/shared (120 each). SSE2 continuation uses 40 wrapper bytes plus an 8-byte temporary continuation save.

ARM64 W1 and W16 spills were classified by tracing row/constant loads and register copies to every stack-store source, then matching stack reloads. `stp` contributes two sources but one instruction. Saved C registers are excluded.

| Spilled value | W1 stored sources | W1 reloaded destinations | W16 stored sources | W16 reloaded destinations |
| --- | ---: | ---: | ---: | ---: |
| Feature values/vectors | 0 | 0 | 58 | 59 |
| Threshold vectors | 0 | 0 | 0 | 0 |
| Lane masks | 0 | 0 | 0 | 0 |
| Leaf slots/indices | 0 | 0 | 0 | 0 |
| Loaded leaf values | 40 | 40 | 80 | 80 |
| Sums | 0 | 0 | 0 | 0 |
| Constants pointer | 0 | 0 | 0 | 0 |

W1 stores are 22 `str d` plus nine `stp d,d`; reloads are 30 `ldr d` plus five `ldp d,d`: 66 stack instructions. W16 stores are 46 `str q`, six `stp q,q` and 80 `str d`; reloads are 47 `ldr q`, six `ldp q,q` and 80 `ldr d`: 265 stack instructions. The extra feature reload uses a previously stored value again. For example, W1 loads a leaf into `d1` at assembly line 141, stores it at line 145, and reloads it into `d31` at line 703. W16's first feature spill is `str q31, [sp, #464]` at line 45, after loading row features into `d31`/`d17` at line 31. The JSON records every store's line, source class and stack slot.

The fence changes address handling after a boundary. All 32 SSE2 and 24 AVX2 references materializing `.Lconstants` addresses occur before the first fence; none occur after it. W1/W8/W16/W32/shared have the same counts. The one-chunk kernels have no fence and retain 400 SSE2 / 300 AVX2 constant-address materializations. Continuation and all ARM64 variants materialize the constants base once. Thus pointer threading does not force a single machine address materialization before the first boundary; no extra entry barrier was added.

Fenced ARM64 retains absolute constant offsets. No optional chunk-relative GEP rebasing was added. Threshold `ldp q,q` counts are 137 for continuation, then 27/25/25/24/27/30 for W1/W8/W16/W32/shared/one-chunk. Each kernel loads all 300 threshold vectors. Each also contains 45 leaf-offset materializations above 32 KiB. The assembler therefore loses most threshold pairing in fenced mode; large leaf offsets remain. These are measured results, not repaired with extra barriers or address conventions.

Prepared IR has 600 real feature lane fills at 100 trees and 1800 at 300. Counts match across all three targets. Continuation's reductions below come from `early-cse`; fenced reductions come from the emitter's map. W1 permits repeated-feature reuse within one tree, but none between trees.

| Variant | 100-tree row loads | 100-tree lane fills reused | 300-tree row loads | 300-tree lane fills reused |
| --- | ---: | ---: | ---: | ---: |
| Continuation | 464 | 136 | 1402 | 398 |
| Fenced W1 | 584 | 16 | 1749 | 51 |
| Fenced W8 | 427 | 173 | 1230 | 570 |
| Fenced W16 | 324 | 276 | 925 | 875 |
| Fenced W32 | 228 | 372 | 636 | 1164 |
| Fenced W16 shared | 279 | 321 | 746 | 1054 |
| Fenced W16 one chunk | 324 | 276 | 925 | 875 |

Skipping IR passes does not disable backend merging within a chunk. For example, W1 has 584 prepared-IR loads at 100 trees, but ARM64 loads 468 scalar row values in 430 instructions. With one chunk, W16 still emits 324 IR loads, but ARM64 loads only 99 scalar row values in 74 instructions. The fences bound that backend freedom; the emitter can deliberately carry an SSA feature value across them.

The required feature-3 probes passed after `prepare_module` and after assembly emission:

| Probe | ARM64 IR / asm loads | SSE2 IR / asm loads | AVX2 IR / asm loads |
| --- | ---: | ---: | ---: |
| Trees 0, 5, 20; W8; chunks of 4 | 2 / 2 | 2 / 2 | 2 / 2 |
| Trees 0, 5, 20; W32; chunks of 4 | 1 / 1 | 1 / 1 | 1 / 1 |
| Trees 0, 12; W1; chunks of 8 | 2 / 2 | 2 / 2 | 2 / 2 |

Hot medians are ns/prediction, including the dependent row-selection driver; no floor subtraction. Native ARM64 uses the requested `scalar_bench`, five rounds of 50,000 calls, with alternating variant order. x86 uses the same production model/rows and Darwin assembly under Rosetta, five rounds of 50,000 calls per executable; all rows and a 50,000-call dependent trace are checked first. The C probe alternates kernel/floor measurements, while variants run serially. x86 timing code is Mach-O, unlike the Linux static table, and Rosetta results do not measure native x86 hardware. ARM64 driver floors were 5.51/5.69 ns; Rosetta floors were about 6 ns. [Native 100](probes/fence-amended/hot-100.log), [native 300](probes/fence-amended/hot-300.log), [Rosetta](probes/fence-amended/hot-x86.log), [C probe](probes/fence-amended/hot_x86.c).

| Variant | ARM64 100 | ARM64 300 | Rosetta SSE2 100 | Rosetta SSE2 300 | Rosetta AVX2 100 | Rosetta AVX2 300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuation | 89.65 | 241.87 | 110.92 | 304.08 | 413.24 | 1314.43 |
| Fenced W8 | 92.42 | 244.94 | 116.65 | 310.24 | 391.54 | 1284.51 |
| Fenced W16 | 90.68 | 253.65 | 115.92 | 354.30 | 370.60 | 1146.52 |
| Fenced W32 | 91.84 | 252.43 | 115.07 | 320.35 | 389.93 | 1219.61 |
| Fenced W16 shared | 92.37 | 235.96 | 114.48 | 317.17 | 366.45 | 1227.09 |
| Fenced W16 one chunk | 89.86 | 236.40 | 117.21 | 334.81 | 454.45 | 1355.05 |

Cold results are native ARM64 `100x100`, 3000 samples per condition and row sequence, in ns. The timer resolution was about 41 ns; raw percentiles are not timer-corrected. Hot, 64 MiB data disturbance and other-model disturbance are shown; all conditions and timer-only controls remain in the [continuation](probes/fence-amended/cold-continuation.log), [W8](probes/fence-amended/cold-w8.log) and [W16](probes/fence-amended/cold-w16.log) logs. Runs were sequential.

| Variant | Row sequence | Condition | p50 | p90 | p99 |
| --- | --- | --- | ---: | ---: | ---: |
| Continuation | shuffled | hot | 83 | 84 | 125 |
| Continuation | shuffled | data 64MiB | 875 | 2000 | 3333 |
| Continuation | shuffled | other model | 208 | 250 | 292 |
| Continuation | dependent | hot | 83 | 84 | 125 |
| Continuation | dependent | data 64MiB | 833 | 1166 | 2583 |
| Continuation | dependent | other model | 208 | 250 | 292 |
| Fenced W8 | shuffled | hot | 83 | 125 | 125 |
| Fenced W8 | shuffled | data 64MiB | 750 | 1000 | 2500 |
| Fenced W8 | shuffled | other model | 208 | 250 | 292 |
| Fenced W8 | dependent | hot | 83 | 125 | 125 |
| Fenced W8 | dependent | data 64MiB | 750 | 959 | 2541 |
| Fenced W8 | dependent | other model | 250 | 250 | 333 |
| Fenced W16 | shuffled | hot | 83 | 125 | 125 |
| Fenced W16 | shuffled | data 64MiB | 750 | 917 | 2375 |
| Fenced W16 | shuffled | other model | 250 | 250 | 292 |
| Fenced W16 | dependent | hot | 83 | 125 | 208 |
| Fenced W16 | dependent | data 64MiB | 750 | 958 | 2375 |
| Fenced W16 | dependent | other model | 250 | 250 | 333 |

The tested fences held: loads requested on opposite sides were not merged, on ARM64, SSE2 or AVX2. LLVM did spill. ARM64 W1 spilled loaded leaf values; W16 spilled feature values/vectors and loaded leaf values, with no observed threshold-vector, lane-mask, leaf-slot, sum or constants-pointer spills. Larger reuse windows reduced row loads but increased stack traffic. The default remains continuation mode.
