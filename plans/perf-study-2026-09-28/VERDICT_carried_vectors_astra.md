status: raise

The carry experiment is implemented. The required no-stack test fails because LLVM spills inside continuations. All prediction checks pass. Carrying is slower in the measured ARM64 hot runs. All requested native measurements are complete. The required no-stack gate remains unmet; spills were recorded as experimental results without changing the pinned carry rule.

HEAD is 6cae781bcc69c23d295aac2eb12e87b1f3648e78. The cited anchors matched before edits. Amendment 1 controls the sum packing and zero-capacity fallback.

Files changed: src/jit.rs, tests/differential.rs, tests/scalar_jit.rs, examples/scalar_bench.rs, examples/cold_bench.rs, examples/inspect_code.rs, examples/x86_check.rs, and this report. CODEX_TASK.md, VERDICT.md, codex_run.log, and codex_run2.log existed at entry. No commits, formatting commands, saturating arithmetic, or edits outside the allowed source files.

EmitOptions.carry_features defaults to false, including from_host(). Carry mode uses eight <2 x double> arguments after the existing pointers. Sums occupy ceil(sum_count / 2) vectors in SumId order. Odd sums and odd feature counts leave lane 1 undef; unused arguments are undef. Plans with 15 or more sums take the original emitter path.

carried_features ranks features present in both adjacent chunks by next-chunk lane fills, then feature index. Received features are included because every received feature is used in its receiving chunk. Padding follows the actual comparison-group lane fill in each emitter. Both EvaluateTree arms reuse extracted, already-normalized values. Ordinary loads, early-cse, preserve_nonecc, musttail, constants, and tree ordering remain unchanged.

The requested scalar_bench, cold_bench --carry, inspect_code, and x86_check variants are present. Differential tests exercise carry on/off and both orderings. Additional tests cover feature 3 crossing two boundaries, ranking/capacity, default-off assembly equivalence, the 15–18 sum fallback on both triples with AVX2 on/off, and prediction with 1, 2, 3, 13, 14, 15, 16, and 18 sums including NaNs.

All continuations use matching signatures. Assembly uses vector arguments in v0–v7 / xmm0–xmm7; packing is not split into scalar ABI arguments. Stack references checked in all 24 measured assembly files stay inside local ARM64 frames or x86 frames/red zones. The required test is stricter: it rejects any stack traffic in a continuation. It remains failing and is neither ignored nor weakened to accept spills.

Counts below cover continuation bodies only, excluding the C wrapper, alignment, directives, and constant tables. Row loads count machine instructions reading row memory, including folded loads and ARM64 lane loads; an ARM64 ldp counts once. Scalar feature loads are computed from the plan's per-chunk distinct uses minus carried-in features. Backend load combining and rematerialization can change machine counts. These units differ.

Lane delta is the net increase in register lane pack/extract/shuffle instructions relative to the same ordering with carry off. ARM64 counts mov.d between vector lanes, mov dN,vM[lane], dup, zip, and ext. x86 counts unpack/shuffle/insert/extract/blend/permutation and register scalar-lane moves. Row-memory lane loads and whole-register copies are excluded. The instruction delta includes every backend change. Stack is explicit memory instructions plus a separate push/pop count.

| Trees | Target | Variant | Instructions | Row loads | Scalar feature loads | Added lane ops | Stack memory + push/pop |
|---:|---|---|---:|---:|---:|---:|---:|
| 100 | ARM64 | continuation | 2117 | 429 | 468 | +0 | 0 + 0 |
| 100 | ARM64 | shared control | 2060 | 332 | 358 | +0 | 0 + 0 |
| 100 | ARM64 | carried | 2280 | 296 | 317 | +213 | 11 + 0 |
| 100 | ARM64 | carried + shared | 2129 | 272 | 291 | +94 | 2 + 0 |
| 100 | x86 SSE2 | continuation | 2285 | 470 | 468 | +0 | 0 + 4 |
| 100 | x86 SSE2 | shared control | 2295 | 360 | 358 | +0 | 4 + 6 |
| 100 | x86 SSE2 | carried | 2644 | 318 | 317 | +185 | 134 + 24 |
| 100 | x86 SSE2 | carried + shared | 2445 | 295 | 291 | +74 | 40 + 24 |
| 100 | x86 AVX2 | continuation | 1436 | 470 | 468 | +0 | 0 + 0 |
| 100 | x86 AVX2 | shared control | 1390 | 360 | 358 | +0 | 0 + 0 |
| 100 | x86 AVX2 | carried | 1580 | 318 | 317 | +185 | 60 + 20 |
| 100 | x86 AVX2 | carried + shared | 1421 | 294 | 291 | +76 | 2 + 10 |
| 300 | ARM64 | continuation | 6495 | 1315 | 1419 | +0 | 0 + 0 |
| 300 | ARM64 | shared control | 6247 | 886 | 974 | +0 | 0 + 0 |
| 300 | ARM64 | carried | 6941 | 879 | 938 | +684 | 8 + 0 |
| 300 | ARM64 | carried + shared | 6512 | 682 | 739 | +338 | 0 + 0 |
| 300 | x86 SSE2 | continuation | 6852 | 1414 | 1419 | +0 | 0 + 6 |
| 300 | x86 SSE2 | shared control | 6925 | 977 | 974 | +0 | 46 + 20 |
| 300 | x86 SSE2 | carried | 7947 | 944 | 938 | +557 | 396 + 72 |
| 300 | x86 SSE2 | carried + shared | 7469 | 746 | 739 | +275 | 182 + 70 |
| 300 | x86 AVX2 | continuation | 4289 | 1414 | 1419 | +0 | 0 + 0 |
| 300 | x86 AVX2 | shared control | 4151 | 977 | 974 | +0 | 0 + 0 |
| 300 | x86 AVX2 | carried | 4712 | 944 | 938 | +566 | 168 + 66 |
| 300 | x86 AVX2 | carried + shared | 4288 | 745 | 739 | +283 | 8 + 34 |

| Trees | Target | Row instructions removed by carrying | Added instructions | Row instructions removed by carrying in shared order | Added instructions in shared order |
|---:|---|---:|---:|---:|---:|
| 100 | ARM64 | 133 | +163 | 60 | +69 |
| 100 | x86 SSE2 | 152 | +359 | 65 | +150 |
| 100 | x86 AVX2 | 152 | +144 | 66 | +31 |
| 300 | ARM64 | 436 | +446 | 204 | +265 |
| 300 | x86 SSE2 | 470 | +1095 | 231 | +544 |
| 300 | x86 AVX2 | 470 | +423 | 232 | +137 |

Shared control is included to separate ordering from carrying. At 100/300 trees it already removes 110/445 scalar feature loads relative to model order.

The carry counts are identical for ARM64, x86 SSE2, and x86 AVX2 on these depth-six workloads. The initial wrapper call is excluded; 100 trees have 12 boundaries, 300 trees have 37.

| Trees | Variant | Features per boundary, mean | Max | Total |
|---:|---|---:|---:|---:|
| 100 | continuation | 0.00 | 0 | 0 |
| 100 | shared control | 0.00 | 0 | 0 |
| 100 | carried | 12.58 | 14 | 151 |
| 100 | carried + shared | 5.58 | 9 | 67 |
| 300 | continuation | 0.00 | 0 | 0 |
| 300 | shared control | 0.00 | 0 | 0 |
| 300 | carried | 13.00 | 14 | 481 |
| 300 | carried + shared | 6.35 | 10 | 235 |

Hot medians are native ARM64, five rounds of 50,000 dependent predictions, including the existing driver. Other experiments run concurrently, so timings are indicative. The shared control identifies the cost of carrying within that ordering.

| Target | Trees | Continuation ns | Shared control ns | Carried ns | Carried + shared ns |
|---|---:|---:|---:|---:|---:|
| ARM64 | 100 | 92.02 | 90.73 | 99.20 | 94.96 |
| ARM64 | 300 | 248.69 | 233.89 | 261.65 | 249.27 |
| x86 SSE2 | 100 / 300 | unavailable | unavailable | unavailable | unavailable |
| x86 AVX2 | 100 / 300 | unavailable | unavailable | unavailable | unavailable |

This host is ARM64. x86_check verifies emitted x86 code under Rosetta; it does not measure latency. Native x86 timing was not available.

All commands used LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21. Rustdoc used RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'.

| Gate | Result |
|---|---|
| Before-edit inspect_code, 100x100 | Passed; /tmp/tb-carry-before-77557, 18 files |
| cargo build --offline --release --examples | Passed |
| cargo test --offline --all-targets | Failed only no_carried_value_on_stack; 52 passed, 1 failed before examples |
| cargo test --offline --release --all-targets | Same failure; 52 passed, 1 failed before examples |
| cargo test --offline --examples and release equivalent | Passed all 16 example tests in each profile; run separately because the integration-test failure stops --all-targets |
| cargo test --offline --doc | Passed, 1 test |
| cargo clippy --offline --all-targets -- -D warnings | Passed, including final test revision |
| After-edit inspect_code, 100x100 and 300x100 | Passed; 30 files per workload |
| cmp every pre-existing output | Passed all 18 files, including final emission at /tmp/tb-carry-final-10789 |
| tools/x86_check.sh 100x100 | Passed all six variants, 256 rows each |
| tools/x86_check.sh special --avx2 | Passed all six variants, 256 rows each |
| Additional tools/x86_check.sh 100x100 --avx2 | Passed all six variants, including the full spilling workload |
| scalar_bench 100x100 and 300x100 | Passed verification and timings |
| cold_bench 100x100 --samples 3000, with and without --carry | Passed verification and timings |
| Additional cold runs, carried/shared at 100 and all three at 300 | Passed verification and all 3,000-sample runs |
| git diff --check | Passed |

The x86 linker warns about -no_pie deprecation and unaligned absolute relocation locations in emitted code. Both requested x86 executions nevertheless pass.

Final no-stack test counts at 100 trees, including push/pop: ARM64 11 / 2, x86 SSE2 158 / 64, x86 AVX2 80 / 12 for carried / carried with shared order. No tests were skipped or marked ignored. Ordinary baseline SSE2 already has four push/pop instructions, so the report separates explicit stack operands from push/pop.

Raw outputs are /tmp/tb-carry-debug-tests-final.log, /tmp/tb-carry-release-tests-final.log, /tmp/tb-carry-example-tests.log, /tmp/tb-carry-release-example-tests.log, /tmp/tb-carry-doc.log, /tmp/tb-carry-clippy.log, /tmp/tb-carry-x86.log, /tmp/tb-carry-x86-avx2.log, /tmp/tb-carry-x86-100-avx2.log, /tmp/tb-carry-hot-{100,300}.log, and /tmp/tb-carry-cold-{100,300}-{base,carry,shared}.log. Static counts are in /tmp/tb-carry-static.json; measured assembly and plans are in /tmp/tb-carry-after-93906 and /tmp/tb-carry-after-300-95709.

Cold measurements use 3,000 samples per condition and sequence. Each cell is p50 / p90 / p99 in ns. All are native ARM64. Native x86 SSE2 and AVX2 cold timings are unavailable for every variant and both tree counts. Timer resolution was about 41 ns, so small percentile differences are quantized. The 64 MiB disturbance and other-model execution are outside the timed prediction.

| Trees | Variant | Sequence | Hot | 64 MiB | Other model |
|---:|---|---|---:|---:|---:|
| 100 | continuation | shuffled | 83 / 84 / 125 | 917 / 1500 / 3208 | 208 / 209 / 250 |
| 100 | continuation | dependent | 83 / 84 / 125 | 750 / 1084 / 2584 | 208 / 250 / 292 |
| 100 | carried | shuffled | 83 / 125 / 125 | 1041 / 1459 / 3709 | 208 / 250 / 292 |
| 100 | carried | dependent | 83 / 125 / 167 | 1042 / 2750 / 3958 | 209 / 250 / 333 |
| 100 | carried + shared | shuffled | 83 / 84 / 125 | 875 / 1750 / 3292 | 208 / 209 / 250 |
| 100 | carried + shared | dependent | 83 / 84 / 125 | 750 / 916 / 2417 | 208 / 250 / 292 |
| 300 | continuation | shuffled | 250 / 250 / 291 | 2167 / 2334 / 5667 | 500 / 583 / 667 |
| 300 | continuation | dependent | 250 / 250 / 292 | 2166 / 2333 / 5625 | 500 / 542 / 667 |
| 300 | carried | shuffled | 250 / 292 / 292 | 2166 / 2333 / 5875 | 500 / 583 / 708 |
| 300 | carried | dependent | 250 / 250 / 334 | 2167 / 2375 / 6209 | 542 / 667 / 750 |
| 300 | carried + shared | shuffled | 250 / 250 / 291 | 2125 / 2333 / 5958 | 500 / 500 / 625 |
| 300 | carried + shared | dependent | 250 / 250 / 250 | 2125 / 2334 / 6291 | 500 / 541 / 625 |

With model order, carrying removes 151 scalar feature loads at 100 trees and 481 at 300 trees. It adds 163 / 446 ARM64 instructions, 359 / 1,095 x86 SSE2 instructions, and 144 / 423 x86 AVX2 instructions. In shared order it removes another 67 / 235 scalar loads compared with shared order alone, while adding 69 / 265 ARM64 instructions, 150 / 544 SSE2 instructions, and 31 / 137 AVX2 instructions. Neither carried variant beats the continuation kernel on the measured ARM64 hot median. Native x86 hot performance remains unmeasured.
