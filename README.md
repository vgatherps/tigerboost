# tigerboost

A model-specialized LLVM JIT for single-row prediction over numeric symmetric
trees. Features, thresholds, leaves, and arithmetic are `f64`; output is one raw
scalar score. Models are constructed in memory. JSON/CBM loading, categorical
features, and probability transforms are not implemented.

## Use

```rust
use inkwell::context::Context;
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};
use tigerboost::sort_policy::{SortOptions, plan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Trees::new(
        vec![SingleTree {
            selections: vec![Discriminator {
                feature_idx: FeatureIndex(0),
                threshold: 1.0,
            }],
            leafs: vec![2.0, 5.0],
        }],
        vec![NanMode::AsIs],
        2.0,  // scale
        -1.0, // bias
    )?;
    let context = Context::create();
    let plan = plan(&model, SortOptions::default());
    let predictor = compile(&context, &plan)?;
    drop(plan);
    drop(model);
    assert_eq!(predictor.predict(&[1.0])?, 3.0);
    assert_eq!(predictor.predict(&[2.0])?, 9.0);
    Ok(())
}
```

`Trees::new` validates leaf counts and feature indices. `Trees::predict` is the
independent tree-order reference. Compiled models own their code and copied
constants, so the source `Trees` may be dropped; the LLVM `Context` must outlive
the compiled model. `predict` checks that the row covers every declared feature.
Rows stay read-only during each call and may change between calls. Prediction
writes neither caller memory nor model data.

At original tree level `k`, a strict `feature > threshold` decision sets bit `k`
of the leaf index. Each feature's `NanMode` applies before comparison: `AsIs`
leaves NaN unchanged (ordered comparisons are false), `AsFalse` replaces it with
negative infinity, and `AsTrue` replaces it with positive infinity. `AsTrue` is
not an unconditional true bit: positive infinity is not greater than itself.
A NaN threshold also compares false.

The score is the sum of selected leaves, scaled and then biased. Depth-zero
trees are constants; an empty forest returns bias without applying scale.
Arithmetic may reassociate: bitwise agreement and matching overflow behavior
with the reference are not promised. Comparisons retain NaN/infinity semantics.

## How a prediction runs

`sort_policy::plan(&model, SortOptions { chunk_trees, accumulators })` produces
a plan without LLVM: one action per tree, in tree order, grouped into chunks of
`chunk_trees` whole trees, with leaves accumulated round-robin into
`accumulators` independent sums that are added at the end. The default is eight
trees per chunk and two sums.

`jit::compile(&context, &plan)` emits that plan. All thresholds and all leaf
tables live in one private constant block that every function reaches through a
pointer argument, so inside a function each threshold is one load and each leaf
lookup is one indexed load (on x86-64 the leaf load folds into the add). A tree of
depth `d` is evaluated as `ceil(d/2)` two-lane vector comparisons: its feature
values are loaded into lane pairs, compared against the tree's thresholds (stored
consecutively), and the lane masks are packed with shift-left-insert into two
signed lane values whose sum addresses a leaf table that was permuted at compile
time to match. Odd depths get a constant-false padding level. The permutation is
defined by simulating that lane arithmetic for every bit pattern, and checked to
be a bijection when the model is compiled.

Each chunk is a separate `noinline preserve_nonecc` function that transfers only
the row pointer, the constants pointer, and the running sums to the next chunk
with a `musttail` call; a C `predict` wrapper calls the first one. The split
exists to stop LLVM from hoisting loads across many trees and spilling; eight
trees per chunk is the largest spill-free size measured on ARM64.

`jit::compile_with(&context, &plan, EmitOptions { avx2: true })` selects an
x86-64 emission that compares four lanes then two per tree with the threshold
vector as a memory operand and reads the level bits back with `vmovmskpd`. It
sets `x86-64-v3` on the emitted functions and must not run on a host without
AVX2; `EmitOptions::from_host()` enables it when LLVM reports `+avx2` for the
host. `compile` never uses it.

## Build and checks

Requires LLVM 21 and Inkwell 0.9.0 (pinned in Cargo.toml). With the existing
Homebrew LLVM installation used here on ARM64:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo test --offline --all-targets
cargo test --offline --release --all-targets
RUSTDOCFLAGS="-C link-arg=/opt/homebrew/lib/libzstd.dylib" cargo test --offline --doc
cargo clippy --offline --all-targets -- -D warnings
cargo fmt --check
tools/x86_check.sh special && tools/x86_check.sh special --avx2
```

The macOS ARM64 Cargo configuration links `/opt/homebrew/lib` for zstd;
rustdoc does not inherit that setting.

`tools/x86_check.sh <workload> [--avx2]` emits every benchmark variant for
`x86_64-apple-darwin`, links each kernel with a C driver, runs it under Rosetta 2
(`arch -x86_64`) and compares every prediction bit for bit with the reference.
Rosetta executes the code but its timing says nothing about an x86 CPU; no x86
performance has been measured.

## Measurements

Apple M4 Pro, LLVM 21.1.8, generic target, serial runs. Hot numbers are
`examples/scalar_bench.rs` medians of five rounds of 50,000 dependent calls, ns
per call including about 5 ns of driver. Cold numbers are
`examples/cold_bench.rs` per-call p50 over shuffled rows after 64 MiB of unrelated
read-modify-write traffic. Both workloads are synthetic depth-6 forests with
random feature reuse and globally distinct thresholds.

| 100 trees, 100 features | hot median ns | cold p50 ns |
|---|---:|---:|
| scheduled scalar kernel before this work | 121 | 791 |
| tree-major, 8 trees per chunk, 2 sums | 87 | 792 |

| 300 trees, 100 features | hot median ns |
|---|---:|
| scheduled scalar kernel before this work | ~345 |
| tree-major, 8 trees per chunk, 2 sums | 237 |

Static, 100 trees: ARM64 2253 instructions (22.5 per tree, no continuation stack
traffic); x86-64 2332 (SSE2) or 1486 with `avx2`. Rosetta 2 runs the SSE2 kernel at
106 ns and the scalar kernel it replaced at 168 ns, the same ordering as native
ARM64. Details, the study that led here, and the review record are in
`plans/perf-study-2026-09-28/`.

Commands:

```sh
cargo run --offline --release --example scalar_bench -- 100x100
cargo run --offline --release --example cold_bench -- 100x100 --samples 5000
cargo run --offline --release --example inspect_code -- /tmp/tigerboost-code 100x100
```

Workloads are `<trees>x<features>` (`100x100`, `300x100`, `500x100`, `500x1000`)
plus the historical `mixed`, `reuse-heavy`, `disjoint`, and `wide` fixtures.
Inspection requires a fresh output directory and emits separate ARM64/x86-64
assembly and plan dumps, not MCJIT's executable bytes.
