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
    let predictor = compile(&context, &model)?;
    drop(model);
    assert_eq!(predictor.predict(&[1.0])?, 3.0);
    assert_eq!(predictor.predict(&[2.0])?, 9.0);
    Ok(())
}
```

`Trees::new` validates leaf counts and feature indices. `Trees::predict` is the
independent tree-order reference. Compiled models own their code and copied leaf
tables, so the source `Trees` may be dropped; the LLVM `Context` must outlive the
compiled model. `predict` checks that the row covers every declared feature.
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

## Scheduling

`compile` defaults to scalar comparisons, up to four active trees, and two
independent sums. The compile-time schedule shares feature loads and predicates,
updates original leaf-index bits, and retires/refills trees as they finish.
Prediction does not walk split metadata or run the scheduler.

`jit::compile_with_options` accepts `CompileOptions { order, accumulators,
comparisons }`. Counts use `NonZeroUsize`. Choose `Order::TreeOrder`, one
accumulator, and `ComparisonMode::Scalar` for the direct baseline. Choose
`Order::Rolling(cap)` and `ComparisonMode::Simd2` for opt-in two-lane comparisons
sharing a feature; tree order remains scalar regardless of that option.

The backend uses a generic CPU, LLVM `early-cse`, and backend optimization
level `Less`. Immutable leaf tables and read-only input/function annotations
permit load reuse. LLVM can still reorder instructions and spill: the planner's
active-tree cap is not a bound on physical register usage.

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
```

The macOS ARM64 Cargo configuration links `/opt/homebrew/lib` for zstd;
rustdoc does not inherit that setting. The explicit dylib avoids selecting the
existing Intel zstd in `/usr/local/lib` first on this machine.
Native execution was tested on ARM64. Generic x86-64 assembly was emitted and
assembled, not executed; x86-64 correctness/performance is not established.

## Synthetic measurements

SIMD regressed on reuse-heavy trees: 75.85 ns versus 52.04 ns for tree order.
The scalar default was slightly slower on disjoint trees (about 1–2 ns across
two runs, with overlapping ranges). It improves mixed/wide workloads and stays
near parity on reuse-heavy ones; it is not a universal optimum.

Apple M4 Pro, LLVM 21.1.8, median ns/call including the roughly 5 ns row-selection
driver; five interleaved rounds of 50,000 dependent single-row calls:

| Workload | Reference | Tree order, 1 sum | Default scalar, cap 4 / 2 sums | SIMD, cap 2 / 2 sums |
|---|---:|---:|---:|---:|
| Mixed | 179.07 | 92.25 | 80.43 | 74.38 |
| Reuse-heavy | 176.51 | 52.04 | 51.99 | 75.85 |
| Disjoint | 179.36 | 105.96 | 107.00 | 107.12 |
| Wide | 346.32 | 226.25 | 184.95 | 202.34 |

All cases have depth six and 256 rows. Mixed/reuse-heavy/disjoint have 64 trees
and 16/4/384 features; wide has 128 trees and 128 features. All rows are checked
against the reference before timing. The result-dependent traces visit only
9/2/13/30 distinct rows respectively: these are warm synthetic measurements,
especially the two-row reuse-heavy trace, not deployment-level evidence.

```sh
cargo run --offline --release --example scalar_bench -- mixed
cargo run --offline --release --example inspect_code -- /private/tmp/tigerboost-code-new mixed
```

Both accept `mixed` (default), `reuse-heavy`, `disjoint`, or `wide`.
Inspection requires a fresh output directory and emits separate ARM64/x86-64
assembly, not MCJIT's executable bytes. See [implementation notes](plans/implementation.md)
for compilation times, code sizes, spills, regressions, and review history.
