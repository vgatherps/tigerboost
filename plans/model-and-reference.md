Superseded: see PLAN.md and plans/perf-study-2026-09-28/RESULTS.md
# Model contract and independent reference

## Contract

Features, thresholds, leaf values, and sums use `f64`. Every tree level contributes its original bit position to the leaf index, even when comparisons execute out of order. A scalar depth-d tree has 2^d leaf values.

Separate a predicate's feature/threshold/missing-value semantics from its consumers, each identified by tree and level. Use small typed IDs backed by dense collections; keep implementation-only IDs private. Introduce types with their first callers rather than creating an unused public IR framework.

Output is the scalar raw prediction, including model scale/bias. Empty forests return bias; depth-zero trees contain one constant leaf. The JIT accepts validated in-memory models. JSON extraction is deferred.

## Slice: validation and reference

- Replace panic-only validation with an error result; prevent unchecked shifts, leaf-count/address overflow, invalid feature references, and inconsistent model metadata.
- Resolve depth-zero trees explicitly as constant leaves; do not inherit the sketch's rejection without checking the supported model contract. Document maximum depth from index/address representation, not an arbitrary performance cutoff.
- Correct leaf-count validation, evaluator enumeration, and tree-ID selection in the sketch as their replacement code is introduced.
- Write a straightforward tree-order reference evaluator. Keep it independent of schedule construction and JIT lowering. Provide test access to per-tree leaf indices.
- Preserve comparison behavior for equality, signed zero, infinities, and model-defined NaNs. Arithmetic reassociation permission does not authorize changing which leaf is chosen.

Tests: depths 0/1/2/3/4, malformed leaf counts, repeated feature at several levels, repeated identical predicates, invalid feature references, boundary values immediately below/equal/above a threshold, and bias-only empty-model behavior.

Gate: validated models alone reach codegen; reference leaf indices have explicit hand-derived expectations.

## Keep validation proportional

The naive Rust evaluator is the correctness reference. Use small constructed trees with hand-derived expectations, then compare JIT results against it. No Python/CatBoost installation, native fixture generator, or import compatibility gate. Proceed to scalar codegen after reviewing the reference.
