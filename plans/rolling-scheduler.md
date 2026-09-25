# Rolling admission, feature reuse, and bounded live state

Depends on the scalar JIT baseline. Scheduling is entirely compile-time.

## Representation

Keep three identities distinct: tree/level consumer, model predicate, and emitted comparison result. One predicate result can update several tree bits. Identical predicates require identical feature, threshold representation, and missing-value semantics; sharing only a feature does not make two comparisons equivalent.

The plan records admissions, loaded-feature identity, explicit scalar/SIMD comparisons and their consumers, and tree completion/accumulation. Value production and last use must be recoverable without guessing which earlier vector contains a bit. Use typed dense IDs and small enums; avoid a general compiler IR.

Track active tree indices, retained feature values, pending comparison results, and accumulators separately. Initial policy retains one loaded feature and consumes one comparison pack before creating another. Physical temporary counts may be larger after lowering; verify generated code.

## Slice: scalar scheduling and schedule interpreter

Initial heuristic, to be measured rather than claimed optimal:

1. Choose the first tree deterministically using feature-frequency scores. Fill up to the configured live-tree cap using the admission rule below.
2. Admit subsequent trees by overlap with the active set's remaining feature needs, including the currently retained feature. Break ties by the initial score, then original tree ID. Start with direct scans; add indexing only if compile-time measurements justify it.
3. Prefer the feature with the greatest number of pending split uses among active trees. Break ties by how many trees processing that feature would finish, then feature ID. If no feature serves multiple pending uses, favor a tree with the fewest remaining uses and finish it.
4. Load the selected feature once. Service its pending predicates in deterministic predicate-ID order. For each comparison, attach every matching pending consumer in the active set and apply each result to its original destination bit immediately.
5. On the last required bit, emit the leaf lookup and accumulation, retire the tree, and refill its slot. Do not wait for unrelated active trees to finish.
6. Refilling occurs between consumed comparison packs. Retain the loaded feature while active work, including replacements, can use it; otherwise return to feature selection. A new tree may require a comparison already discarded; recompute it rather than retaining every historic result.
7. Continue until every tree is retired. Every emitted comparison must resolve at least one pending split use, giving a finite progress argument.

The active-tree cap is a bound, not a requirement to keep every slot occupied after unadmitted work is exhausted. Constant trees retire directly. Never revisit a completed tree.

Implement a simple schedule interpreter before lowering this plan. It uses real predicate and consumer types and must match the independent tree-order reference.

## Dedicated checks

- Every original tree/level receives exactly one predicate result at its original bit position.
- Every tree retires exactly once, after all its bits, and releases its active slot.
- Active-tree count never exceeds the cap; comparison results are used only after production and before release.
- Deterministic schedules for equal scores; no dependence on hash iteration order.
- Hand-derived cases for exact-predicate sharing, same-feature/different-threshold load sharing, and repeated features within a tree.
- A case where one tree completes while another remains active: verify replacement admission happens before that other tree finishes.
- A case where replacement benefits from the retained feature, and another where a discarded predicate must be recomputed.
- Random small forests: reference versus interpreted schedule with exact leaf-index agreement.

## Slice: scheduled scalar codegen

Lower the validated plan through the same ownership and memory contract as the baseline JIT. Avoid a second semantics implementation for NaNs or leaf addressing. Keep reference evaluation independent.

Record logical peak live values, comparison/load counts, actual stack traffic, latency, compile time, and code size. Compare live caps such as 1/2/4/8 on the same workloads; choose one simple default from evidence without adding per-CPU planning.

Gate: exact leaf agreement; demonstrable shared loads/comparisons and rolling refill; repeatable latency benefit on reuse-bearing models; regressions on low-reuse/large models explained and resolved before calling the policy finished.
