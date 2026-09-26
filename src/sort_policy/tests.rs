use super::*;
use crate::parsed_tree::{NanMode, SingleTree};

#[test]
fn default_uses_rolling_scalar_with_two_accumulators() {
    let options = SortOptions::default();
    assert!(matches!(options.order, Order::Rolling(cap) if cap.get() == 4));
    assert_eq!(options.accumulators.get(), 2);
    assert!(matches!(options.comparisons, ComparisonMode::Scalar));
}

fn tree(splits: &[(usize, f64)]) -> SingleTree {
    SingleTree {
        selections: splits
            .iter()
            .map(|&(feature, threshold)| Discriminator {
                feature_idx: FeatureIndex(feature),
                threshold,
            })
            .collect(),
        leafs: (0..1usize << splits.len()).map(|i| i as f64).collect(),
    }
}

fn forest(trees: Vec<SingleTree>) -> Trees {
    Trees::new(
        trees,
        vec![
            NanMode::AsIs,
            NanMode::AsFalse,
            NanMode::AsTrue,
            NanMode::AsIs,
        ],
        2.0,
        3.0,
    )
    .unwrap()
}

fn check<'model>(trees: &'model Trees, cap: usize, row: &[f64]) -> Plan<'model> {
    check_mode(trees, cap, row, ComparisonMode::Scalar)
}

fn check_mode<'model>(
    trees: &'model Trees,
    cap: usize,
    row: &[f64],
    mode: ComparisonMode,
) -> Plan<'model> {
    let plan = plan(
        trees,
        SortOptions {
            order: Order::Rolling(NonZeroUsize::new(cap).unwrap()),
            accumulators: NonZeroUsize::new(1).unwrap(),
            comparisons: mode,
        },
    );
    replay(&plan, cap, row);
    plan
}

fn tree_order_plan(trees: &Trees) -> Plan<'_> {
    plan(
        trees,
        SortOptions {
            order: Order::TreeOrder,
            accumulators: NonZeroUsize::new(1).unwrap(),
            comparisons: ComparisonMode::Scalar,
        },
    )
}

fn replay(plan: &Plan<'_>, cap: usize, row: &[f64]) -> f64 {
    let trees = plan.model();
    let mut state = vec![0; trees.trees().len()];
    let mut seen = vec![0usize; state.len()];
    let mut indices = vec![0usize; state.len()];
    let mut active = 0;
    let mut loaded = Vec::new();
    let mut sums = vec![0.0; plan.sum_count()];
    let mut prediction = None;
    for action in plan.actions() {
        assert!(prediction.is_none(), "command after return");
        match action {
            Action::Admit(TreeId(tree)) => {
                assert_eq!(state[*tree], 0, "tree admitted twice");
                state[*tree] = 1;
                active += 1;
                assert!(active <= cap);
            }
            Action::Load { value: id, feature } => {
                let value = row[feature.0];
                let value = match (value.is_nan(), trees.feature_modes()[feature.0]) {
                    (true, NanMode::AsFalse) => f64::NEG_INFINITY,
                    (true, NanMode::AsTrue) => f64::INFINITY,
                    _ => value,
                };
                assert_eq!(id.0, loaded.len(), "feature value ID must be fresh");
                loaded.push(Some((*feature, value)));
            }
            Action::Compare {
                feature,
                comparisons,
            } => {
                let (feature, value) = loaded
                    .get(feature.0)
                    .copied()
                    .flatten()
                    .expect("comparison requires a live feature value");
                assert!((1..=2).contains(&comparisons.len()));
                if comparisons.len() == 2 {
                    assert_ne!(comparisons[0].predicate, comparisons[1].predicate);
                }
                for Comparison {
                    predicate,
                    consumers,
                } in comparisons
                {
                    assert!(
                        !consumers.is_empty(),
                        "comparison must resolve pending work"
                    );
                    let predicate = &plan.predicates()[predicate.0];
                    assert_eq!(feature, predicate.feature_idx);
                    let result = usize::from(value > predicate.threshold);
                    for &Consumer {
                        tree: TreeId(tree),
                        level,
                    } in consumers
                    {
                        assert_eq!(state[tree], 1);
                        let original = &trees.trees()[tree].selections[level];
                        assert_eq!(original.feature_idx, predicate.feature_idx);
                        assert_eq!(original.threshold.to_bits(), predicate.threshold.to_bits());
                        let bit = 1 << level;
                        assert_eq!(seen[tree] & bit, 0, "split consumed twice");
                        seen[tree] |= bit;
                        indices[tree] |= result << level;
                    }
                }
            }
            Action::Release(feature) => {
                loaded
                    .get_mut(feature.0)
                    .expect("feature released before load")
                    .take()
                    .expect("feature released twice");
            }
            Action::Retire {
                tree: TreeId(tree),
                sum,
            } => {
                assert_eq!(state[*tree], 1);
                assert_eq!(
                    seen[*tree],
                    (1 << trees.trees()[*tree].selections.len()) - 1
                );
                state[*tree] = 2;
                active -= 1;
                sums[sum.0] += trees.trees()[*tree].leafs[indices[*tree]];
            }
            Action::AddSums {
                destination,
                source,
            } => {
                sums[destination.0] += sums[source.0];
            }
            Action::ReturnScaled(sum) => {
                prediction = Some(sums[sum.0] * trees.scale() + trees.bias());
            }
            Action::ReturnBias => {
                prediction = Some(trees.bias());
            }
        }
    }
    assert_eq!(active, 0);
    assert!(
        loaded.iter().all(Option::is_none),
        "unreleased feature value"
    );
    assert!(state.iter().all(|&state| state == 2));
    assert_eq!(indices, trees.leaf_indices(row).unwrap());
    let prediction = prediction.expect("plan must return a prediction");
    assert_eq!(prediction, trees.predict(row).unwrap());
    prediction
}

#[test]
fn shares_exact_predicates_and_loads_different_thresholds() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (0, 0.0)]),
        tree(&[(0, 0.0), (0, 1.0)]),
    ]);
    let plan = check(&trees, 2, &[0.5; 4]);
    assert_eq!(plan.predicates.len(), 2);
    assert_eq!(
        plan.actions
            .iter()
            .filter(|a| matches!(a, Action::Load { .. }))
            .count(),
        1
    );
    let comparisons: Vec<_> = plan
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::Compare { comparisons, .. } => {
                assert_eq!(comparisons.len(), 1);
                Some((comparisons[0].predicate, comparisons[0].consumers.len()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(comparisons, [(PredicateId(0), 3), (PredicateId(1), 1)]);
}

#[test]
fn replaces_finished_tree_and_recomputes_discarded_predicate_without_reload() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0)]),
        tree(&[(0, 0.0)]),
        tree(&[(0, 0.0)]),
    ]);
    let plan = check(&trees, 2, &[1.0; 4]);
    let retire_short = plan
        .actions
        .iter()
        .position(|a| {
            *a == Action::Retire {
                tree: TreeId(1),
                sum: SumId(0),
            }
        })
        .unwrap();
    assert_eq!(plan.actions[retire_short + 1], Action::Admit(TreeId(2)));
    assert_eq!(
        plan.actions[retire_short + 2],
        Action::Compare {
            feature: FeatureValueId(0),
            comparisons: vec![Comparison {
                predicate: PredicateId(0),
                consumers: vec![Consumer {
                    tree: TreeId(2),
                    level: 0
                }],
            }],
        }
    );
    let retire_long = plan
        .actions
        .iter()
        .position(|a| {
            *a == Action::Retire {
                tree: TreeId(0),
                sum: SumId(0),
            }
        })
        .unwrap();
    assert!(retire_short + 2 < retire_long);
    assert_eq!(
        plan.actions
            .iter()
            .filter(|a| matches!(
                a,
                Action::Load {
                    feature: FeatureIndex(0),
                    ..
                }
            ))
            .count(),
        1
    );
}

#[test]
fn empty_and_constant_trees_need_no_feature_loads() {
    let empty = forest(vec![]);
    assert_eq!(check(&empty, 1, &[0.0; 4]).actions, [Action::ReturnBias]);
    let constants = forest(vec![tree(&[]), tree(&[]), tree(&[])]);
    assert_eq!(
        check(&constants, 1, &[0.0; 4]).actions,
        [
            Action::Admit(TreeId(0)),
            Action::Retire {
                tree: TreeId(0),
                sum: SumId(0)
            },
            Action::Admit(TreeId(1)),
            Action::Retire {
                tree: TreeId(1),
                sum: SumId(0)
            },
            Action::Admit(TreeId(2)),
            Action::Retire {
                tree: TreeId(2),
                sum: SumId(0)
            },
            Action::ReturnScaled(SumId(0)),
        ]
    );
}

#[test]
fn without_shared_uses_finishes_shortest_tree_first() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0), (2, 0.0)]),
        tree(&[(3, 0.0)]),
    ]);
    let plan = check(&trees, 2, &[1.0; 4]);
    assert_eq!(plan.actions[0], Action::Admit(TreeId(0)));
    assert_eq!(
        plan.actions[2],
        Action::Load {
            value: FeatureValueId(0),
            feature: FeatureIndex(3),
        }
    );
    assert_eq!(
        plan.actions[4],
        Action::Retire {
            tree: TreeId(1),
            sum: SumId(0)
        }
    );
}

#[test]
fn interns_threshold_bits_in_first_occurrence_order() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let trees = forest(vec![tree(&[(0, -0.0), (0, 0.0), (0, nan), (0, nan)])]);
    let plan = check(&trees, 1, &[1.0; 4]);
    let bits: Vec<_> = plan
        .predicates
        .iter()
        .map(|p| p.threshold.to_bits())
        .collect();
    assert_eq!(
        bits,
        [(-0.0_f64).to_bits(), 0.0_f64.to_bits(), nan.to_bits()]
    );
}

#[test]
fn deterministic_varied_forests_obey_caps_and_preserve_leaf_bits() {
    let rows = [
        [-1.0, 0.0, 1.0, 2.0],
        [0.0, -0.0, 0.0, 0.0],
        [f64::NAN; 4],
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
    ];
    for variant in 0..12 {
        let trees = forest(
            (0..8)
                .map(|index| {
                    let splits: Vec<_> = (0..(index + variant) % 5)
                        .map(|level| ((index + level * 3 + variant) % 4, (level % 3) as f64 - 1.0))
                        .collect();
                    tree(&splits)
                })
                .collect(),
        );
        for cap in [1, 2, 4, 16] {
            for mode in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
                let first = check_mode(&trees, cap, &rows[0], mode);
                for row in rows {
                    let next = check_mode(&trees, cap, &row, mode);
                    assert_eq!(first.actions, next.actions);
                }
            }
        }
    }
}

#[test]
fn pair_fanout_retires_both_trees_before_refill_and_leaves_odd_tail_scalar() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (0, 1.0)]),
        tree(&[(0, 1.0), (0, 0.0)]),
        tree(&[(0, 2.0)]),
    ]);
    let plan = check_mode(&trees, 2, &[0.5; 4], ComparisonMode::Simd2);
    let Action::Compare {
        comparisons: pair, ..
    } = &plan.actions[3]
    else {
        panic!("expected pair")
    };
    assert_eq!(pair.len(), 2);
    assert_eq!(
        pair[0].consumers,
        [
            Consumer {
                tree: TreeId(0),
                level: 0
            },
            Consumer {
                tree: TreeId(1),
                level: 1
            }
        ]
    );
    assert_eq!(
        pair[1].consumers,
        [
            Consumer {
                tree: TreeId(0),
                level: 1
            },
            Consumer {
                tree: TreeId(1),
                level: 0
            }
        ]
    );
    assert_eq!(
        plan.actions[4..7],
        [
            Action::Retire {
                tree: TreeId(0),
                sum: SumId(0)
            },
            Action::Retire {
                tree: TreeId(1),
                sum: SumId(0)
            },
            Action::Admit(TreeId(2))
        ]
    );
    let Action::Compare {
        comparisons: tail, ..
    } = &plan.actions[7]
    else {
        panic!("expected scalar tail")
    };
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].predicate, PredicateId(2));
    assert_eq!(
        plan.actions
            .iter()
            .filter(|action| matches!(action, Action::Load { .. }))
            .count(),
        1
    );
}

#[test]
fn released_features_reload_as_fresh_values_in_both_orders() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (0, 1.0), (1, 0.0), (2, 0.0)]),
        tree(&[(0, 2.0), (0, 3.0), (1, 1.0), (2, 1.0)]),
    ]);
    for row in [[0.5; 4], [f64::NAN; 4]] {
        replay(&tree_order_plan(&trees), 1, &row);
        for mode in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
            let plan = check_mode(&trees, 1, &row, mode);
            let loads: Vec<_> = plan
                .actions
                .iter()
                .enumerate()
                .filter_map(|(position, action)| match action {
                    Action::Load {
                        value,
                        feature: FeatureIndex(0),
                    } => Some((position, *value)),
                    _ => None,
                })
                .collect();
            assert_eq!(loads.len(), 2);
            assert_ne!(loads[0].1, loads[1].1);
            assert!(
                plan.actions[loads[0].0 + 1..loads[1].0].contains(&Action::Release(loads[0].1))
            );
        }
    }
}

#[test]
#[should_panic(expected = "comparison requires a live feature value")]
fn replay_rejects_comparison_before_load() {
    let trees = forest(vec![tree(&[(0, 0.0)])]);
    let mut plan = tree_order_plan(&trees);
    plan.actions.swap(1, 2);
    replay(&plan, 1, &[1.0; 4]);
}

#[test]
#[should_panic(expected = "comparison requires a live feature value")]
fn replay_rejects_comparison_after_release() {
    let trees = forest(vec![tree(&[(0, 0.0)])]);
    let mut plan = tree_order_plan(&trees);
    plan.actions.swap(2, 3);
    replay(&plan, 1, &[1.0; 4]);
}

#[test]
#[should_panic(expected = "feature released twice")]
fn replay_rejects_double_release() {
    let trees = forest(vec![tree(&[(0, 0.0)])]);
    let mut plan = tree_order_plan(&trees);
    plan.actions.insert(4, Action::Release(FeatureValueId(0)));
    replay(&plan, 1, &[1.0; 4]);
}

#[test]
fn plan_assigns_constant_leaves_to_sums_and_orders_the_final_reduction() {
    let trees = forest(
        [2.0, 3.0, 5.0, 7.0, 11.0]
            .into_iter()
            .map(|value| SingleTree {
                leafs: vec![value],
                selections: vec![],
            })
            .collect(),
    );
    let options = SortOptions {
        order: Order::TreeOrder,
        accumulators: NonZeroUsize::new(3).unwrap(),
        comparisons: ComparisonMode::Scalar,
    };
    let first = plan(&trees, options);
    let next = plan(&trees, options);
    assert_eq!(first.actions(), next.actions());
    assert_eq!(first.sum_count(), 3);
    let destinations: Vec<_> = first
        .actions()
        .iter()
        .filter_map(|action| match action {
            Action::Retire { tree, sum } => Some((*tree, *sum)),
            _ => None,
        })
        .collect();
    assert_eq!(
        destinations,
        [
            (TreeId(0), SumId(0)),
            (TreeId(1), SumId(1)),
            (TreeId(2), SumId(2)),
            (TreeId(3), SumId(0)),
            (TreeId(4), SumId(1)),
        ]
    );
    assert_eq!(
        &first.actions()[first.actions().len() - 3..],
        [
            Action::AddSums {
                destination: SumId(0),
                source: SumId(1)
            },
            Action::AddSums {
                destination: SumId(0),
                source: SumId(2)
            },
            Action::ReturnScaled(SumId(0)),
        ]
    );
    assert_eq!(replay(&first, 1, &[0.0; 4]), 59.0);
}

#[test]
fn rolling_sum_destinations_follow_completion_order_not_tree_ids() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0), (2, 0.0)]),
        tree(&[(3, 0.0)]),
    ]);
    let options = SortOptions {
        order: Order::Rolling(NonZeroUsize::new(2).unwrap()),
        accumulators: NonZeroUsize::new(4).unwrap(),
        comparisons: ComparisonMode::Simd2,
    };
    let first = plan(&trees, options);
    assert_eq!(first.actions(), plan(&trees, options).actions());
    assert_eq!(first.sum_count(), 2);
    let destinations: Vec<_> = first
        .actions()
        .iter()
        .filter_map(|action| match action {
            Action::Retire { tree, sum } => Some((*tree, *sum)),
            _ => None,
        })
        .collect();
    assert_eq!(destinations, [(TreeId(1), SumId(0)), (TreeId(0), SumId(1))]);
    assert_eq!(replay(&first, 2, &[1.0; 4]), 19.0);
}

#[test]
fn empty_plan_returns_bias_without_scaling_or_reduction() {
    let trees = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    let plan = plan(&trees, SortOptions::default());
    assert_eq!(plan.actions(), [Action::ReturnBias]);
    assert_eq!(replay(&plan, 0, &[]).to_bits(), (-0.0f64).to_bits());
}
