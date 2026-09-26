use super::*;
use crate::parsed_tree::{NanMode, SingleTree};

#[test]
fn default_uses_twelve_register_budgets_scalar_and_two_sums() {
    let options = SortOptions::default();
    assert!(matches!(options.order, Order::Scheduled(b) if b == budget(12, 12)));
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

fn budget(general: usize, floating: usize) -> RegisterBudget {
    RegisterBudget {
        general: NonZeroUsize::new(general).unwrap(),
        floating: NonZeroUsize::new(floating).unwrap(),
    }
}

fn check<'model>(trees: &'model Trees, general: usize, row: &[f64]) -> Plan<'model> {
    check_mode(trees, general, row, ComparisonMode::Scalar)
}

fn check_mode<'model>(
    trees: &'model Trees,
    general: usize,
    row: &[f64],
    mode: ComparisonMode,
) -> Plan<'model> {
    check_budget(trees, budget(general, 8), row, mode)
}

fn check_budget<'model>(
    trees: &'model Trees,
    budget: RegisterBudget,
    row: &[f64],
    mode: ComparisonMode,
) -> Plan<'model> {
    let plan = plan(
        trees,
        SortOptions {
            order: Order::Scheduled(budget),
            accumulators: NonZeroUsize::new(1).unwrap(),
            comparisons: mode,
        },
    )
    .unwrap();
    replay(&plan, Some(budget), row);
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
    .unwrap()
}

fn peak(budget: Option<RegisterBudget>, general: usize, floating: usize) {
    if let Some(budget) = budget {
        assert!(
            general <= budget.general.get(),
            "GP peak {general} exceeds {}",
            budget.general
        );
        assert!(
            floating <= budget.floating.get(),
            "FP peak {floating} exceeds {}",
            budget.floating
        );
    }
}

fn replay(plan: &Plan<'_>, budget: Option<RegisterBudget>, row: &[f64]) -> f64 {
    let trees = plan.model();
    let mut state = vec![0; trees.trees().len()];
    let mut seen = vec![0usize; state.len()];
    let mut indices = vec![0usize; state.len()];
    let mut active = 0;
    let mut loaded: Vec<Option<(FeatureIndex, f64)>> = Vec::new();
    let mut sums = vec![0.0; plan.sum_count()];
    let mut prediction = None;
    for action in plan.actions() {
        assert!(prediction.is_none(), "command after return");
        let floating = sums.len() + loaded.iter().filter(|v| v.is_some()).count();
        match action {
            Action::Admit(TreeId(tree)) => {
                assert_eq!(state[*tree], 0, "tree admitted twice");
                state[*tree] = 1;
                active += usize::from(!trees.trees()[*tree].selections.is_empty());
                peak(budget, 1 + active + 2, floating + 1);
            }
            Action::Load { value: id, feature } => {
                let normalize = usize::from(trees.feature_modes()[feature.0] != NanMode::AsIs);
                peak(budget, 1 + active + 1 + normalize, floating + 1 + normalize);
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
                // Scalar: threshold. Pair: broadcast plus threshold overwritten by mask.
                peak(
                    budget,
                    1 + active + 1 + comparisons.len(),
                    floating + comparisons.len(),
                );
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
                // The selector remains live through address formation and leaf load.
                peak(budget, 1 + active + 2, floating + 1);
                assert_eq!(state[*tree], 1);
                assert_eq!(
                    seen[*tree],
                    (1 << trees.trees()[*tree].selections.len()) - 1
                );
                state[*tree] = 2;
                active -= usize::from(!trees.trees()[*tree].selections.is_empty());
                sums[sum.0] += trees.trees()[*tree].leafs[indices[*tree]];
            }
            Action::AddSums {
                destination,
                source,
            } => {
                peak(budget, 1, floating);
                sums[destination.0] += sums[source.0];
            }
            Action::ReturnScaled(sum) => {
                peak(budget, 1, floating + 1);
                prediction = Some(sums[sum.0] * trees.scale() + trees.bias());
            }
            Action::ReturnBias => {
                peak(budget, 0, 1);
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
    let plan = check(&trees, 6, &[0.5; 4]);
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
fn later_admissions_recompute_a_shared_predicate_without_reloading_the_feature() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0)]),
        tree(&[(0, 0.0)]),
        tree(&[(0, 0.0)]),
    ]);
    let plan = check(&trees, 5, &[1.0; 4]);
    let consumers: Vec<_> = plan
        .actions()
        .iter()
        .filter_map(|action| match action {
            Action::Compare { comparisons, .. } if comparisons[0].predicate == PredicateId(0) => {
                Some(
                    comparisons[0]
                        .consumers
                        .iter()
                        .map(|c| c.tree)
                        .collect::<Vec<_>>(),
                )
            }
            _ => None,
        })
        .collect();
    assert_eq!(consumers, [vec![TreeId(0), TreeId(1)], vec![TreeId(2)]]);
    assert_eq!(
        plan.actions()
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
    assert_eq!(check(&empty, 4, &[0.0; 4]).actions, [Action::ReturnBias]);
    let constants = forest(vec![tree(&[]), tree(&[]), tree(&[])]);
    assert_eq!(
        check(&constants, 3, &[0.0; 4]).actions,
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
fn new_work_starts_with_the_highest_tree_score_not_the_shortest_waiting_tree() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0), (2, 0.0)]),
        tree(&[(3, 0.0)]),
    ]);
    let plan = check(&trees, 6, &[1.0; 4]);
    assert_eq!(admissions(&plan), [TreeId(0), TreeId(1)]);
    let retire = plan
        .actions
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::Retire {
                    tree: TreeId(0),
                    ..
                }
            )
        })
        .unwrap();
    let start = plan
        .actions
        .iter()
        .position(|a| *a == Action::Admit(TreeId(1)))
        .unwrap();
    assert!(retire < start);
}

#[test]
fn interns_threshold_bits_in_first_occurrence_order() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let trees = forest(vec![tree(&[(0, -0.0), (0, 0.0), (0, nan), (0, nan)])]);
    let plan = check(&trees, 4, &[1.0; 4]);
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
fn deterministic_varied_forests_obey_budgets_and_preserve_leaf_bits() {
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
        for general in [4, 6, 8, 16] {
            for mode in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
                let first = check_mode(&trees, general, &rows[0], mode);
                for row in rows {
                    let next = check_mode(&trees, general, &row, mode);
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
    let plan = check_mode(&trees, 6, &[0.5; 4], ComparisonMode::Simd2);
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
        replay(&tree_order_plan(&trees), None, &row);
        for mode in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
            let plan = check_budget(&trees, budget(4, 3), &row, mode);
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
    replay(&plan, None, &[1.0; 4]);
}

#[test]
#[should_panic(expected = "comparison requires a live feature value")]
fn replay_rejects_comparison_after_release() {
    let trees = forest(vec![tree(&[(0, 0.0)])]);
    let mut plan = tree_order_plan(&trees);
    plan.actions.swap(2, 3);
    replay(&plan, None, &[1.0; 4]);
}

#[test]
#[should_panic(expected = "feature released twice")]
fn replay_rejects_double_release() {
    let trees = forest(vec![tree(&[(0, 0.0)])]);
    let mut plan = tree_order_plan(&trees);
    plan.actions.insert(4, Action::Release(FeatureValueId(0)));
    replay(&plan, None, &[1.0; 4]);
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
    let first = plan(&trees, options).unwrap();
    let next = plan(&trees, options).unwrap();
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
    assert_eq!(replay(&first, None, &[0.0; 4]), 59.0);
}

#[test]
fn scheduled_sum_destinations_follow_completion_order_not_tree_ids() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0), (2, 0.0)]),
        tree(&[(0, 0.0)]),
    ]);
    let options = SortOptions {
        order: Order::Scheduled(budget(6, 8)),
        accumulators: NonZeroUsize::new(4).unwrap(),
        comparisons: ComparisonMode::Simd2,
    };
    let first = plan(&trees, options).unwrap();
    assert_eq!(first.actions(), plan(&trees, options).unwrap().actions());
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
    assert_eq!(replay(&first, Some(budget(6, 8)), &[1.0; 4]), 19.0);
}

#[test]
fn empty_plan_returns_bias_without_scaling_or_reduction() {
    let trees = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    let plan = plan(&trees, SortOptions::default()).unwrap();
    assert_eq!(plan.actions(), [Action::ReturnBias]);
    assert_eq!(replay(&plan, None, &[]).to_bits(), (-0.0f64).to_bits());
}

fn admissions(plan: &Plan<'_>) -> Vec<TreeId> {
    plan.actions()
        .iter()
        .filter_map(|action| match action {
            Action::Admit(tree) => Some(*tree),
            _ => None,
        })
        .collect()
}

fn widths(plan: &Plan<'_>) -> Vec<usize> {
    plan.actions()
        .iter()
        .filter_map(|action| match action {
            Action::Compare { comparisons, .. } => Some(comparisons.len()),
            _ => None,
        })
        .collect()
}

#[test]
fn insufficient_budgets_are_rejected_but_constants_do_not_reserve_selectors() {
    let split = forest(vec![tree(&[(1, 0.0)])]);
    for registers in [budget(3, 8), budget(8, 2)] {
        let error = plan(
            &split,
            SortOptions {
                order: Order::Scheduled(registers),
                ..SortOptions::default()
            },
        )
        .err()
        .unwrap();
        assert_eq!(error.required, budget(4, 3));
    }
    for mode in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
        check_budget(&split, budget(4, 3), &[f64::NAN; 4], mode);
        check_budget(&forest(vec![tree(&[]); 3]), budget(3, 2), &[0.0; 4], mode);
        check_budget(&forest(vec![]), budget(1, 1), &[0.0; 4], mode);
    }
}

#[test]
fn pair_starts_two_selectors_and_reuses_threshold_storage_for_its_result() {
    let trees = forest(vec![tree(&[(0, 0.0)]), tree(&[(0, 1.0)])]);
    // 2 selectors + row + 3 GP scratch; sum + feature + broadcast + threshold/mask.
    let pair = check_budget(&trees, budget(6, 4), &[0.5; 4], ComparisonMode::Simd2);
    assert_eq!(widths(&pair), [2]);
    assert_eq!(admissions(&pair), [TreeId(0), TreeId(1)]);
    for registers in [budget(4, 4), budget(6, 3)] {
        let scalar = check_budget(&trees, registers, &[0.5; 4], ComparisonMode::Simd2);
        assert_eq!(widths(&scalar), [1, 1]);
    }
    let two_sums = plan(
        &trees,
        SortOptions {
            order: Order::Scheduled(budget(6, 4)),
            accumulators: NonZeroUsize::new(2).unwrap(),
            comparisons: ComparisonMode::Simd2,
        },
    )
    .unwrap();
    replay(&two_sums, Some(budget(6, 4)), &[0.5; 4]);
    assert_eq!(widths(&two_sums), [1, 1]);
}

#[test]
fn selector_pressure_drains_an_existing_short_tail_before_starting_more_work() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 1.0), (2, 2.0)]),
        tree(&[(0, 0.0), (3, 3.0)]),
        tree(&[(0, 0.0)]),
    ]);
    let plan = check_budget(&trees, budget(5, 4), &[2.5; 4], ComparisonMode::Scalar);
    let tail = plan
        .actions()
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::Retire {
                    tree: TreeId(1),
                    ..
                }
            )
        })
        .unwrap();
    let start = plan
        .actions()
        .iter()
        .position(|a| *a == Action::Admit(TreeId(2)))
        .unwrap();
    let long = plan
        .actions()
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::Retire {
                    tree: TreeId(0),
                    ..
                }
            )
        })
        .unwrap();
    assert!(tail < start && start < long);
}

#[test]
fn floating_pressure_releases_a_feature_before_loading_another_despite_free_selectors() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (0, 1.0), (1, 2.0)]),
        tree(&[(0, 3.0), (2, 4.0), (2, 5.0)]),
        tree(&[(0, 6.0), (3, 7.0)]),
    ]);
    let tight = check_budget(&trees, budget(8, 3), &[2.5; 4], ComparisonMode::Scalar);
    let roomy = check_budget(&trees, budget(8, 8), &[2.5; 4], ComparisonMode::Scalar);
    let loads = |plan: &Plan<'_>| {
        plan.actions()
            .iter()
            .filter_map(|a| match a {
                Action::Load {
                    value,
                    feature: FeatureIndex(0),
                } => Some(*value),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let reloads = loads(&tight);
    assert!(reloads.len() > 1);
    assert_eq!(loads(&roomy).len(), 1);
    assert_ne!(reloads[0], reloads[1]);
    let release = tight
        .actions()
        .iter()
        .position(|a| *a == Action::Release(reloads[0]))
        .unwrap();
    let load_b = tight
        .actions()
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::Load {
                    feature: FeatureIndex(1),
                    ..
                }
            )
        })
        .unwrap();
    assert!(release < load_b);
    let live = tight.actions()[..load_b]
        .iter()
        .fold(0usize, |live, action| match action {
            Action::Admit(_) => live + 1,
            Action::Retire { .. } => live - 1,
            _ => live,
        });
    assert!(
        1 + live + 1 + 2 <= 8,
        "another selector fits; FP storage is the constraint"
    );
}

#[test]
fn feature_use_tree_score_and_eligible_overlap_each_affect_real_choices() {
    let scores = forest(vec![tree(&[(0, 0.0)]), tree(&[(1, 1.0), (1, 2.0)])]);
    let scored = check(&scores, 12, &[1.5; 4]);
    assert_eq!(admissions(&scored)[0], TreeId(1));

    let frequency = forest(vec![tree(&[(0, 0.0), (1, 1.0), (1, 2.0)])]);
    let frequent = check(&frequency, 12, &[1.5; 4]);
    assert!(matches!(
        frequent.actions()[0],
        Action::Load {
            feature: FeatureIndex(1),
            ..
        }
    ));

    // Scores are 11, 3, 9. After two B splits, overlap admits tree 1 before tree 2.
    let overlap = forest(vec![
        tree(&[(0, 0.0), (1, 1.0), (1, 2.0), (1, 3.0)]),
        tree(&[(0, 4.0), (2, 5.0)]),
        tree(&[(3, 6.0), (3, 7.0), (3, 8.0)]),
    ]);
    let overlapping = check_mode(&overlap, 12, &[4.5; 4], ComparisonMode::Simd2);
    assert_eq!(admissions(&overlapping), [TreeId(0), TreeId(1), TreeId(2)]);
}

#[test]
fn a_hundred_trees_with_distinct_thresholds_preserve_indices_with_tight_budgets() {
    let trees = Trees::new(
        (0..100)
            .map(|t| {
                tree(
                    &(0..6)
                        .map(|level| {
                            (
                                (t * 37 + level * 13) % 100,
                                (t * 6 + level) as f64 / 128.0 - 2.0,
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
        vec![NanMode::AsIs; 100],
        1.0,
        0.0,
    )
    .unwrap();
    for registers in [budget(4, 3), budget(12, 12)] {
        for mode in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
            let first = check_budget(&trees, registers, &[0.5; 100], mode);
            let next = check_budget(&trees, registers, &[f64::NAN; 100], mode);
            assert_eq!(first.predicates().len(), 600);
            assert_eq!(first.actions(), next.actions());
        }
    }
}

#[test]
fn pressure_advances_the_shortest_selector_not_an_earlier_same_feature_threshold() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (0, 1.0), (0, 2.0), (2, 3.0), (2, 4.0)]),
        tree(&[(1, 5.0), (1, 6.0), (2, 7.0)]),
    ]);
    let plan = check_budget(&trees, budget(5, 4), &[4.5; 4], ComparisonMode::Scalar);
    let a: Vec<_> = plan
        .actions()
        .iter()
        .filter_map(|action| match action {
            Action::Compare { comparisons, .. }
                if plan.predicates()[comparisons[0].predicate.0].feature_idx == FeatureIndex(2) =>
            {
                Some(comparisons[0].consumers[0].tree)
            }
            _ => None,
        })
        .collect();
    assert_eq!(a, [TreeId(1), TreeId(0), TreeId(0)]);
    let retired: Vec<_> = plan
        .actions()
        .iter()
        .filter_map(|a| match a {
            Action::Retire { tree, .. } => Some(*tree),
            _ => None,
        })
        .collect();
    assert_eq!(retired, [TreeId(1), TreeId(0)]);
}

#[test]
fn mixed_pairs_charge_the_union_of_new_selectors_before_retiring_any() {
    for both_lanes in [false, true] {
        let trees = forest(vec![
            tree(&[(0, 0.0), (0, 0.0), (0, 0.0), (1, 3.0)]),
            tree(if both_lanes {
                &[(1, 3.0), (1, 4.0)]
            } else {
                &[(1, 3.0)]
            }),
            tree(&[(1, 4.0)]),
        ]);
        let plan = check_budget(&trees, budget(6, 4), &[3.5; 4], ComparisonMode::Simd2);
        assert_eq!(admissions(&plan), [TreeId(0), TreeId(1), TreeId(2)]);
        assert_eq!(
            widths(&plan),
            if both_lanes {
                vec![1, 2, 1]
            } else {
                vec![1, 1, 1]
            }
        );
        let first_retire = plan
            .actions()
            .iter()
            .position(|a| matches!(a, Action::Retire { .. }))
            .unwrap();
        let third_start = plan
            .actions()
            .iter()
            .position(|a| *a == Action::Admit(TreeId(2)))
            .unwrap();
        assert!(first_retire < third_start);
    }
}
