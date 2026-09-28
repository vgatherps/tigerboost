use super::*;
use crate::parsed_tree::{FeatureIndex, NanMode, SingleTree};

#[test]
fn default_uses_eight_trees_per_chunk_and_two_sums() {
    let options = SortOptions::default();
    assert_eq!(options.chunk_trees.get(), 8);
    assert_eq!(options.accumulators.get(), 2);
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

fn check<'model>(trees: &'model Trees, row: &[f64]) -> Plan<'model> {
    let plan = plan(
        trees,
        SortOptions {
            accumulators: NonZeroUsize::new(1).unwrap(),
            ..SortOptions::default()
        },
    );
    replay(&plan, row);
    plan
}

fn replay(plan: &Plan<'_>, row: &[f64]) -> f64 {
    let trees = plan.model();
    let mut state = vec![0; trees.trees().len()];
    let mut indices = vec![0usize; state.len()];
    let mut sums = vec![0.0; plan.sum_count()];
    let mut prediction = None;
    for action in plan.actions() {
        assert!(prediction.is_none(), "command after return");
        match action {
            Action::Admit(TreeId(tree)) => {
                assert_eq!(state[*tree], 0, "tree admitted twice");
                state[*tree] = 1;
                assert!(trees.trees()[*tree].selections.is_empty());
            }
            Action::Retire {
                tree: TreeId(tree),
                sum,
            } => {
                assert_eq!(state[*tree], 1);
                assert!(trees.trees()[*tree].selections.is_empty());
                state[*tree] = 2;
                sums[sum.0] += trees.trees()[*tree].leafs[indices[*tree]];
            }
            Action::EvaluateTree {
                tree: TreeId(tree),
                predicates,
                sum,
            } => {
                assert_eq!(state[*tree], 0, "tree evaluated twice");
                let description = &trees.trees()[*tree];
                assert_eq!(
                    predicates.len(),
                    description.selections.len().div_ceil(2) * 2
                );
                for (level, predicate) in predicates
                    .iter()
                    .take(description.selections.len())
                    .enumerate()
                {
                    let predicate = &plan.predicates()[predicate.0];
                    let original = &description.selections[level];
                    assert_eq!(predicate.feature_idx, original.feature_idx);
                    assert_eq!(predicate.threshold.to_bits(), original.threshold.to_bits());
                    let value = row[predicate.feature_idx.0];
                    let value = match (
                        value.is_nan(),
                        trees.feature_modes()[predicate.feature_idx.0],
                    ) {
                        (true, NanMode::AsFalse) => f64::NEG_INFINITY,
                        (true, NanMode::AsTrue) => f64::INFINITY,
                        _ => value,
                    };
                    indices[*tree] |= usize::from(value > predicate.threshold) << level;
                }
                state[*tree] = 2;
                sums[sum.0] += description.leafs[indices[*tree]];
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
    assert!(state.iter().all(|&state| state == 2));
    assert_eq!(indices, trees.leaf_indices(row).unwrap());
    let prediction = prediction.expect("plan must return a prediction");
    assert_eq!(prediction, trees.predict(row).unwrap());
    prediction
}

#[test]
fn tree_major_replay_matches_reference() {
    let trees = forest(vec![
        tree(&[]),
        tree(&[(1, 0.0)]),
        tree(&[(2, 0.0), (0, 1.0), (2, -1.0)]),
    ]);
    let plan = plan(
        &trees,
        SortOptions {
            chunk_trees: NonZeroUsize::new(2).unwrap(),
            ..SortOptions::default()
        },
    );
    for row in [[0.0; 4], [2.0; 4], [f64::NAN; 4], [f64::NEG_INFINITY; 4]] {
        assert_eq!(replay(&plan, &row), trees.predict(&row).unwrap());
    }
}

#[test]
fn empty_and_constant_trees_need_no_feature_loads() {
    let empty = forest(vec![]);
    assert_eq!(check(&empty, &[0.0; 4]).actions, [Action::ReturnBias]);
    let constants = forest(vec![tree(&[]), tree(&[]), tree(&[])]);
    assert_eq!(
        check(&constants, &[0.0; 4]).actions,
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
        accumulators: NonZeroUsize::new(3).unwrap(),
        ..SortOptions::default()
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
    assert_eq!(replay(&first, &[0.0; 4]), 59.0);
}

#[test]
fn empty_plan_returns_bias_without_scaling_or_reduction() {
    let trees = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    let plan = plan(&trees, SortOptions::default());
    assert_eq!(plan.actions(), [Action::ReturnBias]);
    assert_eq!(replay(&plan, &[]).to_bits(), (-0.0f64).to_bits());
}

#[test]
fn plan_is_deterministic_and_chunks_whole_trees() {
    let mut random = 0x4d59_5df4_d0f3_3173u64;
    let mut next = |limit: usize| {
        random ^= random >> 12;
        random ^= random << 25;
        random ^= random >> 27;
        (random.wrapping_mul(0x2545_f491_4f6c_dd1d) % limit as u64) as usize
    };
    for case in 0..96 {
        let trees = forest(
            (0..1 + next(40))
                .map(|_| {
                    let splits: Vec<_> = (0..next(9))
                        .map(|_| (next(4), next(9) as f64 - 4.0))
                        .collect();
                    tree(&splits)
                })
                .collect(),
        );
        let chunk_trees = [1, 3, 8][case % 3];
        let options = SortOptions {
            chunk_trees: NonZeroUsize::new(chunk_trees).unwrap(),
            accumulators: NonZeroUsize::new([1, 2, 4][case / 3 % 3]).unwrap(),
        };
        let first = plan(&trees, options);
        let second = plan(&trees, options);
        assert_eq!(first.actions(), second.actions());
        assert_eq!(first.chunks(), second.chunks());
        assert_eq!(first.sum_count(), second.sum_count());
        assert_eq!(
            first.sum_count(),
            options.accumulators.get().min(trees.trees().len())
        );
        assert_eq!(first.predicates().len(), second.predicates().len());
        for (a, b) in first.predicates().iter().zip(second.predicates()) {
            assert_eq!(a.feature_idx, b.feature_idx);
            assert_eq!(a.threshold.to_bits(), b.threshold.to_bits());
        }
        assert_eq!(
            first.chunks().len(),
            trees.trees().len().div_ceil(chunk_trees)
        );
        let mut seen = vec![false; trees.trees().len()];
        let mut next_action = 0;
        let mut next_predicate = 0;
        for (number, chunk) in first.chunks().iter().enumerate() {
            assert_eq!(chunk.actions.start, next_action);
            next_action = chunk.actions.end;
            let mut chunk_tests = 0;
            let mut chunk_ids = Vec::new();
            let mut admitted = None;
            for action in &first.actions()[chunk.actions.clone()] {
                let (tree, sum) = match action {
                    Action::Admit(tree) => {
                        assert!(admitted.replace(*tree).is_none());
                        assert!(trees.trees()[tree.0].selections.is_empty());
                        continue;
                    }
                    Action::Retire { tree, sum } => {
                        assert_eq!(admitted.take(), Some(*tree));
                        (*tree, *sum)
                    }
                    Action::EvaluateTree {
                        tree,
                        predicates,
                        sum,
                    } => {
                        assert!(admitted.is_none());
                        let description = &trees.trees()[tree.0];
                        let depth = description.selections.len();
                        assert_ne!(depth, 0);
                        assert_eq!(predicates.len(), depth + depth % 2);
                        for (level, predicate) in predicates.iter().enumerate() {
                            assert_eq!(*predicate, PredicateId(next_predicate));
                            next_predicate += 1;
                            let actual = &first.predicates()[predicate.0];
                            if level < depth {
                                let expected = &description.selections[level];
                                assert_eq!(actual.feature_idx, expected.feature_idx);
                                assert_eq!(
                                    actual.threshold.to_bits(),
                                    expected.threshold.to_bits()
                                );
                            } else {
                                assert_eq!(depth % 2, 1);
                                assert_eq!(level, depth);
                                assert_eq!(
                                    actual.feature_idx,
                                    description.selections[0].feature_idx
                                );
                                assert_eq!(actual.threshold, f64::INFINITY);
                            }
                        }
                        chunk_tests += predicates.len();
                        (*tree, *sum)
                    }
                    Action::AddSums { .. } | Action::ReturnScaled(_) | Action::ReturnBias => {
                        assert!(admitted.is_none());
                        assert_eq!(number + 1, first.chunks().len());
                        continue;
                    }
                };
                assert!(!std::mem::replace(&mut seen[tree.0], true));
                assert_eq!(sum, SumId(tree.0 % first.sum_count()));
                chunk_ids.push(tree.0);
            }
            assert!(admitted.is_none(), "chunk splits a constant tree");
            assert_eq!(
                chunk_ids,
                (number * chunk_trees..((number + 1) * chunk_trees).min(trees.trees().len()))
                    .collect::<Vec<_>>()
            );
            assert_eq!(chunk.tests, chunk_tests);
        }
        assert_eq!(next_action, first.actions().len());
        assert_eq!(next_predicate, first.predicates().len());
        assert!(seen.into_iter().all(|seen| seen));
        for row in [[0.0; 4], [2.0; 4], [f64::NAN; 4], [f64::INFINITY; 4]] {
            replay(&first, &row);
        }
    }
}
