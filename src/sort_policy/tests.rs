use super::*;
use crate::jit::compile_plan;
use crate::parsed_tree::{FeatureIndex, NanMode, SingleTree};
use inkwell::context::Context;

#[test]
fn default_uses_eight_trees_per_chunk_and_two_sums() {
    let trees = forest((0..17).map(|_| tree(&[(0, 0.0)])).collect());
    let plan = plan(&trees);
    assert_eq!(plan.sum_count(), 2);
    assert_eq!(
        plan.chunks()
            .iter()
            .map(|chunk| chunk.actions.clone())
            .collect::<Vec<_>>(),
        [0..8, 8..16, 16..17]
    );
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
    let plan = plan_with(trees, 8, 1);
    replay(&plan, row);
    plan
}

fn replay(plan: &Plan<'_>, row: &[f64]) -> f64 {
    let trees = plan.model();
    let mut state = vec![0; trees.trees().len()];
    let mut indices = vec![0usize; state.len()];
    let mut sums = vec![0.0; plan.sum_count()];
    for action in plan.actions() {
        match action {
            Action::Constant {
                tree: TreeId(tree),
                sum,
            } => {
                assert_eq!(state[*tree], 0, "tree evaluated twice");
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
                    .clone()
                    .take(description.selections.len())
                    .enumerate()
                {
                    let predicate = &plan.predicates()[predicate];
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
        }
    }
    assert!(state.iter().all(|&state| state == 2));
    assert_eq!(indices, trees.leaf_indices(row).unwrap());
    let prediction = if trees.trees().is_empty() {
        trees.bias()
    } else {
        for source in 1..sums.len() {
            sums[0] += sums[source];
        }
        sums[0] * trees.scale() + trees.bias()
    };
    assert_eq!(prediction, trees.predict(row).unwrap());
    prediction
}

#[test]
fn plan_replay_matches_reference() {
    let trees = forest(vec![
        tree(&[]),
        tree(&[(1, 0.0)]),
        tree(&[(2, 0.0), (0, 1.0), (2, -1.0)]),
    ]);
    let plan = plan_with(&trees, 2, 2);
    for row in [[0.0; 4], [2.0; 4], [f64::NAN; 4], [f64::NEG_INFINITY; 4]] {
        assert_eq!(replay(&plan, &row), trees.predict(&row).unwrap());
    }
}

#[test]
fn shared_features_replay_matches_reference() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 1.0)]),
        tree(&[]),
        tree(&[(2, 0.0), (0, 1.0), (2, -1.0)]),
        tree(&[(0, -2.0), (1, 2.0)]),
    ]);
    let plan = plan_with(&trees, 2, 2);
    for row in [[0.0; 4], [2.0; 4], [f64::NAN; 4], [f64::NEG_INFINITY; 4]] {
        assert_eq!(replay(&plan, &row), trees.predict(&row).unwrap());
    }
}

#[test]
fn empty_and_constant_trees_need_no_feature_loads() {
    let empty = forest(vec![]);
    assert!(check(&empty, &[0.0; 4]).actions.is_empty());
    let constants = forest(vec![tree(&[]), tree(&[]), tree(&[])]);
    assert_eq!(
        check(&constants, &[0.0; 4]).actions,
        [
            Action::Constant {
                tree: TreeId(0),
                sum: SumId(0)
            },
            Action::Constant {
                tree: TreeId(1),
                sum: SumId(0)
            },
            Action::Constant {
                tree: TreeId(2),
                sum: SumId(0)
            },
        ]
    );
}

#[test]
fn plan_assigns_constant_leaves_to_sums() {
    let trees = forest(
        [2.0, 3.0, 5.0, 7.0, 11.0]
            .into_iter()
            .map(|value| SingleTree {
                leafs: vec![value],
                selections: vec![],
            })
            .collect(),
    );
    let first = plan_with(&trees, 8, 3);
    let next = plan_with(&trees, 8, 3);
    assert_eq!(first.actions(), next.actions());
    assert_eq!(first.sum_count(), 3);
    let destinations: Vec<_> = first
        .actions()
        .iter()
        .filter_map(|action| match action {
            Action::Constant { tree, sum } => Some((*tree, *sum)),
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
    assert_eq!(first.actions().len(), trees.trees().len());
    assert_eq!(replay(&first, &[0.0; 4]), 59.0);
}

#[test]
fn empty_plan_returns_bias_without_scaling_or_reduction() {
    let trees = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    let plan = plan(&trees);
    assert!(plan.actions().is_empty());
    assert_eq!(replay(&plan, &[]).to_bits(), (-0.0f64).to_bits());
}

fn visit_order(plan: &Plan<'_>) -> Vec<TreeId> {
    plan.actions()
        .iter()
        .map(|action| match action {
            Action::Constant { tree, .. } | Action::EvaluateTree { tree, .. } => *tree,
        })
        .collect()
}

#[test]
fn shared_features_order_is_a_permutation_and_deterministic() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0)]),
        tree(&[]),
        tree(&[(2, 0.0), (2, 1.0)]),
        tree(&[(2, 1.0), (3, 0.0)]),
        tree(&[]),
        tree(&[(0, 1.0), (3, 1.0)]),
    ]);
    let first = plan(&trees);
    let second = plan(&trees);
    assert_eq!(first.actions(), second.actions());
    assert_eq!(first.chunks(), second.chunks());
    let order = visit_order(&first);
    assert_eq!(order.len(), trees.trees().len());
    let mut sorted = order.clone();
    sorted.sort_by_key(|tree| tree.0);
    assert_eq!(
        sorted,
        (0..trees.trees().len()).map(TreeId).collect::<Vec<_>>()
    );
    assert_eq!(
        order,
        [
            TreeId(2),
            TreeId(3),
            TreeId(5),
            TreeId(0),
            TreeId(1),
            TreeId(4)
        ]
    );
    let sums: Vec<_> = first
        .actions()
        .iter()
        .map(|action| match action {
            Action::Constant { sum, .. } | Action::EvaluateTree { sum, .. } => *sum,
        })
        .collect();
    assert_eq!(
        sums,
        [SumId(0), SumId(1), SumId(0), SumId(1), SumId(0), SumId(1)]
    );
}

#[test]
fn partition_trees_counts_visit_order() {
    let trees = forest(vec![
        tree(&[(0, 0.0), (1, 0.0)]),
        tree(&[]),
        tree(&[(2, 0.0), (3, 0.0)]),
        tree(&[(0, 1.0), (1, 1.0)]),
    ]);
    let predicates = trees
        .trees()
        .iter()
        .flat_map(|tree| tree.selections.iter().cloned())
        .collect();
    let plan = Plan {
        model: &trees,
        predicates,
        actions: vec![
            Action::EvaluateTree {
                tree: TreeId(3),
                predicates: 0..2,
                sum: SumId(0),
            },
            Action::Constant {
                tree: TreeId(1),
                sum: SumId(1),
            },
            Action::EvaluateTree {
                tree: TreeId(0),
                predicates: 2..4,
                sum: SumId(0),
            },
            Action::EvaluateTree {
                tree: TreeId(2),
                predicates: 4..6,
                sum: SumId(1),
            },
        ],
        sum_count: 2,
        chunks: Vec::new(),
    };
    assert_eq!(
        continuation::partition_trees(&plan, NonZeroUsize::new(2).unwrap()),
        [Chunk { actions: 0..2 }, Chunk { actions: 2..4 },]
    );
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
        let sums = [1, 2, 4][case / 3 % 3];
        let first = plan_with(&trees, chunk_trees, sums);
        let second = plan_with(&trees, chunk_trees, sums);
        assert_eq!(first.actions(), second.actions());
        assert_eq!(first.chunks(), second.chunks());
        assert_eq!(first.sum_count(), second.sum_count());
        assert_eq!(first.sum_count(), sums.min(trees.trees().len()));
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
        let mut visit_position = 0;
        for (number, chunk) in first.chunks().iter().enumerate() {
            assert_eq!(chunk.actions.start, next_action);
            next_action = chunk.actions.end;
            let mut chunk_ids = Vec::new();
            for action in &first.actions()[chunk.actions.clone()] {
                let (tree, sum) = match action {
                    Action::Constant { tree, sum } => {
                        assert!(trees.trees()[tree.0].selections.is_empty());
                        (*tree, *sum)
                    }
                    Action::EvaluateTree {
                        tree,
                        predicates,
                        sum,
                    } => {
                        let description = &trees.trees()[tree.0];
                        let depth = description.selections.len();
                        assert_ne!(depth, 0);
                        assert_eq!(predicates.len(), depth + depth % 2);
                        for (level, predicate) in predicates.clone().enumerate() {
                            assert_eq!(predicate, next_predicate);
                            next_predicate += 1;
                            let actual = &first.predicates()[predicate];
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
                        (*tree, *sum)
                    }
                };
                assert!(!std::mem::replace(&mut seen[tree.0], true));
                assert_eq!(sum, SumId(visit_position % first.sum_count()));
                visit_position += 1;
                chunk_ids.push(tree.0);
            }
            assert_eq!(
                chunk_ids.len(),
                chunk_trees.min(trees.trees().len() - number * chunk_trees)
            );
        }
        assert_eq!(next_action, first.actions().len());
        assert_eq!(next_predicate, first.predicates().len());
        assert!(seen.into_iter().all(|seen| seen));
        for row in [[0.0; 4], [2.0; 4], [f64::NAN; 4], [f64::INFINITY; 4]] {
            replay(&first, &row);
        }
    }
}

#[test]
fn partition_limits_and_sum_clamping_compile() {
    let trees = forest(vec![
        tree(&[]),
        tree(&[(0, 0.0)]),
        tree(&[(1, 0.0), (2, 0.0)]),
    ]);
    let context = Context::create();
    for chunk_trees in [1, 8] {
        let plan = plan_with(&trees, chunk_trees, 8);
        assert_eq!(plan.sum_count(), 3);
        assert_eq!(plan.chunks().len(), if chunk_trees == 1 { 3 } else { 1 });
        let compiled = compile_plan(&context, &plan).unwrap();
        for row in [[0.0; 4], [1.0; 4], [f64::NAN; 4]] {
            assert_eq!(compiled.predict(&row), trees.predict(&row));
        }
    }
}
