use super::*;
use crate::parsed_tree::{NanMode, SingleTree};

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

fn check(trees: &Trees, cap: usize, row: &[f64]) -> Schedule {
    let plan = schedule(trees, NonZeroUsize::new(cap).unwrap());
    let mut state = vec![0; trees.trees().len()];
    let mut seen = vec![0usize; state.len()];
    let mut indices = vec![0usize; state.len()];
    let mut active = 0;
    let mut loaded = None;
    let mut sum = 0.0;
    for action in &plan.actions {
        match action {
            Action::Admit(TreeId(tree)) => {
                assert_eq!(state[*tree], 0, "tree admitted twice");
                state[*tree] = 1;
                active += 1;
                assert!(active <= cap);
            }
            Action::Load(feature) => {
                let value = row[feature.0];
                let value = match (value.is_nan(), trees.feature_modes()[feature.0]) {
                    (true, NanMode::AsFalse) => f64::NEG_INFINITY,
                    (true, NanMode::AsTrue) => f64::INFINITY,
                    _ => value,
                };
                loaded = Some((*feature, value));
            }
            Action::Compare {
                predicate,
                consumers,
            } => {
                assert!(
                    !consumers.is_empty(),
                    "comparison must resolve pending work"
                );
                let predicate = &plan.predicates[predicate.0];
                let (feature, value) = loaded.expect("comparison without a loaded feature");
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
            Action::Retire(TreeId(tree)) => {
                assert_eq!(state[*tree], 1);
                assert_eq!(
                    seen[*tree],
                    (1 << trees.trees()[*tree].selections.len()) - 1
                );
                state[*tree] = 2;
                active -= 1;
                sum += trees.trees()[*tree].leafs[indices[*tree]];
            }
        }
    }
    assert_eq!(active, 0);
    assert!(state.iter().all(|&state| state == 2));
    assert_eq!(indices, trees.leaf_indices(row).unwrap());
    let prediction = if trees.trees().is_empty() {
        trees.bias()
    } else {
        sum * trees.scale() + trees.bias()
    };
    assert_eq!(prediction, trees.predict(row).unwrap());
    plan
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
            .filter(|a| matches!(a, Action::Load(_)))
            .count(),
        1
    );
    let comparisons: Vec<_> = plan
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::Compare {
                predicate,
                consumers,
            } => Some((*predicate, consumers.len())),
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
        .position(|a| *a == Action::Retire(TreeId(1)))
        .unwrap();
    assert_eq!(plan.actions[retire_short + 1], Action::Admit(TreeId(2)));
    assert_eq!(
        plan.actions[retire_short + 2],
        Action::Compare {
            predicate: PredicateId(0),
            consumers: vec![Consumer {
                tree: TreeId(2),
                level: 0
            }],
        }
    );
    let retire_long = plan
        .actions
        .iter()
        .position(|a| *a == Action::Retire(TreeId(0)))
        .unwrap();
    assert!(retire_short + 2 < retire_long);
    assert_eq!(
        plan.actions
            .iter()
            .filter(|a| **a == Action::Load(FeatureIndex(0)))
            .count(),
        1
    );
}

#[test]
fn empty_and_constant_trees_need_no_feature_loads() {
    let empty = forest(vec![]);
    assert!(check(&empty, 1, &[0.0; 4]).actions.is_empty());
    let constants = forest(vec![tree(&[]), tree(&[]), tree(&[])]);
    assert_eq!(
        check(&constants, 1, &[0.0; 4]).actions,
        [
            Action::Admit(TreeId(0)),
            Action::Retire(TreeId(0)),
            Action::Admit(TreeId(1)),
            Action::Retire(TreeId(1)),
            Action::Admit(TreeId(2)),
            Action::Retire(TreeId(2)),
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
    assert_eq!(plan.actions[2], Action::Load(FeatureIndex(3)));
    assert_eq!(plan.actions[4], Action::Retire(TreeId(1)));
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
            let first = check(&trees, cap, &rows[0]);
            for row in rows {
                let next = check(&trees, cap, &row);
                assert_eq!(first.actions, next.actions);
            }
        }
    }
}
