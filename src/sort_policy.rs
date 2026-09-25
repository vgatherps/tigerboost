use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    num::NonZeroUsize,
};

use crate::parsed_tree::{FeatureIndex, Trees};

// sorting and batching policy for trees
pub struct SortOptions {
    pub max_live_trees: NonZeroUsize,
}

#[derive(Default)]
pub struct FeatureScore {
    n_uses: usize,
}

pub fn score_feature_indices(trees: &Trees) -> HashMap<FeatureIndex, FeatureScore> {
    let mut feature_scores = HashMap::<_, FeatureScore>::default();
    for tree in trees.trees() {
        for feature in &tree.selections {
            feature_scores
                .entry(feature.feature_idx)
                .or_default()
                .n_uses += 1;
        }
    }
    feature_scores
}

pub fn score_trees(
    trees: &Trees,
    feature_scores: &HashMap<FeatureIndex, FeatureScore>,
) -> HashMap<usize, usize> {
    let mut tree_scores = HashMap::new();
    for (tree_idx, tree) in trees.trees().iter().enumerate() {
        let mut score = 0;
        for eval in &tree.selections {
            score += feature_scores[&eval.feature_idx].n_uses;
        }
        tree_scores.insert(tree_idx, score);
    }
    tree_scores
}

pub enum Action {
    EvalFeatureFor {
        feature_idx: FeatureIndex,
        evaluator_index: usize,
    },
    EvalFeatureVecFor {
        feature_idx: FeatureIndex,
        evaluator_indices: Vec<usize>,
    },
    // implicit in here whether this is single or vector eval
    ExtractEvalBit {
        evaluator_index: usize,
    },
}

pub struct Evaluator {
    pub tree_index: usize,
    pub level_index: usize,
}

pub fn build_evaluators(trees: &Trees) -> HashMap<usize, Evaluator> {
    let mut evaluators = HashMap::default();

    for (tree_index, tree) in trees.trees().iter().enumerate() {
        for level_index in 0..tree.leafs.len() {
            let eval_idx = evaluators.len();
            evaluators.insert(
                eval_idx,
                Evaluator {
                    tree_index,
                    level_index,
                },
            );
        }
    }

    evaluators
}

pub fn sort_trees(trees: &Trees) -> Vec<Action> {
    let mut actions = Vec::new();
    let mut live_trees: HashSet<usize> = Default::default();

    let evaluators = build_evaluators(trees);

    let feature_scores = score_feature_indices(trees);
    let tree_scores = score_trees(trees, &feature_scores);

    // tree with highest score i.e. most used indices is live
    let first_tree_idx = *tree_scores
        .iter()
        .max_by_key(|(_, score)| *score)
        .unwrap()
        .1;

    // set the live features
    let first_tree = &trees.trees()[first_tree_idx];

    // 1. take the first tree.
    // 2. start appending trees *first* ranked by overlap with active set, THEN, by highest score
    // 2.a we want to maximize parallelism here
    // 3. append trees until we have a batch or no more trees.
    // 4. find the feature with the most uses in the batch
    // 5. decide whether to vectorize or individual eval it.
    // 6. do so, *and*, count 'live vector registers' and 'live tree mask trackers'
    // 7. increase parallel ops while can. either by individual eval or vector.
    // 8. once enough, start retiring indices.
    // 9. if a tree is complete, then it's retired' (this should be an action no?)
    // flop between 7 and 8. balance of reuse of loaded features, clearing them out, and keeping state resources under control.
    // note that as we reach a limit, we can start aggressively finishing eval for trees to retire.
    // and as we retire trees, add the as-of-now most overlapped trees
    // details are not fleshed out yet

    actions
}
