use std::{num::NonZeroUsize, ops::Range};

use crate::parsed_tree::{Discriminator, Trees};

mod continuation;
pub use continuation::Chunk;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumId(pub usize);

const CHUNK_TREES: NonZeroUsize = NonZeroUsize::new(8).unwrap();
const SUMS: usize = 2;

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Constant {
        tree: TreeId,
        sum: SumId,
    },
    EvaluateTree {
        tree: TreeId,
        predicates: Range<usize>,
        sum: SumId,
    },
}

pub struct Plan<'model> {
    model: &'model Trees,
    predicates: Vec<Discriminator>,
    actions: Vec<Action>,
    sum_count: usize,
    chunks: Vec<Chunk>,
}

impl Plan<'_> {
    pub fn model(&self) -> &Trees {
        self.model
    }

    pub fn predicates(&self) -> &[Discriminator] {
        &self.predicates
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    pub fn sum_count(&self) -> usize {
        self.sum_count
    }

    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
}

pub fn plan(model: &Trees) -> Plan<'_> {
    build_plan(model, CHUNK_TREES, SUMS)
}

#[cfg(test)]
pub(crate) fn plan_with(model: &Trees, chunk_trees: usize, sums: usize) -> Plan<'_> {
    assert!(sums > 0);
    build_plan(model, NonZeroUsize::new(chunk_trees).unwrap(), sums)
}

fn build_plan(model: &Trees, chunk_trees: NonZeroUsize, sums: usize) -> Plan<'_> {
    let sum_count = sums.min(model.trees().len().max(1));
    let visit_order = shared_feature_order(model, chunk_trees);
    let mut plan = tree_actions(model, sum_count, &visit_order);
    plan.chunks = continuation::partition_trees(&plan, chunk_trees);
    plan
}

fn tree_actions<'model>(
    trees: &'model Trees,
    sum_count: usize,
    visit_order: &[TreeId],
) -> Plan<'model> {
    let mut plan = Plan {
        model: trees,
        predicates: Vec::new(),
        actions: Vec::new(),
        sum_count,
        chunks: Vec::new(),
    };
    for (visit_position, &tree) in visit_order.iter().enumerate() {
        let description = &trees.trees()[tree.0];
        let sum = SumId(visit_position % sum_count);
        if description.selections.is_empty() {
            plan.actions.push(Action::Constant { tree, sum });
        } else {
            let start = plan.predicates.len();
            plan.predicates
                .extend(description.selections.iter().cloned());
            if !description.selections.len().is_multiple_of(2) {
                plan.predicates.push(Discriminator {
                    feature_idx: description.selections[0].feature_idx,
                    threshold: f64::INFINITY,
                });
            }
            plan.actions.push(Action::EvaluateTree {
                tree,
                predicates: start..plan.predicates.len(),
                sum,
            });
        }
    }
    plan
}

fn shared_feature_order(trees: &Trees, chunk_trees: NonZeroUsize) -> Vec<TreeId> {
    let mut frequencies = vec![0usize; trees.feature_modes().len()];
    for split in trees.trees().iter().flat_map(|tree| tree.selections.iter()) {
        frequencies[split.feature_idx.0] = frequencies[split.feature_idx.0]
            .checked_add(1)
            .expect("feature frequency overflow");
    }

    let mut scores = Vec::with_capacity(trees.trees().len());
    let mut features = Vec::with_capacity(trees.trees().len());
    let mut remaining = Vec::new();
    let mut constants = Vec::new();
    for (tree, description) in trees.trees().iter().enumerate() {
        let score = description
            .selections
            .iter()
            .map(|split| frequencies[split.feature_idx.0])
            .try_fold(0usize, usize::checked_add)
            .expect("tree feature score overflow");
        scores.push(score);
        let mut distinct: Vec<_> = description
            .selections
            .iter()
            .map(|split| split.feature_idx.0)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        features.push(distinct);
        if description.selections.is_empty() {
            constants.push(TreeId(tree));
        } else {
            remaining.push(TreeId(tree));
        }
    }

    let mut order = Vec::with_capacity(trees.trees().len());
    let mut chunk_features = vec![false; trees.feature_modes().len()];
    while !remaining.is_empty() {
        chunk_features.fill(false);
        let seed = remaining
            .iter()
            .enumerate()
            .max_by_key(|&(_, tree)| (scores[tree.0], std::cmp::Reverse(tree.0)))
            .map(|(position, _)| position)
            .unwrap();
        let seed = remaining.remove(seed);
        for &feature in &features[seed.0] {
            chunk_features[feature] = true;
        }
        order.push(seed);

        let mut chunk_len = 1;
        while chunk_len < chunk_trees.get() && !remaining.is_empty() {
            let next = remaining
                .iter()
                .enumerate()
                .max_by_key(|&(_, tree)| {
                    let overlap = features[tree.0]
                        .iter()
                        .filter(|&&feature| chunk_features[feature])
                        .count();
                    (overlap, std::cmp::Reverse(tree.0))
                })
                .map(|(position, _)| position)
                .unwrap();
            let next = remaining.remove(next);
            for &feature in &features[next.0] {
                chunk_features[feature] = true;
            }
            order.push(next);
            chunk_len += 1;
        }
    }
    order.extend(constants);
    order
}

#[cfg(test)]
mod tests;
