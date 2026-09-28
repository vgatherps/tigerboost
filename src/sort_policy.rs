use std::num::NonZeroUsize;

use crate::parsed_tree::{Discriminator, Trees};

mod continuation;
pub use continuation::Chunk;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicateId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumId(pub usize);

#[derive(Clone, Copy, Debug)]
pub struct SortOptions {
    pub chunk_trees: NonZeroUsize,
    pub accumulators: NonZeroUsize,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            chunk_trees: NonZeroUsize::new(8).unwrap(),
            accumulators: NonZeroUsize::new(2).unwrap(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Admit(TreeId),
    Retire {
        tree: TreeId,
        sum: SumId,
    },
    EvaluateTree {
        tree: TreeId,
        predicates: Vec<PredicateId>,
        sum: SumId,
    },
    AddSums {
        destination: SumId,
        source: SumId,
    },
    ReturnScaled(SumId),
    ReturnBias,
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

pub fn plan(model: &Trees, options: SortOptions) -> Plan<'_> {
    let sum_count = options.accumulators.get().min(model.trees().len().max(1));
    let mut plan = tree_major(model, sum_count);
    for source in 1..sum_count {
        plan.actions.push(Action::AddSums {
            destination: SumId(0),
            source: SumId(source),
        });
    }
    plan.actions.push(if model.trees().is_empty() {
        Action::ReturnBias
    } else {
        Action::ReturnScaled(SumId(0))
    });
    plan.chunks = continuation::partition_trees(&plan, options.chunk_trees);
    plan
}

fn tree_major(trees: &Trees, sum_count: usize) -> Plan<'_> {
    let mut plan = Plan {
        model: trees,
        predicates: Vec::new(),
        actions: Vec::new(),
        sum_count,
        chunks: Vec::new(),
    };
    for (tree, description) in trees.trees().iter().enumerate() {
        let tree = TreeId(tree);
        let sum = SumId(tree.0 % sum_count);
        if description.selections.is_empty() {
            plan.actions.push(Action::Admit(tree));
            plan.actions.push(Action::Retire { tree, sum });
        } else {
            let start = plan.predicates.len();
            plan.predicates
                .extend(description.selections.iter().cloned());
            if description.selections.len() % 2 != 0 {
                plan.predicates.push(Discriminator {
                    feature_idx: description.selections[0].feature_idx,
                    threshold: f64::INFINITY,
                });
            }
            plan.actions.push(Action::EvaluateTree {
                tree,
                predicates: (start..plan.predicates.len()).map(PredicateId).collect(),
                sum,
            });
        }
    }
    plan
}

#[cfg(test)]
mod tests;
