use std::{cmp::Reverse, num::NonZeroUsize};

use crate::parsed_tree::{Discriminator, FeatureIndex, Trees};

mod continuation;
pub use continuation::{Chunk, Slots};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicateId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureValueId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Consumer {
    pub tree: TreeId,
    pub level: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ComparisonMode {
    Scalar,
    Simd2,
}

#[derive(Clone, Copy, Debug)]
pub enum Order {
    TreeOrder,
    Scheduled(RegisterBudget),
}

/// Logical working registers, not a guarantee about LLVM's physical allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterBudget {
    pub general: NonZeroUsize,
    pub floating: NonZeroUsize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct InsufficientBudget {
    pub required: RegisterBudget,
}

impl std::fmt::Display for InsufficientBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "schedule needs at least {} general and {} floating registers",
            self.required.general, self.required.floating
        )
    }
}

impl std::error::Error for InsufficientBudget {}

#[derive(Clone, Copy, Debug)]
pub struct SortOptions {
    pub order: Order,
    pub accumulators: NonZeroUsize,
    /// Maximum pack width for scheduled order; tree order remains scalar.
    pub comparisons: ComparisonMode,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            order: Order::Scheduled(RegisterBudget {
                general: NonZeroUsize::new(12).unwrap(),
                floating: NonZeroUsize::new(9).unwrap(),
            }),
            accumulators: NonZeroUsize::new(2).unwrap(),
            comparisons: ComparisonMode::Scalar,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Comparison {
    pub predicate: PredicateId,
    pub consumers: Vec<Consumer>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Admit(TreeId),
    Load {
        value: FeatureValueId,
        feature: FeatureIndex,
    },
    Compare {
        feature: FeatureValueId,
        comparisons: Vec<Comparison>,
    },
    Release(FeatureValueId),
    Retire {
        tree: TreeId,
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

pub fn plan(model: &Trees, options: SortOptions) -> Result<Plan<'_>, InsufficientBudget> {
    let sum_count = options.accumulators.get().min(model.trees().len().max(1));
    let mut plan = match options.order {
        Order::TreeOrder => tree_order(model, sum_count),
        Order::Scheduled(budget) => {
            if !model.trees().is_empty() {
                let has_splits = model.trees().iter().any(|tree| !tree.selections.is_empty());
                let required = RegisterBudget {
                    general: NonZeroUsize::new(3 + usize::from(has_splits)).unwrap(),
                    floating: NonZeroUsize::new(sum_count + 1 + usize::from(has_splits)).unwrap(),
                };
                if budget.general < required.general || budget.floating < required.floating {
                    return Err(InsufficientBudget { required });
                }
            }
            schedule(model, budget, options.comparisons, sum_count)
        }
    };
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
    plan.chunks = continuation::partition(&plan, 16);
    Ok(plan)
}

fn tree_order(trees: &Trees, sum_count: usize) -> Plan<'_> {
    let mut plan = Plan {
        model: trees,
        predicates: Vec::new(),
        actions: Vec::new(),
        sum_count,
        chunks: Vec::new(),
    };
    let mut next_value = 0;
    for (tree, description) in trees.trees().iter().enumerate() {
        let tree = TreeId(tree);
        plan.actions.push(Action::Admit(tree));
        for (level, split) in description.selections.iter().enumerate() {
            let predicate = PredicateId(plan.predicates.len());
            plan.predicates.push(split.clone());
            let value = FeatureValueId(next_value);
            next_value += 1;
            plan.actions.push(Action::Load {
                value,
                feature: split.feature_idx,
            });
            plan.actions.push(Action::Compare {
                feature: value,
                comparisons: vec![Comparison {
                    predicate,
                    consumers: vec![Consumer { tree, level }],
                }],
            });
            plan.actions.push(Action::Release(value));
        }
        plan.actions.push(Action::Retire {
            tree,
            sum: SumId(tree.0 % sum_count),
        });
    }
    plan
}

struct ActiveTree {
    id: TreeId,
    remaining: usize,
}

struct Operation {
    feature: FeatureIndex,
    new_trees: Vec<TreeId>,
    comparisons: Vec<Comparison>,
}

struct Planner<'model> {
    model: &'model Trees,
    predicates: Vec<Discriminator>,
    splits: Vec<Vec<PredicateId>>,
    scores: Vec<usize>,
    waiting: Vec<TreeId>,
    active: Vec<ActiveTree>,
    resident: Vec<(FeatureIndex, FeatureValueId)>,
    actions: Vec<Action>,
    budget: RegisterBudget,
    comparisons: ComparisonMode,
    sum_count: usize,
    retired: usize,
}

fn schedule(
    trees: &Trees,
    budget: RegisterBudget,
    comparisons: ComparisonMode,
    sum_count: usize,
) -> Plan<'_> {
    let mut predicates: Vec<Discriminator> = Vec::new();
    let mut frequencies = vec![0; trees.feature_modes().len()];
    let splits = trees
        .trees()
        .iter()
        .map(|tree| {
            tree.selections
                .iter()
                .map(|split| {
                    frequencies[split.feature_idx.0] += 1;
                    let id = predicates.iter().position(|p| {
                        p.feature_idx == split.feature_idx
                            && p.threshold.to_bits() == split.threshold.to_bits()
                    });
                    PredicateId(id.unwrap_or_else(|| {
                        predicates.push(split.clone());
                        predicates.len() - 1
                    }))
                })
                .collect()
        })
        .collect();
    let scores = trees
        .trees()
        .iter()
        .map(|tree| {
            tree.selections
                .iter()
                .map(|split| frequencies[split.feature_idx.0])
                .sum()
        })
        .collect();
    Planner {
        model: trees,
        predicates,
        splits,
        scores,
        waiting: (0..trees.trees().len()).map(TreeId).collect(),
        active: Vec::new(),
        resident: Vec::new(),
        actions: Vec::new(),
        budget,
        comparisons,
        sum_count,
        retired: 0,
    }
    .run()
}

impl<'model> Planner<'model> {
    fn pending<'a>(
        &'a self,
        tree: &'a ActiveTree,
    ) -> impl Iterator<Item = (usize, PredicateId)> + 'a {
        self.splits[tree.id.0]
            .iter()
            .copied()
            .enumerate()
            .filter(|(level, _)| tree.remaining & (1 << level) != 0)
    }

    fn uses(&self, tree: &ActiveTree, feature: FeatureIndex) -> usize {
        self.pending(tree)
            .filter(|(_, id)| self.predicates[id.0].feature_idx == feature)
            .count()
    }

    fn waiting_order(&self) -> Vec<TreeId> {
        let mut waiting = self.waiting.clone();
        waiting.sort_by_cached_key(|tree| {
            let overlap = self.splits[tree.0]
                .iter()
                .filter(|id| {
                    let feature = self.predicates[id.0].feature_idx;
                    self.resident.iter().any(|&(f, _)| f == feature)
                        || self.active.iter().any(|tree| self.uses(tree, feature) > 0)
                })
                .count();
            (Reverse(overlap), Reverse(self.scores[tree.0]), tree.0)
        });
        waiting
    }

    fn operation(
        &self,
        feature: FeatureIndex,
        width: usize,
        waiting: &[TreeId],
    ) -> Option<Operation> {
        // Pair threshold storage becomes its result mask; it is not another FP slot.
        let scratch = width + 1;
        if 1 + self.active.len() + scratch > self.budget.general.get()
            || self.sum_count + 1 + width > self.budget.floating.get()
        {
            return None;
        }
        let mut op = Operation {
            feature,
            new_trees: Vec::new(),
            comparisons: Vec::new(),
        };
        let mut active: Vec<_> = self.active.iter().collect();
        if 1 + active.len() + 1 + 2 > self.budget.general.get() {
            active.sort_by_key(|tree| (tree.remaining.count_ones(), tree.id.0));
        }
        for tree in active
            .into_iter()
            .map(|t| t.id)
            .chain(waiting.iter().copied())
        {
            let active = self.active.iter().find(|t| t.id == tree);
            if active.is_none()
                && 1 + self.active.len() + op.new_trees.len() + 1 + scratch
                    > self.budget.general.get()
            {
                continue;
            }
            let mut used = false;
            for (level, &predicate) in self.splits[tree.0].iter().enumerate() {
                if active.is_some_and(|t| t.remaining & (1 << level) == 0)
                    || self.predicates[predicate.0].feature_idx != feature
                {
                    continue;
                }
                let position = op.comparisons.iter().position(|c| c.predicate == predicate);
                let position = match position {
                    Some(position) => position,
                    None if op.comparisons.len() < width => {
                        op.comparisons.push(Comparison {
                            predicate,
                            consumers: Vec::new(),
                        });
                        op.comparisons.len() - 1
                    }
                    None => continue,
                };
                op.comparisons[position]
                    .consumers
                    .push(Consumer { tree, level });
                used = true;
            }
            if used && active.is_none() {
                op.new_trees.push(tree);
            }
        }
        (op.comparisons.len() == width).then_some(op)
    }

    fn select(&self, waiting: &[TreeId]) -> Operation {
        let pressured = 1 + self.active.len() + 1 + 2 > self.budget.general.get();
        let shortest = self
            .active
            .iter()
            .min_by_key(|t| (t.remaining.count_ones(), t.id.0));
        (0..self.model.feature_modes().len())
            .filter_map(|index| {
                let feature = FeatureIndex(index);
                if !self.active.iter().any(|t| self.uses(t, feature) > 0)
                    && !waiting.first().is_some_and(|t| {
                        self.splits[t.0]
                            .iter()
                            .any(|id| self.predicates[id.0].feature_idx == feature)
                    })
                {
                    return None;
                }
                let pair = matches!(self.comparisons, ComparisonMode::Simd2)
                    .then(|| self.operation(feature, 2, waiting))
                    .flatten();
                pair.or_else(|| self.operation(feature, 1, waiting))
            })
            .max_by_key(|op| {
                let uses = self
                    .active
                    .iter()
                    .map(|t| self.uses(t, op.feature))
                    .sum::<usize>()
                    + op.new_trees
                        .iter()
                        .map(|t| {
                            self.splits[t.0]
                                .iter()
                                .filter(|id| self.predicates[id.0].feature_idx == op.feature)
                                .count()
                        })
                        .sum::<usize>();
                let consumed = |tree: TreeId| {
                    op.comparisons
                        .iter()
                        .flat_map(|c| &c.consumers)
                        .filter(|c| c.tree == tree)
                        .count()
                };
                let finishes = self
                    .active
                    .iter()
                    .filter(|t| consumed(t.id) == t.remaining.count_ones() as usize)
                    .count();
                let drains = pressured && shortest.is_some_and(|t| consumed(t.id) > 0);
                (
                    drains,
                    uses,
                    finishes,
                    self.resident.iter().any(|&(f, _)| f == op.feature),
                    Reverse(op.feature.0),
                )
            })
            .expect("validated scalar workspace must permit progress")
    }

    fn remaining_uses(&self, feature: FeatureIndex) -> usize {
        self.active
            .iter()
            .map(|t| self.uses(t, feature))
            .sum::<usize>()
            + self
                .waiting
                .iter()
                .map(|t| {
                    self.splits[t.0]
                        .iter()
                        .filter(|id| self.predicates[id.0].feature_idx == feature)
                        .count()
                })
                .sum::<usize>()
    }

    fn release(&mut self, position: usize) {
        let (_, value) = self.resident.remove(position);
        self.actions.push(Action::Release(value));
    }

    fn retire(&mut self, tree: TreeId) {
        self.actions.push(Action::Retire {
            tree,
            sum: SumId(self.retired % self.sum_count),
        });
        self.retired += 1;
    }

    fn run(mut self) -> Plan<'model> {
        let mut next_value = 0;
        while !self.active.is_empty() || !self.waiting.is_empty() {
            let waiting = self.waiting_order();
            if let Some(&tree) = waiting.first()
                && self.splits[tree.0].is_empty()
            {
                self.waiting.retain(|&t| t != tree);
                self.actions.push(Action::Admit(tree));
                self.retire(tree);
                continue;
            }
            let op = self.select(&waiting);
            let loaded = self.resident.iter().any(|&(f, _)| f == op.feature);
            // Loading needs one address register (two with NaN normalization),
            // and normalization one FP replacement: covered by scalar workspace.
            while self.sum_count + self.resident.len() + usize::from(!loaded) + op.comparisons.len()
                > self.budget.floating.get()
            {
                let position = self
                    .resident
                    .iter()
                    .enumerate()
                    .filter(|(_, (f, _))| *f != op.feature)
                    .min_by_key(|(_, (f, _))| (self.remaining_uses(*f), Reverse(f.0)))
                    .map(|(position, _)| position)
                    .expect("operand and scalar workspace fit");
                self.release(position);
            }
            let value = match self.resident.iter().find(|&&(f, _)| f == op.feature) {
                Some(&(_, value)) => value,
                None => {
                    let value = FeatureValueId(next_value);
                    next_value += 1;
                    self.actions.push(Action::Load {
                        value,
                        feature: op.feature,
                    });
                    self.resident.push((op.feature, value));
                    value
                }
            };
            for tree in op.new_trees {
                self.waiting.retain(|&t| t != tree);
                self.actions.push(Action::Admit(tree));
                self.active.push(ActiveTree {
                    id: tree,
                    remaining: (1 << self.splits[tree.0].len()) - 1,
                });
            }
            for comparison in &op.comparisons {
                for consumer in &comparison.consumers {
                    self.active
                        .iter_mut()
                        .find(|t| t.id == consumer.tree)
                        .unwrap()
                        .remaining &= !(1 << consumer.level);
                }
            }
            self.actions.push(Action::Compare {
                feature: value,
                comparisons: op.comparisons,
            });
            while let Some(position) = self
                .resident
                .iter()
                .position(|&(f, _)| self.remaining_uses(f) == 0)
            {
                self.release(position);
            }
            while let Some(position) = self.active.iter().position(|t| t.remaining == 0) {
                let tree = self.active.remove(position);
                self.retire(tree.id);
            }
        }
        Plan {
            model: self.model,
            predicates: self.predicates,
            actions: self.actions,
            sum_count: self.sum_count,
            chunks: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests;
