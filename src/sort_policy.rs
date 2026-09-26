use std::{cmp::Reverse, num::NonZeroUsize};

use crate::parsed_tree::{Discriminator, FeatureIndex, Trees};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FeatureValueId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Consumer {
    pub tree: TreeId,
    pub level: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ComparisonMode {
    Scalar,
    Simd2,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Comparison {
    pub predicate: PredicateId,
    pub consumers: Vec<Consumer>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Action {
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
    Retire(TreeId),
}

pub(crate) struct Schedule {
    pub predicates: Vec<Discriminator>,
    pub actions: Vec<Action>,
}

pub(crate) fn tree_order(trees: &Trees) -> Schedule {
    let mut plan = Schedule {
        predicates: Vec::new(),
        actions: Vec::new(),
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
        plan.actions.push(Action::Retire(tree));
    }
    plan
}

struct ActiveTree {
    id: TreeId,
    remaining: usize,
}

struct Planner {
    predicates: Vec<Discriminator>,
    splits: Vec<Vec<PredicateId>>,
    scores: Vec<usize>,
    waiting: Vec<TreeId>,
    active: Vec<ActiveTree>,
    retained: Option<(FeatureIndex, FeatureValueId)>,
    actions: Vec<Action>,
    limit: usize,
    comparisons: ComparisonMode,
}

pub(crate) fn schedule(
    trees: &Trees,
    max_live_trees: NonZeroUsize,
    comparisons: ComparisonMode,
) -> Schedule {
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
        predicates,
        splits,
        scores,
        waiting: (0..trees.trees().len()).map(TreeId).collect(),
        active: Vec::new(),
        retained: None,
        actions: Vec::new(),
        limit: max_live_trees.get(),
        comparisons,
    }
    .run()
}

impl Planner {
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

    fn fill(&mut self) {
        while self.active.len() < self.limit && !self.waiting.is_empty() {
            let (position, &tree) = self
                .waiting
                .iter()
                .enumerate()
                .max_by_key(|(_, tree)| {
                    let overlap = self.splits[tree.0]
                        .iter()
                        .filter(|id| {
                            let feature = self.predicates[id.0].feature_idx;
                            self.retained.map(|(feature, _)| feature) == Some(feature)
                                || self.active.iter().any(|tree| self.uses(tree, feature) > 0)
                        })
                        .count();
                    (overlap, self.scores[tree.0], Reverse(tree.0))
                })
                .unwrap();
            self.waiting.remove(position);
            self.actions.push(Action::Admit(tree));
            let remaining = (1 << self.splits[tree.0].len()) - 1;
            if remaining == 0 {
                self.actions.push(Action::Retire(tree));
            } else {
                self.active.push(ActiveTree {
                    id: tree,
                    remaining,
                });
            }
        }
    }

    fn select_feature(&self) -> FeatureIndex {
        if let Some((feature, _)) = self.retained
            && self.active.iter().any(|tree| self.uses(tree, feature) > 0)
        {
            return feature;
        }
        let (feature, uses, _) = self
            .predicates
            .iter()
            .map(|predicate| {
                let feature = predicate.feature_idx;
                let mut uses = 0;
                let mut finishes = 0;
                for tree in &self.active {
                    let count = self.uses(tree, feature);
                    uses += count;
                    finishes += usize::from(count == tree.remaining.count_ones() as usize);
                }
                (feature, uses, finishes)
            })
            .max_by_key(|(feature, uses, finishes)| (*uses, *finishes, Reverse(feature.0)))
            .unwrap();
        if uses > 1 {
            return feature;
        }
        let tree = self
            .active
            .iter()
            .min_by_key(|tree| (tree.remaining.count_ones(), tree.id.0))
            .unwrap();
        let (_, predicate) = self.pending(tree).next().unwrap();
        self.predicates[predicate.0].feature_idx
    }

    fn run(mut self) -> Schedule {
        let mut next_value = 0;
        self.fill();
        while !self.active.is_empty() {
            let feature = self.select_feature();
            let value = match self.retained {
                Some((_, value)) => value,
                None => {
                    let value = FeatureValueId(next_value);
                    next_value += 1;
                    self.actions.push(Action::Load { value, feature });
                    self.retained = Some((feature, value));
                    value
                }
            };
            let width = match self.comparisons {
                ComparisonMode::Scalar => 1,
                ComparisonMode::Simd2 => 2,
            };
            let mut comparisons = Vec::new();
            for _ in 0..width {
                let Some(predicate) = self
                    .active
                    .iter()
                    .flat_map(|tree| self.pending(tree))
                    .map(|(_, id)| id)
                    .filter(|id| self.predicates[id.0].feature_idx == feature)
                    .min()
                else {
                    break;
                };
                let mut consumers = Vec::new();
                for tree in &mut self.active {
                    for (level, &id) in self.splits[tree.id.0].iter().enumerate() {
                        let bit = 1 << level;
                        if id == predicate && tree.remaining & bit != 0 {
                            consumers.push(Consumer {
                                tree: tree.id,
                                level,
                            });
                            tree.remaining &= !bit;
                        }
                    }
                }
                comparisons.push(Comparison {
                    predicate,
                    consumers,
                });
            }
            self.actions.push(Action::Compare {
                feature: value,
                comparisons,
            });
            self.active.retain(|tree| {
                if tree.remaining == 0 {
                    self.actions.push(Action::Retire(tree.id));
                    false
                } else {
                    true
                }
            });
            self.fill();
            if self.active.iter().all(|tree| self.uses(tree, feature) == 0) {
                self.actions.push(Action::Release(value));
                self.retained = None;
            }
        }
        Schedule {
            predicates: self.predicates,
            actions: self.actions,
        }
    }
}

#[cfg(test)]
mod tests;
