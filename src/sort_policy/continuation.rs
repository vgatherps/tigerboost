use super::{Action, FeatureValueId, Plan, TreeId};
use std::num::NonZeroUsize;
use std::ops::Range;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Stable argument positions; sum arguments are separate and retain their SumId.
pub struct Slots {
    pub indices: Vec<Option<TreeId>>,
    pub features: Vec<Option<FeatureValueId>>,
}

#[derive(Debug, PartialEq, Eq)]
/// Consecutive commands bounded to limit backend reordering across retirements.
pub struct Chunk {
    pub actions: Range<usize>,
    /// Scalar-equivalent predicates: a SIMD pair counts as two.
    pub tests: usize,
    pub incoming: Slots,
}

fn allocate<T: Copy + PartialEq>(slots: &mut Vec<Option<T>>, value: T) {
    assert!(!slots.contains(&Some(value)));
    if let Some(slot) = slots.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(value);
    } else {
        slots.push(Some(value));
    }
}

fn release<T: Copy + PartialEq>(slots: &mut [Option<T>], value: T) {
    slots
        .iter_mut()
        .find(|slot| **slot == Some(value))
        .unwrap()
        .take();
}

pub(super) fn partition_trees(plan: &Plan<'_>, limit: NonZeroUsize) -> Vec<Chunk> {
    let mut chunks = vec![Chunk {
        actions: 0..0,
        tests: 0,
        incoming: Slots::default(),
    }];
    for (cursor, action) in plan.actions().iter().enumerate() {
        let (tree, tests) = match action {
            Action::Admit(tree) => (Some(*tree), 0),
            Action::EvaluateTree {
                tree, predicates, ..
            } => (Some(*tree), predicates.len()),
            _ => (None, 0),
        };
        if let Some(tree) = tree
            && tree.0 != 0
            && tree.0 % limit.get() == 0
        {
            chunks.push(Chunk {
                actions: cursor..cursor,
                tests: 0,
                incoming: Slots::default(),
            });
        }
        let chunk = chunks.last_mut().unwrap();
        chunk.tests = chunk
            .tests
            .checked_add(tests)
            .expect("chunk test count overflow");
        chunk.actions.end = cursor + 1;
    }
    chunks
}

pub(super) fn partition(plan: &Plan<'_>, limit: usize) -> Vec<Chunk> {
    let actions = plan.actions();
    let mut live = Slots::default();
    let mut chunks = vec![Chunk {
        actions: 0..0,
        tests: 0,
        incoming: live.clone(),
    }];
    let mut cursor = 0;
    while cursor < actions.len() {
        let (width, mut end) = match actions[cursor..]
            .iter()
            .position(|action| matches!(action, Action::Compare { .. }))
        {
            Some(offset) => {
                let Action::Compare { comparisons, .. } = &actions[cursor + offset] else {
                    unreachable!()
                };
                (comparisons.len(), cursor + offset + 1)
            }
            None => (0, actions.len()),
        };
        assert!(width <= limit);
        while end < actions.len()
            && matches!(actions[end], Action::Release(_) | Action::Retire { .. })
        {
            end += 1;
        }
        if chunks.last().unwrap().tests + width > limit {
            chunks.push(Chunk {
                actions: cursor..cursor,
                tests: 0,
                incoming: live.clone(),
            });
        }
        for action in &actions[cursor..end] {
            match action {
                Action::Admit(tree) if !plan.model().trees()[tree.0].selections.is_empty() => {
                    allocate(&mut live.indices, *tree)
                }
                Action::Load { value, .. } => allocate(&mut live.features, *value),
                Action::Retire { tree, .. }
                    if !plan.model().trees()[tree.0].selections.is_empty() =>
                {
                    release(&mut live.indices, *tree)
                }
                Action::Release(value) => release(&mut live.features, *value),
                _ => {}
            }
        }
        let chunk = chunks.last_mut().unwrap();
        chunk.tests += width;
        chunk.actions.end = end;
        cursor = end;
    }
    assert!(live.indices.iter().all(Option::is_none));
    assert!(live.features.iter().all(Option::is_none));
    chunks
}
