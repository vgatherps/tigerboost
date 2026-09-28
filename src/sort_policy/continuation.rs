use super::{Action, Plan};
use std::num::NonZeroUsize;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq)]
/// A run of whole trees emitted as one function, so the backend cannot hoist loads across chunks.
pub struct Chunk {
    pub actions: Range<usize>,
    /// Predicates evaluated in this chunk, padding levels included.
    pub tests: usize,
}

pub(super) fn partition_trees(plan: &Plan<'_>, limit: NonZeroUsize) -> Vec<Chunk> {
    let mut chunks = vec![Chunk {
        actions: 0..0,
        tests: 0,
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
