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
    let mut trees = 0usize;
    for (cursor, action) in plan.actions().iter().enumerate() {
        let tests = match action {
            Action::Admit(_) => Some(0),
            Action::EvaluateTree { predicates, .. } => Some(predicates.len()),
            _ => None,
        };
        if let Some(tests) = tests {
            if trees != 0 && trees.is_multiple_of(limit.get()) {
                chunks.push(Chunk {
                    actions: cursor..cursor,
                    tests: 0,
                });
            }
            trees = trees.checked_add(1).expect("tree count overflow");
            let chunk = chunks.last_mut().unwrap();
            chunk.tests = chunk
                .tests
                .checked_add(tests)
                .expect("chunk test count overflow");
        }
        let chunk = chunks.last_mut().unwrap();
        chunk.actions.end = cursor + 1;
    }
    chunks
}
