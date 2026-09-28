//! Dependent-trace verification shared by the timing examples.
use crate::support::ROWS;
use tigerboost::jit::CompiledModel;
use tigerboost::parsed_tree::Trees;

/// Odd, prediction-dependent step: never a fixed point modulo ROWS.
pub fn next_row(index: usize, prediction: f64) -> usize {
    (index + (prediction as usize | 1)) & (ROWS - 1)
}

/// Checks every row and a `calls`-long dependent trace against `Trees::predict`.
/// Returns the reference scores and the number of distinct rows the trace visits.
pub fn verify(
    model: &Trees,
    compiled: &CompiledModel<'_>,
    rows: &[Vec<f64>],
    calls: usize,
) -> (Vec<f64>, usize) {
    let scores: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
    for (row, &score) in rows.iter().zip(&scores) {
        assert_eq!(compiled.predict(row).unwrap(), score);
    }
    let (mut reference, mut jit, mut floor) = (0, 0, 0);
    let mut visited = [false; ROWS];
    for _ in 0..calls {
        assert_eq!((reference, reference), (jit, floor));
        visited[reference] = true;
        reference = next_row(reference, model.predict(&rows[reference]).unwrap());
        jit = next_row(jit, compiled.predict(&rows[jit]).unwrap());
        floor = next_row(floor, scores[floor]);
    }
    assert_eq!((reference, reference), (jit, floor));
    let distinct = visited.into_iter().filter(|&seen| seen).count();
    assert!(distinct > 1, "dependent chain must visit varied rows");
    assert_ne!(
        reference,
        next_row(reference, scores[reference]),
        "dependent chain settled on one row"
    );
    (scores, distinct)
}
