//! Synthetic dependent-call timings, including the row-selection driver.
use inkwell::context::Context;
use std::{error::Error, hint::black_box, time::Instant};
use tigerboost::jit::{CompiledModel, compile};
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

const FEATURES: usize = 16;
const ROWS: usize = 256;
const CALLS: usize = 50_000;
const ROUNDS: usize = 5;

fn workload() -> (Trees, Vec<[f64; FEATURES]>) {
    let mut state = 42u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as usize
    };
    let trees = (0..64)
        .map(|_| SingleTree {
            selections: (0..6)
                .map(|_| Discriminator {
                    feature_idx: FeatureIndex(random() % FEATURES),
                    threshold: (random() % 2048) as f64 / 256.0 - 4.0,
                })
                .collect(),
            leafs: (0..64).map(|_| (1 + random() % 16) as f64).collect(),
        })
        .collect();
    let rows = (0..ROWS)
        .map(|_| std::array::from_fn(|_| (random() % 2048) as f64 / 256.0 - 4.0))
        .collect();
    (
        Trees::new(trees, vec![NanMode::AsIs; FEATURES], 1.0, 0.0).unwrap(),
        rows,
    )
}

fn next_row(index: usize, prediction: f64) -> usize {
    (index + prediction as usize + 1) & (ROWS - 1)
}

fn verify(
    model: &Trees,
    compiled: &CompiledModel<'_>,
    rows: &[[f64; FEATURES]],
) -> (Vec<f64>, usize) {
    let scores: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
    for (row, &score) in rows.iter().zip(&scores) {
        assert_eq!(compiled.predict(row).unwrap(), score);
    }
    let (mut reference, mut jit, mut floor) = (0, 0, 0);
    let mut visited = [false; ROWS];
    for _ in 0..CALLS {
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

fn measure(predict: impl Fn(usize) -> f64, calls: usize) -> f64 {
    let mut index = 0;
    let start = Instant::now();
    for _ in 0..calls {
        index = next_row(index, black_box(predict(black_box(index))));
    }
    black_box(index);
    start.elapsed().as_secs_f64() * 1e9 / calls as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let (model, rows) = workload();
    let context = Context::create();
    let start = Instant::now();
    let compiled = compile(&context, &model)?;
    let compile_ms = start.elapsed().as_secs_f64() * 1e3;
    let (scores, distinct) = verify(&model, &compiled, &rows);
    let reference = |index: usize| model.predict(black_box(rows[index].as_slice())).unwrap();
    let jit = |index: usize| compiled.predict(black_box(rows[index].as_slice())).unwrap();
    let floor = |index| scores[index];
    measure(reference, 10_000);
    measure(jit, 10_000);
    measure(floor, 10_000);
    let mut samples = [[0.0; 3]; ROUNDS];
    for (round, sample) in samples.iter_mut().enumerate() {
        let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for variant in order {
            sample[variant] = match variant {
                0 => measure(reference, CALLS),
                1 => measure(jit, CALLS),
                2 => measure(floor, CALLS),
                _ => unreachable!(),
            };
        }
    }
    println!("Synthetic: 64 trees, depth 6, {FEATURES} features, {ROWS} rows; generic target");
    println!("Compile: {compile_ms:.3} ms; dependent chain visits {distinct} distinct rows");
    println!(
        "{ROUNDS} rounds x {CALLS} calls; ns/prediction includes driver (floor not subtracted)"
    );
    for (variant, label) in ["Reference", "Scalar JIT", "Driver floor"]
        .into_iter()
        .enumerate()
    {
        let mut values = samples.map(|sample| sample[variant]);
        values.sort_by(f64::total_cmp);
        println!(
            "{label}: median {:.2}, min {:.2}, max {:.2}",
            values[ROUNDS / 2],
            values[0],
            values[ROUNDS - 1]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_agrees_and_dependent_trace_visits_multiple_rows() {
        let (model, rows) = workload();
        let context = Context::create();
        let compiled = compile(&context, &model).unwrap();
        let (scores, distinct) = verify(&model, &compiled, &rows);
        assert_eq!(scores.len(), ROWS);
        assert!(distinct > 1);
    }
}
