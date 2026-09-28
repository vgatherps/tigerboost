//! Single-call prediction latency with cache and code disturbance.
mod support;

use inkwell::context::Context;
use std::{
    error::Error,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use support::{ROWS, Workload, workload};
use tigerboost::jit::{CompiledModel, compile};
use tigerboost::parsed_tree::Trees;
use tigerboost::sort_policy::{SortOptions, plan};

const DEFAULT_SAMPLES: usize = 10_000;
const VERIFY_CALLS: usize = 50_000;
const USAGE: &str = "usage: cold_bench <workload> [--samples N] [--out PATH]";

#[derive(Debug, PartialEq, Eq)]
struct Arguments {
    workload: Workload,
    workload_name: String,
    samples: usize,
    out: Option<PathBuf>,
}

fn parse_workload(name: &str) -> Result<Workload, &'static str> {
    if let Ok(workload) = Workload::parse(name) {
        return Ok(workload);
    }
    let (trees, features) = name.split_once('x').ok_or(USAGE)?;
    let trees = trees.parse::<usize>().map_err(|_| USAGE)?;
    let features = features.parse::<usize>().map_err(|_| USAGE)?;
    if trees == 0 || features == 0 || trees > 10_922 {
        return Err(USAGE);
    }
    Ok(Workload::Distinct(trees, features))
}

fn parse_arguments(arguments: impl IntoIterator<Item = String>) -> Result<Arguments, &'static str> {
    let mut arguments = arguments.into_iter();
    let workload_name = arguments.next().ok_or(USAGE)?;
    let workload = parse_workload(&workload_name)?;
    let mut samples = None;
    let mut out = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--samples" if samples.is_none() => {
                let value = arguments.next().ok_or(USAGE)?;
                let value = value.parse::<usize>().map_err(|_| USAGE)?;
                if value == 0 {
                    return Err(USAGE);
                }
                samples = Some(value);
            }
            "--out" if out.is_none() => {
                out = Some(PathBuf::from(arguments.next().ok_or(USAGE)?));
            }
            _ => return Err(USAGE),
        }
    }
    Ok(Arguments {
        workload,
        workload_name,
        samples: samples.unwrap_or(DEFAULT_SAMPLES),
        out,
    })
}

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn shuffle<T>(values: &mut [T], state: &mut u64) {
    for index in (1..values.len()).rev() {
        let other = (random(state) % (index as u64 + 1)) as usize;
        values.swap(index, other);
    }
}

fn shuffled_rows() -> [usize; ROWS] {
    let mut rows = std::array::from_fn(|index| index);
    let mut state = 0x434f4c44524f5753;
    shuffle(&mut rows, &mut state);
    rows
}

fn next_row(index: usize, prediction: f64) -> usize {
    (index + (prediction as usize | 1)) & (ROWS - 1)
}

fn verify(model: &Trees, compiled: &CompiledModel<'_>, rows: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let scores: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
    for (row, &score) in rows.iter().zip(&scores) {
        assert_eq!(compiled.predict(row).unwrap(), score);
    }
    let (mut reference, mut jit, mut floor) = (0, 0, 0);
    let mut visited = [false; ROWS];
    for _ in 0..VERIFY_CALLS {
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

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u64; 8]);

struct PollutionBuffer {
    lines: Vec<Line>,
    line_order: Vec<usize>,
}

impl PollutionBuffer {
    fn new(bytes: usize, seed: u64) -> Self {
        assert!(bytes > 0 && bytes.is_multiple_of(64));
        let mut line_order: Vec<_> = (0..bytes / 64).collect();
        let mut state = seed;
        shuffle(&mut line_order, &mut state);
        Self {
            lines: vec![Line([0; 8]); bytes / size_of::<Line>()],
            line_order,
        }
    }

    fn touch_pages(&mut self) {
        for index in (0..self.lines.len()).step_by(4096 / size_of::<Line>()) {
            self.lines[index].0[0] = self.lines[index].0[0].wrapping_add(1);
        }
        black_box(&self.lines);
    }

    fn disturb(&mut self) {
        let mut total = 0u64;
        for &line in &self.line_order {
            let word = &mut self.lines[line].0[0];
            *word = word.wrapping_add(1);
            total = total.wrapping_add(*word);
        }
        black_box(total);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowSequence {
    Shuffled,
    Dependent,
}

impl RowSequence {
    fn label(self) -> &'static str {
        match self {
            Self::Shuffled => "shuffled",
            Self::Dependent => "dependent",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    Hot,
    Data256KiB,
    Data4MiB,
    Data16MiB,
    Data64MiB,
    OtherModel,
    TimerOnly,
    TimerAfterData64MiB,
}

impl Condition {
    const ALL: [Self; 8] = [
        Self::Hot,
        Self::Data256KiB,
        Self::Data4MiB,
        Self::Data16MiB,
        Self::Data64MiB,
        Self::OtherModel,
        Self::TimerOnly,
        Self::TimerAfterData64MiB,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Data256KiB => "data 256KiB",
            Self::Data4MiB => "data 4MiB",
            Self::Data16MiB => "data 16MiB",
            Self::Data64MiB => "data 64MiB",
            Self::OtherModel => "other model",
            Self::TimerOnly => "timer only",
            Self::TimerAfterData64MiB => "timer only after data 64MiB",
        }
    }

    fn data_index(self) -> Option<usize> {
        match self {
            Self::Data256KiB => Some(0),
            Self::Data4MiB => Some(1),
            Self::Data16MiB => Some(2),
            Self::Data64MiB | Self::TimerAfterData64MiB => Some(3),
            _ => None,
        }
    }

    fn calls_model(self) -> bool {
        !matches!(self, Self::TimerOnly | Self::TimerAfterData64MiB)
    }
}

#[derive(Clone, Copy)]
struct BenchmarkConfig {
    data_bytes: [usize; 4],
    other_trees: usize,
    other_features: usize,
    warmup_calls: usize,
}

impl BenchmarkConfig {
    fn production() -> Self {
        Self {
            data_bytes: [256 << 10, 4 << 20, 16 << 20, 64 << 20],
            other_trees: 2_000,
            other_features: 100,
            warmup_calls: 1_000,
        }
    }
}

#[derive(Debug, PartialEq)]
struct Statistics {
    mean: f64,
    p50: u64,
    p90: u64,
    p99: u64,
    max: u64,
}

fn statistics(samples: &[u64]) -> Statistics {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let percentile = |q: f64| {
        let index = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[index]
    };
    Statistics {
        mean: samples.iter().map(|&sample| sample as f64).sum::<f64>() / samples.len() as f64,
        p50: percentile(0.50),
        p90: percentile(0.90),
        p99: percentile(0.99),
        max: *sorted.last().unwrap(),
    }
}

struct Measurement {
    sequence: RowSequence,
    condition: Condition,
    samples: Vec<u64>,
}

struct Report {
    workload_name: String,
    tree_count: usize,
    feature_count: usize,
    timer_resolution_ns: f64,
    distinct_rows: usize,
    sample_count: usize,
    measurements: Vec<Measurement>,
}

fn timer_resolution() -> Result<f64, &'static str> {
    let mut smallest = None;
    for _ in 0..100_000 {
        let first = Instant::now();
        let elapsed = Instant::now().duration_since(first);
        if !elapsed.is_zero() && smallest.is_none_or(|current| elapsed < current) {
            smallest = Some(elapsed);
        }
    }
    smallest
        .map(|duration| duration.as_secs_f64() * 1e9)
        .ok_or("Instant produced no nonzero delta in 100000 reads")
}

fn timed_prediction(compiled: &CompiledModel<'_>, row: &[f64]) -> (u64, f64) {
    let start = Instant::now();
    let prediction = compiled.predict(row).unwrap();
    let elapsed = start.elapsed().as_nanos() as u64;
    black_box(prediction);
    (elapsed, prediction)
}

fn timed_empty() -> u64 {
    let start = Instant::now();
    black_box(());
    start.elapsed().as_nanos() as u64
}

#[allow(clippy::too_many_arguments)]
fn measure_condition(
    compiled: &CompiledModel<'_>,
    rows: &[Vec<f64>],
    scores: &[f64],
    row_order: &[usize; ROWS],
    sequence: RowSequence,
    condition: Condition,
    pollution: &mut [PollutionBuffer; 4],
    other: &CompiledModel<'_>,
    other_rows: &[Vec<f64>],
    sample_count: usize,
) -> Vec<u64> {
    let mut row_buffer = vec![0.0; rows[0].len()];
    let mut dependent_index = 0;
    let mut samples = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let row_index = match sequence {
            RowSequence::Shuffled => row_order[sample % ROWS],
            RowSequence::Dependent => dependent_index,
        };
        if let Some(index) = condition.data_index() {
            pollution[index].disturb();
        }
        if condition == Condition::OtherModel {
            let prediction = other
                .predict(black_box(other_rows[sample % ROWS].as_slice()))
                .unwrap();
            black_box(prediction);
        }
        row_buffer.copy_from_slice(&rows[row_index]);
        black_box(row_buffer.as_slice());
        let (elapsed, prediction) = if condition.calls_model() {
            let (elapsed, prediction) = timed_prediction(compiled, row_buffer.as_slice());
            (elapsed, prediction)
        } else {
            (timed_empty(), scores[row_index])
        };
        samples.push(elapsed);
        if sequence == RowSequence::Dependent {
            dependent_index = next_row(dependent_index, prediction);
        }
    }
    samples
}

#[allow(clippy::too_many_arguments)]
fn measure_sequence(
    compiled: &CompiledModel<'_>,
    rows: &[Vec<f64>],
    scores: &[f64],
    row_order: &[usize; ROWS],
    sequence: RowSequence,
    pollution: &mut [PollutionBuffer; 4],
    other: &CompiledModel<'_>,
    other_rows: &[Vec<f64>],
    sample_count: usize,
) -> Vec<Measurement> {
    let mut state = match sequence {
        RowSequence::Shuffled => 0x53485546464c4544,
        RowSequence::Dependent => 0x444550454e44454e,
    };
    let mut order = Condition::ALL;
    let mut kept = Vec::with_capacity(order.len());
    for pass in 0..2 {
        shuffle(&mut order, &mut state);
        for &condition in &order {
            let samples = measure_condition(
                compiled,
                rows,
                scores,
                row_order,
                sequence,
                condition,
                pollution,
                other,
                other_rows,
                sample_count,
            );
            if pass == 1 {
                kept.push(Measurement {
                    sequence,
                    condition,
                    samples,
                });
            }
        }
    }
    kept
}

fn benchmark(arguments: &Arguments, config: BenchmarkConfig) -> Result<Report, Box<dyn Error>> {
    let (model, rows) = workload(arguments.workload);
    let tree_count = model.trees().len();
    let feature_count = model.feature_modes().len();
    let context = Context::create();
    let compiled = compile(&context, &plan(&model, SortOptions::default())?)?;
    let (scores, distinct_rows) = verify(&model, &compiled, &rows);
    let timer_resolution_ns = timer_resolution()?;
    for call in 0..config.warmup_calls {
        black_box(
            compiled
                .predict(black_box(rows[call % ROWS].as_slice()))
                .unwrap(),
        );
    }

    let (other_model, other_rows) = workload(Workload::Distinct(
        config.other_trees,
        config.other_features,
    ));
    let other = compile(&context, &plan(&other_model, SortOptions::default())?)?;
    let mut pollution = std::array::from_fn(|index| {
        PollutionBuffer::new(config.data_bytes[index], 0x504f4c4c5554494f ^ index as u64)
    });
    for buffer in &mut pollution {
        buffer.touch_pages();
    }
    let row_order = shuffled_rows();
    let mut measurements = Vec::with_capacity(Condition::ALL.len() * 2);
    for sequence in [RowSequence::Shuffled, RowSequence::Dependent] {
        measurements.extend(measure_sequence(
            &compiled,
            &rows,
            &scores,
            &row_order,
            sequence,
            &mut pollution,
            &other,
            &other_rows,
            arguments.samples,
        ));
    }
    Ok(Report {
        workload_name: arguments.workload_name.clone(),
        tree_count,
        feature_count,
        timer_resolution_ns,
        distinct_rows,
        sample_count: arguments.samples,
        measurements,
    })
}

fn print_report(report: &Report) {
    println!(
        "Workload {}: {} trees, depth 6, {} features, {ROWS} rows",
        report.workload_name, report.tree_count, report.feature_count
    );
    println!(
        "timer resolution about {:.1} ns",
        report.timer_resolution_ns
    );
    println!(
        "verify dependent chain visits {} distinct rows",
        report.distinct_rows
    );
    for sequence in [RowSequence::Shuffled, RowSequence::Dependent] {
        println!();
        println!(
            "{} ({} samples per condition)",
            sequence.label(),
            report.sample_count
        );
        println!(
            "{:<29} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "condition", "mean", "p50", "p90", "p99", "max"
        );
        for condition in Condition::ALL {
            let measurement = report
                .measurements
                .iter()
                .find(|measurement| {
                    measurement.sequence == sequence && measurement.condition == condition
                })
                .unwrap();
            let values = statistics(&measurement.samples);
            println!(
                "{:<29} {:>9.1} {:>9} {:>9} {:>9} {:>9}",
                condition.label(),
                values.mean,
                values.p50,
                values.p90,
                values.p99,
                values.max
            );
        }
    }
}

fn write_csv(path: &PathBuf, report: &Report) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "condition,ns")?;
    for sequence in [RowSequence::Shuffled, RowSequence::Dependent] {
        for condition in Condition::ALL {
            let measurement = report
                .measurements
                .iter()
                .find(|measurement| {
                    measurement.sequence == sequence && measurement.condition == condition
                })
                .unwrap();
            for sample in &measurement.samples {
                writeln!(
                    writer,
                    "{} {},{}",
                    sequence.label(),
                    condition.label(),
                    sample
                )?;
            }
        }
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments(std::env::args().skip(1))
        .map_err(|error| -> Box<dyn Error> { error.into() })?;
    let report = benchmark(&arguments, BenchmarkConfig::production())?;
    print_report(&report);
    if let Some(path) = &arguments.out {
        write_csv(path, &report)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Result<Arguments, &'static str> {
        parse_arguments(values.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn cli_parses_named_and_generic_workloads() {
        assert_eq!(
            arguments(&["100x100"]).unwrap().workload,
            Workload::Distinct(100, 100)
        );
        assert_eq!(
            arguments(&["300x100"]).unwrap().workload,
            Workload::Distinct(300, 100)
        );
        assert_eq!(arguments(&["mixed"]).unwrap().workload, Workload::Mixed);
        for invalid in ["0x5", "5x0", "abc", "20000x100"] {
            assert_eq!(arguments(&[invalid]), Err(USAGE));
        }
    }

    #[test]
    fn shuffled_sequence_is_a_permutation_and_deterministic() {
        let first = shuffled_rows();
        let second = shuffled_rows();
        assert_eq!(first, second);
        let mut seen = [false; ROWS];
        for index in first {
            assert!(!seen[index]);
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn pollution_touches_every_line() {
        let mut pollution = PollutionBuffer::new(256 << 10, 7);
        assert_eq!(pollution.lines.as_ptr() as usize % 64, 0);
        assert_eq!(pollution.lines.len(), (256 << 10) / 64);
        pollution.disturb();
        for line in &pollution.lines {
            assert_eq!(line.0[0], 1);
        }
    }

    #[test]
    fn verified_kernel_matches_reference_before_timing() {
        let (model, rows) = workload(Workload::Distinct(20, 30));
        let context = Context::create();
        let compiled = compile(&context, &plan(&model, SortOptions::default()).unwrap()).unwrap();
        let (scores, _) = verify(&model, &compiled, &rows);
        assert_eq!(scores.len(), ROWS);
        for (row, score) in rows.iter().zip(scores) {
            assert_eq!(score, model.predict(row).unwrap());
        }
    }

    #[test]
    fn percentiles_are_computed_from_raw_samples() {
        assert_eq!(
            statistics(&[10, 20, 30, 40, 50]),
            Statistics {
                mean: 30.0,
                p50: 30,
                p90: 50,
                p99: 50,
                max: 50,
            }
        );
    }

    #[test]
    fn one_sample_round_trip() {
        let arguments = arguments(&["5x8", "--samples", "20"]).unwrap();
        let report = benchmark(
            &arguments,
            BenchmarkConfig {
                data_bytes: [64, 128, 256, 512],
                other_trees: 5,
                other_features: 8,
                warmup_calls: 10,
            },
        )
        .unwrap();
        assert_eq!(report.measurements.len(), Condition::ALL.len() * 2);
        for sequence in [RowSequence::Shuffled, RowSequence::Dependent] {
            for condition in Condition::ALL {
                let matching: Vec<_> = report
                    .measurements
                    .iter()
                    .filter(|measurement| {
                        measurement.sequence == sequence && measurement.condition == condition
                    })
                    .collect();
                assert_eq!(matching.len(), 1);
                assert_eq!(matching[0].samples.len(), 20);
            }
        }
    }
}
