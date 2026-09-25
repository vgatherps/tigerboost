use super::{Optimization, Options, Pattern, Probe, Result, Sequence, reference};
use std::fmt::Write;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ROWS: usize = 32;
const ROUNDS: usize = 9;
const PASSES: usize = 512;

struct Case {
    name: String,
    probe: Probe,
    rows: Vec<Vec<f64>>,
    samples: Vec<f64>,
}

fn rows(features: usize) -> Vec<Vec<f64>> {
    (0..ROWS)
        .map(|row| {
            (0..features)
                .map(|feature| ((row * 17 + feature * 29) % 257) as f64 / 32.0 - 4.0)
                .collect()
        })
        .collect()
}

fn round_order(round: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| {
            let position = if round.is_multiple_of(2) {
                i
            } else {
                count - i - 1
            };
            (position + round * 37) % count
        })
        .collect()
}

fn batch(probe: &Probe, rows: &[Vec<f64>], passes: usize) -> Result<f64> {
    let start = Instant::now();
    for _ in 0..passes {
        for row in rows {
            black_box(probe.run(black_box(row.as_slice()))?);
        }
    }
    Ok(start.elapsed().as_secs_f64() * 1e9 / (passes * rows.len()) as f64)
}

fn statistics(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() - 1],
    )
}

fn create_run_directory(parent: &Path, name: &str) -> Result<PathBuf> {
    fs::create_dir_all(parent)?;
    let directory = parent.join(name);
    fs::create_dir(&directory)?;
    Ok(directory)
}

pub(super) fn run() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("timing sweep requires cargo run --release".into());
    }
    let run_id = format!(
        "{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        std::process::id()
    );
    let directory = create_run_directory(Path::new("target/cranelift-probe"), &run_id)?;
    fs::write(
        directory.join("Cargo.lock"),
        include_str!("../../Cargo.lock"),
    )?;
    fs::write(
        directory.join("method.txt"),
        format!(
            "rows={ROWS}; rounds={ROUNDS}; passes={PASSES}; calls_per_batch={}\n\
         batch means include checked wrapper, iteration and black_box; no subtraction\n\
         independent calls may overlap in hardware; not isolated single-row latency\n\
         warmup: 8 row passes initially, 1 row pass before each measured batch\n\
         compile_us includes ISA/module setup, verification, regalloc checker, IR/disassembly capture and finalization\n\
         native ISA defaults; CPU feature inference disabled; hot rows and kernels\n",
            ROWS * PASSES
        ),
    )?;
    let mut cases = Vec::new();
    for pattern in [Pattern::Immediate, Pattern::Forward, Pattern::Reverse] {
        for features in [4, 16, 64, 128] {
            let sequence = Sequence::new(pattern, features)?;
            for readonly in [false, true] {
                for can_move in [false, true] {
                    for optimization in [Optimization::None, Optimization::Speed] {
                        for accumulators in [1, 4] {
                            let options = Options {
                                readonly,
                                can_move,
                                optimization,
                                accumulators,
                            };
                            let name = format!(
                                "{pattern:?}-{features}-ro{readonly}-move{can_move}-{}-sums{accumulators}",
                                optimization.setting()
                            );
                            let probe = Probe::compile(&sequence, options)?;
                            let rows = rows(features);
                            for row in &rows {
                                if probe.run(row)? != reference(&sequence, row) {
                                    return Err(format!("reference mismatch: {name}").into());
                                }
                            }
                            let artifact = create_run_directory(&directory, &name)?;
                            for (file, data) in [
                                ("before.clif", probe.before.as_str()),
                                ("after.clif", probe.after.as_str()),
                                ("backend.txt", probe.disassembly.as_str()),
                                ("settings.txt", probe.settings.as_str()),
                            ] {
                                fs::write(artifact.join(file), data)?;
                            }
                            fs::write(
                                artifact.join("metadata.txt"),
                                format!(
                                    "target={}\ncode_bytes={}\ncompile_us={:.3}\n",
                                    probe.target,
                                    probe.code_bytes,
                                    probe.compile_time.as_secs_f64() * 1e6
                                ),
                            )?;
                            cases.push(Case {
                                name,
                                probe,
                                rows,
                                samples: Vec::new(),
                            });
                        }
                    }
                }
            }
        }
    }
    for case in &cases {
        batch(&case.probe, &case.rows, 8)?;
    }
    for round in 0..ROUNDS {
        for index in round_order(round, cases.len()) {
            let case = &mut cases[index];
            batch(&case.probe, &case.rows, 1)?;
            case.samples.push(batch(&case.probe, &case.rows, PASSES)?);
        }
    }
    let mut summary = String::from("variant,code_bytes,compile_us,median_ns,min_ns,max_ns\n");
    let mut samples = String::from("variant,round,mean_ns\n");
    for case in cases {
        let (median, min, max) = statistics(&case.samples);
        writeln!(
            summary,
            "{},{},{:.3},{median:.3},{min:.3},{max:.3}",
            case.name,
            case.probe.code_bytes,
            case.probe.compile_time.as_secs_f64() * 1e6
        )?;
        for (round, sample) in case.samples.iter().enumerate() {
            writeln!(samples, "{},{round},{sample:.3}", case.name)?;
        }
    }
    fs::write(directory.join("summary.csv"), &summary)?;
    fs::write(directory.join("samples.csv"), samples)?;
    println!("{}\n{}", directory.display(), summary);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaving_visits_each_case_once_per_round() {
        for round in 0..ROUNDS {
            let order = round_order(round, 192);
            assert_ne!(order, round_order(round + 1, 192));
            let mut sorted = order;
            sorted.sort_unstable();
            assert_eq!(sorted, (0..192).collect::<Vec<_>>());
        }
        assert_eq!(statistics(&[5.0, 2.0, 1.0, 4.0, 3.0]), (3.0, 1.0, 5.0));
    }

    #[test]
    fn inputs_are_deterministic_and_vary_each_feature() {
        for count in [4, 16, 64, 128] {
            let generated = rows(count);
            assert_eq!(generated, rows(count));
            for feature in 0..count {
                assert!(
                    generated
                        .iter()
                        .skip(1)
                        .all(|row| row[feature] != generated[0][feature])
                );
            }
        }
    }

    #[test]
    fn run_directory_refuses_existing_artifacts() -> Result<()> {
        let name = format!(
            "tigerboost-probe-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let directory = create_run_directory(&std::env::temp_dir(), &name)?;
        assert!(create_run_directory(&std::env::temp_dir(), &name).is_err());
        fs::remove_dir(directory)?;
        Ok(())
    }
}
