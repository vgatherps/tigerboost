use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::error::Error;
use std::time::{Duration, Instant};

#[path = "cranelift_probe/sweep.rs"]
mod sweep;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Immediate,
    Forward,
    Reverse,
}

struct Sequence {
    feature_count: usize,
    uses: Vec<usize>,
}

impl Sequence {
    fn new(pattern: Pattern, feature_count: usize) -> Result<Self> {
        feature_count
            .checked_sub(1)
            .and_then(|i| i.checked_mul(size_of::<f64>()))
            .and_then(|offset| i32::try_from(offset).ok())
            .ok_or("feature count must be nonzero and fit Cranelift load offsets")?;
        let uses = match pattern {
            Pattern::Immediate => (0..feature_count).flat_map(|i| [i, i]).collect(),
            Pattern::Forward => (0..feature_count).chain(0..feature_count).collect(),
            Pattern::Reverse => (0..feature_count).chain((0..feature_count).rev()).collect(),
        };
        Ok(Self {
            feature_count,
            uses,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Optimization {
    None,
    Speed,
}

impl Optimization {
    fn setting(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Speed => "speed",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Options {
    readonly: bool,
    can_move: bool,
    optimization: Optimization,
    accumulators: usize,
}

struct OwnedModule(Option<JITModule>);

impl Drop for OwnedModule {
    fn drop(&mut self) {
        // The entry pointer never escapes Probe; no invocation survives its borrow.
        unsafe { self.0.take().unwrap().free_memory() };
    }
}

struct Probe {
    _module: OwnedModule,
    entry: unsafe extern "C" fn(*const f64) -> f64,
    feature_count: usize,
    target: String,
    settings: String,
    before: String,
    after: String,
    disassembly: String,
    code_bytes: usize,
    compile_time: Duration,
}

impl Probe {
    fn compile(sequence: &Sequence, options: Options) -> Result<Self> {
        if ![1, 4].contains(&options.accumulators) {
            return Err("expected 1 or 4 accumulators".into());
        }
        let start = Instant::now();
        let mut flags = settings::builder();
        flags.set("opt_level", options.optimization.setting())?;
        flags.set("enable_verifier", "true")?;
        flags.set("regalloc_checker", "true")?;
        let isa =
            cranelift_native::builder_with_options(false)?.finish(settings::Flags::new(flags))?;
        let target = isa.triple().to_string();
        let isa_flags: Vec<_> = isa.isa_flags().iter().map(ToString::to_string).collect();
        let settings = format!("{}\n{}", isa.flags(), isa_flags.join("\n"));
        let mut owner = OwnedModule(Some(JITModule::new(JITBuilder::with_isa(
            isa,
            default_libcall_names(),
        ))));
        let module = owner.0.as_mut().unwrap();
        let mut context = module.make_context();
        context.set_disasm(true);
        context
            .func
            .signature
            .params
            .push(AbiParam::new(module.target_config().pointer_type()));
        context
            .func
            .signature
            .returns
            .push(AbiParam::new(types::F64));
        let function = module.declare_function("probe", Linkage::Local, &context.func.signature)?;
        let mut frontend = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_all_blocks();
        let input = builder.block_params(block)[0];
        let mut memory = MemFlags::trusted();
        if options.readonly {
            memory.set_readonly();
        }
        if options.can_move {
            memory.set_can_move();
        }
        let zero = builder.ins().f64const(0.0);
        let mut sums = vec![zero; options.accumulators];
        for (use_index, &feature) in sequence.uses.iter().enumerate() {
            let offset = i32::try_from(feature * size_of::<f64>())?;
            let value = builder.ins().load(types::F64, memory, input, offset);
            let lane = use_index % sums.len();
            sums[lane] = builder.ins().fadd(sums[lane], value);
        }
        let sum = sums
            .into_iter()
            .reduce(|a, b| builder.ins().fadd(a, b))
            .unwrap();
        builder.ins().return_(&[sum]);
        builder.finalize();
        let before = context.func.display().to_string();
        module.define_function(function, &mut context)?;
        context.verify(module.isa())?;
        let after = context.func.display().to_string();
        let compiled = context.compiled_code().ok_or("missing compiled code")?;
        let disassembly = compiled
            .vcode
            .clone()
            .ok_or("missing backend disassembly")?;
        let code_bytes = compiled.code_buffer().len();
        module.finalize_definitions()?;
        // Native C ABI: one pointer parameter and one f64 result, as declared above.
        let entry = unsafe {
            std::mem::transmute::<*const u8, unsafe extern "C" fn(*const f64) -> f64>(
                module.get_finalized_function(function),
            )
        };
        Ok(Self {
            _module: owner,
            entry,
            feature_count: sequence.feature_count,
            target,
            settings,
            before,
            after,
            disassembly,
            code_bytes,
            compile_time: start.elapsed(),
        })
    }

    fn run(&self, row: &[f64]) -> Result<f64> {
        if row.len() < self.feature_count {
            return Err("input row is shorter than the probe's feature count".into());
        }
        // Every offset is in this aligned, immutable slice; self owns the executable code.
        Ok(unsafe { (self.entry)(row.as_ptr()) })
    }
}

fn reference(sequence: &Sequence, row: &[f64]) -> f64 {
    sequence
        .uses
        .iter()
        .fold(0.0, |sum, &feature| sum + row[feature])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["sweep"] {
        return sweep::run();
    }
    let [
        pattern,
        count,
        readonly,
        can_move,
        optimization,
        accumulators,
    ] = args.as_slice()
    else {
        return Err("usage: cranelift_probe sweep | immediate|forward|reverse FEATURE_COUNT ordinary|readonly fixed|movable none|speed 1|4".into());
    };
    let pattern = match pattern.as_str() {
        "immediate" => Pattern::Immediate,
        "forward" => Pattern::Forward,
        "reverse" => Pattern::Reverse,
        _ => return Err("unknown use sequence".into()),
    };
    let options = Options {
        readonly: match readonly.as_str() {
            "ordinary" => false,
            "readonly" => true,
            _ => return Err("expected ordinary or readonly".into()),
        },
        can_move: match can_move.as_str() {
            "fixed" => false,
            "movable" => true,
            _ => return Err("expected fixed or movable".into()),
        },
        optimization: match optimization.as_str() {
            "none" => Optimization::None,
            "speed" => Optimization::Speed,
            _ => return Err("expected none or speed".into()),
        },
        accumulators: accumulators.parse()?,
    };
    let sequence = Sequence::new(pattern, count.parse()?)?;
    let probe = Probe::compile(&sequence, options)?;
    let row: Vec<f64> = (0..sequence.feature_count)
        .map(|i| i as f64 + 0.25)
        .collect();
    println!(
        "Cranelift 0.128.2; target {}; {pattern:?}; {options:?}",
        probe.target
    );
    println!(
        "{}\ncode bytes: {}; compile us (with diagnostics): {:.1}; result: {}",
        probe.settings,
        probe.code_bytes,
        probe.compile_time.as_secs_f64() * 1e6,
        probe.run(&row)?
    );
    println!("before optimization:\n{}", probe.before);
    println!("after optimization:\n{}", probe.after);
    println!("post-allocation backend listing:\n{}", probe.disassembly);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_are_explicit_and_validated() {
        assert_eq!(
            Sequence::new(Pattern::Immediate, 3).unwrap().uses,
            [0, 0, 1, 1, 2, 2]
        );
        assert_eq!(
            Sequence::new(Pattern::Forward, 3).unwrap().uses,
            [0, 1, 2, 0, 1, 2]
        );
        assert_eq!(
            Sequence::new(Pattern::Reverse, 3).unwrap().uses,
            [0, 1, 2, 2, 1, 0]
        );
        assert!(Sequence::new(Pattern::Forward, 0).is_err());
        assert!(Sequence::new(Pattern::Forward, i32::MAX as usize / 8 + 2).is_err());
        assert!(Sequence::new(Pattern::Forward, usize::MAX).is_err());
    }

    #[test]
    fn every_feature_contributes_under_all_options() -> Result<()> {
        for pattern in [Pattern::Immediate, Pattern::Forward, Pattern::Reverse] {
            for feature_count in [16, 64, 128] {
                let sequence = Sequence::new(pattern, feature_count)?;
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
                                let probe = Probe::compile(&sequence, options)?;
                                let row: Vec<_> = (0..feature_count)
                                    .map(|i| (i as f64 - 8.0) * 0.125)
                                    .collect();
                                let expected = reference(&sequence, &row);
                                assert_eq!(expected, 2.0 * row.iter().sum::<f64>());
                                assert_eq!(probe.run(&row)?, expected, "{pattern:?} {options:?}");
                                for feature in 0..row.len() {
                                    let mut changed = row.clone();
                                    changed[feature] += 0.25;
                                    let expected_changed = reference(&sequence, &changed);
                                    assert_eq!(expected_changed, expected + 0.5);
                                    assert_eq!(
                                        probe.run(&changed)?,
                                        expected_changed,
                                        "{pattern:?} {options:?} feature {feature}"
                                    );
                                }
                                assert_eq!(probe.run(&row)?, expected);
                                assert!(!probe.before.is_empty());
                                assert_eq!(probe.before.contains(" readonly"), readonly);
                                assert_eq!(probe.before.contains(" can_move"), can_move);
                                assert!(!probe.after.is_empty());
                                assert!(!probe.disassembly.is_empty());
                                assert!(probe.code_bytes > 0);
                                assert!(probe.run(&row[..feature_count - 1]).is_err());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn executable_survives_moves_and_other_modules_being_freed() -> Result<()> {
        let sequence = Sequence::new(Pattern::Forward, 1)?;
        let options = Options {
            readonly: true,
            can_move: true,
            optimization: Optimization::Speed,
            accumulators: 1,
        };
        let mut probes = vec![Probe::compile(&sequence, options)?];
        let other = Probe::compile(&sequence, options)?;
        let probe = probes.pop().unwrap();
        drop(probes);
        drop(other);
        assert_eq!(probe.run(&[3.5])?, 7.0);
        assert_eq!(probe.run(&[-2.0])?, -4.0);
        assert!(probe.run(&[]).is_err());
        Ok(())
    }

    #[test]
    fn accumulator_count_is_validated() -> Result<()> {
        let sequence = Sequence::new(Pattern::Immediate, 1)?;
        for accumulators in [0, 2, 3, 5, usize::MAX] {
            assert!(
                Probe::compile(
                    &sequence,
                    Options {
                        readonly: true,
                        can_move: true,
                        optimization: Optimization::Speed,
                        accumulators,
                    }
                )
                .is_err()
            );
        }
        Ok(())
    }
}
