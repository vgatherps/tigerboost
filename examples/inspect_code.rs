//! Writes separately emitted target assembly, not MCJIT byte disassembly.
mod support;

use inkwell::context::Context;
use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    num::NonZeroUsize,
    path::PathBuf,
};
use tigerboost::jit::{ComparisonMode, CompileOptions, Order, emit_assembly};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: inspect_code <fresh-output-directory>")?,
    );
    if arguments.next().is_some() {
        return Err("usage: inspect_code <fresh-output-directory>".into());
    }
    fs::create_dir(&output)?;
    let (model, _) = support::workload();
    let context = Context::create();
    for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
        for (name, options) in [
            ("tree-order-1", CompileOptions::default()),
            (
                "rolling-4-2",
                CompileOptions {
                    order: Order::Rolling(NonZeroUsize::new(4).unwrap()),
                    accumulators: NonZeroUsize::new(2).unwrap(),
                    comparisons: ComparisonMode::Scalar,
                },
            ),
            (
                "simd2-rolling-4-2",
                CompileOptions {
                    order: Order::Rolling(NonZeroUsize::new(4).unwrap()),
                    accumulators: NonZeroUsize::new(2).unwrap(),
                    comparisons: ComparisonMode::Simd2,
                },
            ),
        ] {
            let assembly = emit_assembly(&context, &model, options, triple)?;
            let path = output.join(format!("{triple}-{name}.s"));
            File::create_new(&path)?.write_all(assembly.as_bytes())?;
            println!("{}", path.display());
        }
    }
    Ok(())
}
