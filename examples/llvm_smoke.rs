//! Toolchain and ownership check; this is not a latency benchmark.

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::values::{BasicValue, FastMathFlags};
use inkwell::{AddressSpace, OptimizationLevel};
use std::error::Error;

type SumFn = unsafe extern "C" fn(*const f64) -> f64;
type CompileResult<T> = Result<T, Box<dyn Error>>;

struct Kernel<'ctx> {
    // JitFunction retains the engine; its lifetime keeps the Context borrowed.
    function: JitFunction<'ctx, SumFn>,
    pre_ir: String,
    post_ir: String,
    target_assembly: String,
}

impl Kernel<'_> {
    fn sum(&self, row: &[f64]) -> Result<f64, &'static str> {
        if row.len() < 3 {
            return Err("kernel requires at least three features");
        }
        // SAFETY: row supplies three aligned, initialized f64 values, remains
        // immutable during the call, and self.function retains executable memory.
        Ok(unsafe { self.function.call(row.as_ptr()) })
    }
}

fn compile(context: &Context) -> CompileResult<Kernel<'_>> {
    Target::initialize_native(&InitializationConfig::default())?;
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)?;
    let machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Less,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("could not create native target machine")?;
    let module = context.create_module("llvm_smoke");
    module.set_triple(&triple);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    let f64_type = context.f64_type();
    let pointer_type = context.ptr_type(AddressSpace::default());
    let function = module.add_function(
        "sum_three",
        f64_type.fn_type(&[pointer_type.into()], false),
        None,
    );
    function.set_call_conventions(0); // LLVM's C calling convention.
    for (name, value) in [("readonly", 0), ("align", 8), ("dereferenceable", 24)] {
        let kind = Attribute::get_named_enum_kind_id(name);
        assert_ne!(kind, 0, "missing LLVM attribute {name}");
        function.add_attribute(
            AttributeLoc::Param(0),
            context.create_enum_attribute(kind, value),
        );
    }
    // LLVM 21 ModRef.h: ArgMem = 0, Ref = 1, two bits per memory location.
    let memory_kind = Attribute::get_named_enum_kind_id("memory");
    assert_ne!(memory_kind, 0);
    function.add_attribute(
        AttributeLoc::Function,
        context.create_enum_attribute(memory_kind, 1),
    );
    for (name, value) in [("target-cpu", "generic"), ("target-features", "")] {
        function.add_attribute(
            AttributeLoc::Function,
            context.create_string_attribute(name, value),
        );
    }
    let builder = context.create_builder();
    builder.position_at_end(context.append_basic_block(function, "entry"));
    let row = function.get_first_param().unwrap().into_pointer_value();
    let load = |index| -> CompileResult<_> {
        // SAFETY: each index is 0..3; the checked entry guarantees these elements.
        let address = unsafe {
            builder.build_in_bounds_gep(
                f64_type,
                row,
                &[context.i64_type().const_int(index, false)],
                "feature_ptr",
            )?
        };
        let value = builder.build_load(f64_type, address, "feature")?;
        value.as_instruction_value().unwrap().set_alignment(8)?;
        Ok(value.into_float_value())
    };
    let first = load(0)?;
    let second = load(1)?;
    let sum = builder.build_float_add(first, second, "sum")?;
    sum.as_instruction_value()
        .unwrap()
        .set_fast_math_flags(FastMathFlags::AllowReassoc)?;
    let third = load(2)?;
    let total = builder.build_float_add(sum, third, "total")?;
    total
        .as_instruction_value()
        .unwrap()
        .set_fast_math_flags(FastMathFlags::AllowReassoc)?;
    builder.build_return(Some(&total))?;

    module.verify()?;
    let pre_ir = module.print_to_string().to_string();
    module.run_passes(
        "function(early-cse)",
        &machine,
        PassBuilderOptions::create(),
    )?;
    module.verify()?;
    let post_ir = module.print_to_string().to_string();
    // Separate target emission: these are not bytes read back from MCJIT.
    let assembly_module = module.clone();
    let assembly = machine.write_to_memory_buffer(&assembly_module, FileType::Assembly)?;
    let target_assembly = String::from_utf8(assembly.as_slice().to_vec())?;
    let engine = module.create_jit_execution_engine(OptimizationLevel::Less)?;
    // SAFETY: sum_three is exactly C ABI double(const double*), as built above.
    let function = unsafe { engine.get_function::<SumFn>("sum_three")? };
    Ok(Kernel {
        function,
        pre_ir,
        post_ir,
        target_assembly,
    })
}

fn main() -> CompileResult<()> {
    let mut version = [0; 3];
    // SAFETY: these are three valid, distinct writable u32 pointers.
    unsafe { llvm_sys::core::LLVMGetVersion(&mut version[0], &mut version[1], &mut version[2]) };
    let context = Context::create();
    let kernel = compile(&context)?;
    println!(
        "LLVM {}.{}.{}; generic CPU; early-cse; backend Less",
        version[0], version[1], version[2]
    );
    println!("sum([1, 2, 3]) = {}", kernel.sum(&[1.0, 2.0, 3.0])?);
    println!("Pre-pass IR:\n{}", kernel.pre_ir);
    println!("Post-pass IR:\n{}", kernel.post_ir);
    println!(
        "Separate target assembly (not MCJIT byte disassembly):\n{}",
        kernel.target_assembly
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_rows_and_every_feature_affect_the_result() {
        let context = Context::create();
        let kernel = compile(&context).unwrap();
        let mut row = [1.0, 2.0, 4.0];
        assert_eq!(kernel.sum(&row), Ok(7.0));
        for feature in 0..3 {
            row[feature] += 8.0;
            assert_eq!(kernel.sum(&row), Ok(row.iter().sum()));
        }
        assert_eq!(kernel.sum(&[-8.0, 2.5, 4.0, 100.0]), Ok(-1.5));
        assert_eq!(row, [9.0, 10.0, 12.0]);
    }

    #[test]
    fn rejects_short_rows_before_calling_generated_code() {
        let context = Context::create();
        let kernel = compile(&context).unwrap();
        for length in 0..3 {
            assert!(kernel.sum(&[1.0, 2.0][..length]).is_err());
        }
    }

    #[test]
    fn moved_kernel_survives_another_kernels_drop() {
        let context = Context::create();
        let kernel = compile(&context).unwrap();
        let mut kernels = vec![kernel];
        {
            let other_context = Context::create();
            let other = compile(&other_context).unwrap();
            assert_eq!(other.sum(&[10.0, 20.0, 30.0]), Ok(60.0));
        }
        let moved = kernels.pop().unwrap();
        assert_eq!(moved.sum(&[1.0, 2.0, 4.0]), Ok(7.0));
    }

    #[test]
    fn nan_infinity_and_signed_zero_are_not_assumed_away() {
        let context = Context::create();
        let kernel = compile(&context).unwrap();
        for feature in 0..3 {
            let mut row = [1.0, 2.0, 4.0];
            row[feature] = f64::NAN;
            assert!(kernel.sum(&row).unwrap().is_nan());
            row[feature] = f64::INFINITY;
            assert_eq!(kernel.sum(&row), Ok(f64::INFINITY));
            row[feature] = f64::NEG_INFINITY;
            assert_eq!(kernel.sum(&row), Ok(f64::NEG_INFINITY));
        }
        assert!(
            kernel
                .sum(&[f64::INFINITY, f64::NEG_INFINITY, 0.0])
                .unwrap()
                .is_nan()
        );
        assert_eq!(
            kernel.sum(&[-0.0; 3]).unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn ir_records_readonly_memory_and_only_reassociation() {
        let context = Context::create();
        let kernel = compile(&context).unwrap();
        for ir in [&kernel.pre_ir, &kernel.post_ir] {
            assert!(ir.contains("readonly align 8 dereferenceable(24)"), "{ir}");
            assert!(ir.contains("memory(argmem: read)"), "{ir}");
            assert!(ir.contains("\"target-cpu\"=\"generic\""), "{ir}");
            assert_eq!(ir.matches("fadd reassoc double").count(), 2, "{ir}");
            assert_eq!(ir.matches("load double").count(), 3, "{ir}");
            for forbidden in [
                "invariant.load",
                "store ",
                "nnan",
                "ninf",
                "nsz",
                "fast double",
            ] {
                assert!(!ir.contains(forbidden), "{forbidden}: {ir}");
            }
        }
        assert!(kernel.target_assembly.contains("sum_three"));
    }
}
