use crate::parsed_tree::{NanMode, RowLengthError, Trees};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::{BasicValue, FastMathFlags, FloatValue};
use inkwell::{AddressSpace, FloatPredicate, OptimizationLevel};
use std::error::Error;

type PredictFn = unsafe extern "C" fn(*const f64) -> f64;
type CompileResult<T> = Result<T, Box<dyn Error>>;

pub struct CompiledModel<'ctx> {
    // Retains the engine, including executable code and private constant globals.
    function: JitFunction<'ctx, PredictFn>,
    feature_count: usize,
}

impl CompiledModel<'_> {
    pub fn predict(&self, row: &[f64]) -> Result<f64, RowLengthError> {
        if row.len() < self.feature_count {
            return Err(RowLengthError {
                expected: self.feature_count,
                actual: row.len(),
            });
        }
        // SAFETY: row covers all validated feature indices, is aligned and stays
        // immutable during the call. The retained function owns code and tables.
        Ok(unsafe { self.function.call(row.as_ptr()) })
    }
}

pub fn compile<'ctx>(context: &'ctx Context, model: &Trees) -> CompileResult<CompiledModel<'ctx>> {
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
    let module = emit_model(context, model)?;
    module.set_triple(&triple);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.verify()?;
    module.run_passes(
        "function(early-cse)",
        &machine,
        PassBuilderOptions::create(),
    )?;
    module.verify()?;
    let engine = module.create_jit_execution_engine(OptimizationLevel::Less)?;
    // SAFETY: emit_model builds exactly C ABI double(const double*).
    let function = unsafe { engine.get_function::<PredictFn>("predict")? };
    Ok(CompiledModel {
        function,
        feature_count: model.feature_modes().len(),
    })
}

fn emit_model<'ctx>(context: &'ctx Context, model: &Trees) -> CompileResult<Module<'ctx>> {
    let module = context.create_module("tigerboost");
    let builder = context.create_builder();
    let f64_type = context.f64_type();
    let index_type = context.i64_type();
    let zero = index_type.const_zero();
    let function = module.add_function(
        "predict",
        f64_type.fn_type(&[context.ptr_type(AddressSpace::default()).into()], false),
        None,
    );
    function.set_call_conventions(0);
    let attribute = |location, name, value| {
        let kind = Attribute::get_named_enum_kind_id(name);
        assert_ne!(kind, 0, "missing LLVM attribute {name}");
        function.add_attribute(location, context.create_enum_attribute(kind, value));
    };
    attribute(AttributeLoc::Param(0), "readonly", 0);
    attribute(AttributeLoc::Param(0), "align", 8);
    let row_bytes = model.feature_modes().len() * size_of::<f64>();
    if row_bytes != 0 {
        attribute(AttributeLoc::Param(0), "dereferenceable", row_bytes as u64);
    }
    // LLVM 21: Ref=1 in each of four two-bit memory locations, including globals.
    attribute(AttributeLoc::Function, "memory", 0b01_01_01_01);
    for (name, value) in [("target-cpu", "generic"), ("target-features", "")] {
        function.add_attribute(
            AttributeLoc::Function,
            context.create_string_attribute(name, value),
        );
    }
    builder.position_at_end(context.append_basic_block(function, "entry"));
    let row = function.get_first_param().unwrap().into_pointer_value();
    let mut sum = f64_type.const_zero();
    for (tree_number, tree) in model.trees().iter().enumerate() {
        let leaf_count = u32::try_from(tree.leafs.len())
            .map_err(|_| "leaf table exceeds LLVM array length limit")?;
        let leaf_type = f64_type.array_type(leaf_count);
        let leaves = module.add_global(leaf_type, None, &format!("leaves_{tree_number}"));
        leaves.set_linkage(Linkage::Private);
        leaves.set_constant(true);
        leaves.set_alignment(8);
        let values: Vec<_> = tree
            .leafs
            .iter()
            .map(|&leaf| f64_type.const_float(leaf))
            .collect();
        leaves.set_initializer(&f64_type.const_array(&values));
        let mut index = zero;
        for (level, split) in tree.selections.iter().enumerate() {
            // SAFETY: Trees validates the feature index and row byte span;
            // CompiledModel::predict checks that the caller supplies that span.
            let address = unsafe {
                builder.build_in_bounds_gep(
                    f64_type,
                    row,
                    &[index_type.const_int(split.feature_idx.0 as u64, false)],
                    "feature_ptr",
                )?
            };
            let loaded = builder.build_load(f64_type, address, "feature")?;
            loaded.as_instruction_value().unwrap().set_alignment(8)?;
            let mut value = loaded.into_float_value();
            let mode = model.feature_modes()[split.feature_idx.0];
            if mode != NanMode::AsIs {
                let is_nan =
                    builder.build_float_compare(FloatPredicate::UNO, value, value, "is_nan")?;
                let replacement = if mode == NanMode::AsFalse {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                value = builder
                    .build_select(
                        is_nan,
                        f64_type.const_float(replacement),
                        value,
                        "nan_value",
                    )?
                    .into_float_value();
            }
            let bit = builder.build_float_compare(
                FloatPredicate::OGT,
                value,
                f64_type.const_float(split.threshold),
                "split",
            )?;
            let bit = builder.build_int_z_extend(bit, index_type, "bit")?;
            let bit = builder.build_left_shift(
                bit,
                index_type.const_int(level as u64, false),
                "level_bit",
            )?;
            index = builder.build_or(index, bit, "leaf_index")?;
        }
        // SAFETY: exactly depth bits form index, and Trees requires 2^depth leaves.
        let address = unsafe {
            builder.build_in_bounds_gep(
                leaf_type,
                leaves.as_pointer_value(),
                &[zero, index],
                "leaf_ptr",
            )?
        };
        let leaf = builder.build_load(f64_type, address, "leaf")?;
        leaf.as_instruction_value().unwrap().set_alignment(8)?;
        sum = reassociate(builder.build_float_add(sum, leaf.into_float_value(), "sum")?)?;
    }
    let result = if model.trees().is_empty() {
        f64_type.const_float(model.bias())
    } else {
        let scaled = reassociate(builder.build_float_mul(
            sum,
            f64_type.const_float(model.scale()),
            "scaled",
        )?)?;
        reassociate(builder.build_float_add(
            scaled,
            f64_type.const_float(model.bias()),
            "biased",
        )?)?
    };
    builder.build_return(Some(&result))?;
    Ok(module)
}

fn reassociate(value: FloatValue<'_>) -> CompileResult<FloatValue<'_>> {
    if let Some(instruction) = value.as_instruction_value() {
        instruction.set_fast_math_flags(FastMathFlags::AllowReassoc)?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsed_tree::{Discriminator, FeatureIndex, SingleTree};

    #[test]
    fn ir_has_readonly_rows_constant_leaves_and_only_reassociation() {
        let context = Context::create();
        let model = Trees::new(
            vec![SingleTree {
                leafs: vec![1.0, 2.0],
                selections: vec![Discriminator {
                    feature_idx: FeatureIndex(0),
                    threshold: 0.0,
                }],
            }],
            vec![NanMode::AsIs],
            2.0,
            3.0,
        )
        .unwrap();
        let module = emit_model(&context, &model).unwrap();
        module.verify().unwrap();
        let ir = module.print_to_string().to_string();
        for expected in [
            "private constant [2 x double]",
            "readonly align 8 dereferenceable(8)",
            "memory(read)",
            "fcmp ogt double",
            "fadd reassoc double",
            "fmul reassoc double",
        ] {
            assert!(ir.contains(expected), "{expected}: {ir}");
        }
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
}
