use crate::parsed_tree::{Discriminator, FeatureIndex, NanMode, RowLengthError, SingleTree, Trees};
use crate::sort_policy::{Action, Plan, plan};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::IntType;
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, FastMathFlags, FloatValue, PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, OptimizationLevel};
use std::error::Error;

type PredictFn = unsafe extern "C" fn(*const f64) -> f64;
type CompileResult<T> = Result<T, Box<dyn Error>>;
const BACKEND_OPTIMIZATION: OptimizationLevel = OptimizationLevel::Less;
// LLVM 21 preserve_nonecc; only used between matching continuation signatures.
const PRESERVE_NONE_CALL_CONVENTION: u32 = 21;

#[derive(Clone, Copy, Debug, Default)]
pub struct EmitOptions {
    pub avx2: bool,
}

impl EmitOptions {
    pub fn from_host() -> Self {
        Self {
            avx2: supports_x86_64_v3(&TargetMachine::get_host_cpu_features().to_string()),
        }
    }
}

fn supports_x86_64_v3(features: &str) -> bool {
    [
        "+avx2", "+bmi1", "+bmi2", "+fma", "+lzcnt", "+movbe", "+f16c",
    ]
    .iter()
    .all(|required| features.split(',').any(|feature| feature == *required))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TreePath {
    Lanes2,
    Avx2,
}

impl TreePath {
    fn target_cpu(self) -> &'static str {
        match self {
            Self::Lanes2 => "generic",
            Self::Avx2 => "x86-64-v3",
        }
    }
}

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
    compile_with(context, model, EmitOptions::default())
}

pub fn compile_with<'ctx>(
    context: &'ctx Context,
    model: &Trees,
    options: EmitOptions,
) -> CompileResult<CompiledModel<'ctx>> {
    compile_plan_with(context, &plan(model), options)
}

pub fn compile_plan<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
) -> CompileResult<CompiledModel<'ctx>> {
    compile_plan_with(context, plan, EmitOptions::default())
}

fn compile_plan_with<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    options: EmitOptions,
) -> CompileResult<CompiledModel<'ctx>> {
    if options.avx2 && !EmitOptions::from_host().avx2 {
        return Err("AVX2 emission requires x86-64-v3 host features".into());
    }
    Target::initialize_native(&InitializationConfig::default())?;
    let triple = TargetMachine::get_default_triple();
    let (module, _) = prepare_module(context, plan, &triple, options)?;
    let engine = module.create_jit_execution_engine(BACKEND_OPTIMIZATION)?;
    // SAFETY: emit_plan builds exactly C ABI double(const double*).
    let function = unsafe { engine.get_function::<PredictFn>("predict")? };
    Ok(CompiledModel {
        function,
        feature_count: plan.model().feature_modes().len(),
    })
}

/// Separate target emission for inspection, not disassembly of MCJIT's executable bytes.
pub fn emit_assembly(
    context: &Context,
    plan: &Plan<'_>,
    target_triple: &str,
) -> CompileResult<String> {
    emit_assembly_with(context, plan, target_triple, EmitOptions::default())
}

/// Separate target emission with explicit target options.
pub fn emit_assembly_with(
    context: &Context,
    plan: &Plan<'_>,
    target_triple: &str,
    options: EmitOptions,
) -> CompileResult<String> {
    Target::initialize_all(&InitializationConfig::default());
    let triple = TargetTriple::create(target_triple);
    let (module, machine) = prepare_module(context, plan, &triple, options)?;
    let buffer = machine.write_to_memory_buffer(&module, FileType::Assembly)?;
    let bytes = buffer.as_slice();
    Ok(String::from_utf8(
        bytes.strip_suffix(&[0]).unwrap_or(bytes).to_vec(),
    )?)
}

fn prepare_module<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    triple: &TargetTriple,
    options: EmitOptions,
) -> CompileResult<(Module<'ctx>, TargetMachine)> {
    let tree_path = if options.avx2 && triple.as_str().to_bytes().starts_with(b"x86_64-") {
        TreePath::Avx2
    } else {
        TreePath::Lanes2
    };
    let target = Target::from_triple(triple)?;
    let machine = target
        .create_target_machine(
            triple,
            tree_path.target_cpu(),
            "",
            BACKEND_OPTIMIZATION,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or("could not create target machine")?;
    let module = emit_plan(context, plan, triple, tree_path)?;
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.verify()?;
    module.run_passes(
        "function(early-cse)",
        &machine,
        PassBuilderOptions::create(),
    )?;
    module.verify()?;
    Ok((module, machine))
}

fn emit_plan<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    triple: &TargetTriple,
    tree_path: TreePath,
) -> CompileResult<Module<'ctx>> {
    let model = plan.model();
    let module = context.create_module("tigerboost");
    module.set_triple(triple);
    let architecture = triple.as_str().to_bytes();
    let arm64 = architecture.starts_with(b"aarch64") || architecture.starts_with(b"arm64");
    let (ordered_predicates, lane_swaps) = order_lanes(plan, arm64);
    let builder = context.create_builder();
    let f64_type = context.f64_type();
    let index_type = context.i64_type();
    let zero = index_type.const_zero();
    let chunks = plan.chunks();
    let mut chunk_first_slot = Vec::with_capacity(chunks.len());
    let mut threshold_base = vec![0; model.trees().len()];
    let mut leaf_base = vec![0; model.trees().len()];
    let mut entry_count = 0usize;
    for chunk in chunks {
        let first = entry_count;
        chunk_first_slot.push(first);
        let actions = &plan.actions()[chunk.actions.clone()];
        for action in actions {
            if let Action::EvaluateTree {
                tree, predicates, ..
            } = action
            {
                threshold_base[tree.0] = entry_count;
                entry_count = entry_count
                    .checked_add(predicates.len())
                    .ok_or("constant block exceeds addressable entry count")?;
            }
        }
        entry_count = (entry_count - first)
            .checked_next_multiple_of(8)
            .and_then(|padded| first.checked_add(padded))
            .ok_or("constant block exceeds addressable entry count")?;
        for action in actions {
            let (Action::Constant { tree, .. } | Action::EvaluateTree { tree, .. }) = action;
            leaf_base[tree.0] = entry_count;
            entry_count = entry_count
                .checked_add(model.trees()[tree.0].leafs.len())
                .ok_or("constant block exceeds addressable entry count")?;
        }
    }
    entry_count = entry_count.max(1);
    let array_length =
        u32::try_from(entry_count).map_err(|_| "constant block exceeds LLVM array length limit")?;
    let constant_bytes = entry_count
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| isize::try_from(bytes).ok())
        .ok_or("constant block exceeds addressable byte count")? as u64;
    let constant_type = f64_type.array_type(array_length);
    let constants = module.add_global(constant_type, None, "constants");
    constants.set_linkage(Linkage::Private);
    constants.set_constant(true);
    constants.set_alignment(64);
    let mut values = vec![f64_type.const_zero(); entry_count];
    for action in plan.actions() {
        if let Action::EvaluateTree {
            tree, predicates, ..
        } = action
        {
            for (offset, predicate) in ordered_predicates[predicates.clone()].iter().enumerate() {
                values[threshold_base[tree.0] + offset] = f64_type.const_float(predicate.threshold);
            }
        }
    }
    let mut leaf_bias = vec![0; model.trees().len()];
    for (id, tree) in model.trees().iter().enumerate() {
        if !tree.selections.is_empty() {
            let (bias, leaves) = permuted_leaves(tree, lane_swaps[id], tree_path);
            leaf_bias[id] = bias;
            for (offset, leaf) in leaves.into_iter().enumerate() {
                values[leaf_base[id] + offset] = f64_type.const_float(leaf);
            }
        } else {
            for (offset, &leaf) in tree.leafs.iter().enumerate() {
                values[leaf_base[id] + offset] = f64_type.const_float(leaf);
            }
        }
    }
    constants.set_initializer(&f64_type.const_array(&values));
    let pointer_count = 2 + usize::from(arm64);
    let row_type = context.ptr_type(AddressSpace::default());
    let mut parameter_types = vec![row_type.into(); pointer_count];
    parameter_types.resize(pointer_count + plan.sum_count(), f64_type.into());
    let continuation_type = f64_type.fn_type(&parameter_types, false);
    let wrapper = module.add_function("predict", f64_type.fn_type(&[row_type.into()], false), None);
    let functions: Vec<_> = (0..chunks.len())
        .map(|n| {
            module.add_function(
                &format!("continuation{n}"),
                continuation_type,
                Some(Linkage::Internal),
            )
        })
        .collect();
    for (number, function) in functions.iter().copied().chain(Some(wrapper)).enumerate() {
        function.set_call_conventions(if function == wrapper {
            0
        } else {
            PRESERVE_NONE_CALL_CONVENTION
        });
        let attribute = |location, name, value| {
            let kind = Attribute::get_named_enum_kind_id(name);
            assert_ne!(kind, 0, "missing LLVM attribute {name}");
            function.add_attribute(location, context.create_enum_attribute(kind, value));
        };
        attribute(AttributeLoc::Function, "noinline", 0);
        attribute(AttributeLoc::Param(0), "noalias", 0);
        attribute(AttributeLoc::Param(0), "readonly", 0);
        attribute(AttributeLoc::Param(0), "align", 8);
        let row_bytes = model.feature_modes().len() * size_of::<f64>();
        if row_bytes != 0 {
            attribute(AttributeLoc::Param(0), "dereferenceable", row_bytes as u64);
        }
        if function != wrapper {
            attribute(AttributeLoc::Param(1), "noalias", 0);
            attribute(AttributeLoc::Param(1), "readonly", 0);
            attribute(AttributeLoc::Param(1), "align", 64);
            attribute(AttributeLoc::Param(1), "dereferenceable", constant_bytes);
            if arm64 {
                let remaining_bytes = constant_bytes
                    .checked_sub(
                        (chunk_first_slot[number] as u64)
                            .checked_mul(size_of::<f64>() as u64)
                            .ok_or("chunk base exceeds addressable byte count")?,
                    )
                    .ok_or("chunk base exceeds constant block")?;
                attribute(AttributeLoc::Param(2), "noalias", 0);
                attribute(AttributeLoc::Param(2), "readonly", 0);
                attribute(AttributeLoc::Param(2), "align", 8);
                attribute(AttributeLoc::Param(2), "dereferenceable", remaining_bytes);
            }
        }
        // LLVM 21: Ref=1 in each of four two-bit memory locations, including globals.
        attribute(AttributeLoc::Function, "memory", 0b01_01_01_01);
        let features = if tree_path == TreePath::Avx2 && number + 1 < functions.len() {
            "-vzeroupper"
        } else {
            ""
        };
        for (name, value) in [
            ("target-cpu", tree_path.target_cpu()),
            ("target-features", features),
        ] {
            function.add_attribute(
                AttributeLoc::Function,
                context.create_string_attribute(name, value),
            );
        }
    }
    let arguments = |row: PointerValue<'ctx>,
                     constants: PointerValue<'ctx>,
                     chunk: usize|
     -> CompileResult<Vec<BasicMetadataValueEnum<'ctx>>> {
        let mut args = vec![row.into(), constants.into()];
        if arm64 {
            // SAFETY: each chunk starts inside constants, including the empty model's entry.
            let base = unsafe {
                builder.build_in_bounds_gep(
                    f64_type,
                    constants,
                    &[index_type.const_int(chunk_first_slot[chunk] as u64, false)],
                    "chunk_base",
                )?
            };
            args.push(base.into());
        }
        args.resize(
            pointer_count + plan.sum_count(),
            f64_type.get_undef().into(),
        );
        Ok(args)
    };
    builder.position_at_end(context.append_basic_block(wrapper, "entry"));
    // The first chunk has no carried values and initializes its sums to zero.
    let call = builder.build_call(
        functions[0],
        &arguments(
            wrapper.get_first_param().unwrap().into_pointer_value(),
            constants.as_pointer_value(),
            0,
        )?,
        "prediction",
    )?;
    call.set_call_convention(PRESERVE_NONE_CALL_CONVENTION);
    builder.build_return(Some(&call.try_as_basic_value().unwrap_basic()))?;
    let vsli = if arm64
        && plan.actions().iter().any(|action| {
            matches!(action,
                Action::EvaluateTree { predicates, .. } if predicates.len() > 2
            )
        }) {
        Some(
            Intrinsic::find("llvm.aarch64.neon.vsli")
                .unwrap()
                .get_declaration(&module, &[index_type.vec_type(2).into()])
                .unwrap(),
        )
    } else {
        None
    };
    let avx2_intrinsics = (tree_path == TreePath::Avx2).then(|| {
        let declaration = |name| {
            Intrinsic::find(name)
                .unwrap()
                .get_declaration(&module, &[])
                .unwrap()
        };
        (
            (
                declaration("llvm.x86.avx.cmp.pd.256"),
                declaration("llvm.x86.avx.movmsk.pd.256"),
            ),
            (
                declaration("llvm.x86.sse2.cmp.pd"),
                declaration("llvm.x86.sse2.movmsk.pd"),
            ),
        )
    });
    for (chunk_number, chunk) in chunks.iter().enumerate() {
        let function = functions[chunk_number];
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let row = function.get_first_param().unwrap().into_pointer_value();
        let constants = function.get_nth_param(1).unwrap().into_pointer_value();
        let chunk_base = arm64.then(|| function.get_nth_param(2).unwrap().into_pointer_value());
        let leaf_origin = if arm64 {
            chunk_first_slot[chunk_number]
        } else {
            0
        };
        let mut sums: Vec<_> = (0..plan.sum_count())
            .map(|sum| {
                if chunk_number == 0 {
                    f64_type.const_zero()
                } else {
                    function
                        .get_nth_param((pointer_count + sum) as u32)
                        .unwrap()
                        .into_float_value()
                }
            })
            .collect();
        let add_leaf = |slot, sum| -> CompileResult<FloatValue<'ctx>> {
            // SAFETY: each caller bounds slot to the selected tree's leaf block.
            let address = unsafe {
                builder.build_in_bounds_gep(
                    f64_type,
                    chunk_base.unwrap_or(constants),
                    &[slot],
                    "leaf_ptr",
                )?
            };
            let leaf = builder.build_load(f64_type, address, "leaf")?;
            leaf.as_instruction_value().unwrap().set_alignment(8)?;
            reassociate(builder.build_float_add(sum, leaf.into_float_value(), "sum")?)
        };
        let feature_vector = |group: &[Discriminator], first_level: usize, depth: usize| {
            let mut features = f64_type.vec_type(group.len() as u32).get_undef();
            let mut first_value = None;
            for (lane, predicate) in group.iter().enumerate() {
                let value = if first_level + lane == depth {
                    first_value.expect("padding follows a real level")
                } else {
                    let feature = predicate.feature_idx;
                    let mode = model.feature_modes()[feature.0];
                    load_feature(&builder, f64_type, index_type, row, feature, mode)?
                };
                if lane == 0 {
                    first_value = Some(value);
                }
                features = builder.build_insert_element(
                    features,
                    value,
                    context.i32_type().const_int(lane as u64, false),
                    "features",
                )?;
            }
            Ok::<_, Box<dyn Error>>(features)
        };
        for action in &plan.actions()[chunk.actions.clone()] {
            match action {
                Action::EvaluateTree {
                    tree,
                    predicates,
                    sum,
                } if tree_path == TreePath::Avx2 => {
                    let (wide, narrow) = avx2_intrinsics.unwrap();
                    let mut index = zero;
                    for (group_number, group) in
                        ordered_predicates[predicates.clone()].chunks(4).enumerate()
                    {
                        let lanes = group.len();
                        assert!(lanes == 2 || lanes == 4);
                        let first_level = group_number * 4;
                        let slot = threshold_base[tree.0] + first_level;
                        // SAFETY: consecutive slots cover this group's thresholds in constants.
                        let address = unsafe {
                            builder.build_in_bounds_gep(
                                constant_type,
                                constants,
                                &[zero, index_type.const_int(slot as u64, false)],
                                "threshold_ptr",
                            )?
                        };
                        let vector_type = f64_type.vec_type(lanes as u32);
                        let thresholds = builder.build_load(vector_type, address, "thresholds")?;
                        thresholds
                            .as_instruction_value()
                            .unwrap()
                            .set_alignment(8)?;
                        let features = feature_vector(
                            group,
                            first_level,
                            model.trees()[tree.0].selections.len(),
                        )?;
                        let (compare, movemask) = if lanes == 4 { wide } else { narrow };
                        let masks = builder
                            .build_call(
                                compare,
                                &[
                                    features.into(),
                                    thresholds.into(),
                                    context.i8_type().const_int(30, false).into(),
                                ],
                                "splits",
                            )?
                            .try_as_basic_value()
                            .unwrap_basic();
                        let mask = builder
                            .build_call(movemask, &[masks.into()], "mask")?
                            .try_as_basic_value()
                            .unwrap_basic()
                            .into_int_value();
                        let mask = builder.build_int_z_extend(mask, index_type, "group_index")?;
                        let shifted = if first_level == 0 {
                            mask
                        } else {
                            builder.build_left_shift(
                                mask,
                                index_type.const_int(first_level as u64, false),
                                "upper",
                            )?
                        };
                        index = builder.build_or(index, shifted, "leaf_index")?;
                    }
                    let slot = builder.build_int_nuw_add(
                        index,
                        index_type.const_int(leaf_base[tree.0] as u64, false),
                        "leaf_slot",
                    )?;
                    sums[sum.0] = add_leaf(slot, sums[sum.0])?;
                }
                Action::EvaluateTree {
                    tree,
                    predicates,
                    sum,
                } => {
                    let pairs = predicates.len() / 2;
                    let mut accumulator: Option<inkwell::values::VectorValue<'ctx>> = None;
                    for (pair_number, pair) in ordered_predicates[predicates.clone()]
                        .chunks_exact(2)
                        .enumerate()
                    {
                        let slot = threshold_base[tree.0] + pair_number * 2;
                        // SAFETY: adjacent slots cover both thresholds inside constants.
                        let address = unsafe {
                            if let Some(base) = chunk_base {
                                let offset = slot
                                    .checked_sub(chunk_first_slot[chunk_number])
                                    .ok_or("threshold slot precedes chunk base")?;
                                builder.build_in_bounds_gep(
                                    f64_type,
                                    base,
                                    &[index_type.const_int(offset as u64, false)],
                                    "threshold_ptr",
                                )?
                            } else {
                                builder.build_in_bounds_gep(
                                    constant_type,
                                    constants,
                                    &[zero, index_type.const_int(slot as u64, false)],
                                    "threshold_ptr",
                                )?
                            }
                        };
                        let thresholds =
                            builder.build_load(f64_type.vec_type(2), address, "thresholds")?;
                        thresholds
                            .as_instruction_value()
                            .unwrap()
                            .set_alignment(8)?;
                        let features = feature_vector(
                            pair,
                            pair_number * 2,
                            model.trees()[tree.0].selections.len(),
                        )?;
                        let masks = builder.build_float_compare(
                            FloatPredicate::OGT,
                            features,
                            thresholds.into_vector_value(),
                            "splits",
                        )?;
                        let masks = builder.build_int_s_extend(
                            masks,
                            index_type.vec_type(2),
                            "full_masks",
                        )?;
                        accumulator = Some(if let Some(previous) = accumulator {
                            if let Some(vsli) = vsli {
                                builder
                                    .build_call(
                                        vsli,
                                        &[
                                            previous.into(),
                                            masks.into(),
                                            context
                                                .i32_type()
                                                .const_int(pair_number as u64, false)
                                                .into(),
                                        ],
                                        "acc",
                                    )?
                                    .try_as_basic_value()
                                    .unwrap_basic()
                                    .into_vector_value()
                            } else {
                                let shift = index_type.const_int(pair_number as u64, false);
                                let low = index_type.const_int((1u64 << pair_number) - 1, false);
                                let shifted = builder.build_left_shift(
                                    masks,
                                    inkwell::types::VectorType::const_vector(&[shift, shift]),
                                    "shifted",
                                )?;
                                let kept = builder.build_and(
                                    previous,
                                    inkwell::types::VectorType::const_vector(&[low, low]),
                                    "kept",
                                )?;
                                builder.build_or(shifted, kept, "acc")?
                            }
                        } else {
                            masks
                        });
                    }
                    let accumulator = accumulator.expect("evaluated tree must have splits");
                    let v0 = builder
                        .build_extract_element(accumulator, context.i32_type().const_zero(), "v0")?
                        .into_int_value();
                    let v1 = builder
                        .build_extract_element(
                            accumulator,
                            context.i32_type().const_int(1, false),
                            "v1",
                        )?
                        .into_int_value();
                    let shifted = builder.build_left_shift(
                        v1,
                        index_type.const_int(pairs as u64, false),
                        "upper",
                    )?;
                    let raw = builder.build_int_add(v0, shifted, "raw")?;
                    let base = i64::try_from(leaf_base[tree.0] - leaf_origin)?
                        .checked_add(leaf_bias[tree.0])
                        .ok_or("leaf base overflow")?;
                    let slot = builder.build_int_add(
                        raw,
                        index_type.const_int(base as u64, false),
                        "leaf_slot",
                    )?;
                    sums[sum.0] = add_leaf(slot, sums[sum.0])?;
                }
                Action::Constant { tree: tree_id, sum } => {
                    let slot =
                        index_type.const_int((leaf_base[tree_id.0] - leaf_origin) as u64, false);
                    sums[sum.0] = add_leaf(slot, sums[sum.0])?;
                }
            }
        }
        if chunk_number + 1 < chunks.len() {
            let mut args = arguments(row, constants, chunk_number + 1)?;
            for (sum, value) in sums.iter().enumerate() {
                args[pointer_count + sum] = (*value).into();
            }
            let call = builder.build_call(functions[chunk_number + 1], &args, "continued")?;
            call.set_call_convention(PRESERVE_NONE_CALL_CONVENTION);
            call.set_tail_call_kind(inkwell::values::LLVMTailCallKind::LLVMTailCallKindMustTail);
            builder.build_return(Some(&call.try_as_basic_value().unwrap_basic()))?;
        } else {
            let result = if model.trees().is_empty() {
                f64_type.const_float(model.bias())
            } else {
                for source in 1..sums.len() {
                    sums[0] =
                        reassociate(builder.build_float_add(sums[0], sums[source], "reduced")?)?;
                }
                let scaled = reassociate(builder.build_float_mul(
                    sums[0],
                    f64_type.const_float(model.scale()),
                    "scaled",
                )?)?;
                reassociate(builder.build_float_add(
                    scaled,
                    f64_type.const_float(model.bias()),
                    "biased",
                )?)?
            };
            // LLVM inserts the return clear when this chunk uses 256-bit comparisons.
            // Shallow chunks still inherit dirty upper halves from earlier chunks.
            let has_wide_compare = plan.actions()[chunk.actions.clone()].iter().any(|action| {
                matches!(action, Action::EvaluateTree { predicates, .. } if predicates.len() >= 4)
            });
            if tree_path == TreePath::Avx2 && !has_wide_compare {
                let clear = Intrinsic::find("llvm.x86.avx.vzeroupper")
                    .unwrap()
                    .get_declaration(&module, &[])
                    .unwrap();
                builder.build_call(clear, &[], "")?;
            }
            builder.build_return(Some(&result))?;
        }
    }
    Ok(module)
}

fn load_feature<'ctx>(
    builder: &inkwell::builder::Builder<'ctx>,
    f64_type: inkwell::types::FloatType<'ctx>,
    index_type: IntType<'ctx>,
    row: PointerValue<'ctx>,
    feature: FeatureIndex,
    mode: NanMode,
) -> CompileResult<FloatValue<'ctx>> {
    // SAFETY: Trees validates the feature index and row byte span;
    // CompiledModel::predict checks that the caller supplies that span.
    let address = unsafe {
        builder.build_in_bounds_gep(
            f64_type,
            row,
            &[index_type.const_int(feature.0 as u64, false)],
            "feature_ptr",
        )?
    };
    let loaded = builder.build_load(f64_type, address, "feature")?;
    loaded.as_instruction_value().unwrap().set_alignment(8)?;
    let mut value = loaded.into_float_value();
    if mode != NanMode::AsIs {
        let is_nan = builder.build_float_compare(FloatPredicate::UNO, value, value, "is_nan")?;
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
    Ok(value)
}

fn order_lanes(plan: &Plan<'_>, arm64: bool) -> (Vec<Discriminator>, Vec<u16>) {
    let mut predicates = plan.predicates().to_vec();
    let mut swaps = vec![0; plan.model().trees().len()];
    let mut uses = vec![0usize; plan.model().feature_modes().len()];
    for chunk in plan.chunks() {
        uses.fill(0);
        let actions = &plan.actions()[chunk.actions.clone()];
        for action in actions {
            if let Action::EvaluateTree { tree, .. } = action {
                // Padding repeats a feature but is not a tree level.
                for split in &plan.model().trees()[tree.0].selections {
                    uses[split.feature_idx.0] += 1;
                }
            }
        }
        for action in actions {
            if let Action::EvaluateTree {
                tree,
                predicates: range,
                ..
            } = action
            {
                let depth = plan.model().trees()[tree.0].selections.len();
                for (number, pair) in predicates[range.start..range.start + depth]
                    .chunks_exact_mut(2)
                    .enumerate()
                {
                    let shared0 = uses[pair[0].feature_idx.0] > 1;
                    let shared1 = uses[pair[1].feature_idx.0] > 1;
                    if shared0 != shared1 && shared0 == arm64 {
                        pair.swap(0, 1);
                        swaps[tree.0] |= 1 << number;
                    }
                }
            }
        }
    }
    (predicates, swaps)
}

fn raw_of(pattern: u32, pairs: usize, swaps: u16, tree_path: TreePath) -> i64 {
    assert!((1..=16).contains(&pairs));
    let mut lanes = [0i64; 2];
    for pair in 0..pairs {
        for (lane, accumulator) in lanes.iter_mut().enumerate() {
            let source = pair * 2 + (lane ^ usize::from(swaps & (1 << pair) != 0));
            let mask = -i64::from(pattern & (1u32 << source) != 0);
            *accumulator = if tree_path == TreePath::Avx2 {
                *accumulator | ((mask & 1) << (pair * 2 + lane))
            } else if pair == 0 {
                mask
            } else {
                (mask.checked_shl(pair as u32).unwrap()) | (*accumulator & ((1i64 << pair) - 1))
            };
        }
    }
    let upper_shift = if tree_path == TreePath::Avx2 {
        0
    } else {
        pairs as u32
    };
    lanes[0]
        .checked_add(lanes[1].checked_shl(upper_shift).unwrap())
        .unwrap()
}

fn permuted_leaves(tree: &SingleTree, swaps: u16, tree_path: TreePath) -> (i64, Vec<f64>) {
    let depth = tree.selections.len();
    let pairs = depth.div_ceil(2);
    let count = u32::try_from(tree.leafs.len()).unwrap();
    let bias = (0..count)
        .map(|pattern| raw_of(pattern, pairs, swaps, tree_path))
        .min()
        .unwrap()
        .checked_neg()
        .unwrap();
    let mut leaves = vec![0.0; count as usize];
    let mut seen = vec![false; count as usize];
    for pattern in 0..count {
        let raw = raw_of(pattern, pairs, swaps, tree_path);
        let slot = usize::try_from(bias.checked_add(raw).unwrap()).unwrap();
        assert!(
            slot < leaves.len() && !seen[slot],
            "leaf permutation is not a bijection"
        );
        seen[slot] = true;
        leaves[slot] = tree.leafs[pattern as usize];
    }
    assert!(seen.into_iter().all(|present| present));
    (bias, leaves)
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
    use crate::parsed_tree::{Discriminator, FeatureIndex, SingleTree, Trees};
    use crate::sort_policy::{plan, plan_with};
    use inkwell::values::{AnyValue, FunctionValue, InstructionOpcode};

    fn indexed_model(depths: &[usize]) -> Trees {
        Trees::new(
            depths
                .iter()
                .map(|&depth| SingleTree {
                    selections: (0..depth)
                        .map(|level| Discriminator {
                            feature_idx: FeatureIndex(level),
                            threshold: 0.0,
                        })
                        .collect(),
                    leafs: (0..1usize << depth).map(|leaf| leaf as f64).collect(),
                })
                .collect(),
            vec![NanMode::AsIs; depths.iter().copied().max().unwrap_or(0).max(8)],
            1.0,
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn reachable_leaf_patterns_compile_for_depths_1_to_12() {
        let context = Context::create();
        for depth in 1..=12 {
            let model = indexed_model(&[depth]);
            let tree = &model.trees()[0];
            let (bias, leaves) = permuted_leaves(tree, 0, TreePath::Lanes2);
            let pairs = depth.div_ceil(2);
            let mut slots: Vec<_> = (0..tree.leafs.len() as u32)
                .map(|pattern| bias + raw_of(pattern, pairs, 0, TreePath::Lanes2))
                .collect();
            slots.sort_unstable();
            assert_eq!(slots, (0..leaves.len() as i64).collect::<Vec<_>>());
            assert_eq!(leaves.len(), tree.leafs.len());
            let compiled = compile(&context, &model).unwrap();
            for pattern in 0..tree.leafs.len() as u32 {
                let slot = (bias + raw_of(pattern, pairs, 0, TreePath::Lanes2)) as usize;
                let row: Vec<_> = (0..model.feature_modes().len())
                    .map(|level| f64::from(pattern & (1 << level) != 0))
                    .collect();
                assert_eq!(leaves[slot], pattern as f64);
                assert_eq!(compiled.predict(&row), Ok(pattern as f64));
                assert_eq!(compiled.predict(&row), model.predict(&row));
            }
        }
    }

    #[test]
    fn shared_features_choose_target_lanes_and_swap_thresholds() {
        let context = Context::create();
        let model = Trees::new(
            [vec![0, 1, 2, 3, 4, 5, 6, 6], vec![0, 3, 0], vec![0, 3]]
                .into_iter()
                .enumerate()
                .map(|(tree, features)| SingleTree {
                    leafs: (0..1usize << features.len())
                        .map(|leaf| leaf as f64)
                        .collect(),
                    selections: features
                        .into_iter()
                        .enumerate()
                        .map(|(level, feature)| Discriminator {
                            feature_idx: FeatureIndex(feature),
                            threshold: (tree * 10 + level) as f64,
                        })
                        .collect(),
                })
                .collect(),
            vec![NanMode::AsIs; 7],
            1.0,
            0.0,
        )
        .unwrap();
        let plan = plan(&model);
        for (triple, path, arm64, expected) in [
            (
                "aarch64-apple-darwin",
                TreePath::Lanes2,
                true,
                [1, 0, 2, 3, 4, 5, 6, 6],
            ),
            (
                "x86_64-unknown-linux-gnu",
                TreePath::Lanes2,
                false,
                [0, 1, 3, 2, 4, 5, 6, 6],
            ),
            (
                "x86_64-unknown-linux-gnu",
                TreePath::Avx2,
                false,
                [0, 1, 3, 2, 4, 5, 6, 6],
            ),
        ] {
            let (predicates, swaps) = order_lanes(&plan, arm64);
            assert_eq!(swaps, [if arm64 { 1 } else { 2 }, 0, 0]);
            let range = plan
                .actions()
                .iter()
                .find_map(|action| match action {
                    Action::EvaluateTree {
                        tree, predicates, ..
                    } if tree.0 == 0 => Some(predicates.clone()),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                predicates[range]
                    .iter()
                    .map(|p| p.feature_idx.0)
                    .collect::<Vec<_>>(),
                expected
            );
            let module = emit_plan(&context, &plan, &TargetTriple::create(triple), path).unwrap();
            module.verify().unwrap();
            let constants = module
                .get_global("constants")
                .unwrap()
                .get_initializer()
                .unwrap()
                .into_array_value();
            let function = module.get_function("continuation0").unwrap();
            let builder = context.create_builder();
            builder.position_before(
                &function
                    .get_first_basic_block()
                    .unwrap()
                    .get_first_instruction()
                    .unwrap(),
            );
            for (slot, predicate) in predicates.iter().enumerate() {
                let value = builder
                    .build_extract_value(constants, slot as u32, "threshold")
                    .unwrap()
                    .into_float_value()
                    .get_constant()
                    .unwrap()
                    .0;
                assert_eq!(value.to_bits(), predicate.threshold.to_bits());
            }
            let features: Vec<_> = function
                .get_first_basic_block()
                .unwrap()
                .get_instructions()
                .filter(|instruction| instruction.get_opcode() == InstructionOpcode::InsertElement)
                .map(|insert| {
                    let load = insert
                        .get_operand(1)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    let gep = load
                        .get_operand(0)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    gep.get_operand(1)
                        .unwrap()
                        .unwrap_value()
                        .into_int_value()
                        .get_zero_extended_constant()
                        .unwrap() as usize
                })
                .collect();
            assert_eq!(
                features,
                predicates
                    .iter()
                    .map(|p| p.feature_idx.0)
                    .collect::<Vec<_>>()
            );
            if path == TreePath::Avx2 {
                let mut slot = predicates.len().next_multiple_of(8);
                for action in plan.actions() {
                    let Action::EvaluateTree { tree, .. } = action else {
                        unreachable!()
                    };
                    for emitted in 0..model.trees()[tree.0].leafs.len() {
                        let original = if tree.0 == 0 {
                            (emitted & !12) | ((emitted & 4) << 1) | ((emitted & 8) >> 1)
                        } else {
                            emitted
                        };
                        let leaf = builder
                            .build_extract_value(constants, slot as u32, "leaf")
                            .unwrap()
                            .into_float_value()
                            .get_constant()
                            .unwrap()
                            .0;
                        assert_eq!(leaf, original as f64);
                        slot += 1;
                    }
                }
            }
        }
        // Uses in another chunk cannot make a feature shared.
        let isolated = plan_with(&model, 1, 2);
        for arm64 in [false, true] {
            let (_, swaps) = order_lanes(&isolated, arm64);
            assert_eq!(swaps[0], 0);
            let odd = indexed_model(&[3]);
            let (_, swaps) = order_lanes(&super::plan(&odd), arm64);
            assert_eq!(swaps, [0], "synthetic padding cannot make a feature shared");
        }
    }

    #[test]
    fn emits_vector_compares_only() {
        let context = Context::create();
        let model = indexed_model(&[6, 6]);
        let plan = plan_with(&model, 1, 2);
        for (triple, pointer_count) in
            [("aarch64-apple-darwin", 3), ("x86_64-unknown-linux-gnu", 2)]
        {
            let arm64 = pointer_count == 3;
            let module = emit_plan(
                &context,
                &plan,
                &TargetTriple::create(triple),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            for number in 0..2 {
                let function = module
                    .get_function(&format!("continuation{number}"))
                    .unwrap();
                assert_eq!(function.count_params(), pointer_count + 2);
                let ir = function.print_to_string().to_string();
                assert_eq!(ir.matches("fcmp ogt <2 x double>").count(), 3);
                assert_eq!(
                    ir.matches("call <2 x i64> @llvm.aarch64.neon.vsli.v2i64")
                        .count(),
                    if arm64 { 2 } else { 0 }
                );
                assert!(!ir.contains("fcmp ogt double"));
                assert_eq!(ir.matches("load <2 x double>").count(), 3);
                for instruction in function.get_first_basic_block().unwrap().get_instructions() {
                    if instruction.get_opcode() == InstructionOpcode::Load {
                        assert_eq!(instruction.get_alignment().unwrap(), 8);
                    }
                }
            }
            let assembly = emit_assembly(&context, &plan, triple).unwrap();
            if arm64 {
                assert!(assembly.contains("fcmgt.2d"));
                assert!(assembly.contains("sli"));
            } else {
                assert!(!assembly.contains("llvm.aarch64.neon.vsli"));
                assert!(!assembly.contains("ucomisd"));
            }
        }
    }

    #[test]
    fn arm64_ignores_avx2_option() {
        let context = Context::create();
        for model in [mixed_depth_model(), indexed_model(&[0, 1, 3, 5, 6, 8])] {
            for plan in [plan(&model), plan_with(&model, 1, 2)] {
                let triple = "aarch64-apple-darwin";
                assert_eq!(
                    emit_assembly(&context, &plan, triple).unwrap(),
                    emit_assembly_with(&context, &plan, triple, EmitOptions { avx2: true })
                        .unwrap(),
                );
            }
        }
    }

    #[test]
    fn avx2_host_check_requires_every_x86_64_v3_feature() {
        let features = [
            "+avx2", "+bmi1", "+bmi2", "+fma", "+lzcnt", "+movbe", "+f16c",
        ];
        assert!(supports_x86_64_v3(&features.join(",")));
        assert!(!supports_x86_64_v3("+avx2"));
        for missing in features {
            let remaining: Vec<_> = features
                .into_iter()
                .filter(|feature| *feature != missing)
                .collect();
            assert!(
                !supports_x86_64_v3(&remaining.join(",")),
                "missing {missing}"
            );
            assert!(!supports_x86_64_v3(&format!(
                "{},-{}",
                remaining.join(","),
                &missing[1..]
            )));
        }
    }

    #[test]
    fn avx2_emits_folded_compares() {
        let context = Context::create();
        Target::initialize_all(&InitializationConfig::default());
        let triple = "x86_64-unknown-linux-gnu";
        for depth in [1usize, 3, 5, 6, 8] {
            let model = indexed_model(&[depth]);
            let plan = plan_with(&model, 1, 2);
            let (module, _) = prepare_module(
                &context,
                &plan,
                &TargetTriple::create(triple),
                EmitOptions { avx2: true },
            )
            .unwrap();
            let ir = module.print_to_string().to_string();
            let padded = depth.next_multiple_of(2);
            assert_eq!(
                ir.matches("call <4 x double> @llvm.x86.avx.cmp.pd.256")
                    .count(),
                padded / 4
            );
            assert_eq!(
                ir.matches("call <2 x double> @llvm.x86.sse2.cmp.pd")
                    .count(),
                padded % 4 / 2
            );
            assert!(!ir.contains("fcmp"));
            assert!(!ir.contains("vsli"));
            for function in module
                .get_functions()
                .filter(|function| function.count_basic_blocks() != 0)
            {
                for (name, value) in [("target-cpu", "x86-64-v3"), ("target-features", "")] {
                    assert_eq!(
                        function
                            .get_string_attribute(AttributeLoc::Function, name)
                            .unwrap()
                            .get_string_value()
                            .to_str()
                            .unwrap(),
                        value,
                    );
                }
                for block in function.get_basic_blocks() {
                    for instruction in block.get_instructions() {
                        if instruction.get_opcode() == InstructionOpcode::Load {
                            assert_eq!(instruction.get_alignment().unwrap(), 8);
                        }
                    }
                }
            }
            let assembly =
                emit_assembly_with(&context, &plan, triple, EmitOptions { avx2: true }).unwrap();
            let instructions: Vec<_> = assembly
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    line.split_once(char::is_whitespace)
                })
                .collect();
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, _)| *op == "vcmpgt_oqpd")
                    .count(),
                padded.div_ceil(4)
            );
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, operands)| *op == "vcmpgt_oqpd" && operands.contains('('))
                    .count(),
                padded.div_ceil(4),
                "depth={depth}\n{assembly}"
            );
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, _)| *op == "vmovmskpd")
                    .count(),
                padded.div_ceil(4)
            );
            let leaf_adds: Vec<_> = instructions
                .iter()
                .enumerate()
                .filter(|(_, (op, _))| *op == "vaddsd")
                .collect();
            assert_eq!(leaf_adds.len(), 1, "depth={depth}\n{assembly}");
            for (position, (_, operands)) in leaf_adds {
                let (displacement, address) = operands.split_once('(').unwrap();
                assert!(
                    displacement
                        .strip_prefix('-')
                        .unwrap_or(displacement)
                        .bytes()
                        .all(|byte| byte.is_ascii_digit())
                );
                let (address, registers) = address.split_once(')').unwrap();
                let parts: Vec<_> = address.split(',').collect();
                assert!(parts.len() == 2 || parts.len() == 3);
                for register in &parts[..2] {
                    let register = register.strip_prefix('%').unwrap();
                    assert!(!register.is_empty());
                    assert!(
                        register
                            .bytes()
                            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
                    );
                }
                if parts.len() == 3 {
                    assert!(["1", "2", "4", "8"].contains(&parts[2]));
                }
                assert!(registers.starts_with(", %xmm"), "depth={depth}\n{assembly}");
                if position != 0 {
                    let (op, operands) = instructions[position - 1];
                    assert!(
                        !(op == "vmovsd"
                            && operands
                                .split_once(')')
                                .is_some_and(|(_, registers)| registers.starts_with(", %xmm"))),
                        "separate leaf load: depth={depth}\n{assembly}"
                    );
                }
            }
            for forbidden in [
                "ucomisd", "vucomisd", "cmpltpd", "vcmpltpd", "psllq", "pshufd",
            ] {
                assert!(
                    !instructions.iter().any(|(op, _)| *op == forbidden),
                    "{forbidden}: depth={depth}"
                );
            }
        }
    }

    #[test]
    fn avx2_leaf_blocks_without_swaps_follow_chunk_order() {
        let context = Context::create();
        Target::initialize_all(&InitializationConfig::default());
        let model = indexed_model(&[0, 1, 3, 5, 6, 8]);
        let plan = plan_with(&model, 2, 2);
        assert!(order_lanes(&plan, false).1.iter().all(|&swaps| swaps == 0));
        for triple in ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"] {
            let (module, _) = prepare_module(
                &context,
                &plan,
                &TargetTriple::create(triple),
                EmitOptions { avx2: true },
            )
            .unwrap();
            let actual = module
                .get_global("constants")
                .unwrap()
                .get_initializer()
                .unwrap()
                .into_array_value();
            let builder = context.create_builder();
            let inspect = module.add_function(
                "inspect_constants",
                context.void_type().fn_type(&[], false),
                None,
            );
            builder.position_at_end(context.append_basic_block(inspect, "entry"));
            let entry = |index| {
                builder
                    .build_extract_value(actual, index as u32, "leaf")
                    .unwrap()
                    .into_float_value()
                    .get_constant()
                    .unwrap()
                    .0
                    .to_bits()
            };
            let mut slot = 0usize;
            for chunk in plan.chunks() {
                let first = slot;
                let actions = &plan.actions()[chunk.actions.clone()];
                for action in actions {
                    if let Action::EvaluateTree { predicates, .. } = action {
                        for predicate in &plan.predicates()[predicates.clone()] {
                            assert_eq!(entry(slot), predicate.threshold.to_bits());
                            slot += 1;
                        }
                    }
                }
                let padded = first + (slot - first).next_multiple_of(8);
                while slot < padded {
                    assert_eq!(entry(slot), 0.0f64.to_bits());
                    slot += 1;
                }
                for action in actions {
                    let (Action::Constant { tree, .. } | Action::EvaluateTree { tree, .. }) =
                        action;
                    let mut expected: Vec<_> = model.trees()[tree.0]
                        .leafs
                        .iter()
                        .map(|leaf| leaf.to_bits())
                        .collect();
                    let length = expected.len();
                    let mut leaves: Vec<_> = (slot..slot + length).map(entry).collect();
                    if triple != "x86_64-unknown-linux-gnu" {
                        leaves.sort_unstable();
                        expected.sort_unstable();
                    }
                    assert_eq!(leaves, expected);
                    slot += length;
                }
            }
            assert_eq!(actual.get_type().len() as usize, slot);
        }
    }

    #[test]
    fn compile_checks_host_avx2_support() {
        let context = Context::create();
        let model = indexed_model(&[3]);
        let result = compile_with(&context, &model, EmitOptions { avx2: true });
        if EmitOptions::from_host().avx2 {
            assert_eq!(result.unwrap().predict(&[1.0; 8]), model.predict(&[1.0; 8]));
        } else {
            let error = result.err().expect("unsupported AVX2 must be rejected");
            assert_eq!(
                error.to_string(),
                "AVX2 emission requires x86-64-v3 host features"
            );
        }
    }

    fn mixed_depth_model() -> Trees {
        let shared = Discriminator {
            feature_idx: FeatureIndex(2),
            threshold: 1.37,
        };
        Trees::new(
            vec![
                SingleTree {
                    leafs: vec![9.0],
                    selections: vec![],
                },
                SingleTree {
                    leafs: vec![10.0, 11.0],
                    selections: vec![shared.clone()],
                },
                SingleTree {
                    leafs: (20..28).map(|leaf| leaf as f64).collect(),
                    selections: vec![
                        shared,
                        Discriminator {
                            feature_idx: FeatureIndex(0),
                            threshold: 2.61,
                        },
                        Discriminator {
                            feature_idx: FeatureIndex(0),
                            threshold: -0.42,
                        },
                    ],
                },
            ],
            vec![NanMode::AsIs; 3],
            2.0,
            3.0,
        )
        .unwrap()
    }

    fn assert_opaque_constants(function: FunctionValue<'_>, comparisons: usize, leaf_adds: usize) {
        let mut thresholds = 0;
        let mut leaves = 0;
        for instruction in function.get_first_basic_block().unwrap().get_instructions() {
            let is_threshold = instruction.get_opcode() == InstructionOpcode::FCmp;
            let is_leaf = instruction.get_opcode() == InstructionOpcode::FAdd
                && instruction
                    .get_name()
                    .unwrap()
                    .to_bytes()
                    .starts_with(b"sum");
            if is_threshold || is_leaf {
                let value = instruction.get_operand(1).unwrap().unwrap_value();
                let load = value
                    .as_instruction_value()
                    .expect("constant must come from a load");
                assert_eq!(load.get_opcode(), InstructionOpcode::Load);
                assert_eq!(load.get_alignment().unwrap(), 8);
                thresholds += usize::from(is_threshold);
                leaves += usize::from(is_leaf);
            }
        }
        assert_eq!((thresholds, leaves), (comparisons, leaf_adds));
    }

    #[test]
    fn constants_block_layout_matches_plan() {
        let context = Context::create();
        let model = mixed_depth_model();
        let plan = plan(&model);
        let module = emit_plan(
            &context,
            &plan,
            &TargetMachine::get_default_triple(),
            TreePath::Lanes2,
        )
        .unwrap();
        module.verify().unwrap();
        assert_eq!(module.get_globals().count(), 1);
        let constants = module.get_global("constants").unwrap();
        assert_eq!(constants.get_linkage(), Linkage::Private);
        assert!(constants.is_constant());
        assert_eq!(constants.get_alignment(), 64);
        let mut expected: Vec<_> = plan
            .predicates()
            .iter()
            .map(|predicate| predicate.threshold)
            .collect();
        expected.resize(8, 0.0);
        expected.extend([
            24.0, 25.0, 20.0, 21.0, 26.0, 27.0, 22.0, 23.0, 11.0, 10.0, 9.0,
        ]);
        let expected: Vec<_> = expected
            .into_iter()
            .map(|value| context.f64_type().const_float(value))
            .collect();
        assert_eq!(
            constants.get_initializer().unwrap().into_array_value(),
            context.f64_type().const_array(&expected),
        );
        let compiled = compile_plan(&context, &plan).unwrap();
        for row in [[0.0, -1.0, 0.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]] {
            assert_eq!(compiled.predict(&row), model.predict(&row));
        }
    }

    #[test]
    fn one_chunk_model_keeps_wrapper_and_opaque_loads() {
        let context = Context::create();
        let model = mixed_depth_model();
        let plan = plan(&model);
        assert_eq!(plan.chunks().len(), 1);
        Target::initialize_all(&InitializationConfig::default());
        for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            let (module, _) = prepare_module(
                &context,
                &plan,
                &TargetTriple::create(triple),
                EmitOptions::default(),
            )
            .unwrap();
            assert_eq!(
                module
                    .get_functions()
                    .filter(|f| f.get_first_basic_block().is_some())
                    .count(),
                2
            );
            let wrapper = module.get_function("predict").unwrap();
            assert_eq!(wrapper.count_params(), 1);
            let ir = wrapper.print_to_string().to_string();
            assert_eq!(
                ir.matches("call preserve_nonecc double @continuation0")
                    .count(),
                1
            );
            let continuation = module.get_function("continuation0").unwrap();
            assert_opaque_constants(continuation, 3, 3);
            let ir = continuation.print_to_string().to_string();
            assert_eq!(ir.matches("load double").count(), 5);
            assert!(!ir.contains("@constants"));
        }
    }

    #[test]
    fn pointer_parameters_have_bounds() {
        let context = Context::create();
        let model = Trees::new(
            (0..9)
                .map(|tree| SingleTree {
                    leafs: (0..8).map(|leaf| (tree * 8 + leaf) as f64).collect(),
                    selections: [1.37, 2.61, -0.42]
                        .map(|threshold| Discriminator {
                            feature_idx: FeatureIndex(tree),
                            threshold,
                        })
                        .into(),
                })
                .collect(),
            vec![NanMode::AsIs; 9],
            1.0,
            0.0,
        )
        .unwrap();
        let plan = plan(&model);
        for (triple, pointer_count) in
            [("x86_64-unknown-linux-gnu", 2), ("aarch64-apple-darwin", 3)]
        {
            let module = emit_plan(
                &context,
                &plan,
                &TargetTriple::create(triple),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            let constants = module.get_global("constants").unwrap();
            let entries = constants
                .get_initializer()
                .unwrap()
                .into_array_value()
                .get_type()
                .len();
            assert_eq!(entries, 112);
            for (number, chunk_base) in [0, 96].into_iter().enumerate() {
                let function = module
                    .get_function(&format!("continuation{number}"))
                    .unwrap();
                assert_eq!(
                    function
                        .get_param_iter()
                        .filter(|value| value.is_pointer_value())
                        .count(),
                    pointer_count,
                );
                assert_eq!(
                    function.count_params() as usize,
                    pointer_count + plan.sum_count()
                );
                for parameter in 0..pointer_count {
                    assert!(
                        function
                            .get_nth_param(parameter as u32)
                            .unwrap()
                            .is_pointer_value()
                    );
                    for (name, value) in [
                        ("readonly", 0),
                        ("noalias", 0),
                        ("align", if parameter == 1 { 64 } else { 8 }),
                        (
                            "dereferenceable",
                            match parameter {
                                0 => 9 * 8,
                                1 => u64::from(entries) * 8,
                                2 => (u64::from(entries) - chunk_base) * 8,
                                _ => unreachable!(),
                            },
                        ),
                    ] {
                        assert_eq!(
                            function
                                .get_enum_attribute(
                                    AttributeLoc::Param(parameter as u32),
                                    Attribute::get_named_enum_kind_id(name),
                                )
                                .unwrap()
                                .get_enum_value(),
                            value,
                            "{triple} continuation{number} parameter {parameter} {name}",
                        );
                    }
                }
                let instructions: Vec<_> = function
                    .get_first_basic_block()
                    .unwrap()
                    .get_instructions()
                    .collect();
                let thresholds: Vec<_> = instructions
                    .iter()
                    .filter(|instruction| {
                        instruction.get_opcode() == InstructionOpcode::Load
                            && instruction
                                .get_name()
                                .unwrap()
                                .to_bytes()
                                .starts_with(b"threshold")
                    })
                    .collect();
                assert_eq!(thresholds.len(), if number == 0 { 16 } else { 2 });
                for (offset, load) in thresholds.iter().enumerate() {
                    let gep = load
                        .get_operand(0)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    assert_eq!(gep.get_opcode(), InstructionOpcode::GetElementPtr);
                    let (parameter, operand, index) = if pointer_count == 3 {
                        (2, 1, 2 * offset as u64)
                    } else {
                        (1, 2, chunk_base + 2 * offset as u64)
                    };
                    assert_eq!(
                        gep.get_operand(0).unwrap().unwrap_value(),
                        function.get_nth_param(parameter).unwrap(),
                    );
                    assert_eq!(
                        gep.get_operand(operand)
                            .unwrap()
                            .unwrap_value()
                            .into_int_value()
                            .get_zero_extended_constant(),
                        Some(index),
                    );
                }
                if pointer_count == 3 && number == 0 {
                    let call = instructions
                        .iter()
                        .find(|instruction| {
                            instruction
                                .print_to_string()
                                .to_string()
                                .contains("@continuation1(")
                        })
                        .unwrap();
                    let base = call
                        .get_operand(2)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    assert_eq!(base.get_opcode(), InstructionOpcode::GetElementPtr);
                    assert_eq!(
                        base.get_operand(0).unwrap().unwrap_value(),
                        function.get_nth_param(1).unwrap(),
                    );
                    assert_eq!(
                        base.get_operand(1)
                            .unwrap()
                            .unwrap_value()
                            .into_int_value()
                            .get_zero_extended_constant(),
                        Some(96),
                    );
                }
            }
            let assembly = emit_assembly(&context, &plan, triple).unwrap();
            if pointer_count == 2 {
                assert!(
                    assembly.lines().any(|line| { line.contains("cmpltpd") }),
                    "{assembly}"
                );
                assert!(
                    !assembly
                        .lines()
                        .any(|line| { line.contains("movabsq") && line.contains("$.LCPI") }),
                    "{assembly}"
                );
            } else {
                assert!(
                    assembly.lines().any(|line| {
                        let mut words = line.split_whitespace();
                        matches!(words.next(), Some("ldr" | "ldp"))
                            && words
                                .next()
                                .is_some_and(|register| register.starts_with('q'))
                    }),
                    "{assembly}"
                );
                assert!(!assembly.contains("movk"), "{assembly}");
            }
        }
    }

    #[test]
    fn dropped_model_and_last_tree_leaf() {
        let context = Context::create();
        let model = Trees::new(
            (0..70)
                .map(|tree| SingleTree {
                    leafs: (0..64).map(|leaf| (tree * 64 + leaf) as f64).collect(),
                    selections: (0..6)
                        .map(|feature| Discriminator {
                            feature_idx: FeatureIndex(feature),
                            threshold: 1.37,
                        })
                        .collect(),
                })
                .collect(),
            vec![NanMode::AsIs; 6],
            2.0,
            3.0,
        )
        .unwrap();
        let rows = [[0.0; 6], [2.0; 6]];
        assert_eq!(model.leaf_indices(&rows[0]).unwrap()[0], 0);
        assert_eq!(model.leaf_indices(&rows[1]).unwrap()[69], 63);
        let preceding_leaves: usize = model.trees()[..69]
            .iter()
            .map(|tree| tree.leafs.len())
            .sum();
        assert!(preceding_leaves > 4095);
        let expected = rows.map(|row| model.predict(&row).unwrap());
        let mut kernels = Vec::new();
        for chunk_trees in [4, 8] {
            let plan = plan_with(&model, chunk_trees, 2);
            kernels.push(compile_plan(&context, &plan).unwrap());
            drop(plan);
        }
        drop(model);
        for compiled in kernels {
            for (row, expected) in rows.iter().zip(expected) {
                assert_eq!(compiled.predict(row), Ok(expected));
            }
        }
    }

    #[test]
    fn empty_and_constant_forests_still_predict() {
        let context = Context::create();
        for leaves in [vec![], vec![1.5, -2.0, 4.0]] {
            let empty = leaves.is_empty();
            let model = Trees::new(
                leaves
                    .into_iter()
                    .map(|leaf| SingleTree {
                        leafs: vec![leaf],
                        selections: vec![],
                    })
                    .collect(),
                vec![],
                if empty { f64::NAN } else { 2.0 },
                3.0,
            )
            .unwrap();
            let plan = plan(&model);
            let module = emit_plan(
                &context,
                &plan,
                &TargetMachine::get_default_triple(),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            let constants = module.get_global("constants").unwrap();
            let expected = if empty {
                vec![0.0]
            } else {
                vec![1.5, -2.0, 4.0]
            };
            let values: Vec<_> = expected
                .into_iter()
                .map(|value| context.f64_type().const_float(value))
                .collect();
            assert_eq!(
                constants.get_initializer().unwrap().into_array_value(),
                context.f64_type().const_array(&values),
            );
            let compiled = compile_plan(&context, &plan).unwrap();
            assert_eq!(compiled.predict(&[]), Ok(if empty { 3.0 } else { 10.0 }));
            assert_eq!(compiled.predict(&[]), model.predict(&[]));
        }
    }

    #[test]
    fn emits_predict_for_aarch64_and_x86_64() {
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
            1.0,
            0.0,
        )
        .unwrap();
        for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            let assembly = emit_assembly(&context, &plan(&model), triple).unwrap();
            assert!(assembly.contains("predict"), "{triple}: {assembly}");
            assert!(!assembly.contains('\0'));
            assert!(assembly.ends_with('\n'));
        }
    }

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
        let module = emit_plan(
            &context,
            &plan(&model),
            &TargetMachine::get_default_triple(),
            TreePath::Lanes2,
        )
        .unwrap();
        module.verify().unwrap();
        let ir = module.print_to_string().to_string();
        assert_eq!(module.get_functions().count(), 2);
        assert_eq!(module.get_function("predict").unwrap().count_params(), 1);
        assert_eq!(
            ir.matches("call preserve_nonecc double @continuation0")
                .count(),
            1
        );
        let continuation = module.get_function("continuation0").unwrap();
        assert_opaque_constants(continuation, 1, 1);
        for expected in [
            "@constants = private constant [10 x double]",
            "align 64",
            "noalias readonly",
            "readonly align 8 dereferenceable(8)",
            "memory(read)",
            "fcmp ogt <2 x double>",
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

    #[test]
    fn continuations_carry_only_pointers_and_sums() {
        let context = Context::create();
        let depths: Vec<_> = (0..30).map(|tree| tree % 9).collect();
        let model = indexed_model(&depths);
        let plan = plan_with(&model, 4, 2);
        for (triple, pointer_count, tree_path) in [
            ("aarch64-apple-darwin", 3, TreePath::Lanes2),
            ("x86_64-unknown-linux-gnu", 2, TreePath::Lanes2),
            ("x86_64-unknown-linux-gnu", 2, TreePath::Avx2),
        ] {
            let module =
                emit_plan(&context, &plan, &TargetTriple::create(triple), tree_path).unwrap();
            module.verify().unwrap();
            assert_eq!(
                module
                    .get_functions()
                    .filter(|f| f.get_first_basic_block().is_some())
                    .count(),
                plan.chunks().len() + 1
            );
            let first = module.get_function("continuation0").unwrap();
            for (number, chunk) in plan.chunks().iter().enumerate() {
                let depths: Vec<_> = plan.actions()[chunk.actions.clone()]
                    .iter()
                    .map(|action| match action {
                        Action::Constant { tree, .. } | Action::EvaluateTree { tree, .. } => {
                            model.trees()[tree.0].selections.len()
                        }
                    })
                    .collect();
                let comparisons = depths.iter().map(|depth| depth.div_ceil(2)).sum::<usize>();
                let function = module
                    .get_function(&format!("continuation{number}"))
                    .unwrap();
                assert_eq!(
                    function.count_params() as usize,
                    pointer_count + plan.sum_count()
                );
                assert_eq!(function.get_type(), first.get_type());
                for (index, parameter) in function.get_param_iter().enumerate() {
                    if index < pointer_count {
                        assert_eq!(
                            parameter.get_type(),
                            context.ptr_type(AddressSpace::default()).into()
                        );
                    } else {
                        assert_eq!(parameter.get_type(), context.f64_type().into());
                    }
                }
                assert_eq!(
                    function.get_call_conventions(),
                    PRESERVE_NONE_CALL_CONVENTION
                );
                assert!(
                    function
                        .get_enum_attribute(
                            AttributeLoc::Function,
                            Attribute::get_named_enum_kind_id("noinline")
                        )
                        .is_some()
                );
                assert_eq!(
                    function
                        .get_string_attribute(AttributeLoc::Function, "target-features")
                        .unwrap()
                        .get_string_value()
                        .to_str()
                        .unwrap(),
                    if tree_path == TreePath::Avx2 && number + 1 < plan.chunks().len() {
                        "-vzeroupper"
                    } else {
                        ""
                    },
                );
                let ir = function.print_to_string().to_string();
                assert_eq!(
                    ir.matches("load double").count(),
                    depths.iter().sum::<usize>() + depths.len()
                );
                if tree_path == TreePath::Lanes2 {
                    assert_eq!(ir.matches("fcmp ogt <2 x double>").count(), comparisons);
                    assert_opaque_constants(function, comparisons, depths.len());
                }
                let mut lines = ir.lines();
                if number + 1 < plan.chunks().len() {
                    let call = lines.find(|line| line.contains("musttail call")).unwrap();
                    assert!(call.contains(&format!("@continuation{}(", number + 1)));
                    assert!(call.contains("preserve_nonecc"));
                    assert!(lines.next().unwrap().trim().starts_with("ret double"));
                    assert_eq!(ir.matches("call preserve_nonecc").count(), 1);
                } else {
                    assert!(!ir.contains("call preserve_nonecc"));
                }
            }
            let wrapper = module.get_function("predict").unwrap();
            assert_eq!(wrapper.get_call_conventions(), 0);
            assert_eq!(wrapper.count_params(), 1);
            assert_eq!(
                wrapper
                    .print_to_string()
                    .to_string()
                    .matches("call ")
                    .count(),
                1
            );
            let assembly = emit_assembly_with(
                &context,
                &plan,
                triple,
                EmitOptions {
                    avx2: tree_path == TreePath::Avx2,
                },
            )
            .unwrap();
            assert!(assembly.contains("continuation"));
        }
        let compiled = compile_plan(&context, &plan).unwrap();
        for row in [[-1.0; 8], [0.0; 8], [1.0; 8], [f64::NAN; 8]] {
            assert_eq!(compiled.predict(&row), model.predict(&row));
        }
    }
}
