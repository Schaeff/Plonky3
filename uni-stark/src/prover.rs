use alloc::vec::Vec;
use core::iter;

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::CanObserve;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::symbolic_builder::{get_log_quotient_degree, SymbolicAirBuilder};
use crate::{
    Commitments, NextStageTraceCallback, PackedChallenge, PackedVal, Proof, ProverConstraintFolder,
    QuotientInputs, Stage, StarkGenericConfig, StarkProvingKey, State, Val,
};

#[derive(Clone)]
struct Panic;

impl<SC: StarkGenericConfig> NextStageTraceCallback<SC> for Panic {
    fn get_next_stage_trace(&self, _: u32, _: &[Val<SC>]) -> RowMajorMatrix<Val<SC>> {
        panic!()
    }
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    main_trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    prove_with_key::<_, _, Panic>(
        config,
        None,
        air,
        challenger,
        main_trace,
        None,
        public_values,
    )
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove_with_key<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
    T,
>(
    config: &SC,
    proving_key: Option<&StarkProvingKey<SC>>,
    air: &A,
    challenger: &mut SC::Challenger,
    stage_0_trace: RowMajorMatrix<Val<SC>>,
    next_stage_trace_callback: Option<&T>,
    stage_0_public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
    T: NextStageTraceCallback<SC>,
{
    let degree = stage_0_trace.height();
    let log_degree = log2_strict_usize(degree);

    let pcs = config.pcs();
    let trace_domain = pcs.natural_domain_for_degree(degree);

    // Observe the instance
    challenger.observe(Val::<SC>::from_canonical_usize(log_degree));
    // TODO: Might be best practice to include other instance data here; see verifier comment.
    if let Some(proving_key) = proving_key {
        challenger.observe(proving_key.preprocessed_commit.clone())
    };

    // commitments to main trace
    let state = State::new(
        pcs,
        trace_domain,
        challenger,
        log_degree,
        stage_0_public_values,
    );
    let initial_stage = Stage {
        trace: stage_0_trace,
        challenge_count: air.required_challenge_count(1),
    };

    let mut state = state.run_stage(initial_stage);

    while !state
        .processed_stages
        .last()
        .unwrap()
        .challenge_values
        .is_empty()
    {
        let last_processed_stage = state.processed_stages.last().unwrap();
        let trace = next_stage_trace_callback
            .as_ref()
            .expect("witgen callback expected in the presence of challenges")
            .get_next_stage_trace(
                state.stage_id() as u32,
                &last_processed_stage.challenge_values,
            );
        let stage = Stage {
            trace,
            challenge_count: air.required_challenge_count(state.stage_id() + 1),
        };
        state = state.run_stage(stage);
    }

    assert_eq!(state.processed_stages.len(), air.stage_count());

    finish(proving_key, air, state)
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)]
pub fn finish<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    proving_key: Option<&StarkProvingKey<SC>>,
    air: &A,
    mut state: State<SC>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    // #[cfg(debug_assertions)]
    // crate::check_constraints::check_constraints(
    //     air,
    //     &air.preprocessed_trace()
    //         .unwrap_or(RowMajorMatrix::new(vec![], 0)),
    //     &trace,
    //     public_values,
    // );

    let log_quotient_degree = get_log_quotient_degree::<Val<SC>, A>(air, state.public_values.len());

    let quotient_inputs = state.quotient_inputs(log_quotient_degree);

    let preprocessed_on_quotient_domain = proving_key.map(|proving_key| {
        state.on_quotient_domain(
            &proving_key.preprocessed_data,
            quotient_inputs.quotient_domain,
        )
    });

    let traces_on_quotient_domain = state
        .processed_stages
        .iter()
        .map(|s| state.on_quotient_domain(&s.prover_data, quotient_inputs.quotient_domain))
        .collect();

    let quotient_values = quotient_values(
        air,
        preprocessed_on_quotient_domain,
        traces_on_quotient_domain,
        quotient_inputs,
    );

    let (quotient_commit, quotient_data) =
        state.commit_to_quotient(quotient_values, log_quotient_degree);

    // build the commitments

    let commitments = Commitments {
        stages: state
            .processed_stages
            .iter()
            .map(|s| s.commitment.clone())
            .collect(),
        quotient_chunks: quotient_commit,
    };

    let (opened_values, opening_proof) =
        state.get_open_proof(proving_key, quotient_data, log_quotient_degree);

    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: state.get_log_degree(),
    }
}

#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<'a, SC, A, Mat>(
    air: &A,
    preprocessed_on_quotient_domain: Option<Mat>,
    traces_on_quotient_domain: Vec<Mat>,
    quotient_inputs: QuotientInputs<'a, SC>,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let QuotientInputs {
        public_values,
        quotient_domain,
        trace_domain,
        alpha,
    } = quotient_inputs;

    let quotient_size = quotient_domain.size();
    let preprocessed_width = preprocessed_on_quotient_domain
        .as_ref()
        .map(Matrix::width)
        .unwrap_or_default();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_zeroifier.push(Val::<SC>::default());
    }

    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_zeroifier = *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);

            let preprocessed = RowMajorMatrix::new(
                preprocessed_on_quotient_domain
                    .as_ref()
                    .map(|on_quotient_domain| {
                        iter::empty()
                            .chain(on_quotient_domain.vertically_packed_row(i_start))
                            .chain(on_quotient_domain.vertically_packed_row(i_start + next_step))
                            .collect_vec()
                    })
                    .unwrap_or_default(),
                preprocessed_width,
            );

            let stages = traces_on_quotient_domain
                .iter()
                .map(|trace_on_quotient_domain| {
                    RowMajorMatrix::new(
                        iter::empty()
                            .chain(trace_on_quotient_domain.vertically_packed_row(i_start))
                            .chain(
                                trace_on_quotient_domain.vertically_packed_row(i_start + next_step),
                            )
                            .collect_vec(),
                        trace_on_quotient_domain.width(),
                    )
                })
                .collect::<Vec<RowMajorMatrix<PackedVal<SC>>>>();

            let accumulator = PackedChallenge::<SC>::zero();
            let mut folder = ProverConstraintFolder {
                stages,
                preprocessed,
                public_values: public_values.clone(),
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator,
            };
            air.eval(&mut folder);

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.accumulator * inv_zeroifier;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<Val<SC>>>::D)
                    .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                    .collect::<Vec<_>>();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}
