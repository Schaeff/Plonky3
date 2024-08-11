use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::symbolic_builder::{get_log_quotient_degree, SymbolicAirBuilder};
use crate::{
    Commitments, Domain, NextStageTraceCallback, OpenedValues, PackedChallenge, PackedVal, Proof,
    ProverConstraintFolder, Stage, StarkGenericConfig, StarkProvingKey, State, Val,
};

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
    T,
>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    challenges: Vec<Vec<u64>>,
    main_trace: RowMajorMatrix<Val<SC>>,
    next_stage_trace_callback: Option<&T>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
    T: NextStageTraceCallback<SC, Val<SC>>,
{
    prove_with_key(
        config,
        None,
        air,
        challenger,
        challenges,
        main_trace,
        next_stage_trace_callback,
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
    challenges: Vec<Vec<u64>>,
    main_trace: RowMajorMatrix<Val<SC>>,
    next_stage_trace_callback: Option<&T>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
    T: NextStageTraceCallback<SC, Val<SC>>,
{
    let degree = main_trace.height(); // all traces have the same height
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
    let committed_data = State::new(pcs, trace_domain, challenger);
    let stage_zero = Stage {
        trace: main_trace,
        public_values,
    };

    let main_state = run_stage(committed_data, stage_zero);

    challenges
        .into_iter()
        .enumerate()
        .for_each(|(stage, stage_challenges)| {
            let challenge_values = stage_challenges
                .iter()
                .map(|id| {
                    let challenge: SC::Challenge = challenger.sample();
                    (*id, challenge)
                })
                .collect::<BTreeMap<u64, SC::Challenge>>();

            // calculating next stage trace
            let next_stage_trace = next_stage_trace_callback
                .unwrap()
                .get_next_stage_trace(stage as u32, challenge_values); // callback to generate trace, this has to store

            let incoming_stage_data =
                committed_data = run_stage(trace_domain, committed_data, incoming_data);
        });

    finish(
        pcs,
        proving_key,
        log_degree,
        trace_domain,
        challenger,
        air,
        committed_data,
    )
}

pub fn run_stage<'a, SC>(state: State<'a, SC>, stage: Stage<SC>) -> State<'a, SC>
where
    SC: StarkGenericConfig + p3_commit::PolynomialSpace,
{
    let pcs = state.get_pcs();
    let trace_domain = state.get_trace_domain();
    let (trace_commit, trace_data) = info_span!("commit to trace data")
        .in_scope(|| pcs.commit(vec![(trace_domain, stage.trace)]));

    let challenger = state.get_challenger();
    challenger.observe(trace_commit.clone());
    challenger.observe_slice(stage.public_values);

    let mut new_state = State::from(state);
    new_state.trace_commits.push(trace_commit);

    new_state.update_from_stage(stage);
    new_state
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)]
pub fn finish<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    pcs: &<SC as StarkGenericConfig>::Pcs,
    proving_key: Option<&StarkProvingKey<SC>>,
    log_degree: usize,
    trace_domain: <<SC as StarkGenericConfig>::Pcs as Pcs<
        <SC as StarkGenericConfig>::Challenge,
        <SC as StarkGenericConfig>::Challenger,
    >>::Domain,
    challenger: &mut SC::Challenger,
    air: &A,
    committed_data: Option<&mut State<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig + p3_commit::PolynomialSpace,
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

    // changes for challenges
    let log_quotient_degree = get_log_quotient_degree::<Val<SC>, A>(
        air,
        committed_data // the total length of public values and challenges
            .unwrap()
            .public_values
            .iter()
            .fold(0, |length, vals| length + vals.len()),
    );
    let quotient_degree = 1 << log_quotient_degree;

    let alpha: SC::Challenge = challenger.sample_ext_element();

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));

    let preprocessed_on_quotient_domain = proving_key.map(|proving_key| {
        pcs.get_evaluations_on_domain(&proving_key.preprocessed_data, 0, quotient_domain)
    });

    let traces_on_quotient_domain = committed_data.map(|committed_data| {
        committed_data
            .traces
            .iter()
            .map(|trace| pcs.get_evaluations_on_domain(trace, 0, quotient_domain))
            .collect()
    });
    // let permutation_on_quotient_domain =
    //     pcs.get_evaluations_on_domain(&permutation_trace, 0, quotient_domain);
    let public_values = match committed_data {
        Some(data) => data.public_values, // needs to be flattened
        _ => Vec::new(),
    };

    let quotient_values = quotient_values(
        air,
        public_values,
        trace_domain,
        quotient_domain,
        preprocessed_on_quotient_domain,
        traces_on_quotient_domain,
        alpha,
    );

    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
    let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
    let qc_domains = quotient_domain.split_domains(quotient_degree);

    let (quotient_commit, quotient_data) = info_span!("commit to quotient poly chunks")
        .in_scope(|| pcs.commit(izip!(qc_domains, quotient_chunks).collect_vec()));
    challenger.observe(quotient_commit.clone());

    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
    };

    let zeta: SC::Challenge = challenger.sample();
    let zeta_next = trace_domain.next_point(zeta).unwrap();

    let (opened_values, opening_proof) = pcs.open(
        iter::empty()
            .chain(
                proving_key
                    .map(|proving_key| {
                        (&proving_key.preprocessed_data, vec![vec![zeta, zeta_next]])
                    })
                    .into_iter(),
            )
            .chain([
                (&trace_data, vec![vec![zeta, zeta_next]]),
                (
                    &quotient_data,
                    // open every chunk at zeta
                    (0..quotient_degree).map(|_| vec![zeta]).collect_vec(),
                ),
            ])
            .collect_vec(),
        challenger,
    );
    let mut opened_values = opened_values.iter();

    // maybe get values for the preprocessed columns
    let (preprocessed_local, preprocessed_next) = if proving_key.is_some() {
        let value = opened_values.next().unwrap();
        assert_eq!(value.len(), 1);
        assert_eq!(value[0].len(), 2);
        (value[0][0].clone(), value[0][1].clone())
    } else {
        (vec![], vec![])
    };

    // get values for the trace
    let value = opened_values.next().unwrap();
    assert_eq!(value.len(), 1);
    assert_eq!(value[0].len(), 2);
    let trace_local = value[0][0].clone();
    let trace_next = value[0][1].clone();

    // get values for the quotient
    let value = opened_values.next().unwrap();
    assert_eq!(value.len(), quotient_degree);
    let quotient_chunks = value.iter().map(|v| v[0].clone()).collect_vec();

    let opened_values = OpenedValues {
        trace_local,
        trace_next,
        preprocessed_local,
        preprocessed_next,
        quotient_chunks,
    };
    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: log_degree,
    }
}

// TODO: finish
#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_on_quotient_domain: Option<Mat>,
    traces_on_quotient_domain: Vec<Mat>,
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let preprocessed_width = preprocessed_on_quotient_domain
        .as_ref()
        .map(Matrix::width)
        .unwrap_or_default();
    let widths = traces_on_quotient_domain
        .iter()
        .map(|trace| trace.width())
        .collect::<Vec<usize>>();
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
                .zip(widths.iter())
                .map(|(trace_on_quotient_domain, width)| {
                    RowMajorMatrix::new(
                        iter::empty()
                            .chain(trace_on_quotient_domain.vertically_packed_row(i_start))
                            .chain(
                                trace_on_quotient_domain.vertically_packed_row(i_start + next_step),
                            )
                            .collect_vec(),
                        *width,
                    )
                })
                .collect::<Vec<RowMajorMatrix<PackedVal<SC>>>>();

            let accumulator = PackedChallenge::<SC>::zero();
            let mut folder = ProverConstraintFolder {
                stages,
                preprocessed,
                public_values,
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
