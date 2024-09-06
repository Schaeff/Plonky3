use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::{Domain, StarkGenericConfig, Val};

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;
pub type PcsProverData<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::ProverData;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    pub(crate) commitments: Commitments<Com<SC>>,
    pub(crate) opened_values: OpenedValues<SC::Challenge>,
    pub(crate) opening_proof: PcsProof<SC>,
    pub(crate) degree_bits: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Commitments<Com> {
    pub(crate) stages: Vec<Com>, // we need to fix this
    pub(crate) quotient_chunks: Com,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    pub(crate) preprocessed_local: Vec<Challenge>,
    pub(crate) preprocessed_next: Vec<Challenge>,
    pub(crate) stages_local: Vec<Vec<Challenge>>,
    pub(crate) stages_next: Vec<Vec<Challenge>>,
    pub(crate) quotient_chunks: Vec<Vec<Challenge>>,
}

pub struct StarkProvingKey<SC: StarkGenericConfig> {
    pub preprocessed_commit: Com<SC>,
    pub preprocessed_data: PcsProverData<SC>,
}
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    pub preprocessed_commit: Com<SC>,
}

pub struct ProcessedStage<SC: StarkGenericConfig> {
    pub(crate) commitment: Com<SC>,
    pub(crate) prover_data: PcsProverData<SC>,
    pub(crate) challenge_values: Vec<Val<SC>>,
}

/// Updating with each new trace in every stage
pub struct State<'a, SC: StarkGenericConfig> {
    // todo: let each stage also set public values
    pub(crate) public_values: &'a Vec<Val<SC>>,
    pub(crate) processed_stages: Vec<ProcessedStage<SC>>,
    pub(crate) challenger: &'a mut SC::Challenger,
    pcs: &'a <SC>::Pcs,
    trace_domain: Domain<SC>,
    log_degree: usize,
}

pub struct QuotientInputs<'a, SC: StarkGenericConfig> {
    pub public_values: Vec<&'a Vec<Val<SC>>>,
    pub trace_domain: Domain<SC>,
    pub quotient_domain: Domain<SC>,
    pub alpha: SC::Challenge,
}

impl<'a, SC: StarkGenericConfig> State<'a, SC> {
    pub(crate) fn new(
        pcs: &'a <SC as StarkGenericConfig>::Pcs,
        trace_domain: Domain<SC>,
        challenger: &'a mut <SC as StarkGenericConfig>::Challenger,
        log_degree: usize,
        public_values: &'a Vec<Val<SC>>,
    ) -> Self {
        Self {
            public_values,
            processed_stages: Default::default(),
            challenger,
            pcs,
            trace_domain,
            log_degree,
        }
    }

    pub(crate) fn get_log_degree(&self) -> usize {
        self.log_degree
    }

    /// Observe a commitment.
    pub(crate) fn observe_commit(&mut self, trace_commit: Com<SC>) {
        self.challenger.observe(trace_commit);
    }

    /// Get inputs to quotient calculation
    pub(crate) fn quotient_inputs(&mut self, log_quotient_degree: usize) -> QuotientInputs<'a, SC> {
        let alpha: SC::Challenge = self.challenger.sample_ext_element();

        let quotient_domain = self
            .trace_domain
            .create_disjoint_domain(1 << (self.log_degree + log_quotient_degree));

        QuotientInputs {
            public_values: vec![&self.public_values],
            trace_domain: self.trace_domain,
            quotient_domain,
            alpha,
        }
    }

    pub(crate) fn on_quotient_domain(
        &self,
        data: &'a PcsProverData<SC>,
        quotient_domain: Domain<SC>,
    ) -> impl Matrix<Val<SC>> + 'a {
        self.pcs.get_evaluations_on_domain(data, 0, quotient_domain)
    }

    pub(crate) fn commit_to_quotient(
        &mut self,
        quotient_values: Vec<SC::Challenge>,
        log_quotient_degree: usize,
    ) -> (Com<SC>, PcsProverData<SC>) {
        let quotient_domain = self
            .trace_domain
            .create_disjoint_domain(1 << (self.log_degree + log_quotient_degree));

        let quotient_degree = 1 << log_quotient_degree;

        let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
        let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
        let qc_domains = quotient_domain.split_domains(quotient_degree);

        let (quotient_commit, quotient_data) = info_span!("commit to quotient poly chunks")
            .in_scope(|| {
                self.pcs
                    .commit(izip!(qc_domains, quotient_chunks).collect_vec())
            });

        self.observe_commit(quotient_commit.clone());

        (quotient_commit, quotient_data)
    }

    /// Open the quotient values at a given evaluation.
    pub(crate) fn get_open_proof(
        &mut self,
        proving_key: Option<&StarkProvingKey<SC>>,
        quotient_data: PcsProverData<SC>,
        log_quotient_degree: usize,
    ) -> (OpenedValues<SC::Challenge>, PcsProof<SC>) {
        let quotient_degree = 1 << log_quotient_degree;
        let zeta: SC::Challenge = self.challenger.sample(); // ? what
        let zeta_next = self.trace_domain.next_point(zeta).unwrap();

        let (opened_values, opening_proof) = self.pcs.open(
            iter::empty()
                .chain(proving_key.map(|proving_key| {
                    (&proving_key.preprocessed_data, vec![vec![zeta, zeta_next]])
                }))
                .chain(
                    self.processed_stages
                        .iter()
                        .map(|processed_stage| {
                            (&processed_stage.prover_data, vec![vec![zeta, zeta_next]])
                        })
                        .collect_vec(),
                )
                .chain([(
                    &quotient_data,
                    // open every chunk at zeta
                    (0..quotient_degree).map(|_| vec![zeta]).collect_vec(),
                )])
                .collect_vec(),
            self.challenger,
        );

        let mut opened_values = opened_values.iter();

        let (preprocessed_local, preprocessed_next) = if proving_key.is_some() {
            let value = opened_values.next().unwrap();
            assert_eq!(value.len(), 1);
            assert_eq!(value[0].len(), 2);
            (value[0][0].clone(), value[0][1].clone())
        } else {
            (vec![], vec![])
        };

        // get values for traces
        let (stages_local, stages_next): (Vec<_>, Vec<_>) = self
            .processed_stages
            .iter()
            .map(|_| {
                let value = opened_values.next().unwrap();
                assert_eq!(value.len(), 1);
                assert_eq!(value[0].len(), 2);
                (value[0][0].clone(), value[0][1].clone())
            })
            .unzip();

        // get values for the quotient
        let value = opened_values.next().unwrap();
        assert_eq!(value.len(), quotient_degree);
        let quotient_chunks = value.iter().map(|v| v[0].clone()).collect_vec();

        let opened_values = OpenedValues {
            stages_local,
            stages_next,
            preprocessed_local,
            preprocessed_next,
            quotient_chunks,
        };

        (opened_values, opening_proof)
    }

    pub(crate) fn run_stage(mut self, stage: Stage<SC>) -> Self {
        let (commitment, prover_data) = info_span!("commit to trace data")
            .in_scope(|| self.pcs.commit(vec![(self.trace_domain, stage.trace)]));

        self.observe_commit(commitment.clone());

        let challenge_values = (0..stage.challenge_count)
            .map(|_| self.challenger.sample())
            .collect();

        self.processed_stages.push(ProcessedStage {
            prover_data,
            commitment,
            challenge_values,
        });
        self
    }
}

pub struct Stage<SC: StarkGenericConfig> {
    // the witness for this stage
    pub(crate) trace: RowMajorMatrix<Val<SC>>,
    // the number of challenges to be drawn at the end of this stage
    pub(crate) challenge_count: usize,
}

pub trait NextStageTraceCallback<SC: StarkGenericConfig> {
    fn get_next_stage_trace(
        &self,
        trace_stage: u32,
        challenges: &[Val<SC>],
    ) -> RowMajorMatrix<Val<SC>>;
}
