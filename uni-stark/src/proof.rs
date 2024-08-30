use alloc::collections::btree_map::BTreeMap;
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

/// Updating with each new trace in every stage
pub struct State<'a, SC: StarkGenericConfig> {
    pub(crate) trace_commits: Vec<Com<SC>>,
    pub(crate) traces: Vec<PcsProverData<SC>>,
    pub(crate) public_values: Vec<&'a Vec<Val<SC>>>, // should also include challenge values
    pub(crate) challenger: &'a mut SC::Challenger,
    pcs: &'a <SC>::Pcs,
    trace_domain: Domain<SC>,
    log_degree: usize,
}

impl<'a, SC: StarkGenericConfig> State<'a, SC> {
    pub(crate) fn new(
        pcs: &'a <SC as StarkGenericConfig>::Pcs,
        trace_domain: Domain<SC>,
        challenger: &'a mut <SC as StarkGenericConfig>::Challenger,
        log_degree: usize,
    ) -> Self {
        Self {
            trace_commits: Vec::new(),
            traces: Vec::new(),
            public_values: Vec::new(),
            challenger,
            pcs,
            trace_domain,
            log_degree,
        }
    }

    pub(crate) fn get_log_degree(&self) -> usize {
        self.log_degree
    }

    pub(crate) fn get_trace_commits(&self) -> Vec<Com<SC>> {
        self.trace_commits.clone()
    }

    /// Observe a commitment.
    pub(crate) fn observe_commit(
        &mut self,
        trace_commit: Com<SC>,
        public_values: Option<&'a Vec<Val<SC>>>,
    ) {
        self.challenger.observe(trace_commit);
        match public_values {
            Some(public_values) => self.challenger.observe_slice(public_values),
            None => (),
        };
    }

    /// Get inputs to quotient calculation
    pub(crate) fn quotient_inputs(
        &mut self,
        log_quotient_degree: usize,
    ) -> (
        Vec<&'a Vec<Val<SC>>>,
        Domain<SC>,
        Domain<SC>,
        // Option<impl Matrix<Val<SC>> + 'a>,
        // Vec<impl Matrix<Val<SC>> + 'a>,
        SC::Challenge,
    ) {
        let alpha: SC::Challenge = self.challenger.sample_ext_element();

        let quotient_domain = self
            .trace_domain
            .create_disjoint_domain(1 << (self.log_degree + log_quotient_degree));

        (
            self.public_values.clone(),
            self.trace_domain,
            quotient_domain,
            alpha,
        )
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

        self.observe_commit(quotient_commit.clone(), None);

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
                .chain(
                    proving_key
                        .map(|proving_key| {
                            (&proving_key.preprocessed_data, vec![vec![zeta, zeta_next]])
                        })
                        .into_iter(),
                )
                .chain(
                    self.traces
                        .iter()
                        .map(|trace_data| (trace_data, vec![vec![zeta, zeta_next]]))
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
        let (stages_local, stages_next): (Vec<Vec<SC::Challenge>>, Vec<Vec<SC::Challenge>>) = self
            .traces
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

    /// Update the prover data state with the current challenge stage.
    pub(crate) fn run_stage<T>(
        &mut self,
        mut stage: Stage<'a, SC>,
        next_stage_trace_callback: Option<T>, // theoretically, all of the challenges should be held herefirst
    ) where
        T: NextStageTraceCallback<SC, Val<SC>> + Clone,
    {
        stage.get_challenge_values_from_challenger(
            self.challenger,
            next_stage_trace_callback.clone(),
        ); // fills in new trace, publics

        let (trace_commit, trace_data) = info_span!("commit to trace data").in_scope(|| {
            self.pcs
                .commit(vec![(self.trace_domain, stage.trace.unwrap())])
        });

        self.observe_commit(trace_commit.clone(), stage.public_values);

        self.traces.push(trace_data);
        self.trace_commits.push(trace_commit);
    }
}

pub struct Stage<'a, SC: StarkGenericConfig> {
    pub(crate) trace: Option<RowMajorMatrix<Val<SC>>>,
    pub(crate) public_values: Option<&'a Vec<Val<SC>>>,
    pub(crate) referenced_challenges: Option<&'a Vec<u64>>,
    pub(crate) stage_idx: u32,
}

impl<'a, SC: StarkGenericConfig> Stage<'a, SC> {
    pub(crate) fn get_challenge_values_from_challenger<T>(
        // honestly let's just try to fix this now
        &mut self,
        challenger: &mut SC::Challenger, // TODO: wrap this challenger
        next_stage_trace_callback: Option<T>,
    ) where
        T: NextStageTraceCallback<SC, Val<SC>> + Clone,
    {
        let challenges = self.referenced_challenges.map(|challenge_id| {
            challenge_id
                .iter()
                .map(|id| {
                    let challenge: Val<SC> = challenger.sample(); // for gl, should change for other fields
                    (*id, challenge)
                })
                .collect::<BTreeMap<u64, Val<SC>>>()
        });

        self.trace = Some(next_stage_trace_callback.unwrap().get_next_stage_trace(
            self.stage_idx,
            challenges.as_ref().unwrap_or(&BTreeMap::new()),
        ));

        let challenge_values = challenges
            .unwrap_or(BTreeMap::new())
            .into_values()
            .collect::<Vec<Val<SC>>>();

        if !challenge_values.is_empty() {
            self.public_values = Some(&challenge_values) // how do you expose these as publics?
        };
    }
}

pub trait NextStageTraceCallback<SC: StarkGenericConfig, T> {
    fn get_next_stage_trace(
        &self,
        trace_stage: u32,
        challenges: &BTreeMap<u64, Val<SC>>,
    ) -> RowMajorMatrix<T>;
}
