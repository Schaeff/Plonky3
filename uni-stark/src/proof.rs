use alloc::collections::btree_map::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSample};
use p3_commit::Pcs;
use p3_matrix::dense::RowMajorMatrix;
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
    pub(crate) trace: Com,
    pub(crate) quotient_chunks: Com,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    pub(crate) preprocessed_local: Vec<Challenge>,
    pub(crate) preprocessed_next: Vec<Challenge>,
    pub(crate) trace_local: Vec<Challenge>,
    pub(crate) trace_next: Vec<Challenge>,
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
    challenger: &'a mut SC::Challenger,
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

    // TODO: fix
    pub(crate) fn from(self) -> Self {
        let mut new = Self::new(
            self.pcs,
            self.trace_domain,
            self.challenger,
            self.log_degree,
        );
        new.traces.extend(self.traces);
        new.trace_commits = self.trace_commits.clone();
        new.public_values = self.public_values.clone();
        new
    }

    pub(crate) fn get_challenger(&mut self) -> &mut SC::Challenger {
        self.challenger
    }

    pub(crate) fn get_pcs(&self) -> &'a <SC>::Pcs {
        self.pcs
    }

    pub(crate) fn get_trace_domain(&self) -> Domain<SC> {
        self.trace_domain
    }

    pub(crate) fn get_log_degree(&self) -> usize {
        self.log_degree
    }

    pub(crate) fn observe_commit(
        &mut self,
        trace_commit: Com<SC>,
        public_values: Option<&'a Vec<Val<SC>>>,
    ) {
        self.challenger.observe(trace_commit);
        self.challenger.observe_slice(public_values.unwrap());
    }

    pub(crate) fn run_stage<T>(
        &mut self,
        mut stage: Stage<'a, SC>,
        next_stage_trace_callback: Option<&'a T>,
    ) where
        T: NextStageTraceCallback<SC, Val<SC>>,
    {
        stage.get_challenge_values_from_challenger(self.challenger, next_stage_trace_callback); // fills in new trace, publics

        let (trace_commit, trace_data) = info_span!("commit to trace data").in_scope(|| {
            self.pcs
                .commit(vec![(self.trace_domain, stage.trace.unwrap())])
        });

        self.observe_commit(trace_commit.clone(), stage.public_values);

        // let mut new_state = State::from(state);

        self.traces.push(trace_data);
        self.trace_commits.push(trace_commit);
    }
}

pub struct Stage<'a, SC: StarkGenericConfig> {
    pub(crate) trace: Option<RowMajorMatrix<Val<SC>>>,
    pub(crate) public_values: Option<&'a Vec<Val<SC>>>,
    pub(crate) referenced_challenges: Option<Vec<u64>>,
    pub(crate) stage_idx: u32,
}

impl<'a, SC: StarkGenericConfig> Stage<'a, SC> {
    pub(crate) fn get_challenge_values_from_challenger<T>(
        &mut self,
        challenger: &mut SC::Challenger,
        next_stage_trace_callback: Option<&'a T>,
    ) where
        T: NextStageTraceCallback<SC, Val<SC>>,
    {
        let challenge_values = match &self.referenced_challenges {
            None => BTreeMap::new(),
            Some(challenge_id) => challenge_id
                .iter()
                .map(|id| {
                    let challenge: SC::Challenge = challenger.sample();
                    (*id, challenge)
                })
                .collect::<BTreeMap<u64, SC::Challenge>>(),
        };

        self.trace = Some(
            next_stage_trace_callback
                .unwrap()
                .get_next_stage_trace(self.stage_idx, challenge_values),
        );

        let challenge_values = challenge_values
            .values()
            .cloned()
            .collect::<Vec<SC::Challenge>>();

        self.public_values = match self.public_values {
            Some(publics) => Some(publics),
            None => Some(&challenge_values),
        }
    }
}

pub trait NextStageTraceCallback<SC: StarkGenericConfig, T> {
    fn get_next_stage_trace(
        &self,
        trace_stage: u32,
        challenge_values: BTreeMap<u64, SC::Challenge>,
    ) -> RowMajorMatrix<T>;
}
