use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;

use p3_commit::{Pcs, PolynomialSpace, Val};
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use crate::StarkGenericConfig;

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
pub struct UpdatingCommitData<'a, SC: StarkGenericConfig + PolynomialSpace> {
    pub(crate) trace_commits: Vec<Com<SC>>,
    pub(crate) traces: Vec<PcsProverData<SC>>,
    pub(crate) public_values: Vec<&'a Vec<Val<SC>>>, // should also include challenge values
    pub(crate) challenger: &'a mut SC::Challenger,
    pcs: &'a <SC>::Pcs,
    domain: <<SC as StarkGenericConfig>::Pcs as Pcs<
        <SC as StarkGenericConfig>::Challenge,
        <SC as StarkGenericConfig>::Challenger,
    >>::Domain,
}

impl<'a, SC: StarkGenericConfig + PolynomialSpace> UpdatingCommitData<'a, SC> {
    pub(crate) fn new(
        pcs: &'a <SC as StarkGenericConfig>::Pcs,
        domain: <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::Domain,
        challenger: &'a mut <SC as StarkGenericConfig>::Challenger,
    ) -> Self {
        Self {
            trace_commits: Vec::new(),
            traces: Vec::new(),
            public_values: Vec::new(),
            challenger,
            pcs,
            domain,
        }
    }

    pub(crate) fn get_pcs(&self) -> &'a <SC>::Pcs {
        self.pcs
    }

    pub(crate) fn get_domain(
        &self,
    ) -> <<SC as StarkGenericConfig>::Pcs as Pcs<
        <SC as StarkGenericConfig>::Challenge,
        <SC as StarkGenericConfig>::Challenger,
    >>::Domain {
        self.domain
    }

    pub(crate) fn update_stage(&mut self, incoming_data: IncomingData<'a, SC>) {
        self.trace_commits.push(incoming_data.commit);
        self.traces.push(incoming_data.trace);
        self.public_values.push(incoming_data.public_values);
    }
}

pub struct IncomingData<'a, SC: StarkGenericConfig + PolynomialSpace> {
    pub(crate) trace: PcsProverData<SC>,
    pub(crate) public_values: &'a Vec<Val<SC>>,
    pub(crate) commit: Com<SC>,
}

pub trait NextStageTraceCallback<SC: StarkGenericConfig, T> {
    fn get_next_stage_trace(
        &self,
        trace_stage: u32,
        challenge_values: BTreeMap<u64, SC::Challenge>,
    ) -> RowMajorMatrix<T>;
}
