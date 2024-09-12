use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues};

pub trait MultistageAirBuilder: AirBuilderWithPublicValues {
    type Challenge: Clone + Into<Self::Expr>;

    /// Traces from each stage.
    fn multi_stage(&self, stage: usize) -> Self::M;

    /// Challenges from each stage, drawn from the base field
    fn challenges(&self, stage: usize) -> &[Self::Challenge];

    fn stage_public_values(&self, stage: usize) -> &[Self::PublicVar] {
        match stage {
            0 => self.public_values(),
            _ => unimplemented!(),
        }
    }
}

pub trait MultiStageAir<AB: AirBuilder>: Air<AB> {
    fn stage_count(&self) -> usize {
        1
    }

    /// The number of columns in a given higher-stage trace.
    fn multi_stage_width(&self, stage: u32) -> usize {
        match stage {
            0 => self.width(),
            _ => unimplemented!(),
        }
    }

    /// The number of challenges produced at the end of each stage
    fn challenge_count(&self, _stage: u32) -> usize {
        0
    }
}

impl<AB: AirBuilder, A: Air<AB>> MultiStageAir<AB> for A {}
