from .training_modules import AbsTrainingModule, MseTrainingModule, MaskedAbsTrainingModule, \
    MaskedMseTrainingModule, PerceptualTrainingModule


class NeuralRender(
    AbsTrainingModule,
    MseTrainingModule,
    MaskedAbsTrainingModule,
    MaskedMseTrainingModule,
    PerceptualTrainingModule
):
    pass
