from .training_modules import AbsTrainingModule, MseTrainingModule, MaskerAbsTrainingModule, \
    MaskerMseTrainingModule


class NeuralRender(
    AbsTrainingModule,
    MseTrainingModule,
    MaskerAbsTrainingModule,
    MaskerMseTrainingModule
):
    pass
