from .training_modules import AbsTrainingModule, MseTrainingModule, MaskerAbsTrainingModule


class NeuralRender(AbsTrainingModule, MseTrainingModule, MaskerAbsTrainingModule):
    pass
