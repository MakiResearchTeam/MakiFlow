from .training_modules import QCETrainingModule, CETrainingModule, FocalTrainingModule, MakiTrainingModule


class Classificator(CETrainingModule, QCETrainingModule, FocalTrainingModule, MakiTrainingModule):
    pass
