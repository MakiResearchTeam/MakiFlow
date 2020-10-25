from .core.aion import Aion


class TrainerBuilder:
    TYPE = Aion.TYPE
    PARAMS = Aion.PARAMS

    # Contains pairs
    TRAINERS = {}

    @staticmethod
    def register_trainer(trainer_class):
        TrainerBuilder.TRAINERS.update(
            {trainer_class.TYPE: trainer_class}
        )

    @staticmethod
    def trainer_from_dict(model, train_inputs, label_tensors, info_dict):
        trainer_type = info_dict[TrainerBuilder.TYPE]
        params = info_dict[TrainerBuilder.PARAMS]
        trainer_class = TrainerBuilder.TRAINERS.get(trainer_type)
        assert trainer_type is not None, f'There is no trainer with TYPE={trainer_type}'
        trainer_object = trainer_class(
            model=model,
            train_inputs=train_inputs,
            label_tensors=label_tensors
        )
        trainer_object.setup_params(params)
        return trainer_object

