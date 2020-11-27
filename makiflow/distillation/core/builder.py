class DistillatorBuilder:
    TYPE = 'type'
    PARAMS = 'params'

    LAYER_PAIRS = 'layer_pairs'

    # Contains pairs
    DISTILLATORS = {}

    @staticmethod
    def add_distillator(distillator_class):
        DistillatorBuilder.DISTILLATORS.update(
            {distillator_class.__name__: distillator_class}
        )

    @staticmethod
    def distillator_from_dict(teacher, info_dict):
        distillator_type = info_dict[DistillatorBuilder.TYPE]
        distillator_class = DistillatorBuilder.DISTILLATORS.get(distillator_type)
        assert distillator_type is not None, f'There is no distillator with TYPE={distillator_type}'

        params = info_dict[DistillatorBuilder.PARAMS]
        layer_pairs = params[DistillatorBuilder.LAYER_PAIRS]
        distillator_object = distillator_class(
            teacher=teacher, layer_pairs=layer_pairs
        )
        distillator_object.set_params(params)
        return distillator_object


def register_distillator(distillator_class):
    """
    Decorator that add the decorated class to the DistillatorBuilder'
    list of all distillators.

    Parameters
    ----------
    distillator_class : type
        The distillator's class to add.
    """
    print(distillator_class)
    DistillatorBuilder.add_distillator(distillator_class)
    return distillator_class


def build_method(method):
    """
    A dummy marker for methods required during building of a
    distillator object. Serves for the developers as a hint.
    """
    return method
