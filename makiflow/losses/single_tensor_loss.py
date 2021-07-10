from makiflow.core import Loss


class SingleTensorLoss(Loss):
    def __init__(self, tensor_names: list, label_tensors: dict, loss_fn):
        assert len(tensor_names) == 1, f'This loss expects only one tensor in `tensor_names`, ' \
                                       f'but received {len(tensor_names)}. tensor_names={tensor_names}'
        super().__init__(tensor_names, label_tensors, loss_fn)