from makiflow.core import LossFabric


class CustomLoss(LossFabric):
    def __init__(self, tensor_names, label_tensors: dict, loss_fn):
        self._loss_fn = loss_fn
        super().__init__(tensor_names, label_tensors)

    def build_loss(self, prediction, label_tensors):
        return self._loss_fn(prediction, label_tensors)

