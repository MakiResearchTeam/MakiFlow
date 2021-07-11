import makiflow as mf
from .distill_loss import DistillLoss


class MAE(DistillLoss):
    def __init__(
            self, teacher: mf.Model, train_inputs,
            tensor_pair, _teacher_train_graph=False,
            reduction=mf.losses.Loss.REDUCTION_MEAN
    ):
        """
        Performs model distillation using l1 loss.

        Parameters
        ----------
        teacher : mf.Model
            Model to distill.
        train_inputs : list
            Training input makitensors. Will be used for creation of the teacher's training graph.
        tensor_pair : tuple
            First is the student tensor name, second is the teacher tensor name.
        _teacher_train_graph : bool
            Set to true if teacher is a GraphCompiler instance with already compiled graph. Otherwise the
            teachers training graph will be built once again.
        reduction : int
            Type of loss tensor reduction. By default equals to 'Loss.REDUCTION_MEAN`.
        """
        loss_fn = lambda t, lt: mf.losses.MAE.mean_absolute_error(t, lt, reduction)
        super().__init__(teacher, train_inputs, tensor_pair, loss_fn, _teacher_train_graph)
