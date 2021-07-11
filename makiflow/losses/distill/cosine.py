import tensorflow as tf

import makiflow as mf
from .distill_loss import DistillLoss


class Cosine(DistillLoss):
    def __init__(
            self, teacher: mf.Model, train_inputs,
            tensor_pair, _teacher_train_graph=False, axes=-1,
            reduction=mf.losses.Loss.REDUCTION_MEAN
    ):
        """
        Performs model distillation via cosine similarity.

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
        axes : int or list
            Determines along which dimension the similarity is being measured. By default last dimension (-1) is set.
        reduction : int
            Type of loss tensor reduction. By default equals to 'Loss.REDUCTION_MEAN`.
        """
        loss_fn = lambda t, lt: Cosine.cosine_similarity(t, lt, reduction, axes)
        super().__init__(teacher, train_inputs, tensor_pair, loss_fn, _teacher_train_graph)

    @staticmethod
    def cosine_similarity(tensors, label_tensors, reduction, axes):
        student_tensor = tensors[0]
        teacher_tensor = label_tensors[DistillLoss.TEACHER_TENSOR]

        student_tensor = tf.nn.l2_normalize(student_tensor, axis=axes)
        teacher_tensor = tf.nn.l2_normalize(teacher_tensor, axis=axes)

        cos_sim = student_tensor * teacher_tensor

        reduction_fn = mf.losses.Loss.REDUCTION_FN[reduction]
        # We should subtract the scalar_product from ones. However, it does not affect the gradient,
        # therefore, we may omit it to save computation time and memory.
        return -reduction_fn(cos_sim)
