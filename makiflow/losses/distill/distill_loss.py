import makiflow as mf
from makiflow.core.training.trainer.train_graph_compiler import GraphCompiler


class DistillLoss(mf.losses.Loss):
    TEACHER_TENSOR = mf.losses.Loss.LABELS

    def __init__(self, teacher: mf.Model, train_inputs: list, tensor_pair, similarity_fn, _teacher_train_graph=False):
        """
        A generic class for distillation loss.

        Parameters
        ----------
        teacher : mf.Model
            Teacher's model. It is advisable to create it with a batch size of 1 for minimizing
            the memory load.
        train_inputs : list
            Training input makitensors. Will be used for creation of the teacher's training graph.
        tensor_pair : tuple
            First is the student tensor name, second is the teacher tensor name.
        similarity_fn : function
            A function with the following signature: (tensors: list, label_tensors: dict)
        _teacher_train_graph : bool
            Set to true if teacher is a GraphCompiler instance with already compiled graph. Otherwise the
            teachers training graph will be built once again.
        """
        if _teacher_train_graph:
            self._teacher_train_graph = teacher
        else:
            self._teacher_train_graph = GraphCompiler(model=teacher, train_inputs=train_inputs)
            self._teacher_train_graph.compile_training_graph()

        label_tensor = self._teacher_train_graph.get_traingraph_tensor(tensor_pair[1])
        super().__init__(
            tensor_names=[tensor_pair[0]],
            # Teacher tensors are not data sources and should not be passed to the trainer.
            # In order to do that we pass in a tuple (label_tensor, None) instead of
            # solely label_tensor
            label_tensors={DistillLoss.TEACHER_TENSOR: (label_tensor, None)},
            loss_fn=similarity_fn
        )

    @property
    def teacher_train_graph(self):
        return self._teacher_train_graph