from makiflow.core import MakiTrainer
from makiflow.core.training.core.hephaestus import Hephaestus
# It is for the teacher network.
# Hephaestus is required since its a great tool that provides API for interacting with the training graph.
# The functionality of the full trainer (that includes all the API of Hephaestus) is redundant and may even harm.
from abc import ABC, abstractmethod
from makiflow.debug import ExceptionScope
import tensorflow as tf


class Distillator(MakiTrainer, ABC):
    STUDENT = 'STUDENT MODEL'
    TEACHER = 'TEACHER MODEL'

    def _setup_label_placeholders(self):
        return {}

    def get_label_feed_dict_config(self):
        return {}

    def _init(self):
        super()._init()
        self._teacher = None
        self._layer_pairs = None
        self._teacher_train_graph = None
        self._track_layer_losses = False

    # noinspection PyAttributeOutsideInit
    def set_teacher(self, teacher):
        self._teacher = teacher
        self._teacher_train_graph = Hephaestus(self._teacher, train_inputs=super().get_train_inputs_list())

    # noinspection PyAttributeOutsideInit
    def track_layer_losses(self, track=True):
        """
        Sets a marker which indicates whether to track individual losses between layers.

        Parameters
        ----------
        track : bool
            If set to False, the individual losses won't be tracked.
        """
        self._track_layer_losses = track

    # noinspection PyAttributeOutsideInit
    def set_layer_pairs(self, layer_pairs):
        """
        Sets pairs (student_layer_name, teacher_layer_name). Later the teacher's layers will
        be distilled into the student's layers accordingly.

        Notes
        -----
        Actually, the loss is created using MakiTensors with such names. It just was made this way, that
        most of the layers outputs inherit parent layer's name, so we may say `set_layer_pairs` instead of
        `set_makitensor_pairs` for simplicity and abstraction. In case a layer outputs more than one MakiTensor
        (which is the case for RNN layers), the user should pass in its MakiTensor names, the layer's name
        won't work.

        Parameters
        ----------
        layer_pairs : list
            Container tuples (student_layer_name, teacher_layer_name).
        """
        self._layer_pairs = layer_pairs

    def compile_training_graph(self):
        with ExceptionScope(f'{Distillator.STUDENT} Graph compilation'):
            super().compile_training_graph()

        with ExceptionScope(f'{Distillator.TEACHER} Graph compilation'):
            self._teacher_train_graph.compile_training_graph()

    def _build_loss(self):
        losses = []
        for (stud_tensor, stud_name), (teach_tensor, teach_name) in self.get_layer_output_pairs():
            loss = self._build_distill_loss(stud_tensor, teach_tensor)
            if self._track_layer_losses:
                super().track_loss(loss_tensor=loss, loss_name=Distillator.layer_loss_name(stud_name, teach_name))

            losses.append(loss)

        return tf.add_n(losses)

    def get_layer_output_pairs(self):
        assert self._teacher is not None, 'The teacher model is not set.'
        assert self._layer_pairs is not None, 'The layer pairs are not set.'
        assert self._teacher_train_graph.is_compiled(), 'The training graph is not compiled.'

        output_tensor_pairs = []
        for student_layer_name, teacher_layer_name in self._layer_pairs:
            with ExceptionScope(Distillator.STUDENT):
                student_tensor = super().get_traingraph_tensor(student_layer_name)
                student_tuple = (student_tensor, student_layer_name)
            with ExceptionScope(Distillator.TEACHER):
                teacher_tensor = self._teacher_train_graph.get_traingraph_tensor(teacher_layer_name)
                teacher_tuple = (teacher_tensor, teacher_layer_name)

            output_tensor_pairs.append((student_tuple, teacher_tuple))

        return output_tensor_pairs

    @abstractmethod
    def _build_distill_loss(self, student_tensor, teacher_tensor):
        pass

    @staticmethod
    def layer_loss_name(student_layer_name, teacher_layer_name):
        return f'Layer loss: {student_layer_name} / {teacher_layer_name}'


