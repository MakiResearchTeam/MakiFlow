from makiflow.core.training.core.hephaestus import Hephaestus
# It is for the teacher network.
# Hephaestus is required since its a great tool that provides API for interacting with the training graph.
# The functionality of the full trainer (that includes all the API of Hephaestus) is redundant and may even harm.
from abc import ABC, abstractmethod
from makiflow.debug import ExceptionScope
import tensorflow as tf
from .class_decorator import ClassDecorator, overloaded
from makiflow.core import MakiModel, MakiTrainer


class Distillator(ClassDecorator, ABC):
    STUDENT = 'STUDENT MODEL'
    TEACHER = 'TEACHER MODEL'
    DISTILLATION_LOSS = 'DISTILLATION_LOSS'

    def __init__(self, teacher: MakiModel, layer_pairs):
        super().__init__()
        self._teacher = teacher
        self._layer_pairs = layer_pairs
        self._teacher_train_graph = None
        self._track_layer_losses = False
        self._init()

    def _init(self):
        # Used by the subclasses to initialize necessary variables
        pass

    def _call_init(self, obj):
        self._teacher_train_graph = Hephaestus(self._teacher, train_inputs=obj.get_train_inputs_list())

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

    def compile_training_graph(self):
        with ExceptionScope(f'{Distillator.STUDENT} Graph compilation'):
            self.get_student_trainer().compile_training_graph()

        with ExceptionScope(f'{Distillator.TEACHER} Graph compilation'):
            self._teacher_train_graph.compile_training_graph()

    @overloaded
    def _build_loss(self):
        losses = []
        for (stud_tensor, stud_name), (teach_tensor, teach_name) in self.get_layer_output_pairs():
            loss = self._build_distill_loss(stud_tensor, teach_tensor)
            if self._track_layer_losses:
                self.track_loss(loss_tensor=loss, loss_name=Distillator.layer_loss_name(stud_name, teach_name))

            losses.append(loss)

        return tf.add_n(losses)

    @overloaded
    def track_loss(self, loss_tensor, loss_name):
        # This method might not have been overloaded, there is no necessity to do so.
        # However, it is since the code become a bit more transparent.
        self.get_student_trainer().track_loss(loss_tensor=loss_tensor, loss_name=loss_name)

    @overloaded
    def build_loss(self):
        with ExceptionScope(Distillator.DISTILLATION_LOSS + ' construction'):
            distillation_loss = self._build_loss()

        assert distillation_loss is not None, '_build_loss method returned None, but must return the loss scalar.'
        self.get_student_trainer().add_loss(distillation_loss)
        self.get_student_trainer().track_loss(distillation_loss, Distillator.DISTILLATION_LOSS)

        with ExceptionScope(Distillator.STUDENT + ' loss construction'):
            self.get_student_trainer().build_loss()

    @overloaded
    def compile(self):
        self.compile_training_graph()
        self.build_loss()

    def get_layer_output_pairs(self):
        assert self._teacher is not None, 'The teacher model is not set.'
        assert self._layer_pairs is not None, 'The layer pairs are not set.'
        assert self._teacher_train_graph.is_compiled(), 'The training graph is not compiled.'

        student_trainer = self.get_student_trainer()

        output_tensor_pairs = []
        for student_layer_name, teacher_layer_name in self._layer_pairs:
            with ExceptionScope(Distillator.STUDENT):
                student_tensor = student_trainer.get_traingraph_tensor(student_layer_name)
                student_tuple = (student_tensor, student_layer_name)
            with ExceptionScope(Distillator.TEACHER):
                teacher_tensor = self._teacher_train_graph.get_traingraph_tensor(teacher_layer_name)
                teacher_tuple = (teacher_tensor, teacher_layer_name)

            output_tensor_pairs.append((student_tuple, teacher_tuple))

        return output_tensor_pairs

    def get_student_trainer(self) -> MakiTrainer:
        return super().get_obj()

    @abstractmethod
    def _build_distill_loss(self, student_tensor, teacher_tensor):
        pass

    @staticmethod
    def layer_loss_name(student_layer_name, teacher_layer_name):
        return f'Layer loss: {student_layer_name} / {teacher_layer_name}'
