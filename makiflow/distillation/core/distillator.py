# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from makiflow.core.training.trainer.train_graph_compiler import TrainGraphCompiler
# It is for the teacher network.
# Hephaestus is required since its a great tool that provides API for interacting with the training graph.
# The functionality of the full trainer (that includes all the API of Hephaestus) is redundant and may even harm.
from abc import ABC, abstractmethod
from makiflow.core.debug import ExceptionScope
import tensorflow as tf
from makiflow.core.dev import ClassDecorator, overloaded
from makiflow.core import MakiModel, MakiTrainer
from .builder import build_method


class Distillator(ClassDecorator, ABC):
    STUDENT = 'STUDENT MODEL'
    TEACHER = 'TEACHER MODEL'
    DISTILLATION_LOSS = 'DISTILLATION_LOSS'

    LOSS_SCALE = 'scale'
    TRACK_LAYER_LOSSES = 'track_losses'

    def __init__(self, teacher: MakiModel, layer_pairs):
        """
        A decorator that adds distillation functionality to the usual MakiTrainer.

        Parameters
        ----------
        teacher : MakiModel
            The teacher model that will be distilled into the student.
        layer_pairs : list
            Contains tuples (student_layer_name, teacher_layer_name). The teacher's layer named
            `teacher_layer_name` will be distilled into the student's layer named `student_layer_name`.
        """
        super().__init__()
        self._teacher = teacher
        self._layer_pairs = layer_pairs
        self._teacher_train_graph = None
        self._track_layer_losses = False
        self._loss_scale = 1.0
        self._init()

    def set_loss_scale(self, scale):
        """
        The distillation loss will then be scaled by the `scale`.
        However, the unscaled loss value will be tracked. This is useful for
        making fair comparisons.

        Parameters
        ----------
        scale : float
        """
        assert scale > 0.0, 'scale must be positive.'
        self._loss_scale = scale

    @build_method
    def set_params(self, params):
        loss_scale = params.get(Distillator.LOSS_SCALE)
        if loss_scale is not None:
            self.set_loss_scale(loss_scale)

        track_losses = params.get(Distillator.TRACK_LAYER_LOSSES)
        if track_losses is not None:
            self.track_layer_losses(track_losses)

    def _init(self):
        # Used by the subclasses to initialize necessary variables
        pass

    def _call_init(self, obj):
        self._teacher_train_graph = TrainGraphCompiler(self._teacher, train_inputs=obj.get_train_inputs_list())
        # { layer_name: layer }
        layers = self._teacher.get_layers()
        layers_trainable = [(layer_name, False) for layer_name in layers.keys()]
        self._teacher_train_graph.set_layers_trainable(layers_trainable)

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
        self.get_student_trainer().add_loss(distillation_loss * self._loss_scale)
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
