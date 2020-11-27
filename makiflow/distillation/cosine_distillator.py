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

from .core import Distillator
import tensorflow as tf
from makiflow.core.debug import ExceptionScope


class CosineDistillator(Distillator):
    def _init(self):
        super()._init()
        self._axis = [1, 2, 3]

    def set_axis(self, axis):
        """
        Sets the axis (axes) in which the cosine distance is being measured.
        By default entire feature tensors are treated as vectors the cosine distance
        is measured for. In other words default axis=[1, 2, 3] - h, w, c.
        You may set axis=3, so that the cosine distance is measure on individual
        feature vectors in each location of the feature map.

        Parameters
        ----------
        axis : list or int
            The axis in which the cosine distance is being measured.
        """
        self._axis = axis

    def _build_distill_loss(self, student_tensor, teacher_tensor):
        with ExceptionScope('Normalization of the student tensor'):
            student_tensor = tf.nn.l2_normalize(student_tensor, axis=self._axis)
        with ExceptionScope('Normalization of the teacher tensor'):
            teacher_tensor = tf.nn.l2_normalize(teacher_tensor, axis=self._axis)

        cosine_similarity = tf.reduce_sum(student_tensor * teacher_tensor)
        # We should subtract the scalar_product from ones. However, it does not affect the gradient,
        # therefore, we may omit it to save computation time and memory.
        return -cosine_similarity


# For debug
def test_training():
    from makiflow.core.debug import classificator
    BATCH_SIZE = 32
    student, train_in_x = classificator(train_batch_size=BATCH_SIZE)
    teacher = classificator()
    sess = tf.Session()
    student.set_session(sess)
    teacher.set_session(sess)

    print('Setting up the distillator.')
    distillator = CosineDistillator(student, train_inputs=[train_in_x])
    distillator.set_teacher(teacher)

    layer_pairs = [
        ('conv1', 'conv1'),
        ('conv2', 'conv2')
    ]
    distillator.set_layer_pairs(layer_pairs)

    print('Compiling.')
    distillator.compile()

    print('Test training...')
    import numpy as np

    def test_generator():
        image = np.random.randn(BATCH_SIZE, 32, 32, 3)
        while True:
            yield (image,), ()

    gen = test_generator()
    distillator.fit_generator(
        generator=gen,
        optimizer=tf.train.AdamOptimizer(),
        epochs=5,
        iter=10,
    )


def test_exception_scope():
    from makiflow.core.debug import classificator
    BATCH_SIZE = 32
    student, train_in_x = classificator(train_batch_size=BATCH_SIZE)
    teacher = classificator(input_shape=[8, 8, 3])
    print('Setting up the distillator.')
    distillator = CosineDistillator(student, train_inputs=[train_in_x])
    distillator.set_teacher(teacher)

    layer_pairs = [
        ('conv1', 'conv1'),
        ('conv3', 'conv3')
    ]
    distillator.set_layer_pairs(layer_pairs)

    print('Compiling.')
    distillator.compile()


if __name__ == '__main__':
    print('TEST TRAINING.')
    test_training()
    print('TEST EXCEPTION SCOPE')
    # test_exception_scope()
