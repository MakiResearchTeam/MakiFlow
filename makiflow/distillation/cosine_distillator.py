from .core import Distillator
import tensorflow as tf
from makiflow.debug.exception_scope import ExceptionScope


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

        cosine_similarity = tf.reduce_sum(student_tensor * teacher_tensor, axis=self._axis)
        # We should subtract the scalar_product from ones. However, it does not affect the gradient,
        # therefore, we may omit to save computation time and memory
        cosine_distance_ish = -cosine_similarity
        return tf.reduce_mean(cosine_distance_ish)


# For debug
def test_training():
    from makiflow.debug import classificator
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
    from makiflow.debug import classificator
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
