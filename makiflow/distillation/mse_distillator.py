from .core import Distillator
import tensorflow as tf


class MSEDistillator(Distillator):
    def _build_distill_loss(self, student_tensor, teacher_tensor):
        return tf.nn.l2_loss(student_tensor - teacher_tensor)


if __name__ == '__main__':
    from makiflow.debug import classificator

    BATCH_SIZE = 32
    student, train_in_x = classificator(train_batch_size=BATCH_SIZE)
    teacher = classificator()
    sess = tf.Session()
    student.set_session(sess)
    teacher.set_session(sess)

    print('Setting up the distillator.')
    distillator = MSEDistillator(student, train_inputs=[train_in_x])
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
