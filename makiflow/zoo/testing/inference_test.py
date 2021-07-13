import unittest
import tensorflow as tf
import numpy as np

from .generic_test import ModelsTest
import makiflow as mf


class InferenceTest(ModelsTest):
    def setUp(self) -> None:
        super(InferenceTest, self).setUp()
        self.session = tf.Session()

    def tearDown(self) -> None:
        self.session.close()

    def test_predict_model(self):
        # Checks output type and shape
        if self.model_fns is None:
            self.skipTest('model_fns are not provided.')

        x = np.random.randn(1, 32, 32, 3).astype('float32')
        for model_fn in self.model_fns:
            with self.subTest(model_fn):
                model = model_fn(self.x)
                print(f'Test predict {model.name}')
                model.set_session(session=self.session)
                out = model.predict(x)
                self.assertIsInstance(out, np.ndarray, f'Model predict returned {out} but expected np.ndarray')
                self.assertEqual(tuple(out.shape), tuple(model.outputs[0].shape),
                                 'Output shape is inconsistent. Must be '
                                 f'{model.outputs[0].shape} but is '
                                 f'{out.shape}')
