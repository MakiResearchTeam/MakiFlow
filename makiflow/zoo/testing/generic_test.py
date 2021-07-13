import unittest

import makiflow as mf


class ModelBuildingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.x = mf.layers.InputLayer([None, 32, 32, 3], name='input')
        # contains a list of model building callables
        self.model_fns: list = None

    def test_build_model(self):
        if self.model_fns is None:
            self.skipTest('model_fns are not provided.')

        for model_fn in self.model_fns:
            with self.subTest(model_fn):
                model = model_fn(self.x)
                print(f'Built {model.name}')
                self.assertIsInstance(model, mf.Model, f'Model builder returned {model} but expected mf.Model')
