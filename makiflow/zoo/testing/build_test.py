import unittest

from .generic_test import ModelsTest
import makiflow as mf


class BuildTest(ModelsTest):
    def test_build_model(self):
        if self.model_fns is None:
            self.skipTest('model_fns are not provided.')

        for model_fn in self.model_fns:
            with self.subTest(model_fn):
                model = model_fn(self.x)
                print(f'Built {model.name}')
                self.assertIsInstance(model, mf.Model, f'Model builder returned {model} but expected mf.Model')

