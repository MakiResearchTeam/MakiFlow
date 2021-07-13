import unittest
import tensorflow as tf
import numpy as np

import makiflow as mf


class ModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = mf.layers.InputLayer([1, 32, 32, 3], name='input')
        # contains a list of model building callables
        self.model_fns: list = None
