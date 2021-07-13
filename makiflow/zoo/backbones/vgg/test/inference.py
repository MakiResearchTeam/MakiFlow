from makiflow.zoo.testing import InferenceTest
from makiflow.zoo.backbones.vgg import VGG16, VGG19


class DenseNetBuildingTest(InferenceTest):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: VGG16(x, create_model=True),
            lambda x: VGG19(x, create_model=True),
        ]


