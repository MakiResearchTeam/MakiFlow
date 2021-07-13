from makiflow.zoo.testing import ModelBuildingTestCase
from makiflow.zoo.backbones.resnetv1 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
    Little_ResNet20, Little_ResNet110, Little_ResNet32, Little_ResNet44, Little_ResNet56


class DenseNetBuildingTest(ModelBuildingTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: ResNet18(x, create_model=True),
            lambda x: ResNet34(x, create_model=True),
            lambda x: ResNet50(x, create_model=True),
            lambda x: ResNet101(x, create_model=True),
            lambda x: ResNet152(x, create_model=True),
            lambda x: Little_ResNet20(x, create_model=True),
            lambda x: Little_ResNet110(x, create_model=True),
            lambda x: Little_ResNet32(x, create_model=True),
            lambda x: Little_ResNet44(x, create_model=True),
            lambda x: Little_ResNet56(x, create_model=True),
        ]


