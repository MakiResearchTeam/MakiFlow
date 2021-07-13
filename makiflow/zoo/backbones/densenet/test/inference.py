from makiflow.zoo.testing import InferenceTest
from makiflow.zoo.backbones.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264


class DenseNetBuildingTest(InferenceTest):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: DenseNet121(x, create_model=True),
            lambda x: DenseNet161(x, create_model=True),
            lambda x: DenseNet169(x, create_model=True),
            lambda x: DenseNet201(x, create_model=True),
            lambda x: DenseNet264(x, create_model=True)
        ]


