from makiflow.zoo.testing import InferenceTest
from makiflow.zoo.backbones.mobilenetv2 import MobileNetV2_0_75, MobileNetV2_1_0, MobileNetV2_1_3, MobileNetV2_1_4


class MobileNetBuildingTest(InferenceTest):
    def setUp(self) -> None:
        super().setUp()
        self.model_fns = [
            lambda x: MobileNetV2_0_75(x, create_model=True),
            lambda x: MobileNetV2_1_0(x, create_model=True),
            lambda x: MobileNetV2_1_3(x, create_model=True),
            lambda x: MobileNetV2_1_4(x, create_model=True),
        ]


