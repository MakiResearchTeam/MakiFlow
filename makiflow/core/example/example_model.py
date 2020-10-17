from makiflow.core import MakiModel, MakiTensor


class ExampleModel(MakiModel):
    def _get_model_info(self):
        return {
            'param': 'value'
        }

    def __init__(self, in_x: MakiTensor, out_x: MakiTensor, name='ExampleModel'):
        super().__init__(
            outputs=[out_x],
            inputs=[in_x]
        )
        self._name = name
        self._out_x = out_x
        self._in_x = in_x

    def predict(self, data):
        return self.get_session().run(
            self._out_x.get_data_tensor(),
            feed_dict={
                self._in_x.get_data_tensor(): data
            }
        )

    def get_output(self):
        return self._out_x
