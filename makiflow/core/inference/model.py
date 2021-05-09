from tqdm import tqdm
import numpy as np

from .model_serializer import ModelSerializer
from .. import MakiTensor
from .maki_builder import MakiBuilder
from makiflow.core.training.trainer.utils import pack_data
from ...generators import data_iterator


class Model(ModelSerializer):
    INPUTS = 'inputs'
    OUTPUTS = 'outputs'
    NAME = 'name'

    @staticmethod
    def from_json(path: str, input_tensor: MakiTensor = None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        model_info, graph_info = ModelSerializer.load_architecture(path)

        output_names = model_info[Model.OUTPUTS]
        input_names = model_info[Model.INPUTS]
        model_name = model_info[Model.NAME]

        inputs_outputs = MakiBuilder.restore_graph(output_names, graph_info)
        out_x = [inputs_outputs[name] for name in output_names]
        in_x = [inputs_outputs[name] for name in input_names]
        print('Model is restored!')
        return Model(inputs=in_x, outputs=out_x, name=model_name)

    def __init__(self, inputs, outputs, name):
        self.name = name
        super().__init__(outputs, inputs)

    def _get_model_info(self):
        return {
            Model.INPUTS: [in_x.name for in_x in super().get_inputs()],
            Model.OUTPUTS: [out_x.name for out_x in super().get_outputs()],
            Model.NAME: self.name
        }

    def get_batch_size(self):
        return self.get_inputs()[0].shape[0]

    def get_feed_dict_config(self) -> dict:
        feed_dict_config = {}
        for i, x in enumerate(super().get_inputs()):
            feed_dict_config[x] = i

        return feed_dict_config

    def predict(self, *args):
        """
        Performs prediction on the given data.

        Parameters
        ----------
        *args : arrays
            Data order must be the same as the model's inputs.

        Returns
        -------
        list
            Predictions.
        """
        feed_dict_config = self.get_feed_dict_config()
        batch_size = self.get_batch_size() if self.get_batch_size() is not None else 1
        predictions = []
        for data in tqdm(data_iterator(*args, batch_size=batch_size)):
            packed_data = pack_data(feed_dict_config, data)
            predictions += [
                self._session.run(
                    [out_x.tensor for out_x in super().get_outputs()],
                    feed_dict=packed_data)
            ]
        # Group data by the model's inputs
        new_pred = []
        for i in range(len(super().get_outputs())):
            single_preds = []
            for output in predictions:
                # grab i-th data
                single_preds.append(output[i])
            new_pred.append(np.concatenate(single_preds, axis=0)[:len(args[0])])

        return new_pred
