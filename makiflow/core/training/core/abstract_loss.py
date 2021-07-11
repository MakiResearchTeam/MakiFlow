import abc
from collections import OrderedDict
import tensorflow as tf

from makiflow.core.training.core.tensor_provider import TensorProvider


class AbstractLoss(object):
    # Used for creation of unique tensor names for lossless label_tensors dictionary merge.
    _id = 1

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        setattr(obj, f'_{__class__.__name__}__id', AbstractLoss._id)
        AbstractLoss._id += 1
        return obj

    def __init__(self, label_tensors):
        # We need to update tensor names to avoid intersection with tensor names in other losses.
        # Have similar names may cause tensor removal which in turn will cause errors.
        upd_label_tensors = {}
        for k, v in label_tensors.items():
            if isinstance(v, tuple):
                print(f"Label tensor {v[0]} won't be passed to the trainer as a data source")
                # Save only the label_tensor in the dictionary
                label_tensors[k] = v[0]
                continue
            upd_label_tensors[self.get_tensor_name(k)] = v
        self.__unique_label_tensors = OrderedDict(upd_label_tensors)
        # The loss construction method will expect label_tensors to have their names as they were.
        self.__label_tensors = label_tensors
        # The loss tensor will be saved in this var
        self._loss = None

    @abc.abstractmethod
    def build(self, tensor_provider: TensorProvider):
        pass

    @property
    def label_tensors(self) -> OrderedDict:
        """
        Returns
        -------
        OrderedDict
            Dictionary of label tensors. Used for loss_fn.
        """
        return self.__label_tensors.copy()

    @property
    def unique_label_tensors(self) -> OrderedDict:
        """
        Returns
        -------
        OrderedDict
            Dictionary of label tensors. Used by ModularLoss and Trainer.
        """
        return self.__unique_label_tensors.copy()

    @property
    def loss(self):
        """
        Returns
        -------
        tf.Tensor
            The built loss scalar tensor. Returns None is the loss hasn't been built yet.
        """
        return self._loss

    # noinspection PyUnresolvedReferences
    def get_tensor_name(self, tensor_name):
        """
        Appends loss class name and the instance id to make the tensor name unique for this
        particular loss instance.

        Parameters
        ----------
        tensor_name : str

        Returns
        -------
        str
            Update tensor names.
        """
        return tensor_name + '_' + self.__class__.__name__ + '_' + str(self.__id)


del TensorProvider
