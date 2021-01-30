from abc import ABC, abstractmethod
from collections import OrderedDict
from ...debug import ExceptionScope
from .core import TensorProvider, LossInterface
from .loss_ops import MulOp, AddOp, DivOp
from ...dev import ClassDecorator


# There is no way of using inheritance because it comes down
# to a situation when the Operation has to have overloaded __mul__, etc.
# that used the Operation before it was actually defined.
# Therefore, decorator pattern is used to overcome this issue.

# noinspection PyShadowingNames
class MutableLoss(ClassDecorator, ABC):
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            loss2 = ConstantLoss(other)
            return self(MulOp(self.get_obj(), loss2))
        elif isinstance(other, MutableLoss):
            return self(MulOp(self.get_obj(), other.get_obj()))
        else:
            raise ValueError(f'Expected type LossInterface, but received {type(other)}')

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            loss2 = ConstantLoss(other)
            return self(AddOp(self.get_obj(), loss2))
        elif isinstance(other, MutableLoss):
            return self(AddOp(self.get_obj(), other.get_obj()))
        else:
            raise ValueError(f'Expected type LossInterface, but received {type(other)}')

    def __div__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            loss2 = ConstantLoss(other)
            return self(DivOp(self.get_obj(), loss2))
        elif isinstance(other, MutableLoss):
            return self(DivOp(self.get_obj(), other.get_obj()))
        else:
            raise ValueError(f'Expected type LossInterface, but received {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            loss2 = ConstantLoss(other)
            return self(DivOp(self.get_obj(), loss2))
        elif isinstance(other, MutableLoss):
            return self(DivOp(self.get_obj(), other.get_obj()))
        else:
            raise ValueError(f'Expected type LossInterface, but received {type(other)}')


class ConstantLoss(LossInterface):
    def __init__(self, const):
        self._const = float(const)

    def build(self, tensor_provider: TensorProvider):
        return self._const

    def get_label_tensors(self) -> OrderedDict:
        return OrderedDict()


class Loss(LossInterface, ABC):
    __instance_id = 0

    def __new__(cls, *args, **kwargs):
        loss = object.__new__(cls)
        loss.__init__(*args, **kwargs)
        decorator = MutableLoss()
        Loss.__instance_id += 1
        return decorator(loss)

    def __init__(self, tensor_names, label_tensors: dict):
        assert len(tensor_names) != 0
        self._id = Loss.__instance_id
        self._tensor_names = tensor_names
        self._label_tensors = label_tensors

    def build(self, tensor_provider: TensorProvider):
        loss = 0.0
        with ExceptionScope(self.__class__.__name__):
            for tensor_name in self._tensor_names:
                tensor = tensor_provider.get_traingraph_tensor(tensor_name)
                loss = loss + self.build_loss(tensor, self._label_tensors)
        return loss

    @abstractmethod
    def build_loss(self, prediction, label_tensors):
        pass

    def get_label_tensors(self) -> OrderedDict:
        # This guaranties that label tensor names won't overlap during
        # tensor gathering inside the trainer.
        modified_label_tensors = OrderedDict()
        for tensor_name, tensor in self._label_tensors.items():
            tensor_name = f'{self._id}_{tensor_name}'
            modified_label_tensors[tensor_name] = tensor
        return modified_label_tensors


if __name__ == '__main__':
    loss1 = Loss(None, {})
    loss2 = Loss(None, {})
    print((loss1 + loss2) * loss1 * 2 + 1)