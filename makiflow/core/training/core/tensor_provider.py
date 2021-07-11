from abc import abstractmethod


class TensorProvider:
    @abstractmethod
    def get_traingraph_tensor(self, tensor_name):
        pass
