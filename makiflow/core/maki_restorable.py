import json
from abc import abstractmethod, ABC


class MakiRestorable(ABC):
    @abstractmethod
    def to_dict(self):
        pass

    def to_json(self, path):
        params = self.to_dict()

        with open(path, 'w') as f:
            json.dumps(f, params, indent=4)
