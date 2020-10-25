from .hephaestus import Hephaestus


class Aion(Hephaestus):
    """
    Aion is the of eternity. Serializing the trainer does not allow it to vanish completely,
    therefore, making it eternal.
    """
    TYPE = 'type'
    PARAMS = 'params'

    def to_dict(self):
        assert self.TYPE != Aion.TYPE, 'The trainer did not specified its type via static TYPE variable.'
        return {
            Aion.TYPE: self.TYPE,
            Aion.PARAMS: {}
        }

    def set_params(self, params):
        """
        This method must be overloaded if the trainer uses some additional
        parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameters for the trainer.
        """
        pass



