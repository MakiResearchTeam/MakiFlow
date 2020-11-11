from makiflow.core import MakiTensor
from makiflow.core.debug_utils import d_msg
from .embedding_layer import SkeletonEmbeddingLayer
from .utils import make_box
import numpy as np


class Head:
    def __init__(self, coords: MakiTensor, point_indicators: MakiTensor, human_indicators: MakiTensor):
        """
        An entity that encapsulates all the tensors necessary to make predictions on a particular grid.
        It makes makes sure the shapes are synchronized and also collects necessary info for the trainer.

        Parameters
        ----------
        coords : MakiTensor
            Tensor of the regressed coordinates of the skeleton points. Must lie approximately within
            the [-1, 1] interval.
        point_indicators : MakiTensor
            Tensor of binary indicators of whether a particular point of the skeleton is visible.
        human_indicators : MakiTensor
            Tensor of binary indicators of whether a human is present in a particular location
            of the grid.
        """
        self._coords = coords
        self._point_indicators = point_indicators
        self._human_indicators = human_indicators

        self._context = f'SSP Head({coords.get_name()}, {point_indicators.get_name()}, {human_indicators.get_name()})'
        self.__check_dimensionality()
        self.__collect_info()

    def __check_dimensionality(self):
        # All have dimensions [b, h, w, c]
        coords_shape = self._coords.get_shape()
        point_indicators_shape = self._point_indicators.get_shape()
        human_indicators_shape = self._human_indicators.get_shape()

        # Only convolutional networks are supported
        assert len(coords_shape) == 4 and \
            len(point_indicators_shape) == 4 and \
            len(human_indicators_shape) == 4, d_msg(
            self._context,
            'Dimensionality of all tensors must be 4, received '
            f'dim(coords)={len(coords_shape)}, '
            f'dim(point_indicators)={len(point_indicators_shape)}, '
            f'dim(human_indicators)={len(human_indicators_shape)}'
        )

        # Check spatial shape (h, w)
        assert coords_shape[1:-1] == point_indicators_shape[1:-1] and \
            coords_shape[1:-1] == human_indicators_shape[1:-1] and \
            point_indicators_shape[1:-1] == human_indicators_shape[1:-1], d_msg(
            self._context,
            'Spatial shapes are not aligned. Received '
            f'coords_shape={coords_shape}, '
            f'point_indicators_shape={point_indicators_shape}, '
            f'human_indicators_shape={human_indicators_shape}'
        )

        # Check alignment of the number of points between coords and point indicators
        n_coords = coords_shape[-1]
        assert n_coords % 2 == 0, d_msg(
            self._context,
            f'coords must have an even number of channel, received {n_coords}.'
        )

        n_points = n_coords // 2
        assert n_points == point_indicators_shape[-1], d_msg(
            self._context,
            f'Number of points in coords and point_indicators must be the same, '
            f'received {n_points} and {point_indicators_shape[-1]}.'
        )

        # Check whether human_indicators has a single channel
        assert human_indicators_shape[-1] == 1, d_msg(
            self._context,
            f'human_indicators tensor must have 1 channel, received {human_indicators_shape[-1]}.'
        )

    def __collect_info(self):
        self._embedding_layer = self._coords.get_parent_layer()
        assert isinstance(self._embedding_layer, SkeletonEmbeddingLayer), d_msg(
            self._context,
            "coords tensor's parent layer must be of type SkeletonEmbeddingLayer, "
            f"received parent layer type={type(self._embedding_layer)}, "
            f"parent layer name={self._embedding_layer.get_name()}"
        )
        # The method returns a list. Convert it to ndarray
        self._source_embedding = np.array(self._embedding_layer.get_embedding())

        # Make a check of the absolute values of the embedding. They should be normalized and approximately within
        # [-1, 1] interval.
        if np.max(np.abs(self._source_embedding)) > 1.4:
            print(d_msg(
                self._context,
                "It seems the embedding's values are not normalized. This is not an error, but the values should "
                "be centered and lie approximately within the [-1, 1] interval. Received embedding with "
                f"maximum absolute value of {np.max(np.abs(self._source_embedding))}"
            ))
        self._embedding_bounding_box = make_box(self._source_embedding)
        width = self._embedding_bounding_box[2] - self._embedding_bounding_box[0]
        height = self._embedding_bounding_box[3] - self._embedding_bounding_box[1]
        # Determine how much this bounding box differs from the default one.
        # Default box has the following coordinates:
        # - top left point = [-1, -1]
        # - bottom right point = [1, 1]
        self._bbox_configuration = (width / 2.0, height / 2.0)

    def get_bbox(self):
        return self._embedding_bounding_box

    def get_bbox_configuration(self):
        return self._bbox_configuration

    def get_coords(self):
        return self._coords

    def get_point_indicators(self):
        return self._point_indicators

    def get_human_indicators(self):
        return self._human_indicators


# For debug
if __name__ == '__main__':
    from makiflow.layers import InputLayer
    batch_size = 1
    n_points = 10
    offsets = InputLayer(input_shape=[batch_size, 3, 3, n_points * 2], name='offsets')
    coords = SkeletonEmbeddingLayer(embedding_dim=n_points, name='SkeletonEmbedding')(offsets)
    point_indicators = InputLayer(input_shape=[batch_size, 3, 3, n_points], name='point_indicators')
    human_indicators = InputLayer(input_shape=[batch_size, 3, 3, 1], name='human_indicators')

    head = Head(coords, point_indicators, human_indicators)
    print('Bbox configuration:', head.get_bbox_configuration())
    print('Bbox coordinates:', head.get_bbox())

    from makiflow.core.debug_utils import DebugContext

    with DebugContext('Spatial shape checking.'):
        offsets = InputLayer(input_shape=[batch_size, 2, 3, n_points * 2], name='offsets')
        coords = SkeletonEmbeddingLayer(embedding_dim=n_points, name='SkeletonEmbedding')(offsets)
        Head(coords, point_indicators, human_indicators)

    with DebugContext('Points number checking.'):
        offsets = InputLayer(input_shape=[batch_size, 3, 3, (n_points - 1) * 2], name='offsets')
        coords = SkeletonEmbeddingLayer(embedding_dim=n_points, name='SkeletonEmbedding')(offsets)
        Head(coords, point_indicators, human_indicators)

    with DebugContext('Single channeled human indicators checking.'):
        offsets = InputLayer(input_shape=[batch_size, 3, 3, n_points * 2], name='offsets')
        coords = SkeletonEmbeddingLayer(embedding_dim=n_points, name='SkeletonEmbedding')(offsets)
        human_indicators = InputLayer(input_shape=[batch_size, 3, 3, 2], name='human_indicators')
        Head(coords, point_indicators, human_indicators)

    with DebugContext('Coords parent_layer=SkeletonEmbeddingLayer checking.'):
        coords = InputLayer(input_shape=[batch_size, 3, 3, n_points * 2], name='offsets')
        human_indicators = InputLayer(input_shape=[batch_size, 3, 3, 1], name='human_indicators')
        Head(coords, point_indicators, human_indicators)

    with DebugContext('Unnormalized embedding checking. Using gaussian with std=100 as an embedding.'):
        offsets = InputLayer(input_shape=[batch_size, 3, 3, n_points * 2], name='offsets')
        coords = SkeletonEmbeddingLayer(
            embedding_dim=n_points, name='SkeletonEmbedding', custom_embedding=np.random.randn(10, 2) * 100)(offsets)
        Head(coords, point_indicators, human_indicators)
