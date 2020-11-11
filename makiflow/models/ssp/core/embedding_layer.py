from makiflow.core import MakiLayer
from makiflow.core.debug_utils import d_msg
import tensorflow as tf
import numpy as np


class SkeletonEmbeddingLayer(MakiLayer):
    def __init__(self, embedding_dim: int, name: str, custom_embedding: list = None):
        """
        Creates a grid of default skeletons. These skeletons are then trained using
        gradient descent.
        The grids values are in the [-1, 1] values.

        Parameters
        ----------
        embedding_dim : int
            How many points are in the skeleton.
        name : str
            Name of the layer.
        custom_embedding : list of shape [n_points, 2]
            List containing custom skeleton embedding. It must be noted, that the embedding's values must be centered
            and normalized within [-1, 1] interval (or approximately so, you can use larger ones for the purpose
            of more dense coverage of the grid), because it will be put into a grid with values within [-1, 1] interval.
        """
        if not isinstance(embedding_dim, int):
            assert custom_embedding is not None, d_msg(
                name, 'embedding_dim is not of the type int. In this case the custom_embedding is expected to be '
                      'provided, but the custom_embedding=None.'
            )
        else:
            assert embedding_dim > 0, d_msg(
                name, f'embedding_dim must be positive. Received embedding_dim={embedding_dim}'
            )

        if custom_embedding is not None:
            assert len(custom_embedding) > 0, d_msg(
                name, f'custom_embedding is empty. Received custom_embedding with '
                f'len={len(custom_embedding)}'
            )
            assert len(custom_embedding[0]) == 2, d_msg(
                name, f"custom_embedding's points are not 2-dimensional. "
                f"Received custom_embedding with {len(custom_embedding[0])}-dimensional points."
            )

            if not isinstance(custom_embedding, list):
                print(d_msg(
                    name, f'custom_embedding is not a list. Received custom_embedding of '
                    f'type={type(custom_embedding)}.')
                )
                print(d_msg(
                    name, 'Iterating over the custom_embedding to convert it to a list.')
                )
                custom_embedding = self.__embed2list(custom_embedding)
        self._embedding_dim = embedding_dim
        self._custom_embedding = custom_embedding
        if custom_embedding is None:
            print(d_msg(name, 'No custom embedding is provided. Creating a random one.'))
            self._custom_embedding = np.random.uniform(low=-1.0, high=1.0, size=[embedding_dim, 2]).tolist()

        embedding = np.array(self._custom_embedding)
        with tf.name_scope(name):
            self._embedding = tf.Variable(embedding, dtype='float32', name='SkeletonEmbedding')

        super().__init__(
            name=name,
            params=[self._embedding],
            regularize_params=[],
            named_params_dict={self._embedding.name: self._embedding}
        )

    def __embed2list(self, custom_embedding):
        list_embedding = []
        for point in custom_embedding:
            list_point = []
            for coord in point:
                list_point.append(coord)
            list_embedding.append(list_point)
        return list_embedding

    def forward(self, x, computation_mode=MakiLayer.INFERENCE_MODE):
        # Do not add the name_scope since in future it won't be used anyway
        _, h, w, c = x.get_shape().as_list()
        assert c == self._embedding_dim * 2, d_msg(
            self.get_name(),
            'The depth of the input tensor must twice as large as the embedding dimensionality. '
            f'Received input tensor channels={c}, embedding dimensionality*2={self._embedding_dim * 2}'
        )
        offsets = x

        grid = SkeletonEmbeddingLayer.generate_grid_stacked((w, h), self._embedding)
        with tf.name_scope('GridCorrection'):
            corrected_grid = grid + offsets

        return corrected_grid

    @staticmethod
    def generate_grid(size):
        """
        Generates grid with values within [-1, 1] interval with the given size.

        Parameters
        ----------
        size : list or tuple
            Contains width and height of the grid.

        Returns
        -------
        ndarray of shape [h, w, 2]
        """
        w, h = size
        delta_x = 1 / w
        x = np.linspace(start=-1 + delta_x, stop=1 - delta_x, num=w, dtype='float32')

        delta_y = 1 / h
        y = np.linspace(start=-1 + delta_y, stop=1 - delta_y, num=h, dtype='float32')
        return np.stack(np.meshgrid(x, y), axis=-1)

    @staticmethod
    def generate_grid_stacked(size, embedding):
        w, h = size
        # grid - [h, w, 2]
        grid = SkeletonEmbeddingLayer.generate_grid(size)
        # stacked_level - [depth, h, w, 2]
        depth = embedding.shape[0]
        stacked_grid = np.stack([grid] * depth)
        # stacked_grid - [h, w, depth, 2]
        stacked_grid = stacked_grid.transpose([1, 2, 0, 3])

        # embedding - [1, 1, n, 2]
        embedding = tf.broadcast_to(embedding, stacked_grid.shape)
        # Normalize the embedding
        embedding = embedding / np.array([w, h])
        skeleton_grid = stacked_grid + embedding
        skeleton_grid = tf.reshape(skeleton_grid, shape=[h, w, -1])
        return skeleton_grid

    def training_forward(self, x):
        return self.forward(x, MakiLayer.TRAINING_MODE)

    @staticmethod
    def build(params: dict):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()

    def get_embedding(self):
        return self._custom_embedding


# For debug
if __name__ == '__main__':
    # Generate points around a circle
    phi = np.linspace(0, 2 * np.pi, num=100)
    x = np.cos(phi) * 0.7 + [0]
    y = np.sin(phi) * 0.7 + [0]
    points = np.stack([x, y], axis=-1)

    from makiflow.layers import InputLayer

    # RUN A SANITY CHECK FIRST
    in_x = InputLayer(input_shape=[1, 3, 3, 100 * 2], name='offsets')
    # Never pass in a numpy array to the `custom_embedding` argument. Always use list.
    coords_ish = SkeletonEmbeddingLayer(embedding_dim=None, name='TestEmbedding', custom_embedding=points)(in_x)

    print('Coords MakiTensor', coords_ish)
    print('Coords TfTensor', coords_ish.get_data_tensor())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coords = sess.run(
        coords_ish.get_data_tensor(),
        feed_dict={
            in_x.get_data_tensor(): np.zeros(shape=[1, 3, 3, 200], dtype='float32')
        }
    )

    # Visualize the circles
    import matplotlib

    # For some reason matplotlib doesn't want to show the plot when it is called from PyCharm
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    coords = coords.reshape(-1, 2)
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()

    from makiflow.core.debug_utils import DebugContext

    # Check if wrong `embedding_dim` was passed
    print('\nChecking embedding_dim asserts...........................................................................')
    with DebugContext('embedding_dim=None'):
        SkeletonEmbeddingLayer(embedding_dim=None, name='TestEmbedding', custom_embedding=None)(in_x)

    with DebugContext('embedding_dim is not positive'):
        SkeletonEmbeddingLayer(embedding_dim=0, name='TestEmbedding', custom_embedding=None)(in_x)

    print('\nChecking custom_embedding asserts........................................................................')
    with DebugContext('custom_embedding is an empty list'):
        SkeletonEmbeddingLayer(embedding_dim=None, name='TestEmbedding', custom_embedding=[])(in_x)

    with DebugContext("custom_embedding's points are not 2-dimensional"):
        SkeletonEmbeddingLayer(embedding_dim=None, name='TestEmbedding', custom_embedding=[[1]])(in_x)

    print('\nChecking randomizing the embedding. A message must be printed.')
    SkeletonEmbeddingLayer(embedding_dim=1, name='TestEmbedding', custom_embedding=None)
