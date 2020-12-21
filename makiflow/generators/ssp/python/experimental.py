from ..utils import generate_grid, make_boxes, run
import numpy as np


class ExperimentalGenerator:
    def __init__(self, image_size, embedding_dim, grid_sizes, bbox_configs, image_flow, keypoints_flow,
                 keypoints_mask_flow, iou_th=0.2):
        self._image_size = image_size
        self._embedding_dim = embedding_dim
        self._grid_sizes = grid_sizes
        self._bbox_configs = bbox_configs
        self._image_flow = image_flow
        self._keypoints_flow = keypoints_flow
        self._keypoints_mask_flow = keypoints_mask_flow
        self._iou_th = iou_th
        self._sess = None
        self._init()

    def _init(self):
        # Generate grids
        self._grids = {}
        for grid_size in self._grid_sizes:
            h, w = grid_size
            self._grids[grid_size] = generate_grid(w, h, self._embedding_dim)

        # Generate embeddings with certain bbox configurations
        self._embeddings = {}
        for bbox_config in self._bbox_configs:
            embedding = np.zeros((self._embedding_dim, 2), dtype='float32')
            embedding[0] = np.array([-1, -1])
            embedding[1] = np.array([1, 1])
            scale = np.array(bbox_config)
            embedding = embedding * scale
            self._embeddings[bbox_config] = embedding

        w, h = self._image_size
        self._image_scale = np.array([w / 2, h / 2], dtype='float32')

    def get_embedding(self, bbox_config):
        return self._embeddings[bbox_config]

    def get_grid(self, grid_size):
        return self._grids[grid_size]

    def set_session(self, session):
        self._sess = session

    def get_iterator(self, debug=False):
        assert self._sess is not None, 'Session is not set.'
        while True:
            # Load a batch of training data
            image, keypoints, keypoints_mask = self.run_flow()
            # [b, n_points, n_people, 2] - > [b, n_people, n_points, 2]
            keypoints = keypoints.transpose([0, 2, 1, 3])
            keypoints_mask = keypoints_mask.transpose([0, 2, 1, 3])
            keypoints = keypoints * keypoints_mask

            labels = []
            for grid_size, bbox_config in zip(self._grid_sizes, self._bbox_configs):
                coordinates_list, keypoint_indicators_list, human_indicators_list = [], [], []
                for keypoints_example, keypoints_mask_example in zip(keypoints, keypoints_mask):
                    coordinates, keypoint_indicators, human_indicators = self.generate_training_data(
                        grid_size, bbox_config, keypoints_example, keypoints_mask_example
                    )
                    coordinates_list.append(coordinates)
                    keypoint_indicators_list.append(keypoint_indicators)
                    human_indicators_list.append(human_indicators)

                coordinates = np.stack(coordinates_list)
                keypoint_indicators = np.stack(keypoint_indicators_list)
                human_indicators = np.stack(human_indicators_list)
                labels += [coordinates, keypoint_indicators, human_indicators]

            if debug:
                yield (image,), tuple(labels), keypoints, keypoints_mask

            yield (image,), tuple(labels)

    def run_flow(self):
        assert self._sess is not None, 'Session is not set.'
        return self._sess.run(
            [self._image_flow, self._keypoints_flow, self._keypoints_mask_flow]
        )

    def generate_training_data(self, grid_size, bbox_config, keypoints_example, keypoints_indicators):
        grid = self.get_grid(grid_size)
        embedding = self.get_embedding(bbox_config)
        h, w, d, _ = grid.shape
        grid = grid + embedding / np.array([w, h])
        grid_flat = grid.reshape(w * h, -1, 2)

        keypoints_example = keypoints_example
        keypoints_marked = np.concatenate([keypoints_example, keypoints_indicators], axis=-1)

        grid_flat = grid_flat * self._image_scale + self._image_scale
        level_boxes = make_boxes(grid_flat)
        p_boxes = make_boxes(keypoints_marked)
        ious = run(p_boxes, level_boxes)

        # Zero out ious < `iou_th`
        ious = np.where(ious > self._iou_th, ious, 0.0)
        # Pick indeces of the data points which values will be assigned to the level vectors
        indeces = np.argmax(ious, axis=0)

        grid_flat = grid_flat.copy()

        grid_flat[np.arange(grid_size[0] * grid_size[1])] = keypoints_example[indeces]

        # Create the human presense indicator
        non_zero_ious = np.max(ious, axis=0)
        human_presence = np.where(non_zero_ious > 0.0, 1, 0)

        grid = grid_flat.reshape(h, w, d * 2)
        keypoints_indicators = keypoints_indicators[indeces].reshape(h, w, d)
        human_presence = human_presence.reshape(h, w, 1)

        return grid, keypoints_indicators, human_presence
