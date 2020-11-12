import numpy as np
from ..utils import run, generate_level_stacked, make_boxes


def train_data(keypoints, image_info, n_levels, iou_th=0.25, debug=False, embeddings=None):
    # Prepare level tensors
    n_points = keypoints[0][0].shape[0]
    if embeddings is None:
        embedding = np.random.uniform(low=-1.0, high=1.0, size=(n_points, 2))  # *np.array([0.6, 1.2])
        embeddings = [embedding, embedding * np.sqrt(2)]
    levels = {}
    for i in range(n_levels):
        level_size = 2 ** i
        levels_to_concat = []
        for embedding in embeddings:
            level = generate_level_stacked((level_size, level_size), embedding)
            level_flat = level.reshape(level_size ** 2, -1, 2)
            levels_to_concat += [level_flat]
        levels[level_size] = np.concatenate(levels_to_concat)

    # Start preparing training data
    training_tensors = []
    for i, (keypoints_pack, im_size) in enumerate(zip(keypoints, image_info)):
        h, w = im_size
        hw = np.array([w / 2, h / 2])
        keypoints_pack = np.array(keypoints_pack)
        keypoints_xy_only = keypoints_pack[..., :2]
        keypoints_indicators = keypoints_pack[..., 2:]
        levels_pack = {}
        human_counter = 0
        for level_size, level_flat in levels.items():
            level_flat = level_flat * hw + hw
            level_boxes = make_boxes(level_flat)
            p_boxes = make_boxes(keypoints_pack)
            ious = run(p_boxes, level_boxes)

            # Zero out ious < `iou_th`
            ious = np.where(ious > iou_th, ious, 0.0)
            # Pick indeces of the data points which values will be assigned to the level vectors
            indeces = np.argmax(ious, axis=0)

            level_flat = level_flat.copy()
            if debug:
                print('indeces: ', indeces)
                print('indexing: ', keypoints_pack[indeces].shape)
                print('level_size=', level_size)
            level_flat[np.arange(level_size ** 2 * len(embeddings))] = keypoints_xy_only[indeces]

            # Create the human presense indicator
            non_zero_ious = np.max(ious, axis=0)
            human_presense = np.where(non_zero_ious > 0.0, 1, 0)
            levels_pack[level_size] = (level_flat, human_presense, keypoints_indicators[indeces])
            human_counter += human_presense.sum()

        training_tensors += [levels_pack]
        if human_counter == 0:
            pass
            # print(human_counter, 'humans present in image', i)
    return training_tensors


