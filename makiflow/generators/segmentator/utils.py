# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf

import numpy as np

DEGREE2RAD = 180.0 / np.pi


def check_bounds(keypoints, image_size):
    """
    Check range of keypoints and return mask for further usage

    Parameters
    ----------
    keypoints : tf.Tensor
        Tensor of the keypoints, where last axis is x and y coordinate
    image_size : list
        List of [H, W] of the image size

    Returns
    -------
    tf.Tensor
        Binary mask, where:
            0 - correspond to non-visible point
            1 - visible point

    """
    image_size = tf.convert_to_tensor(image_size)
    check_min_x = tf.cast(keypoints[..., 1] > 0.0, dtype=tf.float32)
    check_min_y = tf.cast(keypoints[..., 0] > 0.0, dtype=tf.float32)

    check_min_xy = tf.expand_dims(check_min_x * check_min_y, axis=-1)

    check_x = tf.cast(keypoints[..., 1] < tf.cast(image_size[0], dtype=tf.float32), dtype=tf.float32)
    check_y = tf.cast(keypoints[..., 0] < tf.cast(image_size[1], dtype=tf.float32), dtype=tf.float32)

    check_xy = tf.expand_dims(check_x * check_y, axis=-1)

    return check_min_xy * check_xy


def add_z_dim(x):
    """
    Add additional dimension filled in with ones

    """
    return tf.concat(
        [
            x,
            tf.ones_like(tf.expand_dims(tf.convert_to_tensor(x)[..., 0], axis=-1), dtype=tf.float32, optimize=False)
        ],
        axis=-1
    )


def create_shift_matrix_to_center(image):
    """
    Create two matrixes to shift image to center and back to previous location

    Returns
    -------
    tf.Tensor
        Shift to center
    tf.Tensor
        Shift back to previous location

    """
    if isinstance(image, list):
        image = image[0]

    shift_x = image.get_shape()[1].value // 2
    shift_y = image.get_shape()[0].value // 2

    shift_center = get_shift_matrix(-shift_x, -shift_y)

    shift_back = get_shift_matrix(shift_x, shift_y)

    return shift_center, shift_back


def get_rotate_matrix(image, angle):
    """
    Get rotation matrix for image with certain angle

    """
    if isinstance(image, list):
        image = image[0]

    shift_center, shift_back = create_shift_matrix_to_center(image)

    rot_matrix = tf.stack([
        [tf.math.cos(angle), tf.math.sin(angle), 0.0],
        [(-1) * tf.math.sin(angle), tf.math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])

    full_matrix = tf.matmul(tf.matmul(shift_back, rot_matrix), shift_center)

    return full_matrix


def get_rotate_matrix_batched(images, angle_batched):
    """
    Get rotation matrix for every image in the batch with certain angle in the angle_batched array

    """
    if isinstance(images, list):
        images = images[0]

    return tf.stack(
        [get_rotate_matrix(images[i], angle_batched[i]) for i in range(images.get_shape().as_list()[0])]
    )


def get_shift_matrix(dx, dy):
    """
    Get shift matrix with certain dx and dy shifts by certain axis (x and y)

    """

    return tf.stack([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-dx, -dy, 1.0],
    ])


def get_shift_matrix_batched(dx_batched, dy_batched):
    """
    Get batched shift matrix with certain dx and dy in dx_batched and dy_batched array

    """
    assert dx_batched.get_shape().as_list()[0] == dy_batched.get_shape().as_list()[0]

    return tf.stack(
        [get_shift_matrix(dx_batched[i], dy_batched[i]) for i in range(dx_batched.get_shape().as_list()[0])]
    )


def get_zoom_matrix(image, zoom):
    """
    Get zoom matrix with certain scale `zoom`

    """
    shift_center, shift_back = create_shift_matrix_to_center(image)

    zoom_matrix = tf.stack([
        [zoom, 0.0, 0.0],
        [0.0, zoom, 0.0],
        [0.0, 0.0, 1.0]
    ])

    full_matrix = tf.matmul(tf.matmul(shift_back, zoom_matrix), shift_center)
    return full_matrix


def get_zoom_matrix_batched(images, zoom_batched):
    """
    Get batched zoom matrix for every scale in the `zoom_batched` array

    """
    if isinstance(images, list):
        images = images[0]

    return tf.stack(
        [get_zoom_matrix(images[i], zoom_batched[i]) for i in range(images.get_shape().as_list()[0])]
    )


def apply_transformation(
        image,
        use_rotation=False,
        angle=None,
        use_shift=False,
        dx=None,
        dy=None,
        use_zoom=False,
        zoom_scale=None
):
    """
    Apply transformation to an image

    Returns
    -------
    List
        Batch of the transformed image
    tf.Tensor
        Batch of the transformed keypoints
    """
    if isinstance(image, list):
        for i in range(len(image)):
            image[i] = tf.convert_to_tensor(image[i], dtype=tf.float32)
    else:
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    zoom_matrix = None
    shift_matrix = None
    rotation_matrix = None

    if use_zoom and zoom_scale is not None:
        zoom_scale = tf.convert_to_tensor(zoom_scale, dtype=tf.float32)
        zoom_matrix = get_zoom_matrix(image, zoom_scale)

    if use_shift and dx is not None and dy is not None:
        dx = tf.convert_to_tensor(dx, dtype=tf.float32)
        dy = tf.convert_to_tensor(dy, dtype=tf.float32)
        shift_matrix = get_shift_matrix(dx, dy)

    if use_rotation and angle is not None:
        angle = tf.convert_to_tensor(angle, dtype=tf.float32) / DEGREE2RAD
        rotation_matrix = get_rotate_matrix(image, angle)

    full_matrix = tf.ones([3, 3], dtype=tf.float32)
    use_ones = False

    if rotation_matrix is not None:
        full_matrix = tf.multiply(full_matrix, rotation_matrix)
        use_ones = True

    if shift_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, shift_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, shift_matrix)
        use_ones = True

    if zoom_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, zoom_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, zoom_matrix)
        use_ones = True

    proj_matrix = tf.contrib.image.matrices_to_flat_transforms(tf.transpose(full_matrix))

    if isinstance(image, list):
        transformed_image = []
        for i in range(len(image)):
            transformed_image.append(tf.contrib.image.transform(image[i], proj_matrix))
    else:
        transformed_image = tf.contrib.image.transform([image], proj_matrix)[0]

    return transformed_image


def apply_transformation_batched(
        images,
        use_rotation=False,
        angle_batched=None,
        use_shift=False,
        dx_batched=None,
        dy_batched=None,
        use_zoom=False,
        zoom_scale_batched=None
):
    """
    Apply transformation to images in certain batch (i.e. batch size)

    Returns
    -------
    tf.Tensor
        Batch of transformed images
    tf.Tensor
        Batch of transformed keypoints

    """
    if isinstance(images, list):
        for i in range(len(images)):
            images[i] = tf.convert_to_tensor(images[i], dtype=tf.float32)
        N = images[0].get_shape().as_list()[0]
    else:
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        N = images.get_shape().as_list()[0]

    zoom_matrix = None
    shift_matrix = None
    rotation_matrix = None

    if use_zoom and zoom_scale_batched is not None:
        zoom_scale_batched = tf.convert_to_tensor(zoom_scale_batched, dtype=tf.float32)
        zoom_matrix = get_zoom_matrix_batched(images, zoom_scale_batched)

    if use_shift and dx_batched is not None and dy_batched is not None:
        dx_batched = tf.convert_to_tensor(dx_batched, dtype=tf.float32)
        dy_batched = tf.convert_to_tensor(dy_batched, dtype=tf.float32)
        shift_matrix = get_shift_matrix_batched(dx_batched, dy_batched)

    if use_rotation and angle_batched is not None:
        angle_batched = tf.convert_to_tensor(angle_batched, dtype=tf.float32) / DEGREE2RAD
        rotation_matrix = get_rotate_matrix_batched(images, angle_batched)

    full_matrix = tf.ones([N, 3, 3], dtype=tf.float32)
    use_ones = False

    if rotation_matrix is not None:
        full_matrix = tf.multiply(full_matrix, rotation_matrix)
        use_ones = True

    if shift_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, shift_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, shift_matrix)
        use_ones = True

    if zoom_matrix is not None:
        if not use_ones:
            full_matrix = tf.multiply(full_matrix, zoom_matrix)
        else:
            full_matrix = tf.matmul(full_matrix, zoom_matrix)
        use_ones = True

    proj_matrix = tf.contrib.image.matrices_to_flat_transforms(
        [
            tf.transpose(full_matrix[j])
            for j in range(N)
        ]
    )

    if isinstance(images, list):
        transformed_image = []
        for i in range(len(images)):
            transformed_image.append(tf.contrib.image.transform(images[i], proj_matrix))
    else:
        transformed_image = tf.contrib.image.transform(images, proj_matrix)

    return transformed_image


def random_gen(dist):
    """
    Makes a generator that yields numbers in range [0, len(dist)) in accordance with the dist.

    Parameters
    ----------
    dist : np.ndarray
        Distribution.
    Returns
    -------
    generator
    """
    alphas = np.round(dist / dist.min()).astype('int32')
    num_it = np.sum(alphas)
    sample_data = []
    for i in range(len(alphas)):
        sample_data += [i] * int(alphas[i])
    sample_data = np.array(sample_data)
    np.random.shuffle(sample_data)
    it = 0
    while True:
        if it == num_it:
            np.random.shuffle(sample_data)
            it = 0

        yield sample_data[it]
        it += 1
