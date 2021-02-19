# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Code is taken from
# https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/imagenet_utils.py
#
# =====================================================================

"""Utilities for ImageNet data preprocessing & prediction decoding."""

import numpy as np


PREPROCESS_INPUT_DOC = """
  Preprocesses a tensor or Numpy array encoding a batch of images.

  Usage example with `applications.MobileNet`:

  ```python
  i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
  x = tf.cast(i, tf.float32)
  x = tf.keras.applications.mobilenet.preprocess_input(x)
  core = tf.keras.applications.MobileNet()
  x = core(x)
  model = tf.keras.Model(inputs=[i], outputs=[x])

  image = tf.image.decode_png(tf.io.read_file('file.png'))
  result = model(image)
  ```

  Arguments:
    x: A floating point `numpy.array` or a `tf.Tensor`, 3D or 4D with 3 color
      channels, with values in the range [0, 255].
      The preprocessed data are written over the input data
      if the data types are compatible. To avoid this
      behaviour, `numpy.copy(x)` can be used.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it,
      it defaults to "channels_last").{mode}

  Returns:
      Preprocessed `numpy.array` or a `tf.Tensor` with type `float32`.
      {ret}

  Raises:
      {error}
  """

PREPROCESS_INPUT_MODE_DOC = """
    mode: One of "caffe", "tf" or "torch". Defaults to "caffe".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.
  """

PREPROCESS_INPUT_DEFAULT_ERROR_DOC = """
    ValueError: In case of unknown `mode` or `data_format` argument."""

CHANNELS_LAST = 'channels_last'
CHANNELS_FIRST = 'channels_first'

CAFFE = 'caffe'
TF = 'tf'
TORCH = 'torch'


def preprocess_input(x, data_format=CHANNELS_LAST, mode=CAFFE, use_rgb2bgr=False):
    """
    Preprocesses a tensor or Numpy array encoding a batch of images.

    """
    if mode not in {CAFFE, TF, TORCH}:
        raise ValueError('Unknown mode ' + str(mode))

    if data_format not in {CHANNELS_FIRST, CHANNELS_LAST}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if isinstance(x, np.ndarray):
        return preprocess_numpy_input(
            x, data_format=data_format, mode=mode, use_rgb2bgr=use_rgb2bgr)
    else:
        return preprocess_symbolic_input(
            x, mode=mode, use_rgb2bgr=use_rgb2bgr)


preprocess_input.__doc__ = PREPROCESS_INPUT_DOC.format(
    mode=PREPROCESS_INPUT_MODE_DOC,
    ret='',
    error=PREPROCESS_INPUT_DEFAULT_ERROR_DOC)


def preprocess_numpy_input(x, mode, data_format=CHANNELS_LAST, dtype=np.float32, use_rgb2bgr=False):
    """Preprocesses a Numpy array encoding a batch of images.

    Arguments:
    x: Input array, 3D or 4D.
    data_format: Data format of the image array.
    mode: One of "caffe", "tf" or "torch".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.
    dtype : np.dtype
        Type of the returned and input array
    use_rgb2bgr : bool
        Convert input image to bgr,
        By default equal to None, i.e. will be not used

    Returns:
      Preprocessed Numpy array.
    """
    x = x.astype(dtype, copy=False)

    if mode == TF:
        x /= 127.5
        x -= 1.
        return x
    elif mode == TORCH:
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        # Vgg like normalization
        if use_rgb2bgr:
            if data_format == CHANNELS_FIRST:
                # 'RGB'->'BGR'
                if x.ndim == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
        #         B        G        R
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == CHANNELS_FIRST:
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x.astype(dtype, copy=False)


def preprocess_symbolic_input(x, mode, use_rgb2bgr=False):
    """Preprocesses a tensor encoding a batch of images.

    Arguments:
    x: Input tensor, 3D or 4D.
    mode: One of "caffe", "tf" or "torch".
      - caffe: will convert the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset,
          without scaling.
      - tf: will scale pixels between -1 and 1,
          sample-wise.
      - torch: will scale pixels between 0 and 1 and then
          will normalize each channel with respect to the
          ImageNet dataset.
    use_rgb2bgr : bool
        Convert input image to bgr,
        By default equal to None, i.e. will be not used

    Returns:
      Preprocessed tensor.
    """
    if mode == TF:
        x /= 127.5
        x -= 1.
        return x
    elif mode == TORCH:
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if use_rgb2bgr:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    x = x - mean
    if std is not None:
        x /= std
    return x
