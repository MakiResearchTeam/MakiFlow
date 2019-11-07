import tensorflow as tf
from makiflow.models.segmentation.gen_base import PostMapMethod, MapMethod, PathGenerator, SegmentIterator


class LoadResizeNormalize(MapMethod):
    def __init__(
            self, image_shape, mask_shape,
            normalize=255., image_size=None, mask_size=None,
            squeeze_mask=False, calc_positives=False):
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.image_size = image_size
        self.mask_size = mask_size
        self.squeeze_mask = squeeze_mask
        self.normalize = tf.constant(normalize, dtype=tf.float32)
        self.calc_positives = calc_positives

    def load_data(self, data_paths):
        img_file = tf.read_file(data_paths[PathGenerator.image])
        mask_file = tf.read_file(data_paths[PathGenerator.mask])

        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)

        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)

        if self.squeeze_mask:
            mask = mask[:, :, 0]

        if self.image_size is not None:
            img = tf.image.resize(images=img, size=self.image_size, method=tf.image.ResizeMethod.BILINEAR)

        if self.mask_size is not None:
            mask = tf.image.resize(images=mask, size=self.mask_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = tf.cast(img, dtype=tf.float32)

        if self.normalize is not None:
            img = tf.divide(img, self.normalize)

        mask = tf.cast(mask, dtype=tf.int32)
        return {
            SegmentIterator.image: img,
            SegmentIterator.mask: mask
        }


class LoadDataMethod(MapMethod):
    def __init__(self, image_shape, mask_shape):
        """

        Parameters
        ----------
        image_shape : list
            [image width, image height].
        mask_shape : list
            [mask width, mask height].
        """
        self.image_shape = image_shape
        self.mask_shape = mask_shape

    def load_data(self, data_paths):
        img_file = tf.read_file(data_paths[SegmentIterator.image])
        mask_file = tf.read_file(data_paths[SegmentIterator.mask])

        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)

        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)

        img = tf.cast(img, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.int32)
        return {
            SegmentIterator.image: img,
            SegmentIterator.mask: mask
        }


class ResizePostMethod(PostMapMethod):
    def __init__(self, image_size=None, mask_size=None, image_resize_method=tf.image.ResizeMethod.BILINEAR,
                 mask_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        super().__init__()
        self.image_size = image_size
        self.mask_size = mask_size
        self.image_resize_method = image_resize_method
        self.mask_resize_method = mask_resize_method

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[SegmentIterator.image]
        mask = element[SegmentIterator.mask]

        if self.image_size is not None:
            img = tf.image.resize(images=img, size=self.image_size, method=self.image_resize_method)

        if self.mask_size is not None:
            mask = tf.image.resize(images=mask, size=self.mask_size, method=self.mask_resize_method)

        return {
            SegmentIterator.image: img,
            SegmentIterator.mask: mask
        }


class NormalizePostMethod(PostMapMethod):
    def __init__(self, divider=255):
        super().__init__()
        self.divider = tf.constant(divider, dtype=tf.float32)

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[SegmentIterator.image]

        img = tf.divide(img, self.divider)

        element[SegmentIterator.image] = img
        return element


class SqueezeMaskPostMethod(PostMapMethod):
    def __init__(self):
        super().__init__()

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        mask = element[SegmentIterator.mask]
        mask = mask[:, :, 0]
        element[SegmentIterator.mask] = mask
        return element


class ComputePositivesPostMethod(PostMapMethod):
    def __init__(self, background_class=0, dtype=tf.float32):
        super().__init__()
        self.background = tf.constant(background_class, dtype=dtype)

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)

        mask = element[SegmentIterator.mask]
        mask_shape = mask.get_shape().as_list()
        area = mask_shape[0] * mask_shape[1] # tf.int32
        num_neg = tf.reduce_sum(tf.cast(tf.equal(mask, self.background), dtype=tf.int32))

        num_positives = tf.cast(area - num_neg, dtype=tf.float32)

        element[SegmentIterator.num_positives] = num_positives
        return element
