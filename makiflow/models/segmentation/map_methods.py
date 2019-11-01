import tensorflow as tf
from makiflow.models.segmentation.gen_api import PostMapMethod, MapMethod, SegmentationGenerator


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
        img_file = tf.read_file(data_paths[SegmentationGenerator.image])
        mask_file = tf.read_file(data_paths[SegmentationGenerator.mask])

        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)

        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)

        if self.squeeze_mask:
            mask = mask[:, :, 0]

        if self.image_size is not None:
            img = tf.image.resize(images=img, method=tf.image.ResizeMethod.BILINEAR)

        if self.mask_size is not None:
            mask = tf.image.resize(images=mask, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = tf.cast(img, dtype=tf.float32)

        if self.normalize is not None:
            img = tf.divide(img, self.normalize)

        mask = tf.cast(mask, dtype=tf.int32)
        return {
            MapMethod.image: img,
            MapMethod.mask: mask
        }


class LoadDataMethod(MapMethod):
    def __init__(self, image_shape, mask_shape):
        self.image_shape = image_shape
        self.mask_shape = mask_shape

    def load_data(self, data_paths):
        img_file = tf.read_file(data_paths[SegmentationGenerator.image])
        mask_file = tf.read_file(data_paths[SegmentationGenerator.mask])

        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)

        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)

        img = tf.cast(img, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.int32)
        return {
            MapMethod.image: img,
            MapMethod.mask: mask
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
        img = element[MapMethod.image]
        mask = element[MapMethod.mask]

        if self.image_size is not None:
            img = tf.image.resize(images=img, size=self.image_size, method=self.image_resize_method)

        if self.mask_size is not None:
            mask = tf.image.resize(images=mask, size=self.mask_size, method=self.mask_resize_method)

        return {
            MapMethod.image: img,
            MapMethod.mask: mask
        }


class NormalizePostMethod(PostMapMethod):
    def __init__(self, divider=255):
        super().__init__()
        self.divider = tf.constant(divider, dtype=tf.float32)

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[MapMethod.image]

        img = tf.divide(img, self.divider)

        element[MapMethod.image] = img
        return element


class SqueezeMaskPostMethod(PostMapMethod):
    def __init__(self):
        super().__init__()

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        mask = element[MapMethod.mask]
        mask = mask[:, :, 0]
        element[MapMethod.mask] = mask
        return element


class ComputePositivesPostMethod(PostMapMethod):
    def __init__(self, background_class=0, dtype=tf.float32):
        super().__init__()
        self.background = tf.constant(background_class, dtype=dtype)

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)

        mask = element[MapMethod.mask]
        mask_shape = mask.get_shape().as_list()
        area = mask_shape[1] * mask_shape[2]
        num_neg = tf.reduce_sum(tf.cast(tf.equal(mask, self.background), dtype=tf.float32))

        num_positives = area - num_neg

        element[MapMethod.num_positives] = num_positives
        return element
