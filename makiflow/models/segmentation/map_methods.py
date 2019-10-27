import tensorflow as tf
from makiflow.models.segmentation.gen_api import MapMethod, SegmentationGenerator


class LoadResizeNormalize(MapMethod):
    def __init__(self, image_shape, mask_shape, normalize=255., image_size=None, mask_size=None, squeeze_mask=False):
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.image_size = image_size
        self.mask_size = mask_size
        self.squeeze_mask = squeeze_mask
        self.normalize = tf.constant(normalize, dtype=tf.float32)

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


