import tensorflow as tf


def _bytes_feature(value):
    """
    TODO
    If you want to convert a tensor, you need to serialize it first.
    Use tf.io.serialize_tensor function.
    Parameters
    ----------
    value

    Returns
    -------

    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tensor_to_byte_feature(image):
    serialized_image = tf.io.serialize_tensor(image)
    return _bytes_feature(serialized_image)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

