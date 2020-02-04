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
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tensor_to_byte_feature(tensor, sess=None):
    serialized_image = tf.io.serialize_tensor(tensor)
    if sess is not None:
        serialized_image = sess.run(serialized_image)
    elif sess is None and not tf.executing_eagerly():
        raise RuntimeError("TensorFlow is not executing eagerly. Please provide tf.Session to serialize tensors.")
    return _bytes_feature(serialized_image)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

