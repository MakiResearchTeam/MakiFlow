import tensorflow as tf

LOW_MEMORY_SESS = None
FRACTION_MEMORY_SESS = None


def set_main_gpu(gpu_id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    del os


def get_low_memory_sess(new_sess=False):
    """
    Creates tf.Session which dynamically takes GPU memory. Note that taken memory won't
    be freed.

    Parameters
    ----------
    new_sess : bool
        If `new_sess == False`, the function will return previously created session.
        Set `new_sess` to True if you need a new one.

    Returns
    -------
    tf.Session
    """
    global LOW_MEMORY_SESS
    if LOW_MEMORY_SESS is None or new_sess:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        LOW_MEMORY_SESS = tf.Session(config=config)
    return LOW_MEMORY_SESS


def get_fraction_memory_sess(fraction=0.5, new_sess=False):
    """
    Creates tf.Session which takes only a given fraction of a GPU memory.

    Parameters
    ----------
    fraction : float
        How much of a GPU memory the session will take.
        Example: 0.5 - 50%, 0.7 - 70% of the memory.
    new_sess : bool
        If `new_sess == False`, the function will return previously created session.
        Set `new_sess` to True if you need a new one.

    Returns
    -------
    tf.Session
    """
    global FRACTION_MEMORY_SESS
    if FRACTION_MEMORY_SESS is None or new_sess:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        FRACTION_MEMORY_SESS = tf.Session(config=config)
    return FRACTION_MEMORY_SESS
