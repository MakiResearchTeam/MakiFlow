import tensorflow as tf

LOW_MEMORY_SESS = None
FRACTION_MEMORY_SESS = None


def set_main_gpu(gpu_id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    del os


def get_low_memory_sess(new_sess=False):
    global LOW_MEMORY_SESS
    if LOW_MEMORY_SESS is None or new_sess:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        LOW_MEMORY_SESS = tf.Session(config=config)
    return LOW_MEMORY_SESS


def get_fraction_memory_sess(fraction=0.5, new_sess=False):
    global FRACTION_MEMORY_SESS
    if FRACTION_MEMORY_SESS is None or new_sess:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        FRACTION_MEMORY_SESS = tf.Session(config=config)
    return FRACTION_MEMORY_SESS
