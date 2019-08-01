import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    X = tf.placeholder(dtype=tf.float32, shape=[32, 9])

    def scan_fn(_, x):
        return tf.reduce_mean(x)

    res = tf.scan(
        elems=X,
        fn=scan_fn,
        infer_shape=False,
        initializer=1.0
    )

    x = np.ones((32, 9))*2
    sess = tf.Session()
    print(sess.run(
        res,
        feed_dict={X:x}
    ))
