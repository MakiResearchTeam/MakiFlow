from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import pandas as pd
from makiflow.augmentation.segmentation.balancing.utils import hcv_to_num


def vec_len(vec):
    vec2 = vec * vec
    vec_sum = tf.reduce_sum(vec2)
    return tf.sqrt(vec_sum)


def distance(vec1, vec2):
    diff = vec1 - vec2
    return vec_len(diff)


def normalize(vec):
    vec_l = vec_len(vec)
    return vec / vec_l


# noinspection PyAttributeOutsideInit
class GDBalancer:
    def __init__(
            self, hcv_groups, initial_c, objective='alpha', min_c=5.0, max_c=10000.0
    ):
        """
        NOTE: HCVG - Has Class Vector Group.
        Parameters
        ----------
        hcv_groups : str or ndarray
            str case: path to csv file with HCVGs.
            ndarray case: array of HCVGs.
        objective : str
            The objective function of the algorithm. Options:
            - alpha - normalize LAMBA vector by the sum of HCVGs' cardinalities.
            - geo - normalize LAMBA vector as a normal vector.
        min_c : float
            Minimum cardinality.
        max_c : float
            Maximum cardinality.
        initial_c : ndarray
            Ndarray of shape (number of cardinalities). Initial cardinalities of the HCVGs.
        """
        # for GradientDescentOptimizer optimal lr=1000
        # for AdamOptimizer optimal lr=1
        self.min_amount = min_c
        self.max_amount = max_c
        self._setup_initial_values(hcv_groups)

        # Setup the algorithm
        self.initial_c = initial_c
        self.sess = None
        self.reset(initial_c, objective)

    def _setup_initial_values(self, hcv_groups):
        if isinstance(hcv_groups, str):
            # `hv_groups` is a path to config file
            df = pd.DataFrame.from_csv(hcv_groups)
            hcv_groups = df.get_values()
        self.hcv_groups = hcv_groups
        self.vecs = tf.constant(hcv_groups.T, dtype=tf.float32)
        self.dest_v = tf.placeholder(dtype=tf.float32, shape=[hcv_groups.shape[1], 1])

    # noinspection PyAttributeOutsideInit
    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.variables_initializer([self.cardinalities]))

    # noinspection PyAttributeOutsideInit
    def reset(self, initial_c=None, objective='alpha'):
        """
        Parameters
        ----------
        initial_c : ndarray
            Ndarray of shape (number of cardinalities). Initial cardinalities of the HCVGs.
        objective : str
            The objective function of the algorithm. Options:
            - alpha - normalize LAMBA vector by the sum of HCVGs' cardinalities.
            - geo - normalize LAMBA vector as a normal vector.
        """
        if initial_c is None:
            initial_c = self.initial_c

        self.cardinalities = tf.Variable(
            np.reshape(initial_c, [-1, 1]),
            trainable=True,
            # Clip the value of the weights in the interval [min_amount, max_amount]
            constraint=lambda x: tf.clip_by_value(x, self.min_amount, self.max_amount),
            dtype=tf.float32
        )
        if self.sess is not None:
            self.sess.run(tf.variables_initializer([self.cardinalities]))

        self.scaled_hcvgs = tf.matmul(self.vecs, self.cardinalities)
        self._build_objective(objective)
        self.optimizer = None

    def _build_objective(self, objective):
        if objective == 'alpha':
            norm_vecs = self.scaled_hcvgs / tf.reduce_sum(self.cardinalities)
            self.objective = distance(norm_vecs, self.dest_v)
        elif objective == 'geo':
            norm_vecs = normalize(self.scaled_hcvgs)
            self.objective = distance(norm_vecs, self.dest_v)
        else:
            print('Unknowm objective. Call `reset` with the correct one.')

    def show_deviation(self):
        return self.sess.run(self.deviation_vec)

    def add_reg1(self, alpha, initial_cardinalities):
        init_c = initial_cardinalities.reshape(-1, 1).astype(np.float32)

        ones = np.ones((len(init_c), 1), dtype=np.float32)

        self.deviation_vec = self.cardinalities / init_c - ones
        reg_loss1 = vec_len(self.deviation_vec) * vec_len(self.deviation_vec)
        self.objective = self.objective + alpha * reg_loss1

    def add_reg2(self, alpha, initial_cardinalities):
        init_c = initial_cardinalities / np.sum(initial_cardinalities)
        init_c = init_c.reshape(-1, 1).astype(np.float32)

        cur_cardinalities = self.cardinalities / tf.reduce_sum(self.cardinalities)

        self.deviation_vec = init_c - cur_cardinalities
        reg_loss2 = vec_len(self.deviation_vec) * vec_len(self.deviation_vec)
        self.objective = self.objective + alpha * reg_loss2

    def add_reg3(self, alpha, initial_cardinalities):
        init_c = initial_cardinalities / np.sum(initial_cardinalities)
        init_c = init_c.reshape(-1, 1).astype(np.float32)

        cur_cardinalities = self.cardinalities / tf.reduce_sum(self.cardinalities)

        ones = np.ones((len(init_c), 1), dtype=np.float32)

        self.deviation_vec = cur_cardinalities / init_c - ones
        reg_loss3 = vec_len(self.deviation_vec) * vec_len(self.deviation_vec)
        self.objective = self.objective + alpha * reg_loss3

    def _build_train_op(self, optimizer):
        if self.optimizer != optimizer:
            self.optimizer = optimizer
            self.train_op = self.optimizer.minimize(self.objective)
            self.sess.run(tf.variables_initializer(self.optimizer.variables()))
        return self.train_op

    def optimize(self, pi_vec, optimizer, iterations=10, print_period=5):
        """
        Perform the algorithm on the initial cardinalities.

        Parameters
        ----------
        pi_vec : ndarray
            Optimal class ratio vector. Ndarray of shape (num_classes).
            ith element in `pi_vec` stands for number of HCVs that has class i
            divided by total number of HCVs.
        optimizer : TensorFlow optimizer
            Optimizer for the algorithm.
        iterations : int
            Number of iterations to perform.
        print_period : int
            After each `print_period` iterations supplementary info will be printed.
        """
        # Do reshape since the balancer does matrix multiplication.
        pi_vec = pi_vec.reshape(-1, 1)
        for i in range(iterations):
            _, d = self.sess.run([self._build_train_op(optimizer), self.objective],
                                 feed_dict={self.dest_v: pi_vec}
                                 )
            if i % print_period == 0:
                print(d)
                print(self.get_percentage())

    def save_cardinalities(self, path):
        cardinalities = self.get_weights()
        cardinalities = cardinalities.reshape(-1)
        cardinalities = np.round(cardinalities).astype(np.int32)
        config = {}
        for i in range(len(cardinalities)):
            config[hcv_to_num(self.hcv_groups[i])] = cardinalities[i]
        pd.DataFrame.from_dict(config, orient='index').to_csv(path)

    def get_percentage(self):
        percentage = self.get_scaled_vecs() / self.get_vec_num()
        return np.round(percentage, decimals=2) * 100

    def get_weights(self):
        return self.sess.run(self.cardinalities)

    def get_scaled_vecs(self):
        return self.sess.run(self.scaled_hcvgs)

    def get_vec_num(self):
        return self.sess.run(tf.reduce_sum(self.cardinalities))
