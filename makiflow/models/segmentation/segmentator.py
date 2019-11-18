from __future__ import absolute_import
from makiflow.base import MakiModel, MakiTensor
from makiflow.models.segmentation.gen_base import SegmentIterator
from makiflow.layers import InputLayer
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm

from scipy.special import binom


class Segmentator(MakiModel):
    def __init__(self, input_s: InputLayer, output: MakiTensor, name='MakiSegmentator'):
        self.name = str(name)
        graph_tensors = output.get_previous_tensors()
        graph_tensors.update(output.get_self_pair())
        super().__init__(graph_tensors, outputs=[output], inputs=[input_s])
        self._training_vars_are_ready = False

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            'name': self.name,
            'input_s': self._inputs[0].get_name(),
            'output': self._outputs[0].get_name()
        }

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------SETTING UP TRAINING-----------------------------------------

    # noinspection PyAttributeOutsideInit
    def set_generator(self, generator):
        self._generator = generator
        if not self._set_for_training:
            super()._setup_for_training()
        if not self._training_vars_are_ready:
            self._prepare_training_vars(use_generator=True)

    def _prepare_training_vars(self, use_generator=False):
        out_shape = self._outputs[0].get_shape()
        self.out_w = out_shape[1]
        self.out_h = out_shape[2]
        self.total_predictions = out_shape[1] * out_shape[2]
        self.num_classes = out_shape[-1]
        self.batch_sz = out_shape[0]

        # If generator is used, then the input data tensor will by an image tensor
        # produced by the generator, since it's the input layer.
        self._images = self._input_data_tensors[0]
        if use_generator:
            self._labels = self._generator.get_iterator()[SegmentIterator.mask]
        else:
            self._labels = tf.placeholder(tf.int32, shape=out_shape[:-1], name='labels')

        training_out = self._training_outputs[0]
        self._flattened_logits = tf.reshape(training_out, shape=[-1, self.total_predictions, self.num_classes])
        self._flattened_labels = tf.reshape(self._labels, shape=[-1, self.total_predictions])

        self._ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._flattened_logits, labels=self._flattened_labels
        )

        self._training_vars_are_ready = True
        self._use_generator = use_generator

        self._focal_loss_is_build = False
        self._weighted_focal_loss_is_build = False
        self._weighted_ce_loss_is_build = False
        self._maki_loss_is_build = False
        self._quadratic_ce_loss_is_build = False

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------FOCAL LOSS--------------------------------------------------

    def _build_focal_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self._flattened_labels, depth=self.num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, self._focal_gamma)
        num_positives = tf.reduce_sum(self._focal_num_positives)
        self._focal_loss = tf.reduce_sum(focal_weights * self._ce_loss) / num_positives
        self._final_weighted_focal_loss = self._build_final_loss(self._focal_loss)
        self._focal_loss_is_build = True

    def _setup_focal_loss_inputs(self):
        self._focal_gamma = tf.placeholder(tf.float32, shape=[], name='gamma')
        if self._use_generator:
            self._focal_num_positives = self._generator.get_iterator()[SegmentIterator.num_positives]
        else:
            self._focal_num_positives = tf.placeholder(tf.float32, shape=[self.batch_sz], name='num_positives')

    def _minimize_focal_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._focal_loss_is_build:
            self._setup_focal_loss_inputs()
            self._build_focal_loss()
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._focal_optimizer != optimizer:
            print('New optimizer is used.')
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._focal_train_op

    def fit_focal(self, images, labels, gamma, num_positives, optimizer, epochs=1, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        gamma : int
            Hyper parameter for FocalLoss.
        num_positives : list
            List of ints. Contains number of `positive samples` per image. `Positive sample` is a pixel
            that is responsible for a class other that background.
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_focal_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        train_focal_losses = []
        iterator = None
        try:
            for i in range(epochs):
                images, labels = shuffle(images, labels)
                focal_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    NPbatch = num_positives[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_focal_loss, _ = self._session.run(
                        [self._focal_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch,
                            self._focal_gamma: gamma,
                            self._focal_num_positives: NPbatch
                        })
                    # Use exponential decay for calculating loss and error
                    focal_loss = 0.1*batch_focal_loss + 0.9*focal_loss

                train_focal_losses.append(focal_loss)

                print('Epoch:', i, 'Focal loss: {:0.4f}'.format(focal_loss))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train losses': train_focal_losses}

    def genfit_focal(self, gamma, optimizer, epochs=1, iterations=10, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        gamma : int
            Hyper parameter for MakiLoss.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        iterations : int
            Defines how ones epoch is. One operation is a forward pass
            using one batch.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_focal_loss(optimizer, global_step)

        train_focal_losses = []
        iterator = None
        try:
            for i in range(epochs):
                focal_loss = 0
                iterator = tqdm(range(iterations))

                for _ in iterator:
                    batch_focal_loss, _ = self._session.run(
                        [self._focal_loss, train_op],
                        feed_dict={
                            self._focal_gamma: gamma
                        })
                    # Use exponential decay for calculating loss and error
                    focal_loss = 0.1*batch_focal_loss + 0.9*focal_loss

                train_focal_losses.append(focal_loss)

                print('Epoch:', i, 'Focal loss: {:0.4f}'.format(float(focal_loss)))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train losses': train_focal_losses}

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------MAKI LOSS---------------------------------------------------

    def _create_maki_polynom_part(self, k, sparse_confidences):
        binomial_coeff = binom(self._maki_gamma, k)
        powered_p = tf.pow(-1.0 * sparse_confidences, k)
        return binomial_coeff * powered_p / (1.0 * k)

    def _build_maki_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self._flattened_labels, depth=self.num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        # Create Maki polynomial
        maki_polynomial = tf.constant(0.0)
        for k in range(1, self._maki_gamma+1):
            # Do subtraction because gradient must be with minus as well
            # Maki loss grad: -(1 - p)^gamma / p
            # CE loss grad: - 1 / p
            maki_polynomial -= self._create_maki_polynom_part(k, sparse_confidences) - \
                self._create_maki_polynom_part(k, tf.ones_like(sparse_confidences))

        num_positives = tf.reduce_sum(self._maki_num_positives)
        self._maki_loss = tf.reduce_sum(maki_polynomial + self._ce_loss) / num_positives
        self._final_weighted_maki_loss = self._build_final_loss(self._maki_loss)
        self._maki_loss_is_build = True

    def _setup_maki_loss_inputs(self):
        self._maki_gamma = None
        if self._use_generator:
            self._maki_num_positives = self._generator.get_iterator()[SegmentIterator.num_positives]
        else:
            self._maki_num_positives = tf.placeholder(tf.float32, shape=[self.batch_sz], name='num_positives')

    def _minimize_maki_loss(self, optimizer, global_step, gamma):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._maki_loss_is_build:
            self._setup_maki_loss_inputs()
            self._maki_gamma = gamma
            self._build_maki_loss()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_weighted_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._maki_gamma != gamma:
            print('New gamma is used.')
            self._maki_gamma = gamma
            self._build_maki_loss()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_weighted_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._maki_optimizer != optimizer:
            print('New optimizer is used.')
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_weighted_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._maki_train_op

    def fit_maki(self, images, labels, gamma: int, num_positives, optimizer, epochs=1, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        gamma : int
            Hyper parameter for MakiLoss.
        num_positives : list
            List of ints. Contains number of `positive samples` per image. `Positive sample` is a pixel
            that is responsible for a class other that background.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)
        assert (type(gamma) == int)

        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)

        n_batches = len(images) // self.batch_sz
        train_focal_losses = []
        iterator = None
        try:
            for i in range(epochs):
                images, labels = shuffle(images, labels)
                focal_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    NPbatch = num_positives[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_maki_loss, _ = self._session.run(
                        [self._final_weighted_maki_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch,
                            self._maki_num_positives: NPbatch
                        })
                    # Use exponential decay for calculating loss and error
                    focal_loss = 0.1 * batch_maki_loss + 0.9 * focal_loss

                train_focal_losses.append(focal_loss)

                print('Epoch:', i, 'Focal loss: {:0.4f}'.format(focal_loss))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train losses': train_focal_losses}

    def genfit_maki(self, gamma: int, optimizer, epochs=1, iterations=10, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        gamma : int
            Hyper parameter for MakiLoss.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        iterations : int
            Defines how ones epoch is. One operation is a forward pass
            using one batch.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)
        assert (type(gamma) == int)

        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)

        train_maki_losses = []
        iterator = None
        try:
            for i in range(epochs):
                maki_loss = 0
                iterator = tqdm(range(iterations))

                for _ in iterator:
                    batch_maki_loss, _ = self._session.run(
                        [self._final_weighted_maki_loss, train_op]
                    )
                    # Use exponential decay for calculating loss and error
                    maki_loss = 0.1 * batch_maki_loss + 0.9 * maki_loss

                train_maki_losses.append(maki_loss)

                print('Epoch:', i, 'Maki loss: {:0.4f}'.format(maki_loss))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train losses': train_maki_losses}

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------WEIGHTED FOCAL LOSS-----------------------------------------

    def _build_weighted_focal_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self._flattened_labels, depth=self.num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, self._weighted_focal_gamma)
        flattened_weights = tf.reshape(
            self._weighted_focal_weight_maps, shape=[-1, self.total_predictions]
        )
        num_positives = tf.reduce_sum(self._weighted_focal_num_positives)
        self._weighted_focal_loss = tf.reduce_sum(flattened_weights * focal_weights * self._ce_loss) / num_positives
        self._final_weighted_focal_loss = self._build_final_loss(self._weighted_focal_loss)

        self._weighted_focal_loss_is_build = True

    def _setup_weighted_focal_loss_inputs(self):
        self._weighted_focal_gamma = tf.placeholder(tf.float32, shape=[], name='gamma')
        self._weighted_focal_num_positives = tf.placeholder(tf.float32, shape=[self.batch_sz], name='num_positives')
        self._weighted_focal_weight_maps = tf.placeholder(
            tf.float32, shape=[self.batch_sz, self.out_w, self.out_h], name='weighted_focal_weight_map'
        )

    def _minimize_weighted_focal_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._weighted_focal_loss_is_build:
            self._setup_weighted_focal_loss_inputs()
            self._build_weighted_focal_loss()
            self._weighted_focal_optimizer = optimizer
            self._weighted_focal_train_op = optimizer.minimize(
                self._final_weighted_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._weighted_focal_optimizer != optimizer:
            print('New optimizer is used.')
            self._weighted_focal_optimizer = optimizer
            self._weighted_focal_train_op = optimizer.minimize(
                self._final_weighted_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._weighted_focal_train_op

    def fit_weighted_focal(
            self, images, labels, gamma, num_positives, weight_maps, optimizer, epochs=1, global_step=None
    ):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        gamma : int
            Hyper parameter for FocalLoss.
        num_positives : list
            List of ints. Contains number of `positive samples` per image. `Positive sample` is a pixel
            that is responsible for a class other that background.
        weight_maps : list
            Maps for weighting the loss.
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_weighted_focal_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        train_total_losses = []
        train_focal_losses = []
        iterator = None
        try:
            for i in range(epochs):
                images, labels, num_positives, weight_maps = shuffle(images, labels, num_positives, weight_maps)
                total_loss = 0
                focal_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    NPbatch = num_positives[j * self.batch_sz:(j + 1) * self.batch_sz]
                    WMbatch = weight_maps[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_total_loss, batch_focal_loss, _ = self._session.run(
                        [self._final_weighted_focal_loss, self._weighted_focal_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch,
                            self._weighted_focal_gamma: gamma,
                            self._weighted_focal_num_positives: NPbatch,
                            self._weighted_focal_weight_maps: WMbatch
                        })
                    # Use exponential decay for calculating loss and error
                    total_loss = 0.1*batch_total_loss + 0.9*total_loss
                    focal_loss = 0.1*batch_focal_loss + 0.9*focal_loss

                train_total_losses.append(total_loss)
                train_focal_losses.append(focal_loss)

                print(
                    'Epoch:', i,
                    'Total loss: {:0.4f}'.format(total_loss),
                    'Focal loss: {:0.4f}'.format(focal_loss)
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'total losses': train_total_losses,
                'focal losses': train_focal_losses
            }

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------WEIGHTED CROSSENTROPY LOSS----------------------------------

    def _build_weighted_ce_loss(self):
        # [batch_sz, total_predictions, num_classes]
        flattened_weights = tf.reshape(
            self._weighted_ce_weight_maps, shape=[-1, self.total_predictions]
        )
        self._weighted_ce_loss = tf.reduce_mean(self._ce_loss * flattened_weights)
        self._final_weighted_ce_loss = self._build_final_loss(self._weighted_ce_loss)
        self._weighted_ce_loss_is_build = True

    def _setup_weighted_ce_loss_inputs(self):
        self._weighted_ce_weight_maps = tf.placeholder(
            tf.float32, shape=[self.batch_sz, self.out_w, self.out_h], name='ce_weight_map'
        )

    def _minimize_weighted_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._weighted_ce_loss_is_build:
            self._setup_weighted_ce_loss_inputs()
            self._build_weighted_ce_loss()
            self._weighted_ce_optimizer = optimizer
            self._weighted_ce_train_op = optimizer.minimize(
                self._final_weighted_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._weighted_ce_optimizer != optimizer:
            print('New optimizer is used.')
            self._weighted_ce_optimizer = optimizer
            self._weighted_ce_train_op = optimizer.minimize(
                self._final_weighted_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._weighted_ce_train_op

    def fit_weighted_ce(
            self, images, labels, weight_maps, optimizer, epochs=1, global_step=None
    ):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        weight_maps : list
            Maps for weighting the loss.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
            python dictionary
                Dictionary with all testing data(train error, train cost, test error, test cost)
                for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_weighted_ce_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        iterator = None
        train_total_losses = []
        train_weighted_ce_losses = []
        try:
            for i in range(epochs):
                images, labels, weight_maps = shuffle(images, labels, weight_maps)
                total_loss = 0
                weighted_ce_loss = 0
                iterator = tqdm(range(n_batches))
                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    WMbatch = weight_maps[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_weighted_ce_loss, batch_total_loss, _ = self._session.run(
                        [self._final_weighted_ce_loss, self._weighted_ce_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch,
                            self._weighted_ce_weight_maps: WMbatch
                        }
                    )
                    # Use exponential decay for calculating loss and error
                    total_loss = 0.1*batch_total_loss + 0.9*total_loss
                    weighted_ce_loss = 0.1*batch_weighted_ce_loss + 0.9*weighted_ce_loss

                train_total_losses.append(total_loss)
                train_weighted_ce_losses.append(weighted_ce_loss)
                print(
                    'Epoch:', i,
                    'Total loss: {:0.4f}'.format(total_loss),
                    'CE loss: {:0.4f}'.format(weighted_ce_loss)
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'total losses': train_total_losses,
                'ce losses': train_weighted_ce_losses
            }

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------QUADRATIC CROSSENTROPY LOSS---------------------------------

    def _build_quadratic_ce_loss(self):
        # [batch_sz, total_predictions, num_classes]
        quadratic_ce = self._ce_loss * self._ce_loss / 2.0
        self._quadratic_ce = tf.reduce_mean(quadratic_ce)
        self._final_quadratic_ce_loss = self._build_final_loss(self._quadratic_ce)
        self._quadratic_ce_loss_is_build = True

    def _setup_quadratic_ce_loss_inputs(self):
        pass

    def _minimize_quadratic_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._quadratic_ce_loss_is_build:
            self._setup_quadratic_ce_loss_inputs()
            self._build_quadratic_ce_loss()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._quadratic_ce_optimizer != optimizer:
            print('New optimizer is used.')
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._quadratic_ce_train_op

    def fit_quadratic_ce(
            self, images, labels, optimizer, epochs=1, global_step=None
    ):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        labels : list
            Training masks.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        iterator = None
        train_total_losses = []
        train_quadratic_ce_losses = []
        try:
            for i in range(epochs):
                images, labels = shuffle(images, labels)
                total_loss = 0
                quadratic_ce_loss = 0
                iterator = tqdm(range(n_batches))
                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_quadratic_ce_loss, batch_total_loss, _ = self._session.run(
                        [self._final_quadratic_ce_loss, self._quadratic_ce, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._labels: Lbatch
                        }
                    )
                    # Use exponential decay for calculating loss and error
                    total_loss = 0.1*batch_total_loss + 0.9*total_loss
                    quadratic_ce_loss = 0.1*batch_quadratic_ce_loss + 0.9*quadratic_ce_loss

                train_total_losses.append(total_loss)
                train_quadratic_ce_losses.append(quadratic_ce_loss)
                print(
                    'Epoch:', i,
                    'Total loss: {:0.4f}'.format(total_loss),
                    'CE loss: {:0.4f}'.format(quadratic_ce_loss)
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'train losses': train_total_losses,
                'qudratic ce losses': train_quadratic_ce_losses
            }

    def genfit_quadratic_ce(
            self, optimizer, epochs=1, iterations=10, global_step=None
    ):
        """
        Method for training the model using generator.

        Parameters
        ----------
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        iterations : int
            Defines how ones epoch is. One operation is a forward pass
            using one batch.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        iterator = None
        train_total_losses = []
        train_quadratic_ce_losses = []
        try:
            for i in range(epochs):
                total_loss = 0
                quadratic_ce_loss = 0
                iterator = tqdm(range(iterations))
                for _ in iterator:
                    batch_quadratic_ce_loss, batch_total_loss, _ = self._session.run(
                        [self._final_quadratic_ce_loss, self._quadratic_ce, train_op]
                    )
                    # Use exponential decay for calculating loss and error
                    total_loss = 0.1*batch_total_loss + 0.9*total_loss
                    quadratic_ce_loss = 0.1*batch_quadratic_ce_loss + 0.9*quadratic_ce_loss

                train_total_losses.append(total_loss)
                train_quadratic_ce_losses.append(quadratic_ce_loss)
                print(
                    'Epoch:', i,
                    'Total loss: {:0.4f}'.format(total_loss),
                    'CE loss: {:0.4f}'.format(quadratic_ce_loss)
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'train losses': train_total_losses,
                'qudratic ce losses': train_quadratic_ce_losses
            }
