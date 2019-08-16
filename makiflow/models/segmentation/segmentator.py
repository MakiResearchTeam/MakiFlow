from __future__ import absolute_import
from makiflow.base import MakiModel, MakiTensor
from makiflow.layers import InputLayer
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm


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

    def _prepare_training_vars(self):
        out_shape = self._outputs[0].get_shape()
        self.out_w = out_shape[1]
        self.out_h = out_shape[2]
        self.total_predictions = out_shape[1] * out_shape[2]
        self.num_classes = out_shape[-1]
        self.batch_sz = out_shape[0]

        self._images = self._input_data_tensors[0]
        self._labels = tf.placeholder(tf.int32, shape=out_shape[:-1], name='labels')

        training_out = self._training_outputs[0]
        self._flattened_logits = tf.reshape(training_out, shape=[-1, self.total_predictions, self.num_classes])
        self._flattened_labels = tf.reshape(self._labels, shape=[-1, self.total_predictions])

        self._ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._flattened_logits, labels=self._flattened_labels
        )

        self._training_vars_are_ready = True

        self._focal_loss_is_build = False
        self._weighted_focal_loss_is_build = False
        self._weighted_ce_loss_is_build = False

    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------FOCAL LOSS-----------------------------------------------------------------------------

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
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error mesurement.

        Parameters
        ----------
        images : numpy array
            Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
            labels : numpy array
            Training label for each image in `Xtrain` array with shape (num_images).
            IMPORTANT: ALL LABELS MUST BE NOT ONE-HOT ENCODED, USE SPARSE TRAINING DATA INSTEAD.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        test_period : int
            Test begins each `test_period` epochs. You can set a larger number in order to
            speed up training.

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
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error mesurement.

        Parameters
        ----------
            Xtrain : numpy array
                Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
            Ytrain : numpy array
                Training label for each image in `Xtrain` array with shape (num_images).
                IMPORTANT: ALL LABELS MUST BE NOT ONE-HOT ENCODED, USE SPARSE TRAINING DATA INSTEAD.
            Xtest : numpy array
                Same as `Xtrain` but for testing.
            Ytest : numpy array
                Same as `Ytrain` but for testing.
            optimizer : tensorflow optimizer
                Model uses tensorflow optimizers in order train itself.
            epochs : int
                Number of epochs.
            test_period : int
                Test begins each `test_period` epochs. You can set a larger number in order to
                speed up training.

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
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error mesurement.

        Parameters
        ----------
            Xtrain : numpy array
                Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
            Ytrain : numpy array
                Training label for each image in `Xtrain` array with shape (num_images).
                IMPORTANT: ALL LABELS MUST BE NOT ONE-HOT ENCODED, USE SPARSE TRAINING DATA INSTEAD.
            Xtest : numpy array
                Same as `Xtrain` but for testing.
            Ytest : numpy array
                Same as `Ytrain` but for testing.
            optimizer : tensorflow optimizer
                Model uses tensorflow optimizers in order train itself.
            epochs : int
                Number of epochs.
            test_period : int
                Test begins each `test_period` epochs. You can set a larger number in order to
                speed up training.

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
