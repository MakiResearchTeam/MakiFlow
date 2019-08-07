from __future__ import absolute_import
from makiflow.base import MakiModel, MakiTensor
from makiflow.layers import InputLayer
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class Segmentator(MakiModel):
    def __init__(self, input_s: InputLayer, output: MakiTensor, name='MakiSegmentator'):
        graph_tensors = output.get_previous_tensors()
        graph_tensors.update(output.get_self_pair())
        super().__init__(graph_tensors, outputs=[output], inputs=[input_s])
        self.__training_vars_are_ready = False

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        # TODO
        pass

#-------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------SETTING UP TRAINING--------------------------------------------------------------------

    def __prepare_training_vars(self):
        out_shape = self._outputs[0].get_shape()
        self.out_w = out_shape[1]
        self.out_h = out_shape[2]
        self.total_predictions = out_shape[1] * out_shape[2]
        self.num_classes = out_shape[-1]
        self.batch_sz = out_shape[0]

        self.__labels = tf.placeholder(tf.int32, shape=out_shape[:-1], name='labels')

        training_out = self._training_outputs[0]
        self.__flattened_logits = tf.reshape(training_out, shape=[-1, self.total_predictions, self.num_classes])
        self.__flattened_labels = tf.reshape(self._labels, shape=[-1, self.total_predictions])
        
        self.__ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.__flattened_logits, labels=self.__flattened_labels
        )

        self.__training_vars_are_ready = True

        self.__focal_loss_is_build = False

#-------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------FOCAL LOSS-----------------------------------------------------------------------------

    def __build_focal_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self.__flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self.__flattened_labels, depth=self.num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, self.__gamma)
        num_positives = tf.reduce_sum(self.__num_positives)
        self.__focal_loss = tf.reduce_sum(focal_weights * self.__ce_loss) / num_positives

        self.__focal_loss_is_build = True

    def __setup_focal_loss_inputs(self):        
        self.__gamma = tf.placeholder(tf.float32, shape=[], name='gamma')
        self.__num_positives = tf.placeholder(tf.float32, shape=[self.batch_sz], name='num_positives')

    def __minimize_focal_loss(self, optimizer):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self.__training_vars_are_ready:
            self.__prepare_training_vars()
        
        if not self.__focal_loss_is_build:
            self.__setup_focal_loss_inputs()
            self.__build_focal_loss()
            self.__focal_optimizer = optimizer
            self.__focal_train_op = optimizer.minimize(self.__focal_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self.__focal_optimizer != optimizer:
            self.__focal_optimizer = optimizer
            self.__focal_train_op = optimizer.minimize(self.__focal_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._focal_train_op
        
    def fit_focal(self, images, labels, gamma, num_positives, optimizer, epochs=1, test_period=1):
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
        if not self._set_for_training:
            self._setup_for_training()
        # This is for correct working of tqdm loop. After KeyboardInterrupt it breaks and
        # starts to print progress bar each time it updates.
        # In order to avoid this problem we handle KeyboardInterrupt exception and close
        # the iterator tqdm iterates through manually. Yes, it's ugly, but necessary for
        # convenient working with MakiFlow in Jupyter Notebook. Sometimes it's helpful
        # even for console applications.
        
        train_op = self.__minimize_focal_loss(optimizer)

        n_batches = len(images) // self.__batch_sz
        train_losses = []
        iterator = None
        try:
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                train_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.__batch_sz:(j + 1) * self.__batch_sz]
                    Lbatch = labels[j * self.__batch_sz:(j + 1) * self.__batch_sz]
                    NPbatch = num_positives[j * self.__batch_sz:(j + 1) * self.__batch_sz]
                    batch_loss, _ = self._session.run(
                        [self._focal_loss, train_op],
                        feed_dict={
                            self.__input: Ibatch,
                            self.__labels: Lbatch,
                            self.__gamma: gamma,
                            self.__num_positives: NPbatch
                            })
                    # Use exponential decay for calculating loss and error
                    train_loss += batch_loss
                
                train_loss /= n_batches

                train_losses.append(train_loss)

                print('Epoch:', i, 'Focal loss: {:0.4f}'.format(train_loss))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_losses}

#-------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------WEIGHTED CROSSENTROPY LOSS-------------------------------------------------------------

    def __build_weighted_ce_loss(self):
        # [batch_sz, total_predictions, num_classes]
        flattened_weights = tf.reshape(
            self.__weighted_ce_weight_maps, shape=[-1, self.total_predictions]
        )
        self.__weighted_ce_loss = tf.reduce_mean(self.__ce_loss * flattened_weights)

        self.__weighted_ce_loss_is_build = True

    def __setup_weghted_ce_loss_inputs(self):        
        self.__weighted_ce_weight_maps = tf.placeholder(tf.float32, shape=[self.batch_sz, self.out_w, self.out_h], name='ce weight map')

    def __minimize_weighted_ce_loss(self, optimizer):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self.__training_vars_are_ready:
            self.__prepare_training_vars()
        
        if not self.__focal_loss_is_build:
            self.__setup_focal_loss_inputs()
            self.__build_focal_loss()
            self.__focal_optimizer = optimizer
            self.__focal_train_op = optimizer.minimize(self.__focal_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self.__focal_optimizer != optimizer:
            self.__focal_optimizer = optimizer
            self.__focal_train_op = optimizer.minimize(self.__focal_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._focal_train_op
    
    def fit_weighted_ce(self, images, labels, weight_maps, optimizer, epochs=1, test_period=1):
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
        
        train_op = self.__minimize_weighted_ce_loss(optimizer)

        n_batches = len(images) // self.__batch_sz
        train_losses = []
        iterator = None
        try:
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                train_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.__batch_sz:(j + 1) * self.__batch_sz]
                    Lbatch = labels[j * self.__batch_sz:(j + 1) * self.__batch_sz]
                    NPbatch = num_positives[j * self.__batch_sz:(j + 1) * self.__batch_sz]
                    batch_loss, _ = self._session.run(
                        [self._focal_loss, train_op],
                        feed_dict={
                            self.__input: Ibatch,
                            self.__labels: Lbatch,
                            self.__gamma: gamma,
                            self.__num_positives: NPbatch
                            })
                    # Use exponential decay for calculating loss and error
                    train_loss += batch_loss
                
                train_loss /= n_batches

                train_losses.append(train_loss)

                print('Epoch:', i, 'Focal loss: {:0.4f}'.format(train_loss))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_losses}

