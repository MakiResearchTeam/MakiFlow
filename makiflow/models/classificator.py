from __future__ import absolute_import
from makiflow.base import MakiModel, MakiTensor
from makiflow.layers import InputLayer
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from makiflow.utils import error_rate, sparse_cross_entropy
from copy import copy

EPSILON = np.float32(1e-37)


class Classificator(MakiModel):

	def __init__(self, input: InputLayer, output: MakiTensor, name='MakiClassificator'):
		graph_tensors = copy(output.get_previous_tensors())
		# Add output tensor to `graph_tensors` since it doesn't have it.
		# It is assumed that graph_tensors contains ALL THE TENSORS graph consists of.
		graph_tensors.update(output.get_self_pair())
		outputs = [output]
		inputs = [input]
		super().__init__(graph_tensors, outputs, inputs)
		self.name = str(name)
		self.__batch_sz = input.get_shape()[0]
		self.__input = self._input_data_tensors[0]
		self.__inference_out = self._output_data_tensors[0]
		# For training
		self.__training_vars_are_ready = False

	def __prepare_training_vars(self):
		if not self._set_for_training:
			super()._setup_for_training()
		self.__training_out = self._training_outputs[0]
		self.__labels = tf.placeholder(tf.int32, shape=[self.__batch_sz])
		self.__ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.__training_out, labels=self.__labels
		)
		self.__training_vars_are_ready = True

	def _get_model_info(self):
		input_mt = self._inputs[0]
		output_mt = self._outputs[0]
		return {
			'input_mt': input_mt.get_name(),
			'output_mt': output_mt.get_name(),
			'name': self.name
		}

	def __build_ce_loss(self):
        # [batch_sz, total_predictions, num_classes]
		ce_loss = tf.reduce_mean(self.__ce_loss)
		self.__final_ce_loss = self._build_final_loss(ce_loss)

		self.__ce_loss_is_build = True

	def __minimize_ce_loss(self, optimizer):
		if not self._set_for_training:
			super()._setup_for_training()

		if not self.__training_vars_are_ready:
			self.__prepare_training_vars()
		
		if not self.__ce_loss_is_build:
			# no need to setup any inputs for this loss
			self.__build_ce_loss()
			self.__ce_optimizer = optimizer
			self.__ce_train_op = optimizer.minimize(self.__final_ce_loss, var_list=self._trainable_vars)
			self._session.run(tf.variables_initializer(optimizer.variables()))

		if self.__ce_optimizer != optimizer:
			print('New optimizer is used.')
			self.__ce_optimizer = optimizer
			self.__ce_train_op = optimizer.minimize(self.__final_ce_loss, var_list=self._trainable_vars)
			self._session.run(tf.variables_initializer(optimizer.variables()))

		return self.__ce_train_op

	def pure_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
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
		
		# For training
		cost = (
			tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.__training_out, labels=self.__labels)
			),
			self.__training_out
		)
		train_op = (cost, optimizer.minimize(cost[0], var_list=self._trainable_vars) )
		# Initialize optimizer's variables
		self._session.run(tf.variables_initializer(optimizer.variables()))

		# For testing
		Yish_test = tf.nn.softmax(self.__inference_out)

		n_batches = Xtrain.shape[0] // self.__batch_sz

		train_costs = []
		train_errors = []
		test_costs = []
		test_errors = []
		iterator = None
		try:
			for i in range(epochs):
				Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
				train_cost = np.float32(0)
				train_error = np.float32(0)
				iterator = tqdm(range(n_batches))

				for j in iterator:
					Xbatch = Xtrain[j * self.__batch_sz:(j + 1) * self.__batch_sz]
					Ybatch = Ytrain[j * self.__batch_sz:(j + 1) * self.__batch_sz]
					(train_cost_batch, y_ish), _ = self._session.run(
						train_op,
						feed_dict={self.__input: Xbatch, self.__labels: Ybatch})
					# Use exponential decay for calculating loss and error
					train_cost = 0.99 * train_cost + 0.01 * train_cost_batch
					train_error_batch = error_rate(np.argmax(y_ish, axis=1), Ybatch)
					train_error = 0.99 * train_error + 0.01 * train_error_batch

				# Validating the network on test data
				if i % test_period == 0:
					# For test data
					test_cost = np.float32(0)
					test_predictions = np.zeros(len(Xtest))

					for k in range(len(Xtest) // self.__batch_sz):
						# Test data
						Xtestbatch = Xtest[k * self.__batch_sz:(k + 1) * self.__batch_sz]
						Ytestbatch = Ytest[k * self.__batch_sz:(k + 1) * self.__batch_sz]
						Yish_test_done = self._session.run(Yish_test, feed_dict={self.__input: Xtestbatch}) + EPSILON
						test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
						test_predictions[k * self.__batch_sz:(k + 1) * self.__batch_sz] = np.argmax(Yish_test_done, axis=1)

					# Collect and print data
					test_cost = test_cost / (len(Xtest) // self.__batch_sz)
					test_error = error_rate(test_predictions, Ytest)
					test_errors.append(test_error)
					test_costs.append(test_cost)

					train_costs.append(train_cost)
					train_errors.append(train_error)

					print('Epoch:', i, 'Train accuracy: {:0.4f}'.format(1 - train_error),
							'Train cost: {:0.4f}'.format(train_cost),
							'Test accuracy: {:0.4f}'.format(1 - test_error), 'Test cost: {:0.4f}'.format(test_cost))
		except Exception as ex:
			print(ex)
		finally:
			if iterator is not None:
				iterator.close()
			return {'train costs': train_costs, 'train errors': train_errors,
					'test costs': test_costs, 'test errors': test_errors}

	def evaluate(self, Xtest, Ytest,batch_sz):
		# TODO: for test can be delete
		# Validating the network
		Xtest = Xtest.astype(np.float32)
		Yish_test = tf.nn.softmax(self.__inference_out)
		n_batches = Xtest.shape[0] // batch_sz

		# For train data
		test_cost = 0
		predictions = np.zeros(len(Xtest))
		for k in tqdm(range(n_batches)):
			# Test data
			Xtestbatch = Xtest[k * batch_sz:(k + 1) * batch_sz]
			Ytestbatch = Ytest[k * batch_sz:(k + 1) * batch_sz]
			Yish_test_done = self._session.run(Yish_test, feed_dict={self.__input: Xtestbatch}) + EPSILON
			test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
			predictions[k * batch_sz:(k + 1) * batch_sz] = np.argmax(Yish_test_done, axis=1)

		error = error_rate(predictions, Ytest)
		test_cost = test_cost / (len(Xtest) // batch_sz)
		print('Accuracy:', 1 - error, 'Cost:', test_cost)


